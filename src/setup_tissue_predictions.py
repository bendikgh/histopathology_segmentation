import cv2
import os
import json
import torch
import sys
import numpy as np
import torch.nn.functional as F

sys.path.append(os.getcwd())

from src.utils.constants import IDUN_OCELOT_DATA_PATH as data_dir
from src.utils.constants import CELL_IMAGE_MEAN, CELL_IMAGE_STD
from src.utils.utils import (
    get_torch_image,
    crop_and_resize_tissue_patch,
    crop_and_resize_tissue_faster,
    get_metadata_dict,
)
from src.models import DeepLabV3plusModel
from src.trainable import SegformerTissueTrainable


def create_cropped_tissue_predictions(
    model_tissue: torch.nn.Module,
    partition: str,
    output_folder_name: str,
    device: torch.device = torch.device("cuda"),
    data_dir: str = data_dir,
) -> None:
    """
    Reads tissue images and processes them using a tissue prediction model.
    Saves the results to file.

    Args:
        model_tissue (torch.nn.Module): The deep learning model to apply to each tissue image.
        partition (str): The dataset partition to process, e.g., "train", "test", or "validate".
        device (torch.device): The computing device (CUDA or CPU) on which the model will run.
        ocelot_data_path (str): The base path to the dataset and metadata.
    """

    model_tissue.to(device)
    model_tissue.eval()

    # Getting the correct paths
    tissue_cropped_prediction_save_path = os.path.join(
        data_dir, "predictions", partition, output_folder_name
    )
    tissue_path = os.path.join(data_dir, "images", partition, "tissue_macenko")
    tissue_files = sorted(
        [os.path.join(tissue_path, path) for path in os.listdir(tissue_path)]
    )
    metadata = get_metadata_dict(data_dir)

    # For each image, create a tissue crop and save the cropped version with the same name
    for path in tissue_files:
        image_num = path.split("/")[-1].split(".")[0]

        sample_metadata = metadata["sample_pairs"][image_num]
        x_offset = sample_metadata["patch_x_offset"]
        y_offset = sample_metadata["patch_y_offset"]

        # Format: (3, 1024, 1024), 0-1, torch.float32
        image_torch = get_torch_image(path)
        image_torch = image_torch.unsqueeze(0).to(device)
        with torch.no_grad():
            output = model_tissue(image_torch)

        argmaxed = output.squeeze(0).argmax(dim=0)
        cropped_image = crop_and_resize_tissue_faster(
            image=argmaxed, x_offset=x_offset, y_offset=y_offset
        )

        # results in shape (1024, 1024, 3), which is okay for cv2
        one_hot = F.one_hot(cropped_image, num_classes=3)
        assert one_hot.sum(dim=2).unique().item() == 1

        one_hot = one_hot.cpu().numpy().astype(np.uint8)

        # Save the cropped image to file
        filename = os.path.join(tissue_cropped_prediction_save_path, f"{image_num}.png")
        cv2.imwrite(
            filename=filename,
            img=cv2.cvtColor(one_hot, cv2.COLOR_RGB2BGR),
        )


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    deeplab_model_path: str = (
        # "outputs/models/20240303_205501_deeplabv3plus-tissue-branch_pretrained-1_lr-6e-05_dropout-0.1_backbone-resnet50_epochs-100.pth"
        "outputs/models/best/20240313_002829_deeplabv3plus-tissue-branch_pretrained-1_lr-1e-04_dropout-0.1_backbone-resnet50_normalization-macenko_id-5_best.pth"
    )
    segformer_model_path = (
        "outputs/models/20240422_085251/Segformer_Tissue-Branch_backbone-b0_best.pth"
    )

    print("Setting up tissue predictions!")
    print(f"Device: {device}")
    print(f"Model path: {segformer_model_path}")

    normalization = "macenko"
    batch_size = 2
    pretrained = True
    backbone = "b0"
    pretrained_dataset = "ade"
    resize = 1024

    tissue_trainable = SegformerTissueTrainable(
        normalization=normalization,
        batch_size=batch_size,
        pretrained=pretrained,
        device=device,
        backbone_model=backbone,
        pretrained_dataset=pretrained_dataset,
        resize=resize,
    )

    tissue_model = tissue_trainable.create_model(
        backbone_name=backbone,
        pretrained=pretrained,
        device=device,
        model_path=segformer_model_path,
    )

    folder_name = "cropped_tissue_segformer"

    print("Creating images for train set...")
    create_cropped_tissue_predictions(
        model_tissue=tissue_model, partition="train", output_folder_name=folder_name
    )
    print("Creating images for val set...")
    create_cropped_tissue_predictions(
        model_tissue=tissue_model, partition="val", output_folder_name=folder_name
    )
    print("Creating images for test set...")
    create_cropped_tissue_predictions(
        model_tissue=tissue_model, partition="test", output_folder_name=folder_name
    )
    print("All done!")

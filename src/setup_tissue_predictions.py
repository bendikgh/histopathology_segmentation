import cv2
import os
import json
import torch
import numpy as np
import torch.nn.functional as F

from utils.constants import IDUN_OCELOT_DATA_PATH
from utils.utils import get_torch_image, crop_and_resize_tissue_patch
from deeplabv3.network.modeling import _segm_resnet


def create_cropped_tissue_predictions(
    model_tissue: torch.nn.Module,
    partition: str = "train",
    device: torch.device = torch.device("cuda"),
    ocelot_data_path: str = IDUN_OCELOT_DATA_PATH,
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

    # Getting the correct paths
    tissue_crop_path = os.path.join(
        ocelot_data_path, "annotations", partition, "cropped_tissue"
    )
    tissue_path = os.path.join(ocelot_data_path, "images", partition, "tissue")
    tissue_files = sorted(
        [
            os.path.join(tissue_path, path)
            for path in os.listdir(tissue_path)
            if ".jpg" in path
        ]
    )

    # Reading the metadata
    metadata_path = os.path.join(ocelot_data_path, "metadata.json")
    with open(metadata_path, "r") as file:
        metadata = json.load(file)

    # For each image, create a tissue crop and save the cropped version with the same name
    for path in tissue_files:
        image_num = path.split("/")[-1].split(".")[0]

        # Getting the relevant metadata for the sample
        sample_metadata = metadata["sample_pairs"][image_num]
        tissue_mpp = sample_metadata["tissue"]["resized_mpp_x"]
        cell_mpp = sample_metadata["cell"]["resized_mpp_x"]
        x_offset = sample_metadata["patch_x_offset"]
        y_offset = sample_metadata["patch_y_offset"]

        # Format: (3, 1024, 1024), 0-1, torch.float32
        image_torch = get_torch_image(path)

        # Feed the image into the model
        image_torch = image_torch.unsqueeze(0).to(device)
        model_tissue.to(device)
        with torch.no_grad():
            output = model_tissue(image_torch).squeeze(0)

        argmaxed = output.argmax(dim=0)

        # Crop the image to desired size
        cropped_image = crop_and_resize_tissue_patch(
            argmaxed, tissue_mpp, cell_mpp, x_offset, y_offset
        )

        # Permute puts the channels last, which is fine for cv2.imwrite
        one_hot = F.one_hot(cropped_image, num_classes=3)
        assert one_hot.sum(dim=2).unique().item() == 1

        one_hot = one_hot.cpu().numpy().astype(np.uint8)

        # Save the cropped image to file
        cv2.imwrite(
            filename=os.path.join(tissue_crop_path, f"{image_num}.png"),
            img=cv2.cvtColor(one_hot, cv2.COLOR_RGB2BGR),
        )


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path: str = (
        "outputs/models/2024-01-21_15-48-32_deeplabv3plus_tissue_branch_lr-1e-05_dropout-0.3_backbone-resnet50_epochs-30.pth"
    )

    print("Setting up tissue predictions!")
    print(f"Device: {device}")
    print(f"Model path: {model_path}")

    backbone: str = "resnet50"
    dropout_rate = 0.3
    model_tissue = _segm_resnet(
        name="deeplabv3plus",
        backbone_name=backbone,
        num_classes=3,
        num_channels=3,
        output_stride=8,
        pretrained_backbone=True,
        dropout_rate=dropout_rate,
    )
    model_tissue.load_state_dict(torch.load(model_path))
    model_tissue.to(device)
    model_tissue.eval()

    print("Creating images for train set...")
    create_cropped_tissue_predictions(model_tissue=model_tissue, partition="train")
    print("Creating images for val set...")
    create_cropped_tissue_predictions(model_tissue=model_tissue, partition="val")
    print("Creating images for test set...")
    create_cropped_tissue_predictions(model_tissue=model_tissue, partition="test")
    print("All done!")

import cv2
import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from PIL import Image
from torchvision.transforms.v2.functional import resized_crop
from torchvision.transforms import InterpolationMode
from utils.constants import IDUN_OCELOT_DATA_PATH
from utils.utils import get_torch_image
from deeplabv3.network.modeling import _segm_resnet


def crop_and_resize_tissue_prediction(
    image: torch.Tensor,
    tissue_mpp: float,
    cell_mpp: float,
    x_offset: float,
    y_offset: float,
    input_height: int = 1024,
    input_width: int = 1024,
) -> torch.Tensor:
    """
    Takes in an input image of a tissue prediction and crops and resizes it,
    based on the given MPPs and offsets.

    Args:
        image (torch.Tensor): A 2D tensor of shape (input_height, input_width)
            representing the input image to be cropped and resized.
        tissue_mpp (float): The microscopy pixels per unit for the tissue image.
        cell_mpp (float): The microscopy pixels per unit for the cell image.
        x_offset (float): The horizontal offset (as a fraction of width) for the center
            of the crop area, must be between 0 and 1 inclusive.
        y_offset (float): The vertical offset (as a fraction of height) for the center
            of the crop area, must be between 0 and 1 inclusive.
        input_height (int, optional): The height of the input image. Defaults to 1024.
        input_width (int, optional): The width of the input image. Defaults to 1024.

    Returns:
        torch.Tensor: A tensor of the same shape as the input (input_height, input_width)
            containing the cropped and resized image.

    Raises:
        ValueError: If the input image does not have the expected shape.
        ValueError: If tissue_mpp is less than cell_mpp.
        ValueError: If either offset is not within the [0, 1] range.
        ValueError: If the calculated crop area extends beyond the bounds of the input image.

    """
    if image.shape != (input_height, input_width):
        raise ValueError(
            f"Image shape is not ({input_height}, {input_width}), but {image.shape}"
        )

    if tissue_mpp < cell_mpp:
        raise ValueError(f"Tissue mpp is less than cell mpp: {tissue_mpp} < {cell_mpp}")

    if not (0 <= x_offset <= 1) or not (0 <= y_offset <= 1):
        raise ValueError(f"Offsets are not in the range [0, 1]: {x_offset}, {y_offset}")

    # Calculating crop size and position
    scaling_value = cell_mpp / tissue_mpp
    assert 0 <= scaling_value <= 1

    crop_height = int(input_height * scaling_value)
    crop_width = int(input_width * scaling_value)

    # Note that the offset is the center of the cropped image
    top = int(y_offset * input_height - crop_height / 2)
    left = int(x_offset * input_width - crop_width / 2)

    if top < 0 or top + crop_height > input_height:
        raise ValueError(
            f"Top + crop height is not in the range [0, {input_height}]: {top}"
        )
    if left < 0 or left + crop_width > input_width:
        raise ValueError(
            f"Left + crop width is not in the range [0, {input_width}]: {left}"
        )

    image = image.unsqueeze(0)
    crop: torch.Tensor = resized_crop(
        inpt=image,
        top=top,
        left=left,
        height=crop_height,
        width=crop_width,
        size=(input_height, input_width),
        interpolation=InterpolationMode.NEAREST,
    )
    crop = crop.squeeze(0)

    return crop


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
        cropped_image = crop_and_resize_tissue_prediction(
            argmaxed, tissue_mpp, cell_mpp, x_offset, y_offset
        )

        # NOTE: Permute puts the channels last, which is fine for cv2.imwrite
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

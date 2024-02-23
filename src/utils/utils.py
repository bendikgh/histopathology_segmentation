import argparse
import json
import os
import torch
import cv2

import pandas as pd
import numpy as np

from datetime import datetime
from glob import glob
from monai.transforms import SpatialCrop, Resize
from PIL import Image
from torchvision.transforms import PILToTensor
from utils.constants import (
    DEFAULT_BACKBONE_MODEL,
    DEFAULT_BATCH_SIZE,
    DEFAULT_CHECKPOINT_INTERVAL,
    DEFAULT_DATA_DIR,
    DEFAULT_DROPOUT_RATE,
    DEFAULT_EPOCHS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_PRETRAINED,
    DEFAULT_WARMUP_EPOCHS,
)


def crop_and_upscale_tissue(
    tissue_tensor, offset_tensor, scaling_value, image_size=1024
):
    crop_func = SpatialCrop(
        roi_center=offset_tensor,
        roi_size=image_size * torch.tensor([scaling_value, scaling_value]),
    )
    resize_func = Resize(
        spatial_size=torch.tensor([image_size, image_size]), mode="nearest"
    )

    cropped = crop_func(tissue_tensor)
    resized_tensor = resize_func(cropped)

    return resized_tensor


def get_cell_annotations_in_tissue_coordinates(
    data_object: dict, image_size: int = 1024
):
    """Translates cell labels to the tissue coordinates"""
    offset_tensor = (
        torch.tensor([[data_object["x_offset"], data_object["y_offset"]]]) * image_size
    )
    scaling_value = data_object["cell_mpp"] / data_object["tissue_mpp"]

    cell_annotations = data_object["cell_annotated"]
    cell_coords, cell_labels = cell_annotations[:, :2], cell_annotations[:, 2]

    # Performing the transformation
    center_relative_coords = cell_coords - (image_size // 2)
    scaled_coords = center_relative_coords * scaling_value
    tissue_coords = scaled_coords + offset_tensor

    # Adding the labels and rounding to ints
    tissue_annotations = torch.cat((tissue_coords, cell_labels.unsqueeze(1)), dim=1)
    tissue_annotations = torch.round(tissue_annotations).to(dtype=torch.int)
    return tissue_annotations


def get_partition_from_file_name(file_name: str):
    """
    Returns 'train', 'val' or 'test', depending on which number the file name
    has.
    """
    if int(file_name) <= 400:
        partition_folder = "train"
    elif int(file_name) <= 537:
        partition_folder = "val"
    else:
        partition_folder = "test"

    return partition_folder


def get_image_tensor_from_path(path: str):
    tensor_transformation = PILToTensor()
    with Image.open(path) as img:
        image_tensor = tensor_transformation(img)
    return image_tensor


def get_annotated_cell_data(path: str):
    data_frame = pd.read_csv(path, header=None)
    cell_tensor = torch.tensor(data_frame.values)
    return cell_tensor


def get_metadata(path: str):
    metadata_path = os.path.join(path, "metadata.json")
    with open(metadata_path, "r") as file:
        data = json.load(file)
    return data


def read_data(data_folder_path: str, fetch_images=True) -> tuple:
    """Function for reading the OCELOT data from given file path. Stores the
    result in a dictionary.
    """

    # data = {}
    train_data = {}
    val_data = {}
    test_data = {}

    metadata = get_metadata(path=data_folder_path)

    annotation_path = os.path.join(data_folder_path, "annotations")
    image_path = os.path.join(data_folder_path, "images")

    partition_folders = ["train", "val", "test"]
    file_names = []
    for folder in partition_folders:
        tissue_partition_folder_path = os.path.join(annotation_path, folder, "tissue")
        file_names += [
            f.split(".")[0] for f in os.listdir(tissue_partition_folder_path)
        ]

    for f_name in file_names:
        partition_folder = get_partition_from_file_name(f_name)

        # Finding the appropriate paths for the annotations and the images
        cell_csv_path = (
            os.path.join(annotation_path, partition_folder, "cell", f_name) + ".csv"
        )
        segmented_cell_path = (
            os.path.join(annotation_path, partition_folder, "segmented_cell", f_name)
            + ".png"
        )
        tissue_annotation_path = (
            os.path.join(annotation_path, partition_folder, "tissue", f_name) + ".png"
        )
        tissue_cropped_annotation_path = (
            os.path.join(annotation_path, partition_folder, "cropped_tissue", f_name)
            + ".png"
        )
        tissue_image_path = (
            os.path.join(image_path, partition_folder, "tissue", f_name) + ".jpg"
        )
        cell_image_path = (
            os.path.join(image_path, partition_folder, "cell", f_name) + ".jpg"
        )

        # TODO: Maybe remove this?
        # Skipping files without tumor cells
        if os.path.getsize(cell_csv_path) == 0:
            print(f"Skipped file number {f_name} as the .csv was empty.")
            continue

        cell_annotated_tensor = get_annotated_cell_data(cell_csv_path)
        tissue_annotated_tensor = get_image_tensor_from_path(tissue_annotation_path)
        tissue_cropped_annotated_tensor = get_image_tensor_from_path(
            tissue_cropped_annotation_path
        )
        tissue_image_tensor = get_image_tensor_from_path(tissue_image_path)
        cell_image_tensor = get_image_tensor_from_path(cell_image_path)
        segmneted_cell_tensor = get_image_tensor_from_path(segmented_cell_path)

        data = {}

        if fetch_images:
            data[f_name] = {
                "tissue_annotated": tissue_annotated_tensor,
                "tissue_cropped_annotated": tissue_cropped_annotated_tensor,
                "cell_annotated": cell_annotated_tensor,
                "segmented_cell": segmneted_cell_tensor,
                "tissue_image": tissue_image_tensor,
                "cell_image": cell_image_tensor,
                "cell_mpp": metadata["sample_pairs"][f_name]["cell"]["resized_mpp_x"],
                "tissue_mpp": metadata["sample_pairs"][f_name]["tissue"][
                    "resized_mpp_x"
                ],
                "slide_mpp": metadata["sample_pairs"][f_name]["mpp_x"],
                "x_offset": metadata["sample_pairs"][f_name]["patch_x_offset"],
                "y_offset": metadata["sample_pairs"][f_name]["patch_y_offset"],
            }
        else:
            data[f_name] = {
                "tissue_annotated": tissue_annotation_path,
                "tissue_cropped_annotated": tissue_cropped_annotation_path,
                "cell_annotated": cell_csv_path,
                "segmented_cell": segmented_cell_path,
                "tissue_image": tissue_image_path,
                "cell_image": cell_image_path,
                "cell_mpp": metadata["sample_pairs"][f_name]["cell"]["resized_mpp_x"],
                "tissue_mpp": metadata["sample_pairs"][f_name]["tissue"][
                    "resized_mpp_x"
                ],
                "slide_mpp": metadata["sample_pairs"][f_name]["mpp_x"],
                "x_offset": metadata["sample_pairs"][f_name]["patch_x_offset"],
                "y_offset": metadata["sample_pairs"][f_name]["patch_y_offset"],
            }

        if partition_folder == "train":
            train_data.update(data)
        elif partition_folder == "val":
            val_data.update(data)
        else:
            test_data.update(data)

    return train_data, val_data, test_data


def create_cell_segmentation_image(
    annotated_data: torch.Tensor,
    cell_mpp: float,
    radius: float = 1.4,
    image_size: int = 1024,
):
    pixel_radius = int(radius / cell_mpp)

    # Initialize a 3-channel image
    image = np.zeros((image_size, image_size, 3), dtype="uint8")

    for x, y, label in annotated_data:
        if label == 1:  # Background
            # Create a temporary single-channel image for drawing
            tmp = image[:, :, 2].copy()
            cv2.circle(tmp, (x.item(), y.item()), pixel_radius, 1, -1)
            image[:, :, 2] = tmp
        elif label == 2:  # Tumor
            tmp = image[:, :, 1].copy()
            cv2.circle(tmp, (x.item(), y.item()), pixel_radius, 1, -1)
            image[:, :, 1] = tmp
    mask = np.all(image == [0, 0, 0], axis=-1)
    image[mask] = [1, 0, 0]
    return image


def create_segmented_data(data: dict, annotation_path: str):
    """
    Takes the path of the annotated data, e.g. 'ocelot_data/annotations'
    and creates a folder in each of the 'train', 'val' and 'test' folders,
    called 'segmented cell'. This will contain all the appropriate annotated
    cell images.

    Note:
      - This function takes some time to run, usually around 2.5 minutes
    """

    # Finding and creating the necessary folders
    train_segmented_folder = os.path.join(annotation_path, "train/segmented_cell")
    val_segmented_folder = os.path.join(annotation_path, "val/segmented_cell")
    test_segmented_folder = os.path.join(annotation_path, "test/segmented_cell")

    os.makedirs(train_segmented_folder, exist_ok=True)
    os.makedirs(val_segmented_folder, exist_ok=True)
    os.makedirs(test_segmented_folder, exist_ok=True)

    for data_id, data_object in data.items():
        annotated_data = data_object["cell_annotated"]
        cell_mpp = data_object["cell_mpp"]

        # Figuring out which folder the data object belongs to
        partition = get_partition_from_file_name(data_id)
        if partition == "train":
            image_folder = train_segmented_folder
        elif partition == "val":
            image_folder = val_segmented_folder
        else:
            image_folder = test_segmented_folder

        segmented_cell_image = create_cell_segmentation_image(
            annotated_data=annotated_data, cell_mpp=cell_mpp
        )
        image_path = os.path.join(image_folder, f"{data_id}.png")
        img = Image.fromarray(segmented_cell_image.astype(np.uint8))
        img.save(image_path)


def get_cell_annotation_tensor(data, folder_name):
    cell_annotations = []
    for f_name in sorted(list(data.keys())):
        image_path = os.path.join(folder_name, f"{f_name}.png")
        cell_annotation = get_image_tensor_from_path(image_path)
        cell_annotations.append(cell_annotation)
    return torch.stack(cell_annotations)


def get_tissue_crops_scaled_tensor(data, image_size: int = 1024):
    cell_channels_with_tissue_annotations = []

    for data_id in sorted(list(data.keys())):
        data_object = data[data_id]
        offset_tensor = (
            # For some reason we have to swap y and x here, otherwise the
            # crops seemingly are reversed (though all documentation says otherwise...)
            torch.tensor([data_object["y_offset"], data_object["x_offset"]])
            * image_size
        )
        scaling_value = data_object["cell_mpp"] / data_object["tissue_mpp"]
        tissue_tensor = data_object["tissue_annotated"]
        cell_tensor = data_object["cell_image"]

        cropped_scaled = crop_and_upscale_tissue(
            tissue_tensor, offset_tensor, scaling_value
        )
        cell_tensor_tissue_annotation = torch.cat([cell_tensor, cropped_scaled], 0)

        cell_channels_with_tissue_annotations.append(cell_tensor_tissue_annotation)
    return torch.stack(cell_channels_with_tissue_annotations)


def get_ocelot_files(data_dir: str, partition: str, zoom: str = "cell") -> tuple:
    """
    Retrieves paths to image and annotation files for a specified partition
    and kind.

    Scans the given directory for image and corresponding annotation files,
    based on the specified partition (train, val, test) and kind (cell or
    tissue). Assumes specific directory naming and file organization.

    Args:
        data_dir (str): Root directory of the dataset containing 'annotations'
                        and 'images' subdirectories.
        partition (str): Dataset partition to retrieve files from. Must be
                         'train', 'val', or 'test'.
        kind (str, optional): Type of images to retrieve ('cell' or 'tissue').
                              Defaults to 'cell'.

    Returns:
        tuple: Contains two lists; first with paths to images, second with
               paths to corresponding annotations.

    Raises:
        ValueError: If 'partition' not in ['train', 'val', 'test'].
        ValueError: If 'kind' not in ['cell', 'tissue'].

    """
    # Validation
    valid_partitions = ["train", "val", "test"]
    if partition not in valid_partitions:
        raise ValueError(f"Partition must be one of {valid_partitions}")

    valid_kinds = ["cell", "tissue"]
    if zoom not in valid_kinds:
        raise ValueError(f"Kind must be one of {valid_kinds}")

    # Finding the appropriate files
    final_dir: str = "segmented_cell" if zoom == "cell" else zoom
    target_files: list = glob(
        os.path.join(data_dir, f"annotations/{partition}/{final_dir}/*")
    )
    image_numbers: list = [
        file_name.split("/")[-1].split(".")[0] for file_name in target_files
    ]
    image_files: list = [
        os.path.join(data_dir, f"images/{partition}/{zoom}", image_number + ".jpg")
        for image_number in image_numbers
    ]
    return image_files, target_files


def validate_numpy_image(image: np.ndarray) -> None:
    """
    Validates that the image that was read from file is in the correct format.
    """
    if image.shape != (1024, 1024, 3):
        raise ValueError(f"Image shape is not (1024, 1024, 3), but {image.shape}")
    if image.dtype != np.uint8:
        raise ValueError(f"Image dtype is not np.uint8, but {image.dtype}")
    if image.min() < 0 or image.max() > 255:
        raise ValueError(
            f"Image values are not in the range [0, 255], but {image.min()} - {image.max()}"
        )
    if image.max() <= 1:
        raise ValueError(f"Image values are not in the range [0, 255], but [0, 1]")


def validate_torch_image(image_torch: torch.Tensor) -> None:
    """
    Validates that the image that was read from file is in the correct format.
    """
    if image_torch.shape != (3, 1024, 1024):
        raise ValueError(f"Image shape is not (3, 1024, 1024), but {image_torch.shape}")
    if image_torch.dtype != torch.float32:
        raise ValueError(f"Image dtype is not torch.float32, but {image_torch.dtype}")
    if image_torch.min() < 0 or image_torch.max() > 1:
        raise ValueError(
            f"Image values are not in the range [0, 1], but {image_torch.min()} - {image_torch.max()}"
        )


def get_torch_image(path: str) -> torch.Tensor:
    """
    Takes in a string as a path and returns a torch tensor of the image with
    shape (3, 1024, 1024) and dtype torch.float32, with values in [0, 1].
    """
    # Format: (1024, 1024, 3), 0-255, np.uint8
    image: np.ndarray = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    validate_numpy_image(image)

    # Fixing the format of the image
    image = image.astype(np.float32) / 255.0
    image_torch = torch.from_numpy(image).permute(2, 0, 1)
    validate_torch_image(image_torch)

    return image_torch


def get_save_name(
    model_name: str,
    pretrained: bool,
    learning_rate: float,
    dropout_rate: float | None = None,
    backbone_model: str | None = None,
) -> str:
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    result: str = f"{current_time}"
    result += f"_{model_name}"
    result += f"_pretrained-{pretrained}"
    result += f"_lr-{learning_rate:.0e}"
    if dropout_rate:
        result += f"_dropout-{dropout_rate}"
    if backbone_model:
        result += f"_backbone-{backbone_model}"

    return result


def get_ocelot_args() -> argparse.Namespace:
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train Deeplabv3plus model")
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help="Number of epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=DEFAULT_DATA_DIR,
        help="Path to data directory",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=DEFAULT_CHECKPOINT_INTERVAL,
        help="Checkpoint Interval",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default=DEFAULT_BACKBONE_MODEL,
        help="Backbone model",
    )
    parser.add_argument(
        "--pretrained",
        type=int,
        default=DEFAULT_PRETRAINED,
        help="Pretrained backbone",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=DEFAULT_DROPOUT_RATE,
        help="Dropout rate",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help="Learning rate",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=DEFAULT_WARMUP_EPOCHS,
        help="Warmup epochs",
    )

    args: argparse.Namespace = parser.parse_args()
    return args

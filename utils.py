import json
import os
import torch

import pandas as pd
import torch.nn.functional as F

from enum import Enum
from monai.transforms import SpatialCrop, Resize
from PIL import Image
from torchvision.transforms import PILToTensor

IMAGE_SIZE = 1024


def crop_and_upscale_tissue(
    tissue_tensor, offset_tensor, scaling_value, image_size=1024
):
    crop_func = SpatialCrop(
        roi_center=offset_tensor,
        roi_size=image_size * torch.tensor([scaling_value, scaling_value]),
    )
    resize_func = Resize(spatial_size=torch.tensor([image_size, image_size]))

    cropped = crop_func(tissue_tensor)
    resized_tensor = resize_func(cropped)

    return torch.tensor(resized_tensor, dtype=torch.uint8)


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
    """Returns 'train', 'val' or 'test', depending on which
    number the file name has.
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


def read_data(data_folder_path: str) -> dict:
    """Function for reading the OCELOT data from given file path.
    Stores the result in a dictionary.
    """

    data = {}

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
        tissue_annotation_path = (
            os.path.join(annotation_path, partition_folder, "tissue", f_name) + ".png"
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
        tissue_image_tensor = get_image_tensor_from_path(tissue_image_path)
        cell_image_tensor = get_image_tensor_from_path(cell_image_path)

        data[f_name] = {
            "tissue_annotated": tissue_annotated_tensor,
            "cell_annotated": cell_annotated_tensor,
            "tissue_image": tissue_image_tensor,
            "cell_image": cell_image_tensor,
            "cell_mpp": metadata["sample_pairs"][f_name]["cell"]["resized_mpp_x"],
            "tissue_mpp": metadata["sample_pairs"][f_name]["tissue"]["resized_mpp_x"],
            "slide_mpp": metadata["sample_pairs"][f_name]["mpp_x"],
            "x_offset": metadata["sample_pairs"][f_name]["patch_x_offset"],
            "y_offset": metadata["sample_pairs"][f_name]["patch_y_offset"],
        }

    return data


def transform_points(points, offset, scalar):
    if torch.numel(scalar) == 1:
        # Create an identity matrix of the correct size and scale it by the scalar value
        scalar_size = points.size(-1)
        scalar = torch.eye(scalar_size) * scalar

    scalar = scalar.to(dtype=torch.float)
    points = points.to(dtype=torch.float)
    repeated_offset = offset.repeat(points.size(-2), 1)

    transformed_tensor = F.linear(points.view(-1, points.size(-1)), scalar)
    transformed_tensor = repeated_offset + transformed_tensor
    return transformed_tensor.view(points.size())

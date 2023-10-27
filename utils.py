import json
import os
import torch

import pandas as pd
import torch.nn.functional as F

from enum import Enum
from PIL import Image
from torchvision.transforms import PILToTensor

IMAGE_SIZE = 1024


class Annotation(Enum):
    UNK_UNK = (255, 255)
    UNK_BG = (255, 1)
    UNK_CA = (255, 2)
    BC_UNK = (1, 255)
    BC_non_CA = (1, 1)
    BC_CA = (1, 2)
    TC_UNK = (2, 255)
    TC_non_CA = (2, 1)
    TC_CA = (2, 2)
    UNK_neg1 = (255, -1)
    Cneg1_UNK = (-1, 255)
    Cneg1_neg1 = (-1, -1)
    Cneg1_BG = (-1, 1)
    Cneg1_CA = (-1, 2)
    BC_neg1 = (1, -1)
    TC_neg1 = (2, -1)


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


def load_images_to_tensors(image_folder):
    """
    Load images from a folder into a list as tensors.

    Parameters:
    - image_folder: str, path to the directory containing images.

    Returns:
    - list of tensors.
    """

    # Define a transformation: convert image to tensor
    transform_to_tensor = PILToTensor()

    # Get a list of image file names in the folder
    image_files = [
        f
        for f in os.listdir(image_folder)
        if os.path.isfile(os.path.join(image_folder, f))
    ]
    image_files.sort()

    image_tensors = []

    for image_file in image_files:
        # Open an image
        with Image.open(os.path.join(image_folder, image_file)) as img:
            # Apply the transformation and add to the list
            image_tensors.append(transform_to_tensor(img))

    return image_tensors


def load_csv_to_tensors(directory_path, default_shape=(1, 3)):
    """
    Load multiple CSV files from a directory into a list of tensors.
    Append a tensor of -1 with a default shape if a CSV file is empty.

    Parameters:
    - directory_path: str, path to the directory containing CSV files.
    - default_shape: tuple, shape of the tensor to create if a CSV file is empty.

    Returns:
    - list of tensors.
    """

    tensors = []
    csv_files = [f for f in os.listdir(directory_path) if f.endswith(".csv")]
    csv_files.sort()

    for csv_file in csv_files:
        csv_path = os.path.join(directory_path, csv_file)

        # Check if the file is empty
        if os.path.getsize(csv_path) == 0:
            tensors.append(torch.full(default_shape, -1))
        else:
            # Read CSV file into a DataFrame
            data_frame = pd.read_csv(csv_path, header=None)

            # Convert DataFrame to tensor and add to the list
            tensor = torch.tensor(data_frame.values)
            tensors.append(tensor)

    return tensors


def calculate_ROI_scaling_value(sample_id: str):
    file_path = "ocelot_data/metadata.json"

    with open(file_path, "r") as file:
        data = json.load(file)

    cell_dict = data["sample_pairs"][sample_id]["cell"]
    tissue_dict = data["sample_pairs"][sample_id]["tissue"]

    scaling_value = torch.tensor(
        cell_dict["resized_mpp_x"] / tissue_dict["resized_mpp_x"]
    )

    keys_to_extract = ["x_start", "y_start", "x_end", "y_end"]

    cell_dict = {k: cell_dict[k] for k in keys_to_extract if k in cell_dict}
    tissue_dict = {k: tissue_dict[k] for k in keys_to_extract if k in tissue_dict}

    cell_area = torch.tensor(list(cell_dict.values()))
    tissue_area = torch.tensor(list(tissue_dict.values()))

    return cell_area, tissue_area, scaling_value


def calculate_TC_in_CA(cell_tensors, tissue_tensors, sample_ids: list[str]):
    counting_dict = {}

    # Iterate over pairs of cell and image tensors
    for i, (cell_tensor, tissue_tensor) in enumerate(zip(cell_tensors, tissue_tensors)):
        sample_id = sample_ids[i]

        cell_area, tissue_area, scaling_value = calculate_ROI_scaling_value(sample_id)

        cell_annotations = cell_tensor[:, 2]
        cell_tensor = cell_tensor[:, :2]

        cell_tensor = translate_cell_to_tissue_coordinates(
            cell_tensor, cell_area, tissue_area, scaling_value
        )
        cell_tensor = cell_tensor.to(dtype=torch.int)

        for cell_coordinate, cell_annotation in zip(cell_tensor, cell_annotations):
            x_cor, y_cor = cell_coordinate

            annotation = Annotation(
                (cell_annotation.item(), tissue_tensor[0, y_cor, x_cor].item())
            )

            counting_dict[annotation.name] = counting_dict.get(annotation.name, 0) + 1

    return counting_dict


def translate_cell_to_tissue_coordinates(
    cell_coordinates, cell_area, tissue_area, scaling_value
):
    offset_cell = (
        (cell_area[:2] - tissue_area[:2])
        / torch.tensor(
            [tissue_area[2] - tissue_area[0], tissue_area[3] - tissue_area[1]]
        )
        * IMAGE_SIZE
    )

    offset_cell = offset_cell.to(dtype=torch.int)
    points_in_lfov = transform_points(cell_coordinates, offset_cell, scaling_value)

    return points_in_lfov


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

import os
import torch
from PIL import Image
from torchvision import transforms
import pandas as pd
from enum import Enum
import torch.nn.functional as F
import json

IMAGE_SIZE = 1024
file_path = "ocelot2023_v1.0.1/metadata.json"

with open(file_path, "r") as file:
    data = json.load(file)

SAMPLES = list(data["sample_pairs"].keys())
SAMPLES.sort()


class Annotation(Enum):
    UNK_UNK = (255, 255)
    UNK_1 = (255, 1)
    UNK_2 = (255, 2)
    C1_UNK = (1, 255)
    BC_non_CA = (1, 1)
    BC_CA = (1, 2)
    C2_UNK = (2, 255)
    TC_non_CA = (2, 1)
    TC_CA = (2, 2)
    UNK_neg1 = (255, -1)
    Cneg1_UNK = (-1, 255)
    Cneg1_neg1 = (-1, -1)
    Cneg1_1 = (-1, 1)
    Cneg1_2 = (-1, 2)
    C1_neg1 = (1, -1)
    C2_neg1 = (2, -1)


def load_images_to_tensors(image_folder):
    """
    Load images from a folder into a list as tensors.

    Parameters:
    - image_folder: str, path to the directory containing images.

    Returns:
    - list of tensors.
    """

    # Define a transformation: convert image to tensor
    transform_to_tensor = transforms.PILToTensor()

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
        # print(csv_path)

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


def calculate_TC_in_CA(cell_tensors, image_tensors):
    """
    Count cells with a specific annotation and puts it into a dictionary

    Parameters:
    - cell_tensors: list of tensors, containing cell information.
    - image_tensors: list of tensors, containing image information.

    Returns:
    - dictionary[Annotation] = count
    """

    d = {}

    # Iterate over pairs of cell and image tensors
    for i, tensors in enumerate(zip(cell_tensors, image_tensors)):
        cell_tensor, image_tensor = tensors

        sample = SAMPLES[i]

        cell_area, tissue_area, scalar = calculate_areas_scalar(sample)

        cell_annotation = cell_tensor[:, 2]
        cell_tensor = cell_tensor[:, :2]

        cell_tensor = points_to_pixels(cell_tensor, cell_area, tissue_area, scalar)
        cell_tensor = cell_tensor.to(dtype=torch.int)

        for row, cell_annotation in zip(cell_tensor, cell_annotation):
            width, height = row

            annotation = Annotation(
                (cell_annotation.item(), image_tensor[0, height, width].item())
            )

            d[annotation.name] = d.get(annotation.name, 0) + 1

    return d


def calculate_areas_scalar(sample):
    file_path = "ocelot2023_v1.0.1/metadata.json"

    with open(file_path, "r") as file:
        data = json.load(file)

    cell_dict = data["sample_pairs"][sample]["cell"]
    tissue_dict = data["sample_pairs"][sample]["tissue"]

    scalar = torch.tensor(cell_dict["resized_mpp_x"] / tissue_dict["resized_mpp_x"])

    keys_to_extract = ["x_start", "y_start", "x_end", "y_end"]

    cell_dict = {k: cell_dict[k] for k in keys_to_extract if k in cell_dict}
    tissue_dict = {k: tissue_dict[k] for k in keys_to_extract if k in tissue_dict}

    cell_area = torch.tensor(list(cell_dict.values()))
    tissue_area = torch.tensor(list(tissue_dict.values()))

    return cell_area, tissue_area, scalar


def points_to_pixels(points_in_sfov, cell_area, tissue_area, scalar):
    offset_cell = (
        (cell_area[:2] - tissue_area[:2])
        / torch.tensor(
            [tissue_area[2] - tissue_area[0], tissue_area[3] - tissue_area[1]]
        )
        * IMAGE_SIZE
    )

    points_in_lfov = transform_points(points_in_sfov, offset_cell, scalar)

    points_in_lfov = points_in_lfov.to(dtype=torch.float)

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

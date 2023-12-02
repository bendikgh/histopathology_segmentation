import json
import os
import torch
import cv2

import pandas as pd
import numpy as np
import torch.nn.functional as F

from monai.transforms import SpatialCrop, Resize
from monai.data import DataLoader
from PIL import Image
from torchvision.transforms import PILToTensor

from dataset import OcelotCellDataset

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
    
    # Edit: removed uint8
    # cropped_resized = torch.tensor(resized_tensor, dtype=torch.uint8)

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

        data = {}
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
        if partition_folder == "train":
            train_data.update(data)
        elif partition_folder == "val":
            val_data.update(data)
        else:
            test_data.update(data)

    return train_data, val_data, test_data


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
        if label == 1:
            # Create a temporary single-channel image for drawing
            tmp = image[:, :, 2].copy()
            cv2.circle(tmp, (x.item(), y.item()), pixel_radius, 1, -1)
            image[:, :, 2] = tmp
        elif label == 2:
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
    image_size = 1024

    for data_id in sorted(list(data.keys())):
        data_object = data[data_id]
        offset_tensor = (
            torch.tensor([data_object["x_offset"], data_object["y_offset"]])
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


def get_data_loader(data, segmented_cell_folder, batch_size=2):
    cell_annotations_tensor = get_cell_annotation_tensor(data, segmented_cell_folder)
    tissue_crops_scaled_tensor = get_tissue_crops_scaled_tensor(data)
    dataset = OcelotCellDataset(
        img=tissue_crops_scaled_tensor,
        seg=cell_annotations_tensor,
    )
    return DataLoader(dataset=dataset, batch_size=batch_size)

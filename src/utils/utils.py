import argparse
import cv2
import json
import os
import sys
import torch

sys.path.append(os.getcwd())

import numpy as np

from datetime import datetime
from glob import glob
from PIL import Image
from skimage.feature import peak_local_max
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2.functional import resized_crop
from torch import nn
from typing import List, Tuple

from src.utils.constants import *


def get_partition_from_file_name(file_name: str) -> str:
    """
    Returns 'train', 'val' or 'test', depending on which number the file name
    has.
    """
    if not 1 <= int(file_name) <= 667:
        raise ValueError("Index must be between 1 and 667 inclusive.")

    if int(file_name) <= 400:
        partition_folder = "train"
    elif int(file_name) <= 537:
        partition_folder = "val"
    else:
        partition_folder = "test"

    return partition_folder


def create_cell_segmentation_image(
    annotated_data: torch.Tensor,
    cell_mpp: float,
    radius: float = 1.4,
    image_size: int = 1024,
):
    pixel_radius = int(radius / cell_mpp)

    # Initialize a 3-channel image
    result = np.zeros((image_size, image_size, 3), dtype="uint8")

    for x, y, label in annotated_data:
        if label == 1:  # Background
            # Create a temporary single-channel image for drawing
            tmp = result[:, :, 2].copy()
            cv2.circle(tmp, (x.item(), y.item()), pixel_radius, 1, -1)
            result[:, :, 2] = tmp
        elif label == 2:  # Tumor
            tmp = result[:, :, 1].copy()
            cv2.circle(tmp, (x.item(), y.item()), pixel_radius, 1, -1)
            result[:, :, 1] = tmp
    mask = np.all(result == [0, 0, 0], axis=-1)
    result[mask] = [1, 0, 0]
    return result


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
    train_segmented_folder = os.path.join(annotation_path, "train/cell_mask_images")
    val_segmented_folder = os.path.join(annotation_path, "val/cell_mask_images")
    test_segmented_folder = os.path.join(annotation_path, "test/cell_mask_images")

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


def crop_and_resize_tissue_patch(
    image: torch.Tensor,
    tissue_mpp: float,
    cell_mpp: float,
    x_offset: float,
    y_offset: float,
) -> torch.Tensor:
    """
    Takes in an input image of a tissue patch and crops and resizes it,
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

    Returns:
        torch.Tensor: A tensor of the same shape as the input (input_height, input_width)
            containing the cropped and resized image.

    Raises:
        ValueError: If the input image does not have the expected shape.
        ValueError: If tissue_mpp is less than cell_mpp.
        ValueError: If either offset is not within the [0, 1] range.
        ValueError: If the calculated crop area extends beyond the bounds of the input image.

    """
    if len(image.shape) != 2:
        raise ValueError(f"Input image is not 2D, but {image.shape}")

    input_height = image.shape[0]
    input_width = image.shape[1]

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


def crop_and_resize_tissue_faster(
    image: torch.Tensor,
    x_offset: float,
    y_offset: float,
    scaling_value: float = 0.25,
) -> torch.Tensor:
    """
    Takes in an input image of a tissue patch and crops and resizes it,
    based on the given MPPs and offsets. Does the same as above, but without
    all the checks, making it a little faster.
    """

    input_height, input_width = image.shape[-2:]

    crop_height = int(input_height * scaling_value)
    crop_width = int(input_width * scaling_value)

    # Note that the offset is the center of the cropped image
    top = int(y_offset * input_height - crop_height / 2)
    left = int(x_offset * input_width - crop_width / 2)

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


def get_ocelot_files(
    data_dir: str, partition: str, zoom: str = "cell", macenko: bool = False
) -> tuple:
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
        zoom (str, optional): Type of images to retrieve ('cell' or 'tissue').
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
    image_dir: str = ""
    target_dir: str = ""
    if macenko and zoom == "tissue":
        image_dir = "tissue_macenko"
        target_dir = "tissue"
    elif macenko:
        image_dir = "cell_macenko"
        target_dir = "cell_mask_images"
    elif zoom == "cell":
        image_dir = "cell"
        target_dir = "cell_mask_images"
    else:
        image_dir = "tissue"
        target_dir = "tissue"

    target_files: list = glob(
        os.path.join(data_dir, "annotations", partition, target_dir, "*")
    )
    image_numbers: list = [
        os.path.basename(file).split(".")[0] for file in target_files
    ]
    image_files: list = [
        os.path.join(data_dir, "images", partition, image_dir, image_number + ".jpg")
        for image_number in image_numbers
    ]
    # Sorting by image numbers
    target_files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    image_files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    return image_files, target_files


def get_predicted_tissue(
    data_dir: str, image_train_nums: list, image_val_nums: list
) -> tuple:
    """
    Retrieves paths to predicted tissue annotations for both the train set and the val set.

    Args:
        image_train_nums (list): List of image numbers from the train set.
        image_val_nums (list): List of image numbers from the val set.

    Returns:
        tuple: Contains two lists with paths to predcited tissue annotations
          for the train set and the val set respectively.

    Raises:
        ValueError:
    """

    train_tissue_predicted = glob(
        os.path.join(data_dir, "annotations", "train", "pred_tissue", "*")
    )
    train_tissue_predicted = [
        file
        for file in train_tissue_predicted
        if os.path.basename(file).split(".")[0] in image_train_nums
    ]

    val_tissue_predicted = glob(os.path.join(data_dir, "annotations", "val", "pred_tissue", "*"))
    val_tissue_predicted = [
        file
        for file in val_tissue_predicted
        if os.path.basename(file).split(".")[0] in image_val_nums
    ]

    train_tissue_predicted.sort(key=lambda x: int(os.path.basename(x).split(".")[0]))
    val_tissue_predicted.sort(key=lambda x: int(os.path.basename(x).split(".")[0]))

    return train_tissue_predicted, val_tissue_predicted


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
    **keyword_args,
) -> str:
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    result: str = f"{current_time}/"
    result += f"{model_name}"
    result += f"_pretrained-{pretrained}"
    result += f"_lr-{learning_rate:.0e}"
    if dropout_rate:
        result += f"_dropout-{dropout_rate}"
    if backbone_model:
        result += f"_backbone-{backbone_model}"

    for key, value in keyword_args.items():
        if value is None:
            continue
        result += f"_{key}-{value}"

    result = result.replace(" ", "_")
    result = result.replace("+", "and")
    return result


def get_ground_truth_points(partition: str):
    gt_path = os.path.join(os.getcwd(), "eval_outputs", f"cell_gt_{partition}.json")
    with open(gt_path, "r") as f:
        gt_json = json.load(f)
    return gt_json


def get_point_predictions(softmaxed: torch.Tensor) -> List[Tuple[int, int, int, float]]:

    assert softmaxed.shape == (3, 1024, 1024)

    confidences, predictions = torch.max(softmaxed, dim=0)
    confidences, predictions = confidences.numpy(), predictions.numpy()
    peak_points_pred = peak_local_max(
        confidences,
        min_distance=15,
        labels=np.logical_or(predictions == 1, predictions == 2),
        threshold_abs=0.01,
    )
    xs = []
    ys = []
    probs = []
    ids = []
    for x, y in peak_points_pred:
        probability = confidences[x, y]
        class_id = predictions[x, y]

        # Flipping the class ids
        if class_id == 2:
            class_id = 1
        elif class_id == 1:
            class_id = 2
        xs.append(y.item())
        ys.append(x.item())
        probs.append(probability.item())
        ids.append(class_id)

    return list(zip(xs, ys, ids, probs))


def get_metadata_with_offset(data_dir: str, partition: str) -> List:
    """
    Note: This returns everything from the offset point and beyond.
    This means that train will yield all 600+ samples, not just the first
    400, and that val will yield 260+ samples, not just the 137 that belong
    to val.
    """
    metadata_path = os.path.join(data_dir, "metadata.json")
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    metadata = list(metadata["sample_pairs"].values())[
        DATASET_PARTITION_OFFSETS[partition] :
    ]
    return metadata


def get_metadata_dict(data_dir: str) -> dict:
    metadata_path = os.path.join(data_dir, "metadata.json")
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    return metadata


def get_ocelot_args() -> argparse.Namespace:
    # NOTE: Bool arguments must be 0 or 1, because otherwise argparser will
    # handle the presence of the argument as "true", despite setting it to
    # false

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
        "--model-architecture",
        default=DEFAULT_MODEL_ARCHITECTURE,
        type=str,
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
    parser.add_argument(
        "--break-early",
        type=int,
        default=DEFAULT_BREAK_AFTER_ONE_ITERATION,
        help="Break after one iteration",
    )
    parser.add_argument(
        "--do-save",
        type=int,
        default=DEFAULT_DO_SAVE,
        help="Whether to save and plot or not",
    )
    parser.add_argument(
        "--do-eval",
        type=int,
        default=DEFAULT_DO_EVALUATE,
        help="Whether to evaluate the model after training or not",
    )
    parser.add_argument(
        "--normalization",
        type=str,
        default=DEFAULT_NORMALIZATION,
        help="Which type of normalization to use",
        choices=[
            "off",
            "imagenet",
            "cell",
            "macenko",
            "macenko + cell",
            "macenko + imagenet",
        ],
    )
    parser.add_argument(
        "--id",
        type=str,
        default=None,
        help="trial identifier",
    )
    parser.add_argument(
        "--resize",
        type=int,
        default=None,
        help="resize the input images",
    )
    parser.add_argument(
        "--pretrained-dataset",
        type=str,
        default=None,
        help="The dataset the model should be pretrained on",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="The device to run the training on",
        choices=["cpu", "cuda"],
    )
    parser.add_argument(
        "--leak-labels",
        type=int,
        default=0,
        help="Whether to use tissue-leaking or not",
    )
    parser.add_argument(
        "--loss-function",
        type=str,
        default=0,
        help="Which loss function to use",
        choices=["dice", "dice-ce", "dice-wrapper", "dice-ce-wrapper"],
    )

    args: argparse.Namespace = parser.parse_args()
    return args

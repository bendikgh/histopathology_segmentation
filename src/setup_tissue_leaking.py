import cv2
import json
import os
import torch
import sys
import numpy as np

sys.path.append(os.getcwd())

from src.utils.utils import (
    get_partition_from_file_name,
    crop_and_resize_tissue_patch,
    get_ocelot_files,
)
from src.utils.constants import IDUN_OCELOT_DATA_PATH


def create_and_save_cropped_tissue_annotations(
    data_dir: str, tissue_target_files: list
):
    # Reading metadata
    metadata_path = os.path.join(data_dir, "metadata.json")
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    # Creating cropped tissue images
    for path in tissue_target_files:
        image_num = path.split("/")[-1].split(".")[0]

        sample_metadata = metadata["sample_pairs"][image_num]
        tissue_mpp = sample_metadata["tissue"]["resized_mpp_x"]
        cell_mpp = sample_metadata["cell"]["resized_mpp_x"]
        x_offset = sample_metadata["patch_x_offset"]
        y_offset = sample_metadata["patch_y_offset"]

        # Numpy image, 1-channel
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        # [1, 2, 255] -> [1, 2, 3] -> [0, 1, 2]
        image[image == 255] = 3
        image -= 1

        image_tensor: torch.Tensor = torch.from_numpy(image)
        cropped_image_tensor: torch.Tensor = crop_and_resize_tissue_patch(
            image=image_tensor,
            tissue_mpp=tissue_mpp,
            cell_mpp=cell_mpp,
            x_offset=x_offset,
            y_offset=y_offset
        )
        # One-hot encoding the image
        cropped_image = cropped_image_tensor.numpy()
        cropped_image = np.eye(3, dtype=np.uint8)[cropped_image]

        # Saving the image
        partition = get_partition_from_file_name(image_num)
        image_folder = os.path.join(
            data_dir, "annotations", partition, "cropped_tissue"
        )

        cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(image_folder, f"{image_num}.png"), cropped_image)


if __name__ == "__main__":
    data_dir = IDUN_OCELOT_DATA_PATH
    train_tissue_image_files, train_tissue_target_files = get_ocelot_files(
        data_dir=data_dir, partition="train", zoom="tissue"
    )
    val_tissue_image_files, val_tissue_target_files = get_ocelot_files(
        data_dir=data_dir, partition="val", zoom="tissue"
    )
    test_tissue_image_files, test_tissue_target_files = get_ocelot_files(
        data_dir=data_dir, partition="test", zoom="tissue"
    )

    tissue_target_files = (
        train_tissue_target_files + val_tissue_target_files + test_tissue_target_files
    )
    print("Creating and saving cropped tissue annotations...")
    create_and_save_cropped_tissue_annotations(data_dir, tissue_target_files)
    print("Done!")

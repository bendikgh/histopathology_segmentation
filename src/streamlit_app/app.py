import os
import cv2
import json
import torch
import sys

import numpy as np
import streamlit as st

from torch import nn

sys.path.append(os.getcwd())


from src.models import DeepLabV3plusModel
from src.utils.utils import get_partition_from_file_name, crop_and_resize_tissue_patch
from src.utils.constants import (
    IDUN_OCELOT_DATA_PATH,
    MISSING_IMAGE_NUMBERS,
    MAX_IMAGE_NUMBER,
    CELL_IMAGE_MEAN,
    CELL_IMAGE_STD,
)


def get_models(device: torch.device):
    # tissue_branch_model_path = "outputs/models/20240303_205501_deeplabv3plus-tissue-branch_pretrained-1_lr-6e-05_dropout-0.1_backbone-resnet50_epochs-100.pth"
    tissue_branch_model_path = "outputs/models/best/20240313_002829_deeplabv3plus-tissue-branch_pretrained-1_lr-1e-04_dropout-0.1_backbone-resnet50_normalization-macenko_id-5_best.pth"
    cell_branch_model_path = "outputs/models/20240223_194933_deeplabv3plus-cell-branch_pretrained-1_lr-1e-04_dropout-0.3_backbone-resnet50_epochs-100.pth"

    tissue_branch: nn.Module = DeepLabV3plusModel(
        backbone_name="resnet50",
        num_classes=3,
        num_channels=3,
        pretrained=True,
        dropout_rate=0.3,
    )
    tissue_branch.load_state_dict(torch.load(tissue_branch_model_path))
    tissue_branch.to(device)
    tissue_branch.eval()

    cell_branch: nn.Module = DeepLabV3plusModel(
        backbone_name="resnet50",
        num_classes=3,
        num_channels=6,
        pretrained=True,
        dropout_rate=0.3,
    )
    cell_branch.load_state_dict(torch.load(cell_branch_model_path))
    cell_branch.to(device)
    cell_branch.eval()

    return cell_branch, tissue_branch


def show_images(images: list, labels: list):
    """
    Creates a grid of images with three columns. The number of rows depends on
    the number of images in the list
    """
    if len(images) != len(labels):
        raise ValueError("The number of images and labels must be the same")

    if len(images) % 3 != 0:
        num_rows = len(images) // 3 + 1
    else:
        num_rows = len(images) // 3
    for row_idx in range(num_rows):
        row = st.columns(3)
        for col_idx in range(3):
            if not row_idx * 3 + col_idx < len(images):
                break
            with row[col_idx]:
                st.text(labels[row_idx * 3 + col_idx])
                st.image(images[row_idx * 3 + col_idx], use_column_width=True)


def setup_buttons(max_index: int):

    def button_next_callback():
        if st.session_state.image_index < max_index:
            st.session_state.image_index += 1

    def button_prev_callback():
        if st.session_state.image_index > 0:
            st.session_state.image_index -= 1

    # Setting up button
    button_col1, button_col2, _ = st.columns([1, 1, 8])
    with button_col1:
        st.button("Prev", on_click=button_prev_callback)

    with button_col2:
        st.button("Next", on_click=button_next_callback)


def get_cell_images(image_paths, image_target_paths, idx):
    cell_image = cv2.imread(image_paths[idx])
    cell_image = cv2.cvtColor(cell_image, cv2.COLOR_BGR2RGB)

    cell_target_image = cv2.imread(image_target_paths[idx])
    cell_target_image = cv2.cvtColor(cell_target_image, cv2.COLOR_BGR2RGB) * 255
    return cell_image, cell_target_image


def get_tissue_images(image_paths, image_target_paths, idx):
    tissue_image = cv2.imread(image_paths[idx])
    tissue_image = cv2.cvtColor(tissue_image, cv2.COLOR_BGR2RGB)
    tissue_target = cv2.imread(image_target_paths[idx], cv2.IMREAD_UNCHANGED)
    # tissue_target = cv2.cvtColor(tissue_target, cv2.COLOR_BGR2RGB)
    tissue_target[tissue_target == 255] = 3
    tissue_target -= 1
    tissue_target = np.eye(3)[tissue_target]
    return tissue_image, tissue_target


def get_predicted_images(cell_image, tissue_image, cell_branch, tissue_branch, device):
    # tissue_mean = np.array([0.7593, 0.5743, 0.6942], dtype=np.float32) * 255
    # tissue_std = np.array([0.1900, 0.2419, 0.1838], dtype=np.float32) * 255

    cell_mean = np.array(CELL_IMAGE_MEAN, dtype=np.float32) * 255
    cell_std = np.array(CELL_IMAGE_STD, dtype=np.float32) * 255

    normalized_cell_image = (cell_image.astype(np.float32) - cell_mean) / cell_std
    # normalized_tissue_image = (
    #     tissue_image.astype(np.float32) - tissue_mean
    # ) / tissue_std
    normalized_tissue_image = tissue_image.astype(np.float32) / 255.0

    normalized_tissue_image = torch.from_numpy(normalized_tissue_image).permute(2, 0, 1)
    normalized_cell_image = torch.from_numpy(normalized_cell_image).permute(2, 0, 1)

    normalized_cell_image = normalized_cell_image.unsqueeze(0).to(device)
    normalized_tissue_image = normalized_tissue_image.unsqueeze(0).to(device)

    with torch.no_grad():
        # Predicting tissue
        tissue_output = tissue_branch(normalized_tissue_image).squeeze(0)
        tissue_output = tissue_output.detach().cpu()
        tissue_argmax = tissue_output.argmax(dim=0)
        tissue_predicted_image = tissue_argmax.numpy()
        tissue_predicted_image = np.eye(3)[tissue_predicted_image]

        tissue_prediction_tensor = torch.tensor(tissue_predicted_image).permute(2, 0, 1)
        tissue_prediction_tensor = tissue_prediction_tensor.unsqueeze(0).to(device)
        tissue_prediction_tensor = tissue_prediction_tensor.to(torch.float32)

        # Predicting cell
        cell_tissue_input = torch.cat(
            [normalized_cell_image, tissue_prediction_tensor], dim=1
        )
        cell_tissue_output = cell_branch(cell_tissue_input).squeeze(0)
        cell_tissue_output = cell_tissue_output.detach().cpu()
        cell_tissue_argmax = cell_tissue_output.argmax(dim=0)
        cell_tissue_predicted_image = cell_tissue_argmax.numpy()
        cell_tissue_predicted_image = np.eye(3)[cell_tissue_predicted_image]
    return tissue_predicted_image, cell_tissue_predicted_image


def get_image_dict(base_path: str, image_num: str) -> dict:
    """
    Expects a number with 3 digits and returns a dictionary with the paths to
    the relevant images as well as the relevant metadata
    """
    if len(image_num) != 3:
        raise ValueError("Image number must be a string with 3 digits")

    # Setting up paths
    partition = get_partition_from_file_name(image_num)
    cell_input_image_path = os.path.join(
        base_path, "images", partition, "cell", f"{image_num}.jpg"
    )
    cell_target_image_path = os.path.join(
        base_path, "annotations", partition, "cell_mask_images", f"{image_num}.png"
    )
    tissue_input_image_path = os.path.join(
        base_path, "images", partition, "tissue_macenko", f"{image_num}.jpg"
    )
    tissue_target_image_path = os.path.join(
        base_path, "annotations", partition, "tissue", f"{image_num}.png"
    )
    tissue_predicted_image_path = os.path.join(
        base_path,
        "predictions",
        partition,
        "cropped_tissue_deeplab",
        f"{image_num}.png",
    )
    tissue_cropped_target_image_path = os.path.join(
        base_path, "annotations", partition, "cropped_tissue", f"{image_num}.png"
    )
    metadata_path = os.path.join(base_path, "metadata.json")
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    cell_mpp = metadata["sample_pairs"][image_num]["cell"]["resized_mpp_x"]
    tissue_mpp = metadata["sample_pairs"][image_num]["tissue"]["resized_mpp_x"]
    x_offset = metadata["sample_pairs"][image_num]["patch_x_offset"]
    y_offset = metadata["sample_pairs"][image_num]["patch_y_offset"]

    # Creating dictionary
    image_dict: dict = {
        "cell_input_image_path": cell_input_image_path,
        "cell_target_image_path": cell_target_image_path,
        "tissue_input_image_path": tissue_input_image_path,
        "tissue_target_image_path": tissue_target_image_path,
        "tissue_predicted_image_path": tissue_predicted_image_path,
        "tissue_cropped_target_image_path": tissue_cropped_target_image_path,
        "cell_mpp": cell_mpp,
        "tissue_mpp": tissue_mpp,
        "x_offset": x_offset,
        "y_offset": y_offset,
    }
    return image_dict


def update_image_index(new_index: int | None = None):
    if new_index is not None:
        st.session_state.image_index = new_index
    else:
        st.session_state.image_index = 0


def create_three_digit_number(number: int) -> str:
    """
    Converts an integer to a string with 3 digits
    """
    if number < 0:
        raise ValueError("Number must be greater than 0")
    elif number < 10:
        return f"00{number}"
    elif number < 100:
        return f"0{number}"
    elif number < 1000:
        return str(number)
    else:
        raise ValueError("Number must be less than 1000")


def main():
    # st.set_page_config(layout="wide")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cell_branch, tissue_branch = get_models(device=device)
    st.title("Ocelot Data")

    if "image_index" not in st.session_state:
        st.session_state.image_index = 0

    image_numbers: list = [
        create_three_digit_number(i)
        for i in list(set(range(1, MAX_IMAGE_NUMBER + 1)) - set(MISSING_IMAGE_NUMBERS))
    ]

    chosen_index = st.selectbox(
        "Choose an image number:",
        list(range(len(image_numbers) - 1)),
        index=st.session_state.image_index,
        format_func=lambda x: image_numbers[x],
    )
    st.session_state.image_index = chosen_index
    path_dict = get_image_dict(IDUN_OCELOT_DATA_PATH, image_numbers[chosen_index])

    # Reading stored images
    blank_image = np.zeros((1024, 1024, 3), dtype=np.uint8)

    cell_input_image = cv2.imread(path_dict["cell_input_image_path"])
    cell_input_image = cv2.cvtColor(cell_input_image, cv2.COLOR_BGR2RGB)

    cell_target_image = cv2.imread(path_dict["cell_target_image_path"]) * 255
    cell_target_image = cv2.cvtColor(cell_target_image, cv2.COLOR_BGR2RGB)

    tissue_input_image = cv2.imread(path_dict["tissue_input_image_path"])
    tissue_input_image = cv2.cvtColor(tissue_input_image, cv2.COLOR_BGR2RGB)

    tissue_target_image = cv2.imread(
        path_dict["tissue_target_image_path"], cv2.IMREAD_UNCHANGED
    )
    tissue_target_image[tissue_target_image == 255] = 3
    tissue_target_image -= 1
    tissue_target_image = np.eye(3)[tissue_target_image]

    tissue_predicted_image = cv2.imread(path_dict["tissue_predicted_image_path"])
    tissue_predicted_image = (
        cv2.cvtColor(tissue_predicted_image, cv2.COLOR_BGR2RGB) * 255
    )

    tissue_cropped_target_image = cv2.imread(
        path_dict["tissue_cropped_target_image_path"]
    )
    tissue_cropped_target_image = (
        cv2.cvtColor(tissue_cropped_target_image, cv2.COLOR_BGR2RGB) * 255
    )

    tissue_prediction, cell_prediction = get_predicted_images(
        cell_input_image, tissue_input_image, cell_branch, tissue_branch, device
    )
    tissue_prediction_tensor = torch.from_numpy(tissue_prediction.argmax(axis=2))

    tissue_prediction_cropped = crop_and_resize_tissue_patch(
        tissue_prediction_tensor,
        tissue_mpp=path_dict["tissue_mpp"],
        cell_mpp=path_dict["cell_mpp"],
        x_offset=path_dict["x_offset"],
        y_offset=path_dict["y_offset"],
    )
    tissue_prediction_cropped = tissue_prediction_cropped.numpy()
    tissue_prediction_cropped = (
        np.eye(3, dtype=np.uint8)[tissue_prediction_cropped] * 255
    )

    setup_buttons(max_index=len(image_numbers) - 1)

    st.text(f"Showing images for index {image_numbers[st.session_state.image_index]}")

    labels = [
        "Cell Input",
        "Cell Target",
        "Cell Prediction",
        "Tissue Input Image",
        "Tissue Target Image",
        "Tissue Predicted (File)",
        "Tissue Predicted (Model)",
        "Cropped Target Image",
        "Tissue Predicted (Cropped)",
    ]
    images = [
        cell_input_image,
        cell_target_image,
        cell_prediction,
        tissue_input_image,
        tissue_target_image,
        tissue_predicted_image,
        tissue_prediction,
        tissue_cropped_target_image,
        tissue_prediction_cropped,
    ]
    show_images(images, labels)


if __name__ == "__main__":
    main()

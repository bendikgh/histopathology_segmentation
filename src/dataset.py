import cv2
import os
import sys
import torch
import numpy as np

from torch.utils.data import Dataset
from typing import List, Dict

sys.path.append(os.getcwd())

from src.utils.constants import NUMPY_STANDARD_IMAGE_SHAPE, PYTORCH_STANDARD_IMAGE_SHAPE


class TissueDataset(Dataset):
    def __init__(
        self,
        image_files: list,
        seg_files: list,
        transform=None,
        image_shape=(1024, 1024),
        label_shape=(1024, 1024),
    ) -> None:
        self.image_files: list = image_files
        self.seg_files: list = seg_files
        self.transform = transform
        self.image_shape = image_shape
        self.label_shape = label_shape

        self.pytorch_image_shape = (3, *image_shape)
        self.pytorch_label_shape = (3, *label_shape)

    def __len__(self) -> int:
        return len(self.image_files)

    def _validate_input_image(self, image, seg_image) -> None:
        """Checks if the image and label that are loaded from file are valid for
        the dataset. Expects image to be from 0 to 255, label to be from 0 to 1,
        and the shape to be (1024, 1024, 3). Raises ValueError if not.
        """
        if image.dtype != np.uint8:
            raise ValueError(f"Image is not of type np.uint8")
        if image.max() > 255 or image.min() < 0:
            raise ValueError(f"Image has values outside of 0-255")
        if image.shape != NUMPY_STANDARD_IMAGE_SHAPE:
            raise ValueError(
                f"Image shape is {image.shape}, expected {NUMPY_STANDARD_IMAGE_SHAPE}"
            )

        if seg_image.dtype != np.uint8:
            raise ValueError(f"Label is not of type np.uint8")
        if len(set(np.unique(seg_image)) - set([1, 2, 255])) > 0:
            raise ValueError(f"Label has values outside of 1, 2, 255")
        if seg_image.shape != (1024, 1024):
            raise ValueError(f"Label shape is {seg_image.shape}, expected (1024, 1024)")

    def _validate_return_tensor(
        self, input_image: torch.Tensor, target_image: torch.Tensor
    ) -> None:
        """
        Checks that the format of the returned image and label is correct.
        """
        if input_image.dtype != torch.float32:
            raise ValueError(f"Image is not of type torch.float32")
        if input_image.shape != self.pytorch_image_shape:
            raise ValueError(
                f"Image shape is {input_image.shape}, expected {self.pytorch_image_shape}"
            )

        if target_image.dtype != torch.long:
            raise ValueError(f"Label is not of type torch.long")
        if target_image.shape != (1024, 1024):
            raise ValueError(
                f"Label shape is {target_image.shape}, expected (1024, 1024)"
            )
        if target_image.max() > 2 or target_image.min() < 0:
            raise ValueError(f"Label has values outside of 0-2")

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        seg_path = self.seg_files[idx]

        image: np.ndarray = cv2.imread(image_path)
        image: np.ndarray = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Expecting shape (1024, 1024), i.e. single-channel
        seg_image: np.ndarray = cv2.imread(seg_path, cv2.IMREAD_UNCHANGED)
        self._validate_input_image(image, seg_image)

        # [1, 2, 255] -> [1, 2, 3] -> [0, 1, 2]
        seg_image[seg_image == 255] = 3
        seg_image -= 1

        if self.transform is not None:
            transformed = self.transform(image=image, mask=seg_image)
            image = transformed["image"]
            seg_image = transformed["mask"]

        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        elif image.dtype != np.float32:
            image = image.astype(np.float32)
        seg_image = seg_image.astype(np.int64)

        # Convert image to tensor
        image_torch: torch.Tensor = torch.from_numpy(image).permute(2, 0, 1)
        seg_image_torch: torch.Tensor = torch.from_numpy(seg_image)

        self._validate_return_tensor(image_torch, seg_image_torch)

        return image_torch, seg_image_torch


class CellTissueDataset(Dataset):
    def __init__(
        self,
        cell_image_files: list,
        cell_target_files: list,
        tissue_pred_files: list,
        transform=None,
        debug=True,
        image_shape=(1024, 1024),
        label_shape=(1024, 1024),
    ):
        if not (len(cell_image_files) == len(cell_target_files) == len(tissue_pred_files)):  # fmt: skip
            raise ValueError(
                "The number of cell images, cell targets and tissue images must be the same."
            )

        self.cell_image_files = cell_image_files
        self.cell_target_files = cell_target_files
        self.tissue_pred_files = tissue_pred_files
        self.transform = transform
        self.debug = debug

        # Conventions for image shapes
        self.pytorch_image_output_shape = (6, *image_shape)
        self.numpy_image_output_shape = (*image_shape, 3)  # Currently unused

        self.pytorch_label_output_shape = (6, *label_shape)
        self.numpy_label_output_shape = (*label_shape, 3)  # Currently unused

    def __len__(self) -> int:
        return len(self.cell_image_files)

    def _validate_input_image(self, cell_image, cell_label, tissue_image):
        """Checks if the image and label that are loaded from file are valid for
        the dataset. Expects image to be from 0 to 255, label to be from 0 to 1,
        and the shape to be (3, 1024, 1024). Raises ValueError if not.
        """
        if cell_image.dtype != np.uint8:
            raise ValueError(f"Image is not of type np.uint8")
        if cell_image.max() > 255 or cell_image.min() < 0:
            raise ValueError(f"Image has values outside of 0-255")
        if cell_image.shape != NUMPY_STANDARD_IMAGE_SHAPE:
            raise ValueError(
                f"Image shape is {cell_image.shape}, expected {NUMPY_STANDARD_IMAGE_SHAPE}"
            )

        if cell_label.dtype != np.uint8:
            raise ValueError(f"Label is not of type np.uint8")
        if cell_label.max() != 1 or cell_label.min() != 0:
            raise ValueError(f"Label has values outside of 0-1")
        if cell_label.shape != NUMPY_STANDARD_IMAGE_SHAPE:
            raise ValueError(
                f"Label shape is {cell_label.shape}, expected {NUMPY_STANDARD_IMAGE_SHAPE}"
            )
        if np.unique(cell_label.sum(axis=2)) != 1:
            raise ValueError(
                f"Label is not correctly one-hot encoded, as it has multiple 1s in the same pixel."
            )

        if tissue_image.dtype != np.uint8:
            raise ValueError(f"Tissue image is not of type np.uint8")
        if tissue_image.max() > 255 or tissue_image.min() < 0:
            raise ValueError(f"Tissue image has values outside of 0-255")
        if tissue_image.shape != NUMPY_STANDARD_IMAGE_SHAPE:
            raise ValueError(
                f"Tissue image shape is {tissue_image.shape}, expected {NUMPY_STANDARD_IMAGE_SHAPE}"
            )

    def _validate_return_tensor(self, concatenated_input, cell_label):
        """
        Checks that the format of the returned image and label is correct.
        """
        if concatenated_input.dtype != torch.float32:
            raise ValueError(f"Concatenated image is not of type torch.float32")
        if concatenated_input.shape != self.pytorch_image_output_shape:
            raise ValueError(
                f"Concatenated image shape is {concatenated_input.shape}, expected {self.pytorch_image_output_shape}"
            )
        tissue_pred = concatenated_input[3:]
        if tissue_pred.max() != 1 or tissue_pred.min() != 0:
            raise ValueError(f"Tissue prediction has values outside of 0-1")
        if tissue_pred.sum(dim=0).unique().item() != 1:
            raise ValueError(
                f"Tissue prediction is not correctly one-hot encoded, as it has multiple 1s in the same pixel."
            )

        if cell_label.dtype != torch.long:
            raise ValueError(f"Label is not of type torch.long")
        if cell_label.shape != PYTORCH_STANDARD_IMAGE_SHAPE:
            raise ValueError(
                f"Label shape is {cell_label.shape}, expected {PYTORCH_STANDARD_IMAGE_SHAPE}"
            )
        if cell_label.max() != 1 or cell_label.min() != 0:
            raise ValueError(f"Label has values outside of 0-1")
        if cell_label.sum(dim=0).unique().item() != 1:
            raise ValueError(
                f"Label is not correctly one-hot encoded, as it has multiple 1s in the same pixel."
            )

    def __getitem__(self, idx):
        cell_image_path = self.cell_image_files[idx]
        cell_target_path = self.cell_target_files[idx]
        tissue_pred_path = self.tissue_pred_files[idx]

        cell_image = cv2.imread(cell_image_path)
        cell_label = cv2.imread(cell_target_path)
        tissue_pred = cv2.imread(
            tissue_pred_path
        )  # TODO: Check that this works for both pred and gt

        cell_image = cv2.cvtColor(cell_image, cv2.COLOR_BGR2RGB)
        cell_label = cv2.cvtColor(cell_label, cv2.COLOR_BGR2RGB)
        tissue_pred = cv2.cvtColor(tissue_pred, cv2.COLOR_BGR2RGB)

        if self.debug:
            self._validate_input_image(cell_image, cell_label, tissue_pred)

        if self.transform:
            transformed = self.transform(
                image=cell_image, cell_label=cell_label, tissue_prediction=tissue_pred
            )
            cell_image = transformed["image"]
            cell_label = transformed["cell_label"]
            tissue_pred = transformed["tissue_prediction"]

        # Making sure the image is between 0 and 1 and has the right type
        if cell_image.dtype == np.uint8:
            cell_image = cell_image.astype(np.float32) / 255.0
        elif cell_image.dtype != np.float32:
            cell_image = cell_image.astype(np.float32)
        cell_label = cell_label.astype(np.int64)
        tissue_pred = tissue_pred.astype(np.float32)

        # Converting to PyTorch tensors
        cell_image = torch.from_numpy(cell_image).permute(2, 0, 1)
        cell_label = torch.from_numpy(cell_label).permute(2, 0, 1)
        tissue_pred = torch.from_numpy(tissue_pred).permute(2, 0, 1)

        concatenated_input = torch.cat((cell_image, tissue_pred), dim=0)
        if self.debug:
            self._validate_return_tensor(concatenated_input, cell_label)

        return concatenated_input, cell_label

    def get_cell_annotation_list(self, idx):
        """Returns a list of cell annotations for a given image index"""
        path = self.cell_image_files[idx]
        cell_annotation_path = "annotations".join(path.split("images")).replace(
            "jpg", "csv"
        )
        return np.loadtxt(cell_annotation_path, delimiter=",", dtype=np.int32, ndmin=2)


class CellOnlyDataset(Dataset):
    def __init__(
        self,
        cell_image_files: list,
        cell_target_files: list,
        transform=None,
        image_shape: tuple = (1024, 1024),
        label_shape: tuple = (1024, 1024),
    ):

        if len(cell_image_files) != len(cell_target_files):
            raise ValueError(
                "The number of cell images and cell targets must be the same."
            )

        self.cell_image_files: list = cell_image_files
        self.cell_target_files: list = cell_target_files
        self.transform = transform

        # Conventions for image shapes
        self.pytorch_image_output_shape = (3, *image_shape)
        self.pytorch_label_output_shape = (3, *label_shape)

    def __len__(self):
        return len(self.cell_image_files)

    def _validate_input_image(self, image, label):
        """Checks if the image and label that are loaded from file are valid for
        the dataset. Expects image to be from 0 to 255, label to be from 0 to 1,
        and the shape to be (3, 1024, 1024). Raises ValueError if not.
        """
        if image.dtype != np.uint8:
            raise ValueError(f"Image is not of type np.uint8")
        if image.max() > 255 or image.min() < 0:
            raise ValueError(f"Image has values outside of 0-255")
        if image.shape != NUMPY_STANDARD_IMAGE_SHAPE:
            raise ValueError(
                f"Image shape is {image.shape}, expected {NUMPY_STANDARD_IMAGE_SHAPE}"
            )

        if label.dtype != np.uint8:
            raise ValueError(f"Label is not of type np.uint8")
        if label.max() != 1 or label.min() != 0:
            raise ValueError(f"Label has values outside of 0-1")
        if label.shape != NUMPY_STANDARD_IMAGE_SHAPE:
            raise ValueError(
                f"Label shape is {label.shape}, expected {NUMPY_STANDARD_IMAGE_SHAPE}"
            )
        if np.unique(label.sum(axis=2)) != 1:
            raise ValueError(
                f"Label is not correctly one-hot encoded, as it has multiple 1s in the same pixel."
            )

    def _validate_return_tensor(self, image, label):
        """
        Checks that the format of the returned image and label is correct.
        """

        if image.dtype != torch.float32:
            raise ValueError(f"Image is not of type torch.float32")

        if image.shape != self.pytorch_image_output_shape:
            raise ValueError(
                f"Image shape is {image.shape}, expected {self.pytorch_image_output_shape}"
            )

        if label.dtype != torch.long:
            raise ValueError(f"Label is not of type torch.long")
        if label.shape != self.pytorch_label_output_shape:
            raise ValueError(
                f"Label shape is {label.shape}, expected {self.pytorch_label_output_shape}"
            )
        if label.sum(dim=0).unique().item() != 1:
            raise ValueError(
                f"Label is not correctly one-hot encoded, as it has multiple 1s in the same pixel."
            )

    def __getitem__(self, idx) -> tuple:
        image_path = self.cell_image_files[idx]
        seg_path = self.cell_target_files[idx]

        image = cv2.imread(image_path)
        label = cv2.imread(seg_path)

        image: np.ndarray = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label: np.ndarray = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)

        # Checking type and range of values from file
        self._validate_input_image(image=image, label=label)

        if self.transform is not None:
            transformed = self.transform(image=image, mask=label)
            image = transformed["image"]
            label = transformed["mask"]

        # Making sure the image is between 0 and 1 and has the right type
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        elif image.dtype != np.float32:
            image = image.astype(np.float32)
        label = label.astype(np.int64)

        # Changing to PyTorch tensors for the model
        image = torch.from_numpy(image).permute(2, 0, 1)
        label = torch.from_numpy(label).permute(2, 0, 1)

        self._validate_return_tensor(image=image, label=label)

        return image, label

    def get_image(self, idx) -> torch.Tensor:
        """
        Returns the image tensor for a given index, so that it is possible to
        visualize the image.
        """
        image = cv2.imread(self.cell_image_files[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return torch.tensor(image)

    def get_cell_annotation_list(self, idx):
        """Returns a list of cell annotations for a given image index"""
        path = self.cell_image_files[idx]
        cell_annotation_path = "annotations".join(path.split("images")).replace(
            "jpg", "csv"
        )
        return np.loadtxt(cell_annotation_path, delimiter=",", dtype=np.int32, ndmin=2)


class CellTissueSharingDataset(Dataset):
    def __init__(
        self,
        cell_image_files: list,
        cell_target_files: list,
        tissue_image_files: list,
        tissue_target_files: list,
        transform=None,
        debug=True,
        image_shape=(1024, 1024),
        label_shape=(1024, 1024),
    ):
        if not (len(cell_image_files) == len(cell_target_files) == len(tissue_image_files) == len(tissue_target_files)):  # fmt: skip
            raise ValueError(
                "The number of cell images, cell targets and tissue images must be the same."
            )

        self.cell_image_files = cell_image_files
        self.cell_target_files = cell_target_files
        self.tissue_image_files = tissue_image_files
        self.tissue_target_files = tissue_target_files
        self.transform = transform
        self.debug = debug

        # Conventions for image shapes
        self.pytorch_image_output_shape = (6, *image_shape)
        self.numpy_image_output_shape = (*image_shape, 6)  # Currently unused

        self.pytorch_label_output_shape = (6, *label_shape)
        self.numpy_label_output_shape = (*label_shape, 6)  # Currently unused

    def __len__(self) -> int:
        return len(self.cell_image_files)

    def _validate_input_image(self, cell_image, cell_label, tissue_image, tissue_label):
        """Checks if the image and label that are loaded from file are valid for
        the dataset. Expects image to be from 0 to 255, label to be from 0 to 1,
        and the shape to be (3, 1024, 1024). Raises ValueError if not.
        """
        if cell_image.dtype != np.uint8:
            raise ValueError(f"Image is not of type np.uint8")
        if cell_image.max() > 255 or cell_image.min() < 0:
            raise ValueError(f"Image has values outside of 0-255")
        if cell_image.shape != NUMPY_STANDARD_IMAGE_SHAPE:
            raise ValueError(
                f"Image shape is {cell_image.shape}, expected {NUMPY_STANDARD_IMAGE_SHAPE}"
            )

        if cell_label.dtype != np.uint8:
            raise ValueError(f"Label is not of type np.uint8")
        if cell_label.max() != 1 or cell_label.min() != 0:
            raise ValueError(f"Label has values outside of 0-1")
        if cell_label.shape != NUMPY_STANDARD_IMAGE_SHAPE:
            raise ValueError(
                f"Label shape is {cell_label.shape}, expected {NUMPY_STANDARD_IMAGE_SHAPE}"
            )
        if np.unique(cell_label.sum(axis=2)) != 1:
            raise ValueError(
                f"Label is not correctly one-hot encoded, as it has multiple 1s in the same pixel."
            )

        if tissue_image.dtype != np.uint8:
            raise ValueError(f"Tissue image is not of type np.uint8")
        if tissue_image.max() > 255 or tissue_image.min() < 0:
            raise ValueError(f"Tissue image has values outside of 0-255")
        if tissue_image.shape != NUMPY_STANDARD_IMAGE_SHAPE:
            raise ValueError(
                f"Tissue image shape is {tissue_image.shape}, expected {NUMPY_STANDARD_IMAGE_SHAPE}"
            )

        if tissue_label.dtype != np.uint8:
            raise ValueError(f"Tissue image is not of type np.uint8")
        if tissue_label.max() > 255 or tissue_label.min() < 0:
            raise ValueError(f"Tissue image has values outside of 0-255")
        if tissue_label.shape != NUMPY_STANDARD_IMAGE_SHAPE:
            raise ValueError(
                f"Tissue image shape is {tissue_label.shape}, expected {NUMPY_STANDARD_IMAGE_SHAPE}"
            )
        if np.unique(tissue_label.sum(axis=2)) != 1:
            raise ValueError(
                f"Label is not correctly one-hot encoded, as it has multiple 1s in the same pixel."
            )

    def _validate_return_tensor(self, concatenated_image, concatenated_label):
        """
        Checks that the format of the returned image and label is correct.
        """
        if concatenated_image.dtype != torch.float32:
            raise ValueError(f"Concatenated image is not of type torch.float32")
        if concatenated_image.shape != self.pytorch_image_output_shape:
            raise ValueError(
                f"Concatenated image shape is {concatenated_image.shape}, expected {self.pytorch_image_output_shape}"
            )

        if concatenated_label.dtype != torch.long:
            raise ValueError(f"Label is not of type torch.long")
        if concatenated_label.shape != self.pytorch_label_output_shape:
            raise ValueError(
                f"Label shape is {concatenated_label.shape}, expected {self.pytorch_label_output_shape}"
            )
        if concatenated_label.max() != 1 or concatenated_label.min() != 0:
            raise ValueError(f"Label has values outside of 0-1")
        if concatenated_label.sum(dim=0).unique().item() != 2:
            raise ValueError(
                f"Label is not correctly one-hot encoded, as it has multiple 1s in the same pixel."
            )

    def __getitem__(self, idx):
        cell_image_path = self.cell_image_files[idx]
        cell_target_path = self.cell_target_files[idx]
        tissue_image_path = self.tissue_image_files[idx]
        tissue_target_path = self.tissue_target_files[idx]

        cell_image = cv2.imread(cell_image_path)
        cell_label = cv2.imread(cell_target_path)
        tissue_image = cv2.imread(tissue_image_path)
        tissue_label = cv2.imread(tissue_target_path, cv2.IMREAD_UNCHANGED)

        # [1, 2, 255] -> [1, 2, 3] -> [0, 1, 2]
        tissue_label[tissue_label == 255] = 3
        tissue_label -= 1

        tissue_label = np.eye(3, dtype=np.uint8)[tissue_label]

        cell_image = cv2.cvtColor(cell_image, cv2.COLOR_BGR2RGB)
        cell_label = cv2.cvtColor(cell_label, cv2.COLOR_BGR2RGB)

        tissue_image = cv2.cvtColor(tissue_image, cv2.COLOR_BGR2RGB)
        # tissue_label = cv2.cvtColor(tissue_label, cv2.COLOR_BGR2RGB)

        if self.debug:
            self._validate_input_image(
                cell_image, cell_label, tissue_image, tissue_label
            )

        if self.transform:
            transformed = self.transform(
                image=cell_image,
                tissue_image=tissue_image,
                cell_label=cell_label,
                tissue_label=tissue_label,
            )
            cell_image = transformed["image"]
            cell_label = transformed["cell_label"]
            tissue_image = transformed["tissue_image"]
            tissue_label = transformed["tissue_label"]

        # Making sure the image is between 0 and 1 and has the right type
        if cell_image.dtype == np.uint8:
            cell_image = cell_image.astype(np.float32) / 255.0
        elif cell_image.dtype != np.float32:
            cell_image = cell_image.astype(np.float32)
        cell_label = cell_label.astype(np.int64)

        if tissue_image.dtype == np.uint8:
            tissue_image = tissue_image.astype(np.float32) / 255.0
        elif tissue_image.dtype != np.float32:
            tissue_image = tissue_image.astype(np.float32)
        tissue_label = cell_label.astype(np.int64)

        # Converting to PyTorch tensors
        cell_image = torch.from_numpy(cell_image).permute(2, 0, 1)
        cell_label = torch.from_numpy(cell_label).permute(2, 0, 1)

        tissue_image = torch.from_numpy(tissue_image).permute(2, 0, 1)
        tissue_label = torch.from_numpy(tissue_label).permute(2, 0, 1)

        concatenated_image = torch.cat((cell_image, tissue_image), dim=0)
        concatenated_label = torch.cat((cell_label, tissue_label), dim=0)

        if self.debug:
            self._validate_return_tensor(concatenated_image, concatenated_label)

        return concatenated_image, concatenated_label, idx

    def get_cell_annotation_list(self, idx):
        """Returns a list of cell annotations for a given image index"""
        path = self.cell_image_files[idx]
        cell_annotation_path = "annotations".join(path.split("images")).replace(
            "jpg", "csv"
        )
        return np.loadtxt(cell_annotation_path, delimiter=",", dtype=np.int32, ndmin=2)


class SegformerSharingDataset(Dataset):

    def __init__(
        self,
        cell_image_files: list,
        cell_target_files: list,
        tissue_image_files: list,
        tissue_target_files: list,
        metadata: Dict,
        transform=None,
        debug=True,
    ):
        len1 = len(cell_image_files)
        len2 = len(cell_target_files)
        len3 = len(tissue_image_files)
        len4 = len(tissue_target_files)
        if not (len1 == len2 == len3 == len4):
            raise ValueError(
                "The number of cell images, cell targets and tissue images must be the same."
            )

        self.cell_image_files = cell_image_files
        self.cell_target_files = cell_target_files
        self.tissue_image_files = tissue_image_files
        self.tissue_target_files = tissue_target_files
        self.transform = transform
        self.debug = debug
        self.metadata = metadata

    def __len__(self) -> int:
        return len(self.cell_image_files)

    def __getitem__(self, idx):
        cell_image_path = self.cell_image_files[idx]
        cell_target_path = self.cell_target_files[idx]
        tissue_image_path = self.tissue_image_files[idx]
        tissue_target_path = self.tissue_target_files[idx]

        # Checking that the index points to the same image number
        num1 = cell_image_path.split("/")[-1].split(".")[0]
        num2 = cell_target_path.split("/")[-1].split(".")[0]
        num3 = tissue_image_path.split("/")[-1].split(".")[0]
        num4 = tissue_target_path.split("/")[-1].split(".")[0]

        assert num1 == num2 == num3 == num4

        metadata = self.metadata["sample_pairs"][num1]
        offset_x = metadata["patch_x_offset"]
        offset_y = metadata["patch_y_offset"]

        cell_image = cv2.imread(cell_image_path)
        cell_label = cv2.imread(cell_target_path)
        tissue_image = cv2.imread(tissue_image_path)
        tissue_label = cv2.imread(tissue_target_path, cv2.IMREAD_UNCHANGED)
        # UNCHANGED because of encoding

        cell_image = cv2.cvtColor(cell_image, cv2.COLOR_BGR2RGB)
        cell_label = cv2.cvtColor(cell_label, cv2.COLOR_BGR2RGB)
        tissue_image = cv2.cvtColor(tissue_image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            transformed = self.transform(
                cell_image=cell_image,
                cell_label=cell_label,
                tissue_image=tissue_image,
                tissue_label=tissue_label,
            )
            cell_image = transformed["cell_image"]
            cell_label = transformed["cell_label"]
            tissue_image = transformed["tissue_image"]
            tissue_label = transformed["tissue_label"]

        # Fixing types and scaling
        if cell_image.dtype == np.uint8:
            cell_image = cell_image.astype(np.float32) / 255.0
        if tissue_image.dtype == np.uint8:
            tissue_image = tissue_image.astype(np.float32) / 255.0
        cell_label = cell_label.astype(np.int64)
        tissue_label = tissue_label.astype(np.int64)

        # [1, 2, 255] -> [1, 2, 3] -> [0, 1, 2]
        tissue_label[tissue_label == 255] = 3
        tissue_label -= 1
        # one-hot
        tissue_label = np.eye(3)[tissue_label]

        # Converting to PyTorch tensors
        cell_image = torch.from_numpy(cell_image).permute(2, 0, 1)
        cell_label = torch.from_numpy(cell_label).permute(2, 0, 1)
        tissue_image = torch.from_numpy(tissue_image).permute(2, 0, 1)
        tissue_label = torch.from_numpy(tissue_label).permute(2, 0, 1)

        image = torch.cat((cell_image, tissue_image), dim=0)
        mask = torch.cat((cell_label, tissue_label), dim=0)
        offset = torch.tensor([offset_x, offset_y])
        return image, mask, offset

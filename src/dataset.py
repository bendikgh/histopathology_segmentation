import torch
import cv2
import numpy as np

from monai.data import ImageDataset
from PIL import Image
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset

from utils.constants import PYTORCH_STANDARD_IMAGE_SHAPE, NUMPY_STANDARD_IMAGE_SHAPE


class TissueDataset(ImageDataset):
    def __init__(self, image_files, seg_files, transform=None) -> None:
        self.image_files = image_files
        self.seg_files = seg_files
        self.to_tensor = ToTensor()
        self.transform = transform
        self.translator = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        seg_path = self.seg_files[idx]

        image = self.to_tensor(Image.open(image_path).convert("RGB"))
        seg_image = self.to_tensor(Image.open(seg_path)) * 255

        # Setting values to 0, 1, 2, instead of 1, 2, 255
        seg_image[seg_image == 255.0] = 3
        seg_image -= 1

        # One-hot encoding
        seg = self.translator[seg_image.int()].squeeze().permute((2, 0, 1))

        if self.transform:
            transformed = self.transform(
                image=np.array(image.permute((1, 2, 0))),
                mask=np.array(seg.permute((1, 2, 0))),
            )
            image = torch.tensor(transformed["image"]).permute((2, 0, 1))
            seg = torch.tensor(transformed["mask"]).permute((2, 0, 1))

        return image, seg


class TissueLeakingDataset(ImageDataset):
    def __init__(
        self, input_files, cell_seg_files, tissue_seg_files, transform=None
    ) -> None:
        self.image_files = input_files
        self.cell_seg_files = cell_seg_files
        self.tissue_seg_files = tissue_seg_files
        self.to_tensor = ToTensor()
        self.transform = transform

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        cell_seg_path = self.cell_seg_files[idx]
        tissue_seg_path = self.tissue_seg_files[idx]

        image = self.to_tensor(Image.open(image_path).convert("RGB"))
        cell_seg = self.to_tensor(Image.open(cell_seg_path).convert("RGB")) * 255

        # Represented as 0-1 encoding
        tissue_seg = self.to_tensor(Image.open(tissue_seg_path).convert("P")) * 255

        if self.transform:
            transformed = self.transform(
                image=np.array(image.permute((1, 2, 0))),
                mask1=np.array(cell_seg.permute((1, 2, 0))),
                mask2=np.array(tissue_seg.permute((1, 2, 0))),
            )
            image = torch.tensor(transformed["image"]).permute((2, 0, 1))
            cell_seg = torch.tensor(transformed["mask1"]).permute((2, 0, 1))
            tissue_seg = torch.tensor(transformed["mask2"]).permute((2, 0, 1))

        image = torch.cat((image, tissue_seg), dim=0)

        return image, cell_seg

    def get_cell_annotation_list(self, idx):
        """Returns a list of cell annotations for a given image index"""
        path = self.image_files[idx]
        cell_annotation_path = "annotations".join(path.split("images")).replace(
            "jpg", "csv"
        )
        return np.loadtxt(cell_annotation_path, delimiter=",", dtype=np.int32, ndmin=2)


class CellTissueDataset(Dataset):
    def __init__(
        self,
        cell_image_files: list,
        cell_target_files: list,
        tissue_pred_files: list,
        transform=None,
        debug=True
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
        if concatenated_input.shape != (6, 1024, 1024):
            raise ValueError(
                f"Concatenated image shape is {concatenated_input.shape}, expected (6, 1024, 1024)"
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
        tissue_pred = cv2.imread(tissue_pred_path)

        cell_image = cv2.cvtColor(cell_image, cv2.COLOR_BGR2RGB)
        cell_label = cv2.cvtColor(cell_label, cv2.COLOR_BGR2RGB)
        tissue_pred = cv2.cvtColor(tissue_pred, cv2.COLOR_BGR2RGB)

        if self.debug:
            self._validate_input_image(cell_image, cell_label, tissue_pred)

        if self.transform:
            transformed = self.transform(
                image=cell_image, mask1=cell_label, mask2=tissue_pred
            )
            cell_image = transformed["image"]
            cell_label = transformed["mask1"]
            tissue_pred = transformed["mask2"]

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
        output_shape: tuple = (1024, 1024),
    ):

        if len(cell_image_files) != len(cell_target_files):
            raise ValueError(
                "The number of cell images and cell targets must be the same."
            )

        self.cell_image_files: list = cell_image_files
        self.cell_target_files: list = cell_target_files
        self.transform = transform
        self.output_shape: tuple = output_shape

        # Conventions for image shapes
        self.pytorch_image_output_shape = (3, *self.output_shape)
        self.numpy_image_output_shape = (*self.output_shape, 3)  # Currently unused

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
        if label.shape != self.pytorch_image_output_shape:
            raise ValueError(
                f"Label shape is {label.shape}, expected {self.pytorch_image_output_shape}"
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

import torch
import cv2
import numpy as np

from monai.data import ImageDataset
from PIL import Image
from torchvision.transforms import ToTensor
from torchvision.transforms import functional as F
from torch.nn.functional import softmax
from torch.utils.data import Dataset

from utils.constants import PYTORCH_STANDARD_IMAGE_SHAPE, NUMPY_STANDARD_IMAGE_SHAPE


class OcelotTissueDataset(ImageDataset):

    def __init__(self, image_files, seg_files, device=None):
        super().__init__(image_files, seg_files)
        self.device = device
        self.processed_images, self.processed_labels = self.process_data()

    def process_data(self):
        processed_images = []
        processed_labels = []
        for index in range(len(self)):
            image, label = super().__getitem__(index)

            label[np.logical_or(label == 1, label == 255)] = 0
            label[label == 2] = 1

            processed_images.append(torch.tensor(image).permute((2, 0, 1)))
            processed_labels.append(torch.tensor(label))
        return torch.stack(processed_images).to(self.device), torch.stack(
            processed_labels
        ).to(self.device)

    def __getitem__(self, index):
        return self.processed_images[index], self.processed_labels[index]

    def __len__(self):
        return super().__len__()


class CellOnlyDatasetOld(ImageDataset):
    def __init__(self, image_files, seg_files, transform=None) -> None:
        self.image_files = image_files
        self.seg_files = seg_files
        self.to_tensor = ToTensor()
        self.transform = transform

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        seg_path = self.seg_files[idx]

        image = self.to_tensor(Image.open(image_path).convert("RGB"))
        seg = self.to_tensor(Image.open(seg_path).convert("RGB")) * 255

        if self.transform:
            transformed = self.transform(
                image=np.array(image.permute((1, 2, 0))),
                mask=np.array(seg.permute((1, 2, 0))),
            )
            image = torch.tensor(transformed["image"]).permute((2, 0, 1))
            seg = torch.tensor(transformed["mask"]).permute((2, 0, 1))

        return image, seg

    def get_cell_annotation_list(self, idx):
        """Returns a list of cell annotations for a given image index"""
        path = self.image_files[idx]
        cell_annotation_path = "annotations".join(path.split("images")).replace(
            "jpg", "csv"
        )
        return np.loadtxt(cell_annotation_path, delimiter=",", dtype=np.int32, ndmin=2)


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


class CellTissueDataset(ImageDataset):
    def __init__(
        self, image_files, seg_files, image_tissue_files, transform=None
    ) -> None:
        self.image_files = image_files
        self.seg_files = seg_files
        self.image_tissue_files = image_tissue_files

        self.to_tensor = ToTensor()
        self.transform = transform

    def __getitem__(self, idx):

        # Cell
        image_path = self.image_files[idx]
        seg_path = self.seg_files[idx]

        image = self.to_tensor(Image.open(image_path).convert("RGB"))
        seg = self.to_tensor(Image.open(seg_path).convert("RGB")) * 255

        # tissue
        tissue_image_path = self.image_tissue_files[idx]
        tissue_image = (
            self.to_tensor(Image.open(tissue_image_path).convert("RGB")) * 255
        )

        if self.transform:
            transformed = self.transform(
                image=np.array(image.permute((1, 2, 0))),
                mask1=np.array(seg.permute((1, 2, 0))),
                mask2=np.array(tissue_image.permute((1, 2, 0))),
            )
            image = torch.tensor(transformed["image"]).permute((2, 0, 1))
            seg = torch.tensor(transformed["mask1"]).permute((2, 0, 1))
            tissue_image = torch.tensor(transformed["mask2"]).permute((2, 0, 1))

        image = torch.cat((image, tissue_image), dim=0)

        return image, seg

    def get_cell_annotation_list(self, idx):
        """Returns a list of cell annotations for a given image index"""
        path = self.image_files[idx]
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
        self.cell_image_files: list = cell_image_files
        self.cell_target_files: list = cell_target_files
        self.transform = transform
        self.output_shape: tuple = output_shape

        # Conventions for image shapes
        self.pytorch_image_output_shape = (3, *self.output_shape)
        self.numpy_image_output_shape = (*self.output_shape, 3)  # Currently unused

    def __len__(self):
        return len(self.cell_image_files)

    def _check_image_validity(self, image, label):
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

    def _check_returned_tensor_validity(self, image, label):
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
        self._check_image_validity(image=image, label=label)

        if self.transform is not None:
            transformed = self.transform(image=image, mask=label)
            image = transformed["image"]
            label = transformed["mask"]

        # Making sure the image is between 0 and 1
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        elif image.dtype != np.float32:
            image = image.astype(np.float32)
        label = label.astype(np.int64)

        # Changing to PyTorch tensors for the model
        image = torch.from_numpy(image).permute(2, 0, 1)
        label = torch.from_numpy(label).permute(2, 0, 1)

        self._check_returned_tensor_validity(image=image, label=label)

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

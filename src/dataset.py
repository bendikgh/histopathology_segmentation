import torch
import numpy as np

from monai.data import ImageDataset
from PIL import Image
from torchvision.transforms import ToTensor
from torch.nn.functional import softmax
from torch.utils.data import Dataset


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


class CellOnlyDataset(ImageDataset):
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


class SegformerDataset(Dataset):
    def __init__(self, image_files, seg_files, transform=None, preprocessor=None):
        self.image_files = image_files
        self.seg_files = seg_files
        self.to_tensor = ToTensor()
        self.transform = transform
        self.preprocessor = preprocessor

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Cell
        image_path = self.image_files[idx]
        seg_path = self.seg_files[idx]

        image = self.to_tensor(Image.open(image_path).convert("RGB"))

        # Note: If seg_path always is from 0 to 1, then this is like multiplying
        # by 1/255 (because that's what ToTensor() does) and then multiplying by
        # 255 again, which really doesn't do anything other than convert it to a
        # tensor
        label = self.to_tensor(Image.open(seg_path).convert("RGB")) * 255

        # TODO: Possibly add asserts to make sure they are 255 before transforms
        # and then another assert to check that they are between 0 and 1 before
        # returning?

        if self.transform:
            transformed = self.transform(
                image=np.array(image.permute((1, 2, 0))),
                mask=np.array(label.permute((1, 2, 0))),
            )
            image = torch.tensor(transformed["image"]).permute((2, 0, 1))
            label = torch.tensor(transformed["mask"]).permute((2, 0, 1))

        if self.preprocessor:
            preprocessed = self.preprocessor(
                (255 * image).to(torch.uint8), label, return_tensors="pt"
            )
            image, label = torch.tensor(preprocessed["pixel_values"]).squeeze(
                0
            ), torch.tensor(preprocessed["labels"]).squeeze(0)

        return image, label

    def get_image(self, idx):
        return self.to_tensor(Image.open(self.image_files[idx]).convert("RGB")) * 255

    def get_cell_annotation_list(self, idx):
        """Returns a list of cell annotations for a given image index"""
        path = self.image_files[idx]
        cell_annotation_path = "annotations".join(path.split("images")).replace(
            "jpg", "csv"
        )
        return np.loadtxt(cell_annotation_path, delimiter=",", dtype=np.int32, ndmin=2)

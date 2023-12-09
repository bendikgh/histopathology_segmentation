import torch
import numpy as np

from monai.data import ArrayDataset, ImageDataset
from PIL import Image
from torchvision.transforms import ToTensor
from torch.nn.functional import softmax


class OcelotTissueDataset(ImageDataset):
    @classmethod
    def decode_target(cls, target):
        return target.argmax(1)

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


class TissueDataset(ImageDataset):
    def __init__(self, image_files, seg_files, transform=None) -> None:
        self.image_files = image_files
        self.seg_files = seg_files
        self.to_tensor = ToTensor()
        self.transform = transform

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        seg_path = self.seg_files[idx]

        image = self.to_tensor(Image.open(image_path).convert("RGB"))
        seg_image = self.to_tensor(Image.open(seg_path)) * 255

        seg = torch.zeros(3, 1024, 1024)
        unique_values = [1.0, 2.0, 255.0]

        for channel, value in enumerate(unique_values):
            seg[channel] = seg_image == value

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
    

class CellTissueDataset(ImageDataset):
    def __init__(self, image_files, seg_files, image_tissue_files, transform=None) -> None:
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
        seg = self.to_tensor(Image.open(seg_path).convert("RGB"))*255
        
        # tissue
        image_path = self.image_tissue_files[idx]
        image_tissue = self.to_tensor(Image.open(image_path).convert("RGB"))

        # max_values, _ = image_tissue.max(0, keepdim=True)
        image_tissue = softmax(image_tissue, 0)

        if self.transform:
            transformed = self.transform(
                image=np.array(image.permute((1, 2, 0))),
                mask1=np.array(seg.permute((1, 2, 0))),
                mask2=np.array(image_tissue.permute((1, 2, 0))),
            )
            image = torch.tensor(transformed["image"]).permute((2, 0, 1))
            seg = torch.tensor(transformed["mask1"]).permute((2, 0, 1))
            image_tissue = torch.tensor(transformed["mask2"]).permute((2, 0, 1))

        image = torch.cat((image, image_tissue), dim=0)

        return image, seg

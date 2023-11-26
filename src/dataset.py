import torch
import numpy as np

from monai.data import ArrayDataset, ImageDataset
from PIL import Image
from torchvision.transforms import ToTensor


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


class OcelotCellDataset(ArrayDataset):
    @classmethod
    def decode_target(cls, target):
        return target.argmax(1)

    # def __init__(self, cell_tensors, cell_annotations, device=None):
    #     super().__init__(img=cell_tensors, seg=cell_annotations)
    #     self.device = device
    #     self.processed_images, self.processed_labels = self.process_data()

    # def process_data(self):
    #     processed_images = []
    #     processed_labels = []
    #     for index in range(len(self)):
    #         image, label = super().__getitem__(index)
    #         processed_images.append(torch.tensor(image))
    #         processed_labels.append(torch.tensor(label))

    #     return torch.stack(processed_images).to(self.device), torch.stack(
    #         processed_labels
    #     ).to(self.device)

    # def __getitem__(self, index):
    #     return self.processed_images[index], self.processed_labels[index].squeeze()

    # def __len__(self):
    #     return super().__len__()


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

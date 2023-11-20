import torch
import numpy as np

from monai.data import ArrayDataset, ImageDataset


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

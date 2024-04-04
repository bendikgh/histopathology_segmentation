import os
import sys
import torch

import albumentations as A
import torch.nn as nn

from monai.losses import DiceLoss
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    get_polynomial_decay_schedule_with_warmup,
)

sys.path.append(os.getcwd())

# Local imports
from ocelot23algo.user.inference import Deeplabv3CellOnlyModel

from src.dataset import CellOnlyDataset
from src.utils.constants import CELL_IMAGE_MEAN, CELL_IMAGE_STD, IDUN_OCELOT_DATA_PATH
from src.utils.utils import get_metadata_with_offset, get_ocelot_files
from src.utils.training import train
from src.models import DeepLabV3plusModel


class Trainable:

    name: str
    macenko_normalize: bool
    device: torch.device
    batch_size: int
    model: nn.Module
    dataloader: DataLoader

    def train(
        self,
        num_epochs: int,
        loss_function,
        optimizer,
        device: torch.device,
        checkpoint_interval: int,
        break_after_one_iteration: bool,
        scheduler,
        do_save_model_and_plot: bool = True,
    ):
        train(
            num_epochs=num_epochs,
            train_dataloader=self.dataloader,
            model=self.model,
            loss_function=loss_function,
            optimizer=optimizer,
            save_name="",  # TODO
            device=device,
            checkpoint_interval=checkpoint_interval,
            break_after_one_iteration=break_after_one_iteration,
            scheduler=scheduler,
            do_save_model_and_plot=do_save_model_and_plot,
        )

    def create_transforms(self, normalization):
        train_transform_list = [
            A.GaussianBlur(),
            A.GaussNoise(var_limit=(0.1, 0.3), p=0.5),
            A.ColorJitter(
                brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.5
            ),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
        ]

        if "imagenet" in normalization:
            train_transform_list.append(A.Normalize())
        elif "cell" in normalization:
            train_transform_list.append(
                A.Normalize(mean=CELL_IMAGE_MEAN, std=CELL_IMAGE_STD)
            )
        return A.Compose(train_transform_list)

    def __str__(self):
        return f"Trainable: {self.name}"

    def __repr__(self):
        return str(self)


class DeeplabCellOnlyTrainable(Trainable):

    def __init__(self, normalization: str, batch_size: int, device):
        super().__init__()

        self.name = "DeeplabV3+ Cell-Only"
        self.macenko_normalize = "macenko" in normalization
        self.batch_size = batch_size

        self.transforms = self.create_transforms(normalization)
        self.dataloader = self.create_dataloader(IDUN_OCELOT_DATA_PATH)
        self.model = self.create_model(backbone_model="resnet50", pretrained=True)
        self.model.to(device)

    def create_dataloader(self, data_dir: str):
        image_files, target_files = get_ocelot_files(
            data_dir=data_dir,
            partition="train",
            zoom="cell",
            macenko=self.macenko_normalize,
        )

        # Create dataset and dataloader
        dataset = CellOnlyDataset(
            cell_image_files=image_files,
            cell_target_files=target_files,
            transform=self.transforms,
        )

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
        )

        return dataloader

    def create_model(
        self, backbone_model: str, pretrained: bool, dropout_rate: float = 0.3
    ):
        model = DeepLabV3plusModel(
            backbone_name=backbone_model,
            num_classes=3,
            num_channels=3,
            pretrained=pretrained,
            dropout_rate=dropout_rate,
        )
        return model

    def create_evaluation_model(self, partition: str):
        metadata = get_metadata_with_offset(
            data_dir=IDUN_OCELOT_DATA_PATH, partition=partition
        )
        return Deeplabv3CellOnlyModel(metadata=metadata, cell_model=self.model)


if __name__ == "__main__":

    device = torch.device("cuda")

    trainer = DeeplabCellOnlyTrainable("imagenet + macenko", 2, device)

    loss_function = DiceLoss(softmax=True)
    learning_rate = 1e-4
    optimizer = AdamW(trainer.model.parameters(), lr=learning_rate)
    scheduler = get_polynomial_decay_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=1,
        power=1,
    )

    trainer.train(
        num_epochs=1,
        loss_function=loss_function,
        optimizer=optimizer,
        device=torch.device("cuda"),
        checkpoint_interval=1,
        break_after_one_iteration=True,
        scheduler=scheduler,
    )
    print(trainer)

import os
import sys
import torch

import albumentations as A
import torch.nn as nn

from abc import ABC, abstractmethod
from monai.losses import DiceLoss
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    get_polynomial_decay_schedule_with_warmup,
)
from typing import Union, Optional


sys.path.append(os.getcwd())

# Local imports
from ocelot23algo.user.inference import Deeplabv3CellOnlyModel, EvaluationModel

from src.dataset import CellOnlyDataset
from src.utils.constants import CELL_IMAGE_MEAN, CELL_IMAGE_STD, IDUN_OCELOT_DATA_PATH
from src.utils.metrics import create_cellwise_evaluation_function
from src.utils.utils import get_metadata_with_offset, get_ocelot_files
from src.utils import training
from src.models import DeepLabV3plusModel


class Trainable(ABC):

    name: str
    macenko_normalize: bool
    pretrained: bool
    device: torch.device
    batch_size: int
    model: nn.Module
    dataloader: DataLoader

    tissue_file_folder: str

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
    ) -> Union[str, None]:
        """Returns the path of the best model."""

        best_model_path = training.train(
            num_epochs=num_epochs,
            train_dataloader=self.dataloader,
            model=self.model,
            loss_function=loss_function,
            optimizer=optimizer,
            save_name=self.get_save_name(),  # TODO
            device=device,
            checkpoint_interval=checkpoint_interval,
            break_after_one_iteration=break_after_one_iteration,
            scheduler=scheduler,
            do_save_model_and_plot=do_save_model_and_plot,
            validation_function=self.get_evaluation_function(partition="val"),
        )

        return best_model_path

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

    def get_save_name(self, **kwargs) -> str:
        result: str = ""
        result += f"{self.name}"

        for key, value in kwargs.items():
            if value is None:
                continue
            result += f"_{key}-{value}"

        result = result.replace(" ", "_")
        result = result.replace("+", "and")

        return result

    @abstractmethod
    def create_dataloader(self, data_dir: str) -> DataLoader:
        pass

    @abstractmethod
    def create_model(self, backbone_model: str, pretrained: bool) -> nn.Module:
        pass

    @abstractmethod
    def create_evaluation_model(self, partition: str) -> EvaluationModel:
        pass

    def get_evaluation_function(self, partition: str):
        evaluation_function = create_cellwise_evaluation_function(
            evaluation_model=self.create_evaluation_model(partition=partition),
            tissue_file_folder=self.tissue_file_folder,
        )
        return evaluation_function

    def __str__(self):
        return f"Trainable: {self.name}"

    def __repr__(self):
        return str(self)


class DeeplabCellOnlyTrainable(Trainable):

    def __init__(
        self,
        normalization: str,
        batch_size: int,
        pretrained: bool,
        device: torch.device,
    ):
        super().__init__()

        self.name = "DeeplabV3+ Cell-Only"
        self.macenko_normalize = "macenko" in normalization
        self.batch_size = batch_size
        self.pretrained = pretrained
        self.tissue_file_folder = "images/val/tissue_macenko"

        self.transforms = self.create_transforms(normalization)
        self.dataloader = self.create_dataloader(IDUN_OCELOT_DATA_PATH)
        self.model = self.create_model(
            backbone_model="resnet50", pretrained=self.pretrained
        )
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
        self,
        backbone_model: str,
        pretrained: bool,
        dropout_rate: float = 0.3,
        model_path: Optional[str] = None,
    ):
        model = DeepLabV3plusModel(
            backbone_name=backbone_model,
            num_classes=3,
            num_channels=3,
            pretrained=pretrained,
            dropout_rate=dropout_rate,
        )
        if model_path is not None:
            model.load_state_dict(torch.load(model_path))
        return model

    def create_evaluation_model(self, partition: str):
        metadata = get_metadata_with_offset(
            data_dir=IDUN_OCELOT_DATA_PATH, partition=partition
        )
        return Deeplabv3CellOnlyModel(metadata=metadata, cell_model=self.model)


if __name__ == "__main__":

    device = torch.device("cuda")

    trainable = DeeplabCellOnlyTrainable(
        normalization="imagenet + macenko",
        batch_size=2,
        pretrained=True,
        device=device,
    )

    loss_function = DiceLoss(softmax=True)
    learning_rate = 1e-4
    optimizer = AdamW(trainable.model.parameters(), lr=learning_rate)
    scheduler = get_polynomial_decay_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=1,
        power=1,
    )

    model_path = trainable.train(
        num_epochs=1,
        loss_function=loss_function,
        optimizer=optimizer,
        device=torch.device("cuda"),
        checkpoint_interval=1,
        break_after_one_iteration=True,
        scheduler=scheduler,
    )

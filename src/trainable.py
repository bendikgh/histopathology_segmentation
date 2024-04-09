import cv2
import os
import sys
import time
import torch

import albumentations as A
import torch.nn as nn

from abc import ABC, abstractmethod
from datetime import datetime
from glob import glob
from monai.losses import DiceLoss
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    get_polynomial_decay_schedule_with_warmup,
)
from typing import Union, Optional, List


sys.path.append(os.getcwd())

# Local imports
from ocelot23algo.user.inference import (
    Deeplabv3CellOnlyModel,
    Deeplabv3TissueCellModel,
    Deeplabv3TissueFromFile,
    EvaluationModel,
    SegformerCellOnlyModel,
    SegformerTissueFromFile,
)

from src.dataset import CellOnlyDataset, CellTissueDataset
from src.utils.constants import (
    CELL_IMAGE_MEAN,
    CELL_IMAGE_STD,
    DEFAULT_TISSUE_MODEL_PATH,
    IDUN_OCELOT_DATA_PATH,
)
from src.utils.metrics import create_cellwise_evaluation_function
from src.utils.utils import get_metadata_with_offset, get_ocelot_files
from src.utils import training
from src.models import CustomSegformerModel, DeepLabV3plusModel


class Trainable(ABC):

    name: str
    macenko_normalize: bool
    pretrained: bool
    device: torch.device
    batch_size: int
    model: nn.Module
    dataloader: DataLoader
    backbone_model: str

    def __init__(
        self,
        normalization: str,
        batch_size: int,
        pretrained: bool,
        device: torch.device,
        backbone_model: str,
    ):
        self.macenko_normalize = "macenko" in normalization
        self.batch_size = batch_size
        self.pretrained = pretrained
        self.device = device
        self.backbone_model = backbone_model

        self.train_transforms = self.create_transforms(normalization)
        self.val_transforms = self.create_transforms(normalization, partition="val")
        self.dataloader = self.create_train_dataloader(IDUN_OCELOT_DATA_PATH)
        self.model = self.create_model(
            backbone_name=self.backbone_model,
            pretrained=self.pretrained,
            device=self.device,
        )

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

    def _create_transform_list(self, normalization, partition: str = "train") -> List:
        if partition == "train":
            transform_list = [
                A.GaussianBlur(),
                A.GaussNoise(var_limit=(0.1, 0.3), p=0.5),
                A.ColorJitter(
                    brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.5
                ),
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
            ]
        else:
            transform_list = []
        if "imagenet" in normalization:
            transform_list.append(A.Normalize())
        elif "cell" in normalization:
            transform_list.append(A.Normalize(mean=CELL_IMAGE_MEAN, std=CELL_IMAGE_STD))
        return transform_list

    def create_transforms(self, normalization, partition: str = "train"):
        transform_list = self._create_transform_list(normalization, partition=partition)
        return A.Compose(transform_list)

    def get_save_name(self, **kwargs) -> str:
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

        result: str = f"{current_time}/"
        result += f"{self.name}"
        result += f"_backbone-{self.backbone_model}"

        for key, value in kwargs.items():
            if value is None:
                continue
            result += f"_{key}-{value}"

        result = result.replace(" ", "_")
        result = result.replace("+", "and")

        return result

    @abstractmethod
    def create_train_dataloader(self, data_dir: str) -> DataLoader:
        pass

    @abstractmethod
    def create_model(
        self, backbone_name: str, pretrained: bool, device: torch.device
    ) -> nn.Module:
        pass

    @abstractmethod
    def create_evaluation_model(self, partition: str) -> EvaluationModel:
        pass

    def get_tissue_folder(self, partition: str) -> str:
        return f"images/{partition}/tissue_macenko"

    def get_evaluation_function(self, partition: str):
        evaluation_function = create_cellwise_evaluation_function(
            evaluation_model=self.create_evaluation_model(partition=partition),
            tissue_file_folder=self.get_tissue_folder(partition=partition),
            transform=self.val_transforms,
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
        self.name = "DeeplabV3+ Cell-Only"
        super().__init__(
            normalization=normalization,
            batch_size=batch_size,
            pretrained=pretrained,
            device=device,
            backbone_model="resnet50",
        )

    def create_train_dataloader(self, data_dir: str):
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
            transform=self.train_transforms,
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
        backbone_name: str,
        pretrained: bool,
        device: torch.device,
        dropout_rate: float = 0.3,
        model_path: Optional[str] = None,
    ):
        model = DeepLabV3plusModel(
            backbone_name=backbone_name,
            num_classes=3,
            num_channels=3,
            pretrained=pretrained,
            dropout_rate=dropout_rate,
        )
        if model_path is not None:
            model.load_state_dict(torch.load(model_path))
        model.to(device)
        return model

    def create_evaluation_model(self, partition: str):
        metadata = get_metadata_with_offset(
            data_dir=IDUN_OCELOT_DATA_PATH, partition=partition
        )
        return Deeplabv3CellOnlyModel(metadata=metadata, cell_model=self.model)


class DeeplabTissueCellTrainable(Trainable):

    def __init__(
        self,
        normalization: str,
        batch_size: int,
        pretrained: bool,
        device: torch.device,
        leak_labels: bool = False,
    ):
        self.leak_labels = leak_labels
        if self.leak_labels:
            self.name = "DeeplabV3+ Tissue-Leaking"
            self.tissue_training_file_path = "annotations/train/cropped_tissue/*"
        else:
            self.name = "DeeplabV3+ Tissue-Cell"
            self.tissue_training_file_path = (
                "annotations/train/predicted_cropped_tissue/*"
            )
        super().__init__(
            normalization=normalization,
            batch_size=batch_size,
            pretrained=pretrained,
            device=device,
            backbone_model="resnet50",
        )

    def create_transforms(self, normalization, partition: str = "train"):
        transform_list = self._create_transform_list(normalization, partition=partition)
        return A.Compose(
            transform_list,
            additional_targets={"mask1": "mask", "mask2": "mask"},
        )

    def create_train_dataloader(self, data_dir: str):
        # Getting cell files
        train_cell_image_files, train_cell_target_files = get_ocelot_files(
            data_dir=data_dir,
            partition="train",
            zoom="cell",
            macenko=self.macenko_normalize,
        )
        train_image_nums = [
            x.split("/")[-1].split(".")[0] for x in train_cell_image_files
        ]

        # Getting tissue files
        train_tissue_predicted = glob(
            os.path.join(data_dir, self.tissue_training_file_path)
        )

        # Making sure only the appropriate numbers are used
        train_tissue_predicted = [
            file
            for file in train_tissue_predicted
            if file.split("/")[-1].split(".")[0] in train_image_nums
        ]

        train_tissue_predicted.sort(key=lambda x: int(x.split("/")[-1].split(".")[0]))

        train_dataset = CellTissueDataset(
            cell_image_files=train_cell_image_files,
            cell_target_files=train_cell_target_files,
            tissue_pred_files=train_tissue_predicted,
            transform=self.train_transforms,
        )

        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            drop_last=True,
            shuffle=True,
        )
        return train_dataloader

    def create_model(
        self,
        backbone_name: str,
        pretrained: bool,
        device: torch.device,
        model_path: Optional[str] = None,
    ):
        model = DeepLabV3plusModel(
            backbone_name=backbone_name,
            num_classes=3,
            num_channels=6,
            pretrained=pretrained,
            dropout_rate=0.3,
        )
        if model_path is not None:
            model.load_state_dict(torch.load(model_path))
        model.to(device)
        return model

    def create_evaluation_model(self, partition: str):
        metadata = get_metadata_with_offset(
            data_dir=IDUN_OCELOT_DATA_PATH, partition=partition
        )
        if self.leak_labels:
            return Deeplabv3TissueFromFile(
                metadata=metadata,
                cell_model=self.model,
            )
        else:
            return Deeplabv3TissueCellModel(
                metadata=metadata,
                cell_model=self.model,
                tissue_model_path=DEFAULT_TISSUE_MODEL_PATH,
            )


class SegformerCellOnlyTrainable(Trainable):

    def __init__(
        self,
        normalization: str,
        batch_size: int,
        pretrained: bool,
        device: torch.device,
        backbone_model: str,
        pretrained_dataset: str,
        resize: Optional[int],
    ):
        self.name = "Segformer Cell-Only"
        self.pretrained_dataset = pretrained_dataset
        self.resize = resize
        super().__init__(
            normalization=normalization,
            batch_size=batch_size,
            pretrained=pretrained,
            device=device,
            backbone_model=backbone_model,
        )

    def build_transform_function_with_extra_transforms(
        self, transforms, extra_transform_image
    ):

        def transform(image, mask):
            transformed = transforms(image=image, mask=mask)
            transformed_image, transformed_label = (
                transformed["image"],
                transformed["mask"],
            )
            transformed_image = extra_transform_image(image=transformed_image)["image"]
            return {"image": transformed_image, "mask": transformed_label}

        return transform

    def create_transforms(self, normalization, partition: str = "train"):
        transform_list = self._create_transform_list(normalization, partition=partition)
        transforms = A.Compose(transform_list)

        if self.resize is not None:
            extra_transform_image = A.Resize(height=self.resize, width=self.resize)
            transforms = self.build_transform_function_with_extra_transforms(
                transforms=transforms, extra_transform_image=extra_transform_image
            )

        return transforms

    def create_train_dataloader(self, data_dir: str) -> DataLoader:
        train_image_files, train_seg_files = get_ocelot_files(
            data_dir=data_dir,
            partition="train",
            zoom="cell",
            macenko=self.macenko_normalize,
        )
        if self.resize is not None:
            image_shape = (self.resize, self.resize)
        else:
            image_shape = (1024, 1024)

        train_dataset = CellOnlyDataset(
            cell_image_files=train_image_files,
            cell_target_files=train_seg_files,
            transform=self.train_transforms,
            image_shape=image_shape,
        )
        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
        )

        return train_dataloader

    def create_model(
        self,
        backbone_name: str,
        pretrained: bool,
        device: torch.device,
        model_path: Optional[str] = None,
    ) -> nn.Module:
        model = CustomSegformerModel(
            backbone_name=backbone_name,
            num_classes=3,
            num_channels=3,
            pretrained_dataset=self.pretrained_dataset,
        )
        if model_path is not None:
            model.load_state_dict(torch.load(model_path))
        model.to(device)
        return model

    def create_evaluation_model(self, partition: str) -> EvaluationModel:
        metadata = get_metadata_with_offset(
            data_dir=IDUN_OCELOT_DATA_PATH, partition=partition
        )
        return SegformerCellOnlyModel(metadata=metadata, cell_model=self.model)


class SegformerTissueCellTrainable(Trainable):

    def __init__(
        self,
        normalization: str,
        batch_size: int,
        pretrained: bool,
        device: torch.device,
        backbone_model: str,
        pretrained_dataset: str,
        resize: Optional[int],
        leak_labels: bool = False,
    ):
        self.leak_labels = leak_labels
        if self.leak_labels:
            self.name = "Segformer Tissue-Leaking"
            self.tissue_training_file_path = "annotations/train/cropped_tissue/*"
        else:
            self.name = "Segformer Tissue-Cell"
            self.tissue_training_file_path = (
                "annotations/train/predicted_cropped_tissue/*"
            )
        self.pretrained_dataset = pretrained_dataset
        self.resize = resize
        super().__init__(
            normalization=normalization,
            batch_size=batch_size,
            pretrained=pretrained,
            device=device,
            backbone_model=backbone_model,
        )

    def build_transform_function_with_extra_transforms(
        self, transforms, extra_transform_cell_tissue
    ):
        def transform(image, mask1, mask2):
            # First transforms
            transformed = transforms(image=image, mask1=mask1, mask2=mask2)

            transformed_cell = transformed["image"]
            transformed_label = transformed["mask1"]
            transformed_tissue = transformed["mask2"]

            # Additional transforms (usually resize)
            transformed = extra_transform_cell_tissue(
                image=transformed_cell, tissue=transformed_tissue
            )

            transformed_cell = transformed["image"]
            transformed_tissue = transformed["tissue"]

            return {
                "image": transformed_cell,
                "mask1": transformed_label,
                "mask2": transformed_tissue,
            }

        return transform

    def create_transforms(self, normalization, partition: str = "train"):
        transform_list = self._create_transform_list(normalization, partition=partition)
        transforms = A.Compose(transform_list)

        if self.resize is not None:
            extra_transform_cell_tissue = A.Compose(
                [
                    A.Resize(
                        height=self.resize,
                        width=self.resize,
                        interpolation=cv2.INTER_NEAREST,
                    )
                ],
                additional_targets={"tissue": "image"},
            )

            transforms = self.build_transform_function_with_extra_transforms(
                transforms=transforms,
                extra_transform_cell_tissue=extra_transform_cell_tissue,
            )

        return transforms

    def create_train_dataloader(self, data_dir: str) -> DataLoader:
        train_cell_image_files, train_cell_seg_files = get_ocelot_files(
            data_dir=data_dir,
            partition="train",
            zoom="cell",
            macenko=self.macenko_normalize,
        )

        train_image_nums = [
            x.split("/")[-1].split(".")[0] for x in train_cell_image_files
        ]

        train_tissue_predicted = glob(
            os.path.join(data_dir, self.tissue_training_file_path)
        )

        train_tissue_predicted = [
            file
            for file in train_tissue_predicted
            if file.split("/")[-1].split(".")[0] in train_image_nums
        ]

        train_tissue_predicted.sort(key=lambda x: int(x.split("/")[-1].split(".")[0]))

        if self.resize is not None:
            image_shape = (self.resize, self.resize)
        else:
            image_shape = (1024, 1024)

        train_dataset = CellTissueDataset(
            cell_image_files=train_cell_image_files,
            cell_target_files=train_cell_seg_files,
            transform=self.train_transforms,
            tissue_pred_files=train_tissue_predicted,
            image_shape=image_shape,
        )

        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
        )
        return train_dataloader

    def create_model(
        self,
        backbone_name: str,
        pretrained: bool,
        device: torch.device,
        model_path: Optional[str] = None,
    ) -> nn.Module:

        model = CustomSegformerModel(
            backbone_name=backbone_name,
            num_classes=3,
            num_channels=6,
            pretrained_dataset=self.pretrained_dataset,
        )

        if model_path is not None:
            model.load_state_dict(torch.load(model_path))

        model.to(device)
        return model

    def create_evaluation_model(self, partition: str) -> EvaluationModel:
        metadata = get_metadata_with_offset(
            data_dir=IDUN_OCELOT_DATA_PATH, partition=partition
        )
        return SegformerTissueFromFile(
            metadata=metadata,
            cell_model=self.model,
            tissue_model_path=None,
            device=self.device,
        )


def main():
    # General training params
    do_save: bool = False
    do_eval: bool = True
    num_epochs = 10
    batch_size = 2
    warmup_epochs = 0
    learning_rate = 1e-4
    checkpoint_interval = 10
    break_after_one_iteration = False

    # Model specific params
    normalization = "macenko"
    pretrained = True
    backbone_model = "b3"
    pretrained_dataset = "cityscapes"
    resize = 1024
    leak_labels = True

    device = torch.device("cuda")

    trainable = SegformerCellOnlyTrainable(
        normalization=normalization,
        batch_size=batch_size,
        pretrained=pretrained,
        device=device,
        backbone_model=backbone_model,
        pretrained_dataset=pretrained_dataset,
        resize=resize,
        # leak_labels=leak_labels,
    )

    loss_function = DiceLoss(softmax=True)
    optimizer = AdamW(trainable.model.parameters(), lr=learning_rate)
    scheduler = get_polynomial_decay_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_epochs,
        num_training_steps=num_epochs,
        power=1,
    )

    print(f"num_epochs: {num_epochs}")
    start = time.time()
    best_model_path = trainable.train(
        num_epochs=num_epochs,
        loss_function=loss_function,
        optimizer=optimizer,
        device=torch.device("cuda"),
        checkpoint_interval=checkpoint_interval,
        break_after_one_iteration=break_after_one_iteration,
        scheduler=scheduler,
    )
    end = time.time()
    print(f"Training finished! Took {end - start:.2f} seconds.")

    if not do_eval:
        return

    if do_save:
        trainable.model = trainable.create_model(
            backbone_name=backbone_model,
            pretrained=pretrained,
            device=trainable.device,
            model_path=best_model_path,
        )
        print(f"Updated model with the best from training.")

    # Evaluation
    trainable.model.eval()
    val_evaluation_function = trainable.get_evaluation_function(partition="val")
    test_evaluation_function = trainable.get_evaluation_function(partition="test")

    val_score = val_evaluation_function("val")
    print(f"val score: {val_score}")
    test_score = test_evaluation_function("test")
    print(f"test score: {test_score}")


if __name__ == "__main__":
    main()

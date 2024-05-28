import cv2
import os
import sys
import time
import torch

import albumentations as A
import torch.nn as nn

from abc import ABC, abstractmethod
from datetime import datetime
from functools import partial
from glob import glob
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
    Deeplabv3TissueFromFile,
    EvaluationModel,
    SegformerCellOnlyModel,
    SegformerTissueFromFile,
)
from ocelot23algo.user.inference import (
    SegformerJointPred2InputModel as SegformerJointPred2InputModule,
    SegformerAdditiveJointPred2DecoderModel as SegformerTissueToCellDecoderModule,
)

from src.dataset import (
    CellOnlyDataset,
    CellTissueDataset,
    SegformerJointPred2InputDataset,
    TissueDataset,
)
from src.utils.constants import *
from src.utils.metrics import (
    create_cellwise_evaluation_function,
    create_tissue_evaluation_function,
)
from src.utils.utils import (
    get_metadata_dict,
    get_metadata_with_offset,
    get_ocelot_files,
)
from src.utils import training
from src.utils.training import run_training_joint_pred2input
from src.models import (
    CustomSegformerModel,
    DeepLabV3plusModel,
    SegformerJointPred2InputModel,
    SegformerAdditiveJointPred2DecoderModel,
    ViTUNetModel,
)
from src.loss import DiceLossWrapper


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
        data_dir: str,
        exclude_bad_images: bool = False
    ):
        self.macenko_normalize = "macenko" in normalization
        self.batch_size = batch_size
        self.pretrained = pretrained
        self.device = device
        self.backbone_model = backbone_model
        self.normalization = normalization
        self.exclude_bad_images = exclude_bad_images

        self.train_transforms = self.create_transforms(normalization)
        self.val_transforms = self.create_transforms(normalization, partition="val")
        self.dataloader = self.create_train_dataloader(data_dir=data_dir)
        self.model = self.create_model(
            backbone_name=self.backbone_model,
            pretrained=self.pretrained,
            device=self.device,
        )
        self.creation_time = datetime.now().strftime("%Y%m%d_%H%M%S")

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
            save_name=self.get_save_name(),
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

        # TODO: Consider making this a variable instead of a function, i.e.
        # have it be generated once
        result: str = f"{self.creation_time}/"
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
        return os.path.join("images", partition, "tissue_macenko")

    def get_evaluation_function(self, partition: str):
        evaluation_function = create_cellwise_evaluation_function(
            evaluation_model=self.create_evaluation_model(partition=partition),
            tissue_file_folder=self.get_tissue_folder(partition=partition),
            transform=self.val_transforms,
            partition=partition,
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
        data_dir: str,
        exclude_bad_images=False,
    ):
        self.name = "DeeplabV3+ Cell-Only"
        super().__init__(
            normalization=normalization,
            batch_size=batch_size,
            pretrained=pretrained,
            device=device,
            backbone_model="resnet50",
            data_dir=data_dir,
            exclude_bad_images=exclude_bad_images,
        )

    def create_train_dataloader(self, data_dir: str):
        image_files, target_files = get_ocelot_files(
            data_dir=data_dir,
            partition="train",
            zoom="cell",
            macenko=self.macenko_normalize,
            exclude_bad_images=self.exclude_bad_images,
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
        data_dir: str,
        leak_labels: bool = False,
        exclude_bad_images=False,
    ):
        self.leak_labels = leak_labels
        if self.leak_labels:
            self.name = "DeeplabV3+ Tissue-Leaking"
            self.tissue_training_file_path = os.path.join(
                "annotations", "train", "cropped_tissue", "*"
            )
        else:
            self.name = "DeeplabV3+ Tissue-Cell"
            self.tissue_training_file_path = os.path.join(
                "predictions", "train", "cropped_tissue_deeplab", "*"
            )
        super().__init__(
            normalization=normalization,
            batch_size=batch_size,
            pretrained=pretrained,
            device=device,
            backbone_model="resnet50",
            data_dir=data_dir,
            exclude_bad_images=exclude_bad_images,
        )

    def get_tissue_folder(self, partition: str) -> str:
        if self.leak_labels:
            return os.path.join("annotations", partition, "cropped_tissue")
        else:
            return os.path.join("predictions", partition, "cropped_tissue_deeplab")

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
            exclude_bad_images=self.exclude_bad_images,
        )
        train_image_nums = [
            os.path.basename(x).split(".")[0] for x in train_cell_image_files
        ]

        # Getting tissue files
        train_tissue_predicted = glob(
            os.path.join(data_dir, self.tissue_training_file_path)
        )

        # Making sure only the appropriate numbers are used
        train_tissue_predicted = [
            file
            for file in train_tissue_predicted
            if os.path.basename(file).split(".")[0] in train_image_nums
        ]

        train_tissue_predicted.sort(
            key=lambda x: int(os.path.basename(x).split(".")[0])
        )

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
        return Deeplabv3TissueFromFile(
            metadata=metadata,
            cell_model=self.model,
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
        cell_image_input_size: Optional[int],
        data_dir: str,
        exclude_bad_images=False,
    ):
        self.name = "Segformer Cell-Only"
        self.pretrained_dataset = pretrained_dataset
        self.cell_image_input_size = cell_image_input_size
        super().__init__(
            normalization=normalization,
            batch_size=batch_size,
            pretrained=pretrained,
            device=device,
            backbone_model=backbone_model,
            data_dir=data_dir,
            exclude_bad_images=exclude_bad_images,
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

        if self.cell_image_input_size is not None:
            extra_transform_image = A.Resize(
                height=self.cell_image_input_size, width=self.cell_image_input_size
            )
            transforms = self.build_transform_function_with_extra_transforms(
                transforms=transforms, extra_transform_image=extra_transform_image
            )

        return transforms

    def _create_dataloader(self, data_dir: str, partition: str):
        image_files, target_files = get_ocelot_files(
            data_dir=data_dir,
            partition=partition,
            zoom="cell",
            macenko=self.macenko_normalize,
            exclude_bad_images=self.exclude_bad_images,
        )
        if self.cell_image_input_size is not None:
            image_shape = (self.cell_image_input_size, self.cell_image_input_size)
        else:
            image_shape = (1024, 1024)

        if partition == "train":
            transforms = self.train_transforms
            shuffle = True
        else:
            transforms = self.val_transforms
            shuffle = False

        dataset = CellOnlyDataset(
            cell_image_files=image_files,
            cell_target_files=target_files,
            transform=transforms,
            image_shape=image_shape,
        )
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=True,
        )
        return dataloader

    def create_train_dataloader(self, data_dir: str) -> DataLoader:
        return self._create_dataloader(data_dir, partition="train")

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


class SegformerTissueTrainable(Trainable):

    def __init__(
        self,
        normalization: str,
        batch_size: int,
        pretrained: bool,
        device: torch.device,
        backbone_model: str,
        pretrained_dataset: str,
        data_dir: str,
        tissue_image_input_size: Optional[int] = 1024,
        oversample=False,
        exclude_bad_images=False,
    ):
        self.name = "Segformer Tissue-Branch"
        self.pretrained_dataset = pretrained_dataset
        self.tissue_image_input_size = tissue_image_input_size
        self.oversample = oversample

        super().__init__(
            normalization=normalization,
            batch_size=batch_size,
            pretrained=pretrained,
            device=device,
            backbone_model=backbone_model,
            data_dir=data_dir,
            exclude_bad_images=exclude_bad_images,
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

        if self.tissue_image_input_size is not None:
            extra_transform_image = A.Resize(
                height=self.tissue_image_input_size, width=self.tissue_image_input_size
            )
            transforms = self.build_transform_function_with_extra_transforms(
                transforms=transforms, extra_transform_image=extra_transform_image
            )

        return transforms

    def _create_dataloader(self, data_dir, partition: str) -> DataLoader:
        tissue_image_files, tissue_target_files = get_ocelot_files(
            data_dir=data_dir,
            partition=partition,
            zoom="tissue",
            macenko=self.macenko_normalize,
            exclude_bad_images=self.exclude_bad_images,
        )
        if partition == "train":
            shuffle = True
            transform = self.train_transforms
        else:
            shuffle = False
            transform = self.val_transforms

        if self.oversample and partition == "train":
            for i in INDICES_TISSUE_MOST_CANCER:
                # Oversampling with the 100 tissue images with the most cancer to even the distribution of pixel types
                tissue_image_files.append(tissue_image_files[i])
                tissue_target_files.append(tissue_target_files[i])

        dataset = TissueDataset(
            image_files=tissue_image_files,
            seg_files=tissue_target_files,
            transform=transform,
            image_shape=(self.tissue_image_input_size, self.tissue_image_input_size),
        )

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            drop_last=True,
            shuffle=shuffle,
        )

        return dataloader

    def create_train_dataloader(self, data_dir: str) -> DataLoader:
        return self._create_dataloader(data_dir=data_dir, partition="train")

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

    def get_evaluation_function(self, partition: str):
        evaluation_function = create_tissue_evaluation_function(
            model=self.model,
            dataloader=self._create_dataloader(
                data_dir=IDUN_OCELOT_DATA_PATH, partition=partition
            ),
            loss_function=DiceLossWrapper(softmax=True, to_onehot_y=True),
            device=self.device,
        )
        return evaluation_function

    def create_evaluation_model(self, partition: str) -> EvaluationModel:
        return None


class SegformerTissueCellTrainable(Trainable):

    def __init__(
        self,
        normalization: str,
        batch_size: int,
        pretrained: bool,
        device: torch.device,
        backbone_model: str,
        pretrained_dataset: str,
        cell_image_input_size: Optional[int],
        data_dir: str,
        leak_labels: bool = False,
        debug: bool = False,
        exclude_bad_images=False,
    ):
        self.leak_labels = leak_labels
        self.debug = debug
        if self.leak_labels:
            self.name = "Segformer Tissue-Leaking"
        else:
            self.name = "Segformer Tissue-Cell"
        
        self.tissue_training_file_path = os.path.join(self.get_tissue_folder("train"), "*")

        self.pretrained_dataset = pretrained_dataset
        self.cell_image_input_size = cell_image_input_size
        super().__init__(
            normalization=normalization,
            batch_size=batch_size,
            pretrained=pretrained,
            device=device,
            backbone_model=backbone_model,
            data_dir=data_dir,
            exclude_bad_images=exclude_bad_images,
        )

    def get_tissue_folder(self, partition: str) -> str:
        if self.leak_labels:
            return os.path.join("annotations", partition, "cropped_tissue")
        else:
            return os.path.join("predictions", partition, "cropped_tissue_segformer_exp6")

    def build_transform_function_with_extra_transforms(
        self, transforms, extra_transform_cell_tissue
    ):
        def transform(image, cell_label, tissue_prediction):
            # First transforms
            if cell_label is not None:
                transformed = transforms(
                    image=image,
                    cell_label=cell_label,
                    tissue_prediction=tissue_prediction,
                )
                transformed_label = transformed["cell_label"]
            else:
                transformed = transforms(
                    image=image, tissue_prediction=tissue_prediction
                )
                transformed_label = None

            transformed_cell = transformed["image"]
            transformed_tissue = transformed["tissue_prediction"]

            # Additional transforms (usually resize)
            transformed = extra_transform_cell_tissue(
                image=transformed_cell, tissue=transformed_tissue
            )

            transformed_cell = transformed["image"]
            transformed_tissue = transformed["tissue"]

            return {
                "image": transformed_cell,
                "cell_label": transformed_label,
                "tissue_prediction": transformed_tissue,
            }

        return transform

    def create_transforms(self, normalization, partition: str = "train"):
        transform_list = self._create_transform_list(normalization, partition=partition)
        transforms = A.Compose(
            transform_list,
            additional_targets={"cell_label": "mask", "tissue_prediction": "mask"},
        )

        if self.cell_image_input_size is not None:
            resize_function = A.Resize(
                height=self.cell_image_input_size,
                width=self.cell_image_input_size,
                interpolation=cv2.INTER_NEAREST,
            )
            extra_transform_cell_tissue = A.Compose(
                [resize_function], additional_targets={"tissue": "image"}
            )

            transforms = self.build_transform_function_with_extra_transforms(
                transforms=transforms,
                extra_transform_cell_tissue=extra_transform_cell_tissue,
            )

        return transforms

    def create_train_dataloader(self, data_dir: str) -> DataLoader:
        train_cell_image_files, train_cell_target_files = get_ocelot_files(
            data_dir=data_dir,
            partition="train",
            zoom="cell",
            macenko=self.macenko_normalize,
            exclude_bad_images=self.exclude_bad_images
        )

        train_image_nums = [
            os.path.basename(x).split(".")[0] for x in train_cell_image_files
        ]

        train_tissue_predicted = glob(
            os.path.join(data_dir, self.tissue_training_file_path)
        )

        train_tissue_predicted = [
            file
            for file in train_tissue_predicted
            if os.path.basename(file).split(".")[0] in train_image_nums
        ]

        train_tissue_predicted.sort(
            key=lambda x: int(os.path.basename(x).split(".")[0])
        )

        if self.cell_image_input_size is not None:
            image_shape = (self.cell_image_input_size, self.cell_image_input_size)
        else:
            image_shape = (1024, 1024)

        train_dataset = CellTissueDataset(
            cell_image_files=train_cell_image_files,
            cell_target_files=train_cell_target_files,
            tissue_pred_files=train_tissue_predicted,
            transform=self.train_transforms,
            image_shape=image_shape,
            debug=self.debug,
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


class SegformerJointPred2InputTrainable(Trainable):

    def __init__(
        self,
        normalization: str,
        batch_size: int,
        pretrained: bool,
        device: torch.device,
        backbone_model: str,
        pretrained_dataset: str,
        data_dir: str,
        debug: bool = False,
        cell_image_input_size: int = 512,
        tissue_image_input_size: int = 1024,
        exclude_bad_images=False,
        weight_loss = False,
        freeze_tissue = False
    ):
        self.pretrained_dataset = pretrained_dataset
        self.name = "Segformer Sharing"
        self.cell_image_input_size = cell_image_input_size
        self.resize_dict = {
            "cell": cell_image_input_size,
            "tissue": tissue_image_input_size,
        }
        self.freeze_tissue = freeze_tissue
        self.weight_loss = weight_loss

        super().__init__(
            normalization=normalization,
            batch_size=batch_size,
            pretrained=pretrained,
            device=device,
            backbone_model=backbone_model,
            data_dir=data_dir,
            exclude_bad_images=exclude_bad_images,
        )

    def create_transforms(self, normalization, partition: str = "train"):
        return None

    def _transform(self, regular_transform, resize_transform, image, mask):
        """
        Transformation that applies regular_transform to both image and mask,
        and resize_transform to the image only.
        """
        transformed = regular_transform(image=image, mask=mask)
        transformed_image = transformed["image"]
        transformed_label = transformed["mask"]

        transformed_image = resize_transform(image=transformed_image)["image"]
        return {"image": transformed_image, "mask": transformed_label}

    def _create_dual_transform(
        self, normalization, partition: str = "train", kind: str = "cell"
    ):
        if kind not in ["cell", "tissue"]:
            raise ValueError(f"kind must be either 'cell' or 'tissue'. Got {kind}")
        if partition not in ["train", "val", "test"]:
            raise ValueError(
                f"partition must be either 'train', 'val', or 'test'. Got {partition}"
            )

        resize = self.resize_dict[kind]
        regular_transform_list = self._create_transform_list(
            normalization, partition=partition
        )
        regular_transform = A.Compose(regular_transform_list)
        resize_transform = A.Resize(height=resize, width=resize)

        return partial(self._transform, regular_transform, resize_transform)

    def create_train_dataloader(self, data_dir: str) -> DataLoader:
        train_cell_image_files, train_cell_target_files = get_ocelot_files(
            data_dir=data_dir,
            partition="train",
            zoom="cell",
            macenko=self.macenko_normalize,
            exclude_bad_images=self.exclude_bad_images,
        )
        train_tissue_image_files, train_tissue_target_files = get_ocelot_files(
            data_dir=data_dir,
            partition="train",
            zoom="tissue",
            macenko=self.macenko_normalize,
            exclude_bad_images=self.exclude_bad_images
        )

        # Removing image numbers from tissue images to match cell and tissue
        image_numbers = [
            os.path.basename(x).split(".")[0] for x in train_cell_image_files
        ]
        train_tissue_image_files = [
            file
            for file in train_tissue_image_files
            if os.path.basename(file).split(".")[0] in image_numbers
        ]
        train_tissue_target_files = [
            file
            for file in train_tissue_target_files
            if os.path.basename(file).split(".")[0] in image_numbers
        ]

        len1 = len(train_cell_image_files)
        len2 = len(train_cell_target_files)
        len3 = len(train_tissue_image_files)
        len4 = len(train_tissue_target_files)
        assert len1 == len2 == len3 == len4

        metadata = get_metadata_dict(data_dir=data_dir)

        cell_transform = self._create_dual_transform(
            normalization=self.normalization, partition="train", kind="cell"
        )
        tissue_transform = self._create_dual_transform(
            normalization=self.normalization, partition="train", kind="tissue"
        )

        train_dataset = SegformerJointPred2InputDataset(
            cell_image_files=train_cell_image_files,
            cell_target_files=train_cell_target_files,
            tissue_image_files=train_tissue_image_files,
            tissue_target_files=train_tissue_target_files,
            metadata=metadata,
            cell_transform=cell_transform,
            tissue_transform=tissue_transform,
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

        model = SegformerJointPred2InputModel(
            backbone_model=backbone_name,
            pretrained_dataset=self.pretrained_dataset,
            input_image_size=self.cell_image_input_size,
            output_image_size=1024,
        )
        if model_path is not None:
            model.load_state_dict(torch.load(model_path))
        model.to(device)
        return model

    def create_evaluation_model(self, partition: str) -> EvaluationModel:
        metadata = get_metadata_with_offset(
            data_dir=IDUN_OCELOT_DATA_PATH, partition=partition
        )
        cell_transform = self._create_dual_transform(
            normalization=self.normalization, partition=partition, kind="cell"
        )
        tissue_transform = self._create_dual_transform(
            normalization=self.normalization, partition=partition, kind="tissue"
        )
        return SegformerJointPred2InputModule(
            metadata=metadata,
            cell_model=self.model,
            cell_transform=cell_transform,
            tissue_transform=tissue_transform,
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
            save_name=self.get_save_name(),
            device=device,
            checkpoint_interval=checkpoint_interval,
            break_after_one_iteration=break_after_one_iteration,
            scheduler=scheduler,
            do_save_model_and_plot=do_save_model_and_plot,
            validation_function=self.get_evaluation_function(partition="val"),
            training_func=run_training_joint_pred2input,
            weight_loss=self.weight_loss,
            freeze_tissue=self.freeze_tissue,
        )

        return best_model_path


class SegformerAdditiveJointPred2DecoderTrainable(SegformerJointPred2InputTrainable):

    def create_model(
        self,
        backbone_name: str,
        pretrained: bool,
        device: torch.device,
        model_path: Optional[str] = None,
    ) -> nn.Module:

        model = SegformerAdditiveJointPred2DecoderModel(
            backbone_model=backbone_name,
            pretrained_dataset=self.pretrained_dataset,
            input_image_size=self.cell_image_input_size,
            output_image_size=1024,
            freeze_tissue=self.freeze_tissue,
        )
        if model_path is not None:
            model.load_state_dict(torch.load(model_path))
        model.to(device)
        return model

    def create_evaluation_model(self, partition: str) -> EvaluationModel:
        metadata = get_metadata_with_offset(
            data_dir=IDUN_OCELOT_DATA_PATH, partition=partition
        )
        cell_transform = self._create_dual_transform(
            normalization=self.normalization, partition=partition, kind="cell"
        )
        tissue_transform = self._create_dual_transform(
            normalization=self.normalization, partition=partition, kind="tissue"
        )
        return SegformerTissueToCellDecoderModule(
            metadata=metadata,
            cell_model=self.model,
            cell_transform=cell_transform,
            tissue_transform=tissue_transform,
        )


class ViTUnetTrainable(Trainable):

    def __init__(
        self,
        normalization: str,
        batch_size: int,
        pretrained: bool,
        device: torch.device,
        backbone_model: str,
        pretrained_dataset: str,
        data_dir: str,
        cell_image_input_size: Optional[int] = 1024,
        exclude_bad_images=False,
    ):
        self.name = "ViTUnet"
        self.pretrained_dataset = pretrained_dataset
        self.cell_image_input_size = cell_image_input_size
        super().__init__(
            normalization=normalization,
            batch_size=batch_size,
            pretrained=pretrained,
            device=device,
            backbone_model=backbone_model,
            data_dir=data_dir,
            exclude_bad_images=exclude_bad_images,
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

        if self.cell_image_input_size is not None:
            extra_transform_image = A.Resize(
                height=self.cell_image_input_size, width=self.cell_image_input_size
            )
            transforms = self.build_transform_function_with_extra_transforms(
                transforms=transforms, extra_transform_image=extra_transform_image
            )

        return transforms

    def _create_dataloader(self, data_dir, partition: str) -> DataLoader:
        tissue_image_files, tissue_target_files = get_ocelot_files(
            data_dir=data_dir,
            partition=partition,
            zoom="tissue",
            macenko=self.macenko_normalize,
            exclude_bad_images=self.exclude_bad_images,
        )
        if partition == "train":
            shuffle = True
            transform = self.train_transforms
        else:
            shuffle = False
            transform = self.val_transforms

        dataset = TissueDataset(
            image_files=tissue_image_files,
            seg_files=tissue_target_files,
            transform=transform,
            image_shape=(self.cell_image_input_size, self.cell_image_input_size),
        )

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            drop_last=True,
            shuffle=shuffle,
        )

        return dataloader

    def create_train_dataloader(self, data_dir: str) -> DataLoader:
        return self._create_dataloader(data_dir=data_dir, partition="train")

    def create_model(
        self,
        backbone_name: str,
        pretrained: bool,
        device: torch.device,
        model_path: Optional[str] = None,
    ) -> nn.Module:

        model = ViTUNetModel(
            pretrained_dataset=self.pretrained_dataset if pretrained else None, input_spatial_shape=self.cell_image_input_size
        )
        if model_path is not None:
            model.load_state_dict(torch.load(model_path))
        model.to(device)
        return model

    def get_evaluation_function(self, partition: str):
        evaluation_function = create_tissue_evaluation_function(
            model=self.model,
            dataloader=self._create_dataloader(
                data_dir=IDUN_OCELOT_DATA_PATH, partition=partition
            ),
            loss_function=DiceLossWrapper(softmax=True, to_onehot_y=True),
            device=self.device,
        )
        return evaluation_function

    def create_evaluation_model(self, partition: str) -> EvaluationModel:
        return None


def main():
    # General training params
    do_save: bool = False
    do_eval: bool = True
    num_epochs = 4
    batch_size = 2
    warmup_epochs = 0
    learning_rate = 1e-4
    checkpoint_interval = 10
    break_after_one_iteration = False

    # Model specific params
    normalization = "macenko"
    pretrained = True
    backbone_model = "b0"
    pretrained_dataset = "ade"
    cell_image_input_size = 512
    leak_labels = False

    device = torch.device("cuda")

    trainable = SegformerTissueTrainable(
        normalization=normalization,
        batch_size=batch_size,
        pretrained=pretrained,
        device=device,
        backbone_model=backbone_model,
        pretrained_dataset=pretrained_dataset,
        tissue_image_input_size=cell_image_input_size,
    )

    # loss_function = DiceLoss(softmax=True)
    loss_function = DiceLossWrapper(softmax=True, to_onehot_y=True)
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
        do_save_model_and_plot=do_save,
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

    val_score = val_evaluation_function()
    print(f"val score: {val_score}")
    test_score = test_evaluation_function()
    print(f"test score: {test_score}")


if __name__ == "__main__":
    main()

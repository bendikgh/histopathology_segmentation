import argparse
import cv2
import json
import os
import sys
import torch

import albumentations as A
import seaborn as sns

from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    get_polynomial_decay_schedule_with_warmup,
)

sys.path.append(os.getcwd())

from src.dataset import CellTissueSharingDataset
from src.models import TissueCellSharingSegformerModel
from src.utils.metrics import create_cellwise_evaluation_function
from src.utils.training import train, run_training_sharing, run_validation_sharing
from src.utils.utils import (
    get_metadata_with_offset,
    get_ocelot_files,
    get_save_name,
    get_ocelot_args,
)
from src.utils.constants import CELL_IMAGE_MEAN, CELL_IMAGE_STD
from src.loss import DiceLossForTissueCellSharing

from ocelot23algo.user.inference import SegformerTissueFromFile


def build_transform(transforms, extra_transform_cell_tissue):

    def transform(image, tissue_image, cell_label, tissue_label):

        transformed = transforms(
            image=image,
            tissue_image=tissue_image,
            cell_label=cell_label,
            tissue_label=tissue_label,
        )

        transformed_cell_image = transformed["image"]
        transformed_cell_label = transformed["cell_label"]
        transformed_tissue_image = transformed["tissue_image"]
        transformed_tissue_label = transformed["tissue_label"]

        transformed = extra_transform_cell_tissue(
            image=transformed_cell_image, tissue_image=transformed_tissue_image
        )
        transformed_cell_image, transformed_tissue_image = (
            transformed["image"],
            transformed["tissue_image"],
        )

        return {
            "image": transformed_cell_image,
            "cell_label": transformed_cell_label,
            "tissue_image": transformed_tissue_image,
            "tissue_label": transformed_tissue_label,
        }

    return transform


def main():
    sns.set_theme()
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args: argparse.Namespace = get_ocelot_args()
    num_epochs: int = args.epochs
    batch_size: int = args.batch_size
    data_dir: str = args.data_dir
    checkpoint_interval: int = args.checkpoint_interval
    backbone_model: str = args.backbone
    dropout_rate: float = args.dropout
    learning_rate: float = args.learning_rate
    pretrained: bool = args.pretrained
    warmup_epochs: int = args.warmup_epochs
    do_save: bool = args.do_save
    do_eval: bool = args.do_eval
    break_after_one_iteration: bool = args.break_early
    normalization: str = args.normalization
    pretrained_dataset: str = args.pretrained_dataset
    resize: int = args.resize
    device: int = torch.device(args.device)
    id: str = args.id

    print("Training with the following parameters:")
    print(f"Data directory: {data_dir}")
    print(f"Number of epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Backbone model: {backbone_model}")
    print(f"Dropout rate: {dropout_rate}")
    print(f"Learning rate: {learning_rate}")
    print(f"Checkpoint interval: {checkpoint_interval}")
    print(f"Pretrained: {pretrained}")
    print(f"Warmup epochs: {warmup_epochs}")
    print(f"Do save: {do_save}")
    print(f"Do eval: {do_eval}")
    print(f"Break after one iteration: {break_after_one_iteration}")
    print(f"Device: {device}")
    print(f"Normalization: {normalization}")
    print(f"Resize: {resize}")
    print(f"pretrained dataset: {pretrained_dataset}")
    print(f"ID: {id}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")

    # Find the correct files

    train_transform_list = [
        A.GaussianBlur(),
        A.GaussNoise(var_limit=(0.1, 0.3), p=0.5),
        A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
    ]
    val_transform_list = []

    macenko = "macenko" in normalization
    if "imagenet" in normalization:
        train_transform_list.append(A.Normalize())
        val_transform_list.append(A.Normalize())
    elif "cell" in normalization:
        train_transform_list.append(
            A.Normalize(mean=CELL_IMAGE_MEAN, std=CELL_IMAGE_STD)
        )
        val_transform_list.append(A.Normalize(mean=CELL_IMAGE_MEAN, std=CELL_IMAGE_STD))

    train_transforms = A.Compose(
        train_transform_list,
        additional_targets={
            "tissue_image": "image",
            "cell_label": "mask",
            "tissue_label": "mask",
        },
    )
    val_transforms = A.Compose(
        val_transform_list,
        additional_targets={
            "tissue_image": "image",
            "cell_label": "mask",
            "tissue_label": "mask",
        },
    )

    if resize is not None:

        extra_transform_cell_tissue = A.Compose(
            [A.Resize(height=resize, width=resize, interpolation=cv2.INTER_NEAREST)],
            additional_targets={"tissue_image": "image"},
        )

        train_transforms = build_transform(
            transforms=train_transforms,
            extra_transform_cell_tissue=extra_transform_cell_tissue,
        )
        val_transforms = build_transform(
            transforms=val_transforms,
            extra_transform_cell_tissue=extra_transform_cell_tissue,
        )

    train_cell_image_files, train_cell_seg_files = get_ocelot_files(
        data_dir=data_dir, partition="train", zoom="cell", macenko=macenko
    )
    val_cell_image_files, val_cell_seg_files = get_ocelot_files(
        data_dir=data_dir, partition="val", zoom="cell", macenko=macenko
    )

    train_tissue_image_files, train_tissue_target_files = get_ocelot_files(
        data_dir=data_dir, partition="train", zoom="tissue", macenko=macenko
    )
    val_tissue_image_files, val_tissue_target_files = get_ocelot_files(
        data_dir=data_dir, partition="val", zoom="tissue", macenko=macenko
    )

    ## Tissue
    train_image_nums = [x.split("/")[-1].split(".")[0] for x in train_cell_image_files]
    val_image_nums = [x.split("/")[-1].split(".")[0] for x in val_cell_image_files]

    # Making sure only the appropriate numbers are used
    train_tissue_image_files = [
        file
        for file in train_tissue_image_files
        if file.split("/")[-1].split(".")[0] in train_image_nums
    ]
    val_tissue_image_files = [
        file
        for file in val_tissue_image_files
        if file.split("/")[-1].split(".")[0] in val_image_nums
    ]

    train_tissue_target_files = [
        file
        for file in train_tissue_target_files
        if file.split("/")[-1].split(".")[0] in train_image_nums
    ]
    val_tissue_target_files = [
        file
        for file in val_tissue_target_files
        if file.split("/")[-1].split(".")[0] in val_image_nums
    ]

    train_tissue_image_files.sort(key=lambda x: int(x.split("/")[-1].split(".")[0]))
    val_tissue_image_files.sort(key=lambda x: int(x.split("/")[-1].split(".")[0]))
    train_tissue_target_files.sort(key=lambda x: int(x.split("/")[-1].split(".")[0]))
    val_tissue_target_files.sort(key=lambda x: int(x.split("/")[-1].split(".")[0]))

    # Create dataset and dataloader
    train_dataset = CellTissueSharingDataset(
        cell_image_files=train_cell_image_files,
        cell_target_files=train_cell_seg_files,
        tissue_image_files=train_tissue_image_files,
        tissue_target_files=train_tissue_target_files,
        transform=train_transforms,
        image_shape=(resize, resize) if resize else (1024, 1024),
    )

    val_dataset = CellTissueSharingDataset(
        cell_image_files=val_cell_image_files,
        cell_target_files=val_cell_seg_files,
        tissue_image_files=val_tissue_image_files,
        tissue_target_files=val_tissue_target_files,
        transform=train_transforms,
        image_shape=(resize, resize) if resize else (1024, 1024),
    )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    # val_dataloader = DataLoader(
    #     dataset=val_dataset,
    #     batch_size=batch_size,
    #     drop_last=True,
    # )

    metadata_path = os.path.join(data_dir, "metadata.json")
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    model = TissueCellSharingSegformerModel(
        backbone_name=backbone_model,
        num_classes=3,
        num_channels=6,
        pretrained_dataset=pretrained_dataset,
        metadata=list(metadata["sample_pairs"].values()),
    )
    model.to(device)

    loss_function = DiceLossForTissueCellSharing(softmax=True)

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = get_polynomial_decay_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_epochs,
        num_training_steps=num_epochs,
        power=1,
    )

    val_metadata = get_metadata_with_offset(data_dir=data_dir, partition="val")
    test_metadata = get_metadata_with_offset(data_dir=data_dir, partition="test")

    val_evaluation_model = SegformerTissueFromFile(
        metadata=val_metadata, cell_model=model, device=device, tissue_model_path=None
    )
    test_evaluation_model = SegformerTissueFromFile(
        metadata=test_metadata, cell_model=model, device=device, tissue_model_path=None
    )

    transform_val_test = None
    if resize:
        transform_val_test = A.Compose(
            [A.Resize(height=resize, width=resize, interpolation=cv2.INTER_NEAREST)],
            additional_targets={"tissue": "image"},
        )
    val_evaluation_function = create_cellwise_evaluation_function(
        evaluation_model=val_evaluation_model,
        tissue_file_folder="annotations/val/predicted_cropped_tissue",
        transform=transform_val_test,
    )
    test_evaluation_function = create_cellwise_evaluation_function(
        evaluation_model=test_evaluation_model,
        tissue_file_folder="annotations/test/predicted_cropped_tissue",
        transform=transform_val_test,
    )

    save_name = get_save_name(
        model_name="segformer-tissue-cell-sharing",
        pretrained=pretrained,
        learning_rate=learning_rate,
        dropout_rate=dropout_rate,
        backbone_model=backbone_model,
        normalization=normalization,
        pretrained_dataset=pretrained_dataset,
        resize=resize,
        id=id,
    )
    print(f"Save name: {save_name}")

    best_model_path = train(
        num_epochs=num_epochs,
        train_dataloader=train_dataloader,
        validation_function=val_evaluation_function,
        model=model,
        loss_function=loss_function,
        optimizer=optimizer,
        device=device,
        save_name=save_name,
        checkpoint_interval=checkpoint_interval,
        break_after_one_iteration=break_after_one_iteration,
        scheduler=scheduler,
        do_save_model_and_plot=do_save,
        training_func=run_training_sharing,
    )

    print("Training complete!")
    if not do_eval:
        return

    print(f"Best model: {best_model_path}\n")
    print(f"Calculating validation score")
    val_mf1 = val_evaluation_function(partition="val")
    print(f"Validation mF1: {val_mf1:.4f}")

    print(f"\nCalculating test score")
    test_mf1 = test_evaluation_function(partition="test")
    print(f"Test mF1: {test_mf1:.4f}")


if __name__ == "__main__":
    main()

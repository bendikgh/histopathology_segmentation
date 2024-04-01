import argparse
import torch
import albumentations as A
import seaborn as sns
from copy import copy

from monai.losses import DiceLoss
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    get_polynomial_decay_schedule_with_warmup,
)

from dataset import CellOnlyDataset
from models import CustomSegformerModel
from ocelot23algo.user.inference import SegformerCellOnlyModel
from src.utils.metrics import predict_and_evaluate
from utils.training import train
from utils.utils import (
    get_metadata_with_offset,
    get_ocelot_files,
    get_save_name,
    get_ocelot_args,
)
from utils.constants import CELL_IMAGE_MEAN, CELL_IMAGE_STD


def build_transform(transforms, extra_transform_image):
    def transform(image, mask):
        transformed = transforms(image=image, mask=mask)
        transformed_image, transformed_label = transformed["image"], transformed["mask"]
        transformed_image = extra_transform_image(image=transformed_image)["image"]
        return {"image": transformed_image, "mask": transformed_label}

    return transform


def main():
    sns.set_theme()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args: argparse.Namespace = get_ocelot_args()
    num_epochs: int = args.epochs
    batch_size: int = args.batch_size
    data_dir: str = args.data_dir
    checkpoint_interval: int = args.checkpoint_interval
    backbone_model: str = args.backbone
    learning_rate: float = args.learning_rate
    pretrained: bool = args.pretrained
    warmup_epochs: int = args.warmup_epochs
    do_save: bool = args.do_save
    do_eval: bool = args.do_eval
    break_after_one_iteration: bool = args.break_early
    normalization: str = args.normalization
    pretrained_dataset: str = args.pretrained_dataset
    resize: int = args.resize
    id: str = args.id

    print("Training with the following parameters:")
    print(f"Data directory: {data_dir}")
    print(f"Number of epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Backbone model: {backbone_model}")
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

    train_transforms = A.Compose(train_transform_list)
    val_transforms = A.Compose(val_transform_list)

    if resize is not None:
        extra_transform_image = A.Resize(height=resize, width=resize)

        train_transforms = build_transform(
            transforms=train_transforms, extra_transform_image=extra_transform_image
        )
        val_transforms = build_transform(
            transforms=val_transforms, extra_transform_image=extra_transform_image
        )

    train_image_files, train_seg_files = get_ocelot_files(
        data_dir=data_dir, partition="train", zoom="cell", macenko=macenko
    )
    val_image_files, val_seg_files = get_ocelot_files(
        data_dir=data_dir, partition="val", zoom="cell", macenko=macenko
    )

    # Create dataset and dataloader
    train_dataset = CellOnlyDataset(
        cell_image_files=train_image_files,
        cell_target_files=train_seg_files,
        transform=train_transforms,
        image_shape=(resize, resize) if resize else (1024, 1024),
    )
    val_dataset = CellOnlyDataset(
        cell_image_files=val_image_files,
        cell_target_files=val_seg_files,
        transform=val_transforms,
        image_shape=(resize, resize) if resize else (1024, 1024),
    )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        drop_last=True,
    )

    model = CustomSegformerModel(
        backbone_name=backbone_model,
        num_classes=3,
        num_channels=3,
        pretrained_dataset=pretrained_dataset,
    )
    model.to(device)

    loss_function = DiceLoss(softmax=True)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = get_polynomial_decay_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_epochs,
        num_training_steps=num_epochs,
        power=1,
    )
    save_name = get_save_name(
        model_name="deeplabv3plus-cell-only",
        pretrained=pretrained,
        learning_rate=learning_rate,
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
        val_dataloader=val_dataloader,
        model=model,
        loss_function=loss_function,
        optimizer=optimizer,
        device=device,
        save_name=save_name,
        checkpoint_interval=checkpoint_interval,
        break_after_one_iteration=break_after_one_iteration,
        scheduler=scheduler,
        do_save_model_and_plot=do_save,
    )

    print("Training complete!")
    if not (do_save and do_eval):
        return

    val_metadata = get_metadata_with_offset(data_dir=data_dir, partition="val")
    val_evaluation_model = SegformerCellOnlyModel(
        metadata=val_metadata, cell_model=model
    )

    print(f"Best model: {best_model_path}\n")
    print("Calculating validation score:")
    val_mf1 = predict_and_evaluate(
        evaluation_model=val_evaluation_model,
        partition="val",
        tissue_file_folder="images/val/tissue_macenko",
    )
    print(f"Validation mF1: {val_mf1:.4f}")
    print("\nCalculating test score:")
    test_metadata = get_metadata_with_offset(data_dir=data_dir, partition="test")
    test_evaluation_model = SegformerCellOnlyModel(
        metadata=test_metadata, cell_model=model
    )
    test_mf1 = predict_and_evaluate(
        evaluation_model=test_evaluation_model,
        partition="test",
        tissue_file_folder="images/test/tissue_macenko",
    )
    print(f"Test mF1: {test_mf1:.4f}")


if __name__ == "__main__":
    main()

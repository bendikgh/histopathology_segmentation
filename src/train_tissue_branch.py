import argparse
import torch
import albumentations as A
import seaborn as sns

from glob import glob
from torch.utils.data import DataLoader
from loss import DiceLossWrapper
from torch.optim import AdamW
from transformers import (
    get_polynomial_decay_schedule_with_warmup,
)

from dataset import TissueDataset
from models import DeepLabV3plusModel
from utils.training import train
from utils.utils import get_ocelot_files, get_save_name, get_ocelot_args
from utils.constants import CELL_IMAGE_MEAN, CELL_IMAGE_STD


def main():
    sns.set_theme()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    break_after_one_iteration: bool = args.break_early
    normalization: str = args.normalization
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
    print(f"Break after one iteration: {break_after_one_iteration}")
    print(f"Device: {device}")
    print(f"Normalization: {normalization}")
    print(f"Id: {id}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")    

    train_transform_list = [
        A.GaussianBlur(blur_limit=(3, 7), p=0.5),
        A.GaussNoise(var_limit=(0.1, 0.3), p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.3, saturation=0.2, hue=0.1),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
    ]
    val_transform_list = []

    macenko = False
    if normalization == "imagenet":
        train_transform_list.append(A.Normalize())
        val_transform_list.append(A.Normalize())
    elif normalization == "cell":
        train_transform_list.append(
            A.Normalize(mean=CELL_IMAGE_MEAN, std=CELL_IMAGE_STD)
        )
        val_transform_list.append(A.Normalize(mean=CELL_IMAGE_MEAN, std=CELL_IMAGE_STD))
    elif normalization == "macenko":
        macenko = True
    elif normalization == "macenko + cell":
        macenko = True
        train_transform_list.append(
            A.Normalize(mean=CELL_IMAGE_MEAN, std=CELL_IMAGE_STD)
        )
        val_transform_list.append(A.Normalize(mean=CELL_IMAGE_MEAN, std=CELL_IMAGE_STD))
    elif normalization == "macenko + imagenet":
        macenko = True
        train_transform_list.append(A.Normalize())
        val_transform_list.append(A.Normalize())

    train_tissue_image_files, train_tissue_target_files = get_ocelot_files(
        data_dir=data_dir, partition="train", zoom="tissue", macenko=macenko
    )
    val_tissue_image_files, val_tissue_target_files = get_ocelot_files(
        data_dir=data_dir, partition="val", zoom="tissue", macenko=macenko
    )
    
    train_transforms = A.Compose(train_transform_list)
    val_transforms = A.Compose(val_transform_list)

    train_tissue_dataset = TissueDataset(
        image_files=train_tissue_image_files,
        seg_files=train_tissue_target_files,
        transform=train_transforms,
    )
    val_tissue_dataset = TissueDataset(
        image_files=val_tissue_image_files,
        seg_files=val_tissue_target_files,
        transform=val_transforms,
    )

    train_tissue_dataloader = DataLoader(
        dataset=train_tissue_dataset,
        batch_size=batch_size,
        drop_last=False,
        shuffle=True,
    )
    val_tissue_dataloader = DataLoader(
        dataset=val_tissue_dataset,
        batch_size=batch_size,
        drop_last=False,
    )

    model = DeepLabV3plusModel(
        backbone_name=backbone_model,
        num_classes=3,
        num_channels=3,
        pretrained=pretrained,
        dropout_rate=dropout_rate,
    )
    model.to(device)

    loss_function = DiceLossWrapper(to_onehot_y=True, softmax=True)

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = get_polynomial_decay_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_epochs,
        num_training_steps=num_epochs,
        power=1,
    )
    save_name = get_save_name(
        model_name="deeplabv3plus-tissue-branch",
        pretrained=pretrained,
        learning_rate=learning_rate,
        dropout_rate=dropout_rate,
        backbone_model=backbone_model,
        id=id
    )
    print(f"Save name: {save_name}")
    train(
        num_epochs=num_epochs,
        train_dataloader=train_tissue_dataloader,
        val_dataloader=val_tissue_dataloader,
        model=model,
        loss_function=loss_function,
        optimizer=optimizer,
        device=device,
        save_name=save_name,
        checkpoint_interval=checkpoint_interval,
        break_after_one_iteration=break_after_one_iteration,
        scheduler=scheduler,
        do_save_model_and_plot=do_save,  # NOTE: Important to change this before training
    )

if __name__ == "__main__":
    main()

import argparse
import os
import torch
import albumentations as A
import seaborn as sns

from glob import glob
from monai.losses import DiceLoss
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    get_polynomial_decay_schedule_with_warmup,
)

# from deeplabv3.network.modeling import _segm_resnet
from models import DeepLabV3plusModel
from utils.training import train
from utils.utils import get_ocelot_files, get_save_name, get_ocelot_args


# Function for crop and scale tissue image
from dataset import CellTissueDataset


def main():
    sns.set_theme()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Parse command line arguments
    args: argparse.Namespace = get_ocelot_args()
    num_epochs = args.epochs
    batch_size = args.batch_size
    data_dir = args.data_dir
    checkpoint_interval = args.checkpoint_interval
    backbone_model = args.backbone
    dropout_rate = args.dropout
    learning_rate = args.learning_rate
    pretrained = args.pretrained
    warmup_epochs = args.warmup_epochs
    do_save: bool = args.do_save
    break_after_one_iteration: bool = args.break_early

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
    print(f"Number of GPUs: {torch.cuda.device_count()}")

    train_cell_image_files, train_cell_target_files = get_ocelot_files(
        data_dir=data_dir, partition="train", zoom="cell"
    )
    val_cell_image_files, val_cell_target_files = get_ocelot_files(
        data_dir=data_dir, partition="val", zoom="cell"
    )

    train_tissue_predicted = glob(
        os.path.join(data_dir, "annotations/train/pred_tissue/*")
    )
    val_tissue_predicted = glob(os.path.join(data_dir, "annotations/val/pred_tissue/*"))
    train_tissue_predicted.sort(key=lambda x: int(x.split("/")[-1].split(".")[0]))
    val_tissue_predicted.sort(key=lambda x: int(x.split("/")[-1].split(".")[0]))

    # Create dataset and dataloader
    train_transforms = A.Compose(
        [
            A.GaussianBlur(blur_limit=(3, 7), p=0.5),
            A.GaussNoise(var_limit=(0.1, 0.3), p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.3, saturation=0.2, hue=0.1, p=1),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
        ],
        additional_targets={"mask1": "mask", "mask2": "mask"},
    )
    # val_transforms = A.Compose(
    #     [A.Normalize()], additional_targets={"mask1": "mask", "mask2": "mask"}
    # )

    train_cell_tissue_dataset = CellTissueDataset(
        cell_image_files=train_cell_image_files,
        cell_target_files=train_cell_target_files,
        tissue_pred_files=train_tissue_predicted,
        transform=train_transforms,
    )
    val_cell_tissue_dataset = CellTissueDataset(
        cell_image_files=val_cell_image_files,
        cell_target_files=val_cell_target_files,
        tissue_pred_files=val_tissue_predicted,
        # transform=val_transforms,
    )

    train_cell_tissue_dataloader = DataLoader(
        dataset=train_cell_tissue_dataset,
        batch_size=batch_size,
        drop_last=True,
        shuffle=True,
    )
    val_cell_tissue_dataloader = DataLoader(
        dataset=val_cell_tissue_dataset, batch_size=batch_size
    )

    model: nn.Module = DeepLabV3plusModel(
        backbone_name=backbone_model,
        num_classes=3,
        num_channels=6,
        pretrained=pretrained,
        dropout_rate=dropout_rate,
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
        model_name="deeplabv3plus-cell-branch",
        pretrained=pretrained,
        learning_rate=learning_rate,
        dropout_rate=dropout_rate,
        backbone_model=backbone_model,
    )
    print(f"Save name: {save_name}")

    train(
        num_epochs=num_epochs,
        train_dataloader=train_cell_tissue_dataloader,
        val_dataloader=val_cell_tissue_dataloader,
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


if __name__ == "__main__":
    main()

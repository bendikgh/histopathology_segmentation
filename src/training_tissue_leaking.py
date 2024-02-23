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

from dataset import CellTissueDataset
from models import DeepLabV3plusModel
from utils.training import train
from utils.utils import get_ocelot_files, get_save_name, get_ocelot_args


def main():
    sns.set_theme()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    print(f"Device: {device}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")

    # Find the correct files
    train_cell_seg = sorted(
        glob(os.path.join(data_dir, "annotations/train/segmented_cell/*"))
    )
    train_tissue_seg = []
    train_input_img = []

    for img_path in train_cell_seg:
        ending = img_path.split("/")[-1].split(".")[0]
        tissue_seg_path = glob(
            os.path.join(data_dir, "annotations/train/cropped_tissue/" + ending + "*")
        )[0]
        input_img_path = glob(
            os.path.join(data_dir, "images/train/cell/" + ending + "*")
        )[0]
        train_tissue_seg.append(tissue_seg_path)
        train_input_img.append(input_img_path)

    val_cell_seg = sorted(
        glob(os.path.join(data_dir, "annotations/val/segmented_cell/*"))
    )
    val_tissue_seg = []
    val_input_img = []

    for img_path in val_cell_seg:
        ending = img_path.split("/")[-1].split(".")[0]
        tissue_seg_path = glob(
            os.path.join(data_dir, "annotations/val/cropped_tissue/" + ending + "*")
        )[0]
        input_img_path = glob(
            os.path.join(data_dir, "images/val/cell/" + ending + "*")
        )[0]
        val_tissue_seg.append(tissue_seg_path)
        val_input_img.append(input_img_path)

    # Create dataset and dataloader
    transforms = A.Compose(
        [
            A.GaussianBlur(blur_limit=(3, 7), p=0.5),
            A.GaussNoise(var_limit=(0.1, 0.3), p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.3, saturation=0.2, hue=0.1, p=1),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
        ],
        additional_targets={"mask1": "mask", "mask2": "mask"},
    )

    train_dataset = CellTissueDataset(
        cell_image_files=train_input_img,
        cell_target_files=train_cell_seg,
        tissue_pred_files=train_tissue_seg,
        transform=transforms,
        debug=False,
    )
    val_dataset = CellTissueDataset(
        cell_image_files=val_input_img,
        cell_target_files=val_cell_seg,
        tissue_pred_files=val_tissue_seg,
        debug=False,
    )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        drop_last=True,
        shuffle=True,
    )
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        drop_last=True,
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
        model_name="deeplabv3plus-tissue-leaking",
        pretrained=pretrained,
        learning_rate=learning_rate,
        dropout_rate=dropout_rate,
        backbone_model=backbone_model,
    )
    print(f"Save name: {save_name}")

    train(
        num_epochs=num_epochs,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        model=model,
        loss_function=loss_function,
        optimizer=optimizer,
        device=device,
        save_name=save_name,
        checkpoint_interval=checkpoint_interval,
        break_after_one_iteration=True,
        scheduler=scheduler,
        do_save_model_and_plot=False,
    )


if __name__ == "__main__":
    main()

import argparse
import os
import torch
import albumentations as A

from glob import glob
from monai.losses import DiceLoss
from torch.utils.data import DataLoader
from torch.optim import Adam

from deeplabv3.network.modeling import _segm_resnet
from train_utils import train
from dataset import TissueLeakingDataset


def main():
    default_epochs = 2
    default_batch_size = 2
    default_data_dir = "ocelot_data"
    default_checkpoint_interval = 5
    default_backbone_model = "resnet34"
    default_dropout_rate = 0.3
    default_learning_rate = 1e-4

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train Deeplabv3plus model")
    parser.add_argument(
        "--epochs", type=int, default=default_epochs, help="Number of epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=default_batch_size, help="Batch size"
    )
    parser.add_argument(
        "--data-dir", type=str, default=default_data_dir, help="Path to data directory"
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=default_checkpoint_interval,
        help="Checkpoint Interval",
    )
    parser.add_argument(
        "--backbone", type=str, default=default_backbone_model, help="Backbone model"
    )
    parser.add_argument(
        "--dropout", type=float, default=default_dropout_rate, help="Dropout rate"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=default_learning_rate,
        help="Learning rate",
    )

    args = parser.parse_args()

    num_epochs = args.epochs
    batch_size = args.batch_size
    data_dir = args.data_dir
    checkpoint_interval = args.checkpoint_interval
    backbone_model = args.backbone
    dropout_rate = args.dropout
    learning_rate = args.learning_rate

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Training with the following parameters:")
    print(f"Data directory: {data_dir}")
    print(f"Number of epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Backbone model: {backbone_model}")
    print(f"Dropout rate: {dropout_rate}")
    print(f"Learning rate: {learning_rate}")
    print(f"Checkpoint interval: {checkpoint_interval}")
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

    train_dataset = TissueLeakingDataset(
        input_files=train_input_img,
        cell_seg_files=train_cell_seg,
        tissue_seg_files=train_tissue_seg,
        transform=transforms,
    )
    val_dataset = TissueLeakingDataset(
        input_files=val_input_img,
        cell_seg_files=val_cell_seg,
        tissue_seg_files=val_tissue_seg,
    )

    train_dataloader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, drop_last=True, shuffle=True
    )
    val_dataloader = DataLoader(
        dataset=val_dataset, batch_size=batch_size, drop_last=True
    )

    # Create model and optimizer
    model = _segm_resnet(
        name="deeplabv3plus",
        backbone_name=backbone_model,
        num_classes=3,
        output_stride=8,
        pretrained_backbone=True,
        dropout_rate=dropout_rate,
        num_channels=6,
    )
    model.to(device)

    loss_function = DiceLoss(softmax=True)
    optimizer = Adam(model.parameters(), lr=learning_rate)

    train(
        num_epochs=num_epochs,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        model=model,
        loss_function=loss_function,
        optimizer=optimizer,
        device=device,
        checkpoint_interval=checkpoint_interval,
        break_after_one_iteration=False,
        dropout_rate=dropout_rate,
        backbone=backbone_model,
        model_name="tissue_leaking",
    )


if __name__ == "__main__":
    main()

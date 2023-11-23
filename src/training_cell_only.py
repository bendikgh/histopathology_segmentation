import argparse
import os
import torch

from glob import glob
from monai.losses import DiceLoss
from monai.data import ImageDataset
from torch.utils.data import DataLoader
from torch.optim import Adam

from deeplabv3.network.modeling import _segm_resnet
from train_utils import train


def main():
    default_epochs = 2
    default_batch_size = 2

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train Deeplabv3plus model")
    parser.add_argument(
        "--epochs", type=int, default=default_epochs, help="Number of epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=default_batch_size, help="Batch size"
    )
    parser.add_argument(
        "--data_dir", type=str, default="ocelot_data", help="Path to data directory"
    )
    args = parser.parse_args()

    num_epochs = args.epochs
    batch_size = args.batch_size
    data_dir = args.data_dir

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Training with the following parameters:")
    print(f"Number of epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Device: {device}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")

    train_seg_files = glob(os.path.join(data_dir, "annotations/train/segmented_cell/*"))
    train_image_numbers = [
        file_name.split("/")[-1].split(".")[0] for file_name in train_seg_files
    ]
    train_image_files = [
        os.path.join(data_dir, "images/train/cell", image_number + ".jpg")
        for image_number in train_image_numbers
    ]

    val_seg_files = glob(os.path.join(data_dir, "annotations/val/segmented_cell/*"))
    val_image_numbers = [
        file_name.split("/")[-1].split(".")[0] for file_name in val_seg_files
    ]
    val_image_files = [
        os.path.join(data_dir, "images/val/cell", image_number + ".jpg")
        for image_number in val_image_numbers
    ]

    # Create dataset and dataloader
    train_dataset = ImageDataset(
        image_files=train_image_files, seg_files=train_seg_files
    )
    val_dataset = ImageDataset(image_files=val_image_files, seg_files=val_seg_files)

    train_dataloader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )
    val_dataloader = DataLoader(dataset=val_dataset)

    model = _segm_resnet(
        name="deeplabv3plus",
        backbone_name="resnet50",
        num_classes=3,
        output_stride=8,
        pretrained_backbone=True,
    )
    model.to(device)

    loss_function = DiceLoss()
    optimizer = Adam(model.parameters(), lr=1e-3)

    train(
        num_epochs=num_epochs,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        model=model,
        loss_function=loss_function,
        optimizer=optimizer,
        device=device,
        checkpoint_interval=5,
        break_after_one_iteration=True,
    )


if __name__ == "__main__":
    main()

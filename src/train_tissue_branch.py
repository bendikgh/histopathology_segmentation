import argparse
import os
import torch
import albumentations as A

from glob import glob
from monai.losses import DiceLoss
from torch.utils.data import DataLoader
from torch.optim import Adam

from deeplabv3.network.modeling import _segm_resnet
from utils.utils_train import train
from dataset import TissueDataset


def main():
    default_epochs = 2
    default_batch_size = 2
    default_data_dir = "ocelot_data"
    default_checkpoint_interval = 5
    default_backbone_model = "resnet50"
    default_dropout_rate = 0.3
    default_learning_rate = 1e-4
    default_pretrained = True
    default_warmup_epochs = 2
    default_decay_rate = 0.99

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
    parser.add_argument(
        "--pretrained", type=int, default=default_pretrained, help="Pretrained backbone"
    )
    parser.add_argument(
        "--warmup-epochs", type=int, default=default_warmup_epochs, help="Warmup epochs"
    )
    parser.add_argument(
        "--decay-rate", type=float, default=default_decay_rate, help="Decay rate"
    )

    args = parser.parse_args()

    num_epochs = args.epochs
    batch_size = args.batch_size
    data_dir = args.data_dir
    checkpoint_interval = args.checkpoint_interval
    backbone_model = args.backbone
    dropout_rate = args.dropout
    learning_rate = args.learning_rate
    pretrained = args.pretrained
    warmup_epochs = args.warmup_epochs
    decay_rate = args.decay_rate

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    print(f"Decay rate: {decay_rate}")
    print(f"Device: {device}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")

    # Find the correct files
    train_tissue_seg_files = glob(os.path.join(data_dir, "annotations/train/tissue/*"))
    train_tissue_image_numbers = [
        file_name.split("/")[-1].split(".")[0] for file_name in train_tissue_seg_files
    ]
    train_tissue_image_files = [
        os.path.join(data_dir, "images/train/tissue", image_number + ".jpg")
        for image_number in train_tissue_image_numbers
    ]

    val_tissue_seg_files = glob(os.path.join(data_dir, "annotations/val/tissue/*"))
    val_tissue_image_numbers = [
        file_name.split("/")[-1].split(".")[0] for file_name in val_tissue_seg_files
    ]
    val_tissue_image_files = [
        os.path.join(data_dir, "images/val/tissue", image_number + ".jpg")
        for image_number in val_tissue_image_numbers
    ]

    # Create dataset and dataloader
    transforms = A.Compose(
        [
            A.GaussianBlur(blur_limit=(3, 7), p=0.5),  # You can adjust the blur limit
            A.GaussNoise(
                var_limit=(0.1, 0.3), p=0.5
            ),  # Adjust var_limit for noise intensity
            A.ColorJitter(brightness=0.2, contrast=0.3, saturation=0.2, hue=0.1, p=1),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
        ]
    )

    train_tissue_dataset = TissueDataset(
        image_files=train_tissue_image_files,
        seg_files=train_tissue_seg_files,
        transform=transforms,
    )
    val_tissue_dataset = TissueDataset(
        image_files=val_tissue_image_files, seg_files=val_tissue_seg_files
    )

    train_tissue_dataloader = DataLoader(
        dataset=train_tissue_dataset, batch_size=batch_size, drop_last=True, shuffle=True
    )
    val_tissue_dataloader = DataLoader(dataset=val_tissue_dataset, batch_size=batch_size, drop_last=True)

    # Create model and optimizer
    model = _segm_resnet(
        name="deeplabv3plus",
        backbone_name=backbone_model,
        num_classes=3,
        num_channels=3,
        output_stride=8,
        pretrained_backbone=pretrained,
        dropout_rate=dropout_rate
    )
    model.to(device)

    loss_function = DiceLoss(softmax=True)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)

    train(
        num_epochs=num_epochs,
        train_dataloader=train_tissue_dataloader,
        val_dataloader=val_tissue_dataloader,
        model=model,
        loss_function=loss_function,
        optimizer=optimizer,
        device=device,
        checkpoint_interval=checkpoint_interval,
        break_after_one_iteration=False,
        dropout_rate=dropout_rate,
        backbone=backbone_model,
        model_name = "tissue_branch",
        scheduler=scheduler,
        warmup_scheduler=warmup_scheduler,
        warmup_epochs=warmup_epochs
    )


if __name__ == "__main__":
    main()

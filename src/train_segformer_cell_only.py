import argparse
import os
import torch
import albumentations as A
import seaborn as sns

from transformers import (
    SegformerForSemanticSegmentation,
    SegformerConfig,
    SegformerImageProcessor,
    get_polynomial_decay_schedule_with_warmup,
)
from glob import glob
from monai.losses import DiceLoss
from torch.utils.data import DataLoader
from torch.optim import AdamW
from dataset import SegformerDataset
from datetime import datetime

from src.utils.utils_train import (
    run_training_segformer,
    run_validation_segformer,
    train,
)

sns.set_theme()


def main():
    default_epochs = 2
    default_batch_size = 2
    default_data_dir = "ocelot_data"
    default_checkpoint_interval = 5
    default_backbone_model = "resnet50"
    default_learning_rate = 1e-4
    default_pretrained = True
    default_warmup_epochs = 2

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

    args = parser.parse_args()

    num_epochs = args.epochs
    batch_size = args.batch_size
    data_dir = args.data_dir
    checkpoint_interval = args.checkpoint_interval
    backbone_model = args.backbone
    learning_rate = args.learning_rate
    pretrained = args.pretrained
    warmup_epochs = args.warmup_epochs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Training with the following parameters:")
    print(f"Data directory: {data_dir}")
    print(f"Number of epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Backbone model: {backbone_model}")
    print(f"Learning rate: {learning_rate}")
    print(f"Checkpoint interval: {checkpoint_interval}")
    print(f"Pretrained: {pretrained}")
    print(f"Warmup epochs: {warmup_epochs}")
    print(f"Device: {device}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")

    # Preparing data
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

    test_seg_files = glob(os.path.join(data_dir, "annotations/test/segmented_cell/*"))
    test_image_numbers = [
        file_name.split("/")[-1].split(".")[0] for file_name in test_seg_files
    ]
    test_image_files = [
        os.path.join(data_dir, "images/test/cell", image_number + ".jpg")
        for image_number in test_image_numbers
    ]

    # Preparing datasets
    image_processor = SegformerImageProcessor(do_resize=False)
    train_dataset = SegformerDataset(
        train_image_files, train_seg_files, transform=image_processor.preprocess
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = SegformerDataset(
        val_image_files, val_seg_files, transform=image_processor.preprocess
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_dataset = SegformerDataset(
        test_image_files, test_seg_files, transform=image_processor.preprocess
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Creating model
    configuration = SegformerConfig(num_labels=3, num_channels=3)
    model = SegformerForSemanticSegmentation(configuration)
    model.to(device)

    # Setting training parameters
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = get_polynomial_decay_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_epochs,
        num_training_steps=num_epochs,
        power=1,
    )
    loss_fn = DiceLoss(softmax=True, to_onehot_y=True)
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_name = f"segformer_tryout_{current_time}"

    train(
        num_epochs=num_epochs,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        model=model,
        loss_function=loss_fn,
        optimizer=optimizer,
        device=device,
        save_name=save_name,
        checkpoint_interval=checkpoint_interval,
        break_after_one_iteration=False,
        scheduler=scheduler,
        training_func=run_training_segformer,
        validation_function=run_validation_segformer,
    )


if __name__ == "__main__":
    main()

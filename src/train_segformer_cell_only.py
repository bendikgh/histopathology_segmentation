import argparse
import torch
import albumentations as A
import seaborn as sns

from transformers import (
    SegformerForSemanticSegmentation,
    SegformerConfig,
    get_polynomial_decay_schedule_with_warmup,
    AutoImageProcessor,
    SegformerModel,
)
from monai.losses import DiceLoss
from torch.utils.data import DataLoader
from torch.optim import AdamW

from dataset import CellOnlyDataset
from utils.training import (
    run_training_segformer,
    run_validation_segformer,
    train,
)
from utils.utils import get_ocelot_files, get_save_name, get_ocelot_args


def main():
    sns.set_theme()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args: argparse.Namespace = get_ocelot_args()
    num_epochs = args.epochs
    batch_size = args.batch_size
    data_dir = args.data_dir
    checkpoint_interval = args.checkpoint_interval
    learning_rate = args.learning_rate
    warmup_epochs = args.warmup_epochs
    pretrained = args.pretrained

    print("Training with the following parameters:")
    print(f"Data directory: {data_dir}")
    print(f"Number of epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Checkpoint interval: {checkpoint_interval}")
    print(f"Warmup epochs: {warmup_epochs}")
    print(f"Device: {device}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")

    # Preparing data
    train_image_files, train_seg_files = get_ocelot_files(
        data_dir=data_dir, partition="train", zoom="cell"
    )
    val_image_files, val_seg_files = get_ocelot_files(
        data_dir=data_dir, partition="val", zoom="cell"
    )

    # Creating model
    configuration = configuration = SegformerConfig(
        num_labels=3,
        num_channels=3,
        depths=[3, 4, 18, 3],  # MiT-b3
        hidden_sizes=[64, 128, 320, 512],
        decoder_hidden_size=768,
    )
    model = SegformerForSemanticSegmentation(configuration)
    model.to(device)

    if pretrained:
        # Use the pre-trained encoder but keep the same decoder
        encoder = SegformerModel.from_pretrained("nvidia/mit-b3")
        encoder.to(device)
        model.segformer = encoder

        # Get the image dimensions from the pre-trained image processor
        image_processor = AutoImageProcessor.from_pretrained("nvidia/mit-b3")
        output_shape = (image_processor.size["height"], image_processor.size["width"])
    else:
        output_shape = (1024, 1024)

    # Preparing datasets
    train_trainsforms = A.Compose(
        [
            A.GaussianBlur(blur_limit=(3, 7), p=0.5),
            A.GaussNoise(var_limit=(0.1, 0.3), p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.3, saturation=0.2, hue=0.1, p=1),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Resize(height=output_shape[0], width=output_shape[1]),
            A.Normalize(),
        ]
    )
    val_transforms = A.Compose(
        [
            A.Resize(height=output_shape[0], width=output_shape[1]),
            A.Normalize(),
        ]
    )

    train_dataset = CellOnlyDataset(
        train_image_files,
        train_seg_files,
        transform=train_trainsforms,
        output_shape=output_shape,
    )
    val_dataset = CellOnlyDataset(
        val_image_files,
        val_seg_files,
        transform=val_transforms,
        output_shape=output_shape,
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Setting training parameters
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = get_polynomial_decay_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_epochs,
        num_training_steps=num_epochs,
        power=1,
    )
    loss_fn = DiceLoss(softmax=True, to_onehot_y=True)

    save_name = get_save_name(
        model_name="segformer-cell-only",
        pretrained=pretrained,
        learning_rate=learning_rate,
    )
    print(f"Save name: {save_name}")

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
        break_after_one_iteration=True,
        scheduler=scheduler,
        training_func=run_training_segformer,
        validation_function=run_validation_segformer,
        do_save_model_and_plot=False,
    )


if __name__ == "__main__":
    main()

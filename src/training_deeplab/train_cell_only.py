import argparse
import os
import sys
from time import time
import torch
import albumentations as A
import seaborn as sns


from monai.losses import DiceLoss
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    get_polynomial_decay_schedule_with_warmup,
)

sys.path.append(os.getcwd())

from src.dataset import CellOnlyDataset
from src.models import DeepLabV3plusModel
from src.utils.metrics import create_cellwise_evaluation_function
from src.utils.training import train
from src.utils.utils import (
    get_metadata_with_offset,
    get_ocelot_files,
    get_save_name,
    get_ocelot_args,
)
from src.utils.constants import CELL_IMAGE_MEAN, CELL_IMAGE_STD
from ocelot23algo.user.inference import Deeplabv3CellOnlyModel


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
    do_eval: bool = args.do_eval
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
    print(f"Do eval: {do_eval}")
    print(f"Break after one iteration: {break_after_one_iteration}")
    print(f"Device: {device}")
    print(f"Normalization: {normalization}")
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

    macenko = "macenko" in normalization
    if "imagenet" in normalization:
        train_transform_list.append(A.Normalize())
    elif "cell" in normalization:
        train_transform_list.append(
            A.Normalize(mean=CELL_IMAGE_MEAN, std=CELL_IMAGE_STD)
        )

    train_image_files, train_seg_files = get_ocelot_files(
        data_dir=data_dir, partition="train", zoom="cell", macenko=macenko
    )

    # Create dataset and dataloader
    train_transforms = A.Compose(train_transform_list)
    train_dataset = CellOnlyDataset(
        cell_image_files=train_image_files,
        cell_target_files=train_seg_files,
        transform=train_transforms,
    )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    model = DeepLabV3plusModel(
        backbone_name=backbone_model,
        num_classes=3,
        num_channels=3,
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

    # Creating evaluation models
    val_metadata = get_metadata_with_offset(data_dir=data_dir, partition="val")
    test_metadata = get_metadata_with_offset(data_dir=data_dir, partition="test")

    val_evaluation_model = Deeplabv3CellOnlyModel(
        metadata=val_metadata, cell_model=model
    )
    test_evaluation_model = Deeplabv3CellOnlyModel(
        metadata=test_metadata, cell_model=model
    )

    val_evaluation_function = create_cellwise_evaluation_function(
        evaluation_model=val_evaluation_model,
        tissue_file_folder="images/val/tissue_macenko",
    )
    test_evaluation_function = create_cellwise_evaluation_function(
        evaluation_model=test_evaluation_model,
        tissue_file_folder="images/test/tissue_macenko",
    )

    save_name = get_save_name(
        model_name="deeplabv3plus-cell-only",
        pretrained=pretrained,
        learning_rate=learning_rate,
        dropout_rate=dropout_rate,
        backbone_model=backbone_model,
        normalization=normalization,
        id=id,
    )
    print(f"Save name: {save_name}")

    start_time = time()
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
    )

    end_time = time()
    print(f"Training complete! Took: {end_time - start_time:.2f} seconds.")
    if not do_eval:
        return

    # Use the best model for evaluation, if it was saved
    print(f"Best model: {best_model_path}\n")
    if do_save:
        print(f"Loading the best model!")
        model.load_state_dict(torch.load(best_model_path))

    print(f"Calculating validation score")
    val_mf1 = val_evaluation_function(partition="val")
    print(f"Validation mF1: {val_mf1:.4f}")

    print(f"\nCalculating test score")
    test_mf1 = test_evaluation_function(partition="test")
    print(f"Test mF1: {test_mf1:.4f}")


if __name__ == "__main__":
    main()

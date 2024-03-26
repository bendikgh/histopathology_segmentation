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

from models import DeepLabV3plusModel
from utils.training import train
from utils.utils import (
    get_metadata_with_offset,
    get_ocelot_files,
    get_save_name,
    get_ocelot_args,
)
from utils.constants import (
    CELL_IMAGE_MEAN,
    CELL_IMAGE_STD,
    DEFAULT_TISSUE_MODEL_PATH,
)
from utils.metrics import predict_and_evaluate, predict_and_evaluate_v2
from ocelot23algo.user.inference import Deeplabv3TissueCellModel
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

    # Setting up transforms
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

    # Getting cell files
    train_cell_image_files, train_cell_target_files = get_ocelot_files(
        data_dir=data_dir, partition="train", zoom="cell", macenko=macenko
    )
    val_cell_image_files, val_cell_target_files = get_ocelot_files(
        data_dir=data_dir, partition="val", zoom="cell", macenko=macenko
    )

    train_image_nums = [x.split("/")[-1].split(".")[0] for x in train_cell_image_files]
    val_image_nums = [x.split("/")[-1].split(".")[0] for x in val_cell_image_files]

    # Getting tissue files
    train_tissue_predicted = glob(
        os.path.join(data_dir, "annotations/train/predicted_cropped_tissue/*")
    )
    val_tissue_predicted = glob(
        os.path.join(data_dir, "annotations/val/predicted_cropped_tissue/*")
    )

    # Making sure only the appropriate numbers are used
    train_tissue_predicted = [
        file
        for file in train_tissue_predicted
        if file.split("/")[-1].split(".")[0] in train_image_nums
    ]
    val_tissue_predicted = [
        file
        for file in val_tissue_predicted
        if file.split("/")[-1].split(".")[0] in val_image_nums
    ]

    train_tissue_predicted.sort(key=lambda x: int(x.split("/")[-1].split(".")[0]))
    val_tissue_predicted.sort(key=lambda x: int(x.split("/")[-1].split(".")[0]))

    # Create dataset and dataloader
    train_transforms = A.Compose(
        train_transform_list,
        additional_targets={"mask1": "mask", "mask2": "mask"},
    )
    val_transforms = A.Compose(
        val_transform_list, additional_targets={"mask1": "mask", "mask2": "mask"}
    )

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
        transform=val_transforms,
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
        normalization=normalization,
        id=id,
    )
    print(f"Save name: {save_name}")

    best_model_path = train(
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

    print("Training complete!")
    if not do_eval:
        return

    val_metadata = get_metadata_with_offset(data_dir=data_dir, partition="val")
    val_evaluation_model = Deeplabv3TissueCellModel(
        metadata=val_metadata,
        cell_model=model,
        tissue_model_path=DEFAULT_TISSUE_MODEL_PATH,
    )

    print(f"Best model: {best_model_path}\n")
    print("Calculating validation score:")

    val_mf1 = predict_and_evaluate_v2(
        evaluation_model=val_evaluation_model,
        partition="val",
        tissue_file_folder="images/val/tissue_macenko",
    )
    print(f"Validation mF1: {val_mf1:.4f}")

    test_metadata = get_metadata_with_offset(data_dir=data_dir, partition="test")
    test_evaluation_model = Deeplabv3TissueCellModel(
        metadata=test_metadata,
        cell_model=model,
        tissue_model_path=DEFAULT_TISSUE_MODEL_PATH,
    )
    print("\nCalculating test score:")
    test_mf1 = predict_and_evaluate_v2(
        evaluation_model=test_evaluation_model,
        partition="test",
        tissue_file_folder="images/test/tissue_macenko",
    )
    print(f"Test mF1: {test_mf1:.4f}")


if __name__ == "__main__":
    main()

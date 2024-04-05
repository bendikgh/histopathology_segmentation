import argparse
import cv2
import os
import sys
import torch

import albumentations as A
import seaborn as sns

from glob import glob
from monai.losses import DiceLoss
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    get_polynomial_decay_schedule_with_warmup,
)
from time import time

sys.path.append(os.getcwd())

from src.dataset import CellTissueDataset
from src.models import CustomSegformerModel
from src.utils.metrics import create_cellwise_evaluation_function
from src.utils.training import train
from src.utils.utils import (
    get_metadata_with_offset,
    get_ocelot_files,
    get_save_name,
    get_ocelot_args,
)
from src.utils.constants import CELL_IMAGE_MEAN, CELL_IMAGE_STD
from ocelot23algo.user.inference import SegformerTissueFromFile


def build_transform(transforms, extra_transform_cell_tissue):

    def transform(image, mask1, mask2):
        # First transforms
        transformed = transforms(image=image, mask1=mask1, mask2=mask2)

        transformed_cell = transformed["image"]
        transformed_label = transformed["mask1"]
        transformed_tissue = transformed["mask2"]

        # Additional transforms (usually resize)
        transformed = extra_transform_cell_tissue(
            image=transformed_cell, extra_image=transformed_tissue
        )

        transformed_cell = transformed["image"]
        transformed_tissue = transformed["extra_image"]

        return {
            "image": transformed_cell,
            "mask1": transformed_label,
            "mask2": transformed_tissue,
        }

    return transform


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
    pretrained_dataset: str = args.pretrained_dataset
    resize: int = args.resize
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
    print(f"Resize: {resize}")
    print(f"pretrained dataset: {pretrained_dataset}")
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

    train_transforms = A.Compose(
        train_transform_list, additional_targets={"mask1": "mask", "mask2": "mask"}
    )

    if resize is not None:

        # if resize == (512, 512) and pretrained_dataset != "ade":
        #     raise ValueError("Resize to 512 is only supported for ADE20K.")

        extra_transform_cell_tissue = A.Compose(
            [A.Resize(height=resize, width=resize, interpolation=cv2.INTER_NEAREST)],
            additional_targets={"extra_image": "image"},
        )

        train_transforms = build_transform(
            transforms=train_transforms,
            extra_transform_cell_tissue=extra_transform_cell_tissue,
        )

    train_cell_image_files, train_cell_seg_files = get_ocelot_files(
        data_dir=data_dir, partition="train", zoom="cell", macenko=macenko
    )

    train_image_nums = [x.split("/")[-1].split(".")[0] for x in train_cell_image_files]

    # Getting tissue files
    train_tissue_predicted = glob(
        os.path.join(data_dir, "annotations/train/predicted_cropped_tissue/*")
    )

    # Making sure only the appropriate numbers are used
    train_tissue_predicted = [
        file
        for file in train_tissue_predicted
        if file.split("/")[-1].split(".")[0] in train_image_nums
    ]

    train_tissue_predicted.sort(key=lambda x: int(x.split("/")[-1].split(".")[0]))

    # Create dataset and dataloader
    image_shape = (resize, resize) if resize else (1024, 1024)
    train_dataset = CellTissueDataset(
        cell_image_files=train_cell_image_files,
        cell_target_files=train_cell_seg_files,
        transform=train_transforms,
        tissue_pred_files=train_tissue_predicted,
        image_shape=image_shape,
    )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    model = CustomSegformerModel(
        backbone_name=backbone_model,
        num_classes=3,
        num_channels=6,
        pretrained_dataset=pretrained_dataset,
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

    val_metadata = get_metadata_with_offset(data_dir=data_dir, partition="val")
    test_metadata = get_metadata_with_offset(data_dir=data_dir, partition="test")

    val_evaluation_model = SegformerTissueFromFile(
        metadata=val_metadata,
        cell_model=model,
        tissue_model_path=None,
        device=device,
    )
    test_evaluation_model = SegformerTissueFromFile(
        metadata=test_metadata,
        cell_model=model,
        tissue_model_path=None,
        device=device,
    )

    val_test_transform = None
    
    if resize:
        val_test_transform = A.Compose(
            [A.Resize(height=resize, width=resize, interpolation=cv2.INTER_NEAREST)],
            additional_targets={"tissue": "image"},
        )
    val_evaluation_function = create_cellwise_evaluation_function(
        evaluation_model=val_evaluation_model,
        tissue_file_folder="annotations/val/predicted_cropped_tissue",
        transform=val_test_transform,
    )
    test_evaluation_function = create_cellwise_evaluation_function(
        evaluation_model=test_evaluation_model,
        tissue_file_folder="annotations/test/predicted_cropped_tissue",
        transform=val_test_transform,
    )

    save_name = get_save_name(
        model_name="segformer-tissue-cell",
        pretrained=pretrained,
        learning_rate=learning_rate,
        backbone_model=backbone_model,
        normalization=normalization,
        pretrained_dataset=pretrained_dataset,
        resize=resize,
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
    if do_save:
        model.load_state_dict(torch.load(best_model_path))

    print(f"Best model: {best_model_path}\n")
    print(f"Calculating validation score")
    val_mf1 = val_evaluation_function(partition="val")
    print(f"Validation mF1: {val_mf1:.4f}")

    print(f"\nCalculating test score")
    test_mf1 = test_evaluation_function(partition="test")
    print(f"Test mF1: {test_mf1:.4f}")


if __name__ == "__main__":
    main()

import argparse
import os
import sys
import torch
import time
import seaborn as sns

from monai.losses import DiceLoss, DiceCELoss
from torch.optim import AdamW
from transformers import (
    get_polynomial_decay_schedule_with_warmup,
)
from datetime import timedelta

sys.path.append(os.getcwd())

from src.loss import DiceCELossWrapper, DiceLossWrapper
from src.utils.utils import (
    get_ocelot_args,
)
from src.trainable import (
    SegformerCellOnlyTrainable,
    SegformerTissueCellTrainable,
    DeeplabTissueCellTrainable,
    DeeplabCellOnlyTrainable,
    SegformerSharingTrainable,
    SegformerTissueTrainable,
    SegformerTissueToCellDecoderTrainable,
    Trainable,
    ViTUnetTrainable,
)


def get_trainable(
    model_architecture: str,
    normalization: str,
    batch_size: int,
    pretrained: bool,
    device: torch.device,
    backbone_model: str,
    pretrained_dataset: str,
    resize: int,
    leak_labels: bool,
    data_dir: str,
) -> Trainable:
    trainable: Trainable
    if model_architecture == "segformer_cell_only":
        trainable = SegformerCellOnlyTrainable(
            normalization=normalization,
            batch_size=batch_size,
            pretrained=pretrained,
            device=device,
            backbone_model=backbone_model,
            pretrained_dataset=pretrained_dataset,
            resize=resize,
            data_dir=data_dir,
        )
    elif model_architecture == "segformer_tissue_branch":
        trainable = SegformerTissueTrainable(
            normalization=normalization,
            batch_size=batch_size,
            pretrained=pretrained,
            device=device,
            backbone_model=backbone_model,
            pretrained_dataset=pretrained_dataset,
            resize=resize,
            data_dir=data_dir,
        )
    elif model_architecture == "segformer_cell_branch":
        trainable = SegformerTissueCellTrainable(
            normalization=normalization,
            batch_size=batch_size,
            pretrained=pretrained,
            device=device,
            backbone_model=backbone_model,
            pretrained_dataset=pretrained_dataset,
            resize=resize,
            leak_labels=leak_labels,
            data_dir=data_dir,
        )
    elif model_architecture == "segformer_sharing":
        trainable = SegformerSharingTrainable(
            normalization=normalization,
            batch_size=batch_size,
            pretrained=pretrained,
            device=device,
            backbone_model=backbone_model,
            pretrained_dataset=pretrained_dataset,
            resize=resize,
            data_dir=data_dir,
        )
    elif model_architecture == "segformer_sum_sharing":
        trainable = SegformerTissueToCellDecoderTrainable(
            normalization=normalization,
            batch_size=batch_size,
            pretrained=pretrained,
            device=device,
            backbone_model=backbone_model,
            pretrained_dataset=pretrained_dataset,
            resize=resize,
            data_dir=data_dir,
        )
    elif model_architecture == "deeplab_cell_only":
        trainable = DeeplabCellOnlyTrainable(
            normalization=normalization,
            batch_size=batch_size,
            pretrained=pretrained,
            device=device,
            data_dir=data_dir,
        )
    elif model_architecture == "deeplab_tissue_cell":
        trainable = DeeplabTissueCellTrainable(
            normalization=normalization,
            batch_size=batch_size,
            pretrained=pretrained,
            device=device,
            leak_labels=leak_labels,
            data_dir=data_dir,
        )
    elif model_architecture == "vit_unet":
        trainable = ViTUnetTrainable(
            normalization=normalization,
            batch_size=batch_size,
            pretrained=pretrained,
            device=device,
            backbone_model=backbone_model,
            pretrained_dataset=pretrained_dataset,
            resize=resize,
            data_dir=data_dir,
        )
    else:
        raise ValueError("Invalid model name")

    return trainable


def main():
    sns.set_theme()

    args: argparse.Namespace = get_ocelot_args()
    num_epochs: int = args.epochs
    batch_size: int = args.batch_size
    data_dir: str = args.data_dir
    checkpoint_interval: int = args.checkpoint_interval
    backbone_model: str = args.backbone
    model_architecture: str = args.model_architecture
    learning_rate: float = args.learning_rate
    pretrained: bool = args.pretrained
    warmup_epochs: int = args.warmup_epochs
    do_save: bool = args.do_save
    do_eval: bool = args.do_eval
    break_after_one_iteration: bool = args.break_early
    normalization: str = args.normalization
    pretrained_dataset: str = args.pretrained_dataset
    resize: int = args.resize
    device = torch.device(args.device)
    id: str = args.id
    leak_labels = args.leak_labels
    loss_function_arg = args.loss_function

    print("Training with the following parameters:")
    print(f"Data directory: {data_dir}")
    print(f"Model architecture: {model_architecture}")
    print(f"Number of epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Backbone model: {backbone_model}")
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
    print(f"Leak labels: {leak_labels}")
    print(f"Loss function: {loss_function_arg}")
    print(f"ID: {id}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")

    trainable = get_trainable(
        model_architecture=model_architecture,
        normalization=normalization,
        batch_size=batch_size,
        pretrained=pretrained,
        device=device,
        backbone_model=backbone_model,
        pretrained_dataset=pretrained_dataset,
        resize=resize,
        leak_labels=leak_labels,
        data_dir=data_dir
    )

    if loss_function_arg == "dice":
        loss_function = DiceLoss(softmax=True, include_background=False)
    elif loss_function_arg == "dice-ce":
        loss_function = DiceCELoss(softmax=True, include_background=False)
    elif loss_function_arg == "dice-wrapper":
        loss_function = DiceLossWrapper(softmax=True, to_onehot_y=True)
    elif loss_function_arg == "dice-ce-wrapper":
        loss_function = DiceCELossWrapper(softmax=True, to_onehot_y=True)
    optimizer = AdamW(trainable.model.parameters(), lr=learning_rate)
    scheduler = get_polynomial_decay_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_epochs,
        num_training_steps=num_epochs,
        power=1,
    )

    print(f"Save name: {trainable.get_save_name()}")

    start = time.time()
    best_model_path = trainable.train(
        num_epochs=num_epochs,
        loss_function=loss_function,
        optimizer=optimizer,
        device=torch.device("cuda"),
        checkpoint_interval=checkpoint_interval,
        break_after_one_iteration=break_after_one_iteration,
        scheduler=scheduler,
        do_save_model_and_plot=do_save,
    )
    end = time.time()
    total_time = end - start
    print(
        f"Training finished! Time: {total_time:.2f}s, aka {str(timedelta(seconds=total_time))}."
    )

    if not do_eval:
        return

    if do_save:
        trainable.model = trainable.create_model(
            backbone_name=backbone_model,
            pretrained=pretrained,
            device=trainable.device,
            model_path=best_model_path,
        )
        print(f"Updated model with the best from training: {best_model_path}")

    # Evaluation
    trainable.model.eval()
    val_evaluation_function = trainable.get_evaluation_function(partition="val")
    test_evaluation_function = trainable.get_evaluation_function(partition="test")

    val_score = val_evaluation_function()
    print(f"val score: {val_score}")
    test_score = test_evaluation_function()
    print(f"test score: {test_score}")


if __name__ == "__main__":
    main()

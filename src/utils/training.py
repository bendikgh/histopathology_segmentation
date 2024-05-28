import time
import torch
import os

import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

# from torch.nn.functional import interpolate
from torch import nn
from tqdm import tqdm
from typing import Union, List

from src.utils.utils import save_model


def plot_losses(training_losses: List, val_scores: List, save_path: str):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    ax1.plot(training_losses, "b-", label="Training Loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Training Loss", color="b")

    ax2.plot(val_scores, "r-", label="Validation Score")
    ax2.set_ylabel("Validation Score", color="r")
    ax2.grid(False)

    # Fixing legend
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    fig.legend(
        lines + lines2,
        labels + labels2,
        loc="upper center",  # Locating it at the upper center of the bounding box
        bbox_to_anchor=(0.85, 1.01),  # Centered horizontally, placed below the axes)
    )

    plt.title("Training loss and validation score")
    plt.savefig(save_path)
    plt.close()


def run_training_joint_pred2input(
    model: nn.Module,
    train_dataloader: DataLoader,
    optimizer,
    loss_function,
    device,
    epoch,
    break_after_one_iteration: bool = False,
    weight_loss = False,
):
    model.train()
    training_loss = 0.0

    lam = 0.5
    if weight_loss:
        if epoch < 30:
            lam = 0.2

    for cell_images, tissue_images, cell_labels, tissue_labels, offsets in tqdm(
        train_dataloader
    ):

        cell_images, tissue_images = cell_images.to(device), tissue_images.to(device)
        cell_labels, tissue_labels = cell_labels.to(device), tissue_labels.to(device)

        optimizer.zero_grad()
        cell_logits, tissue_logits = model(cell_images, tissue_images, offsets)

        # loss = loss_function(cell_logits, cell_masks)
        cell_loss = loss_function(cell_logits, cell_labels)
        tissue_loss = loss_function(tissue_logits, tissue_labels)
        
        loss = (1-lam)*cell_loss + lam*tissue_loss

        loss.backward()
        optimizer.step()

        training_loss += loss.item()

        if break_after_one_iteration:
            print("Breaking after one iteration")
            break

    if not break_after_one_iteration:
        training_loss /= len(train_dataloader)

    return training_loss


def run_training(
    model: nn.Module,
    train_dataloader: DataLoader,
    optimizer,
    loss_function,
    device,
    epoch,
    break_after_one_iteration: bool = False,
    weight_loss = False,
) -> float:
    model.train()
    training_loss = 0.0

    for images, masks in tqdm(train_dataloader):
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_function(outputs, masks)
        loss.backward()
        optimizer.step()

        training_loss += loss.item()

        if break_after_one_iteration:
            print("Breaking after one iteration")
            break

    if not break_after_one_iteration:
        training_loss /= len(train_dataloader)

    return training_loss


def run_validation(
    model: nn.Module,
    val_dataloader: DataLoader,
    loss_function,
    device,
    break_after_one_iteration: bool = False,
) -> float:
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, masks in tqdm(val_dataloader):
            images, masks = images.to(device), masks.to(device)

            outputs = model(images)
            loss = loss_function(outputs, masks)

            val_loss += loss.item()
            if break_after_one_iteration:
                print("Breaking after one iteration")
                break

    if not break_after_one_iteration:
        val_loss /= len(val_dataloader)
    return val_loss


def write_logs(
    save_path: str, epoch: int, start: float, training_losses: List, val_scores: List
) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    end = time.time()
    with open(save_path, "w") as file:
        file.write(
            f"Number of epochs: {epoch + 1}, total time: {end - start:.3f} seconds \n"
        )
        file.write(f"training_losses = {str(training_losses)}\n")
        file.write(f"val_scores = {str(val_scores)}\n")


def train(
    num_epochs: int,
    train_dataloader: DataLoader,
    model: nn.Module,
    loss_function,
    optimizer,
    device,
    save_name: str,
    checkpoint_interval: int = 5,
    break_after_one_iteration: bool = False,
    scheduler=None,
    training_func=run_training,
    validation_function=run_validation,
    do_save_model_and_plot: bool = True,
    weight_loss = False,
) -> Union[str, None]:
    """
    validation_function: takes in (...)
    """

    start = time.time()
    training_losses = []
    val_scores = []
    highest_val_score = -float("inf")
    best_model_save_path: Union[str, None] = None

    model_dir = os.path.join("outputs", "models")
    plot_dir = os.path.join("outputs", "plots")
    log_dir = os.path.join("outputs", "logs")
    log_file_path = os.path.join(log_dir, f"{save_name}.txt")

    for epoch in range(num_epochs):
        model.train()
        training_loss = training_func(
            model=model,
            train_dataloader=train_dataloader,
            optimizer=optimizer,
            loss_function=loss_function,
            device=device,
            break_after_one_iteration=break_after_one_iteration,
            epoch=epoch,
            weight_loss=weight_loss,
        )
        training_losses.append(training_loss)

        if scheduler is not None:
            scheduler.step()

        model.eval()
        with torch.no_grad():
            val_score = validation_function(
                break_after_one_iteration=break_after_one_iteration
            )
        val_scores.append(val_score)

        # Plotting results and saving model
        if val_score > highest_val_score and do_save_model_and_plot:
            highest_val_score = val_score
            best_model_save_path = os.path.join(model_dir, f"{save_name}_best.pth")
            save_model(model, best_model_save_path)
            print(f"New best model saved after {epoch + 1} epochs!")

        at_checkpoint_interval = (epoch + 1) % checkpoint_interval == 0
        at_last_epoch = (epoch + 1) == num_epochs

        if do_save_model_and_plot and (at_checkpoint_interval or at_last_epoch):
            model_path_suffix = f"{save_name}_epochs-{epoch + 1}.pth"
            model_save_path = os.path.join(model_dir, model_path_suffix)
            save_model(model, model_save_path)

            plot_path_suffix = f"{save_name}_epochs-{epoch + 1}.png"
            plot_save_path = os.path.join(plot_dir, plot_path_suffix)
            plot_losses(training_losses, val_scores, save_path=plot_save_path)

            # Writing logs
            write_logs(
                save_path=log_file_path,
                epoch=epoch,
                start=start,
                training_losses=training_losses,
                val_scores=val_scores,
            )
            print(
                f"Saved model and logs, and plotted results after {epoch + 1} epochs!"
            )

        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch + 1}/{num_epochs} - Learning rate: {lr:.6f}  - Training loss: {training_loss:.4f} - Validation score: {val_score:.4f}"
        )
    return best_model_save_path

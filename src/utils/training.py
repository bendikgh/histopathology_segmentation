import time
import torch
import os

import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torch.nn.functional import interpolate
from torch import nn
from tqdm import tqdm


def plot_losses(training_losses, val_losses, save_path: str):
    plt.plot(training_losses, label="Training loss")
    plt.plot(val_losses, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and validation loss")
    plt.legend()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def run_training_segformer(
    model: nn.Module,
    train_dataloader: DataLoader,
    optimizer,
    loss_function,
    device,
    break_after_one_iteration: bool = False,
) -> float:
    model.train()
    training_loss = 0.0

    for images, labels in tqdm(train_dataloader):
        images, labels = images.to(device), labels.to(device)
        labels = labels.argmax(dim=1)

        optimizer.zero_grad()
        outputs = model(images).logits
        outputs = interpolate(
            outputs,
            size=labels.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        loss = loss_function(outputs, labels.unsqueeze(1))

        loss.backward()
        optimizer.step()
        training_loss += loss.item()

        if break_after_one_iteration:
            print("Breaking after one iteration")
            break
    if not break_after_one_iteration:
        training_loss /= len(train_dataloader)
    return training_loss


def run_validation_segformer(
    model: nn.Module,
    val_dataloader: DataLoader,
    loss_function,
    device,
    break_after_one_iteration: bool = False,
) -> float:
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in tqdm(val_dataloader):
            images, labels = images.to(device), labels.to(device)
            labels = labels.argmax(dim=1)
            outputs = model(images).logits
            outputs = interpolate(
                outputs,
                size=labels.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            loss = loss_function(outputs, labels.unsqueeze(1))

            val_loss += loss.item()
            if break_after_one_iteration:
                print("Breaking after one iteration")
                break
    if not break_after_one_iteration:
        val_loss /= len(val_dataloader)
    return val_loss


def run_training(
    model: nn.Module,
    train_dataloader: DataLoader,
    optimizer,
    loss_function,
    device,
    break_after_one_iteration: bool = False,
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


def train(
    num_epochs: int,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
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
):
    start = time.time()
    training_losses = []
    val_losses = []
    lowest_val_loss = float("inf")

    for epoch in range(num_epochs):
        training_loss = training_func(
            model=model,
            train_dataloader=train_dataloader,
            optimizer=optimizer,
            loss_function=loss_function,
            device=device,
            break_after_one_iteration=break_after_one_iteration,
        )
        training_losses.append(training_loss)

        if scheduler:
            scheduler.step()

        val_loss = validation_function(
            model=model,
            val_dataloader=val_dataloader,
            loss_function=loss_function,
            device=device,
            break_after_one_iteration=break_after_one_iteration,
        )
        val_losses.append(val_loss)

        # Plotting results and saving model
        if val_loss < lowest_val_loss and do_save_model_and_plot:
            lowest_val_loss = val_loss
            best_model_save_path = f"outputs/models/{save_name}_best.pth"
            os.makedirs(os.path.dirname(best_model_save_path), exist_ok=True)
            torch.save(model.state_dict(), best_model_save_path)
            print(f"New best model saved after {epoch + 1} epochs!")

        if do_save_model_and_plot and (
            (epoch + 1) % checkpoint_interval == 0 or (epoch + 1) == num_epochs
        ):
            model_save_path = f"outputs/models/{save_name}_epochs-{epoch + 1}.pth"
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            torch.save(model.state_dict(), model_save_path)
            plot_losses(
                training_losses, val_losses, save_path=f"outputs/plots/{save_name}.png"
            )

            # Writing logs
            log_file_path = f"outputs/logs/{save_name}.txt"
            os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
            with open(log_file_path, "w") as file:
                file.write(
                    f"Number of epochs: {epoch + 1}, total time: {time.time() - start:.3f} seconds \n"
                )
                file.write(f"training_losses = {str(training_losses)}\n")
                file.write(f"val_losses = {str(val_losses)}\n")
            print(
                f"Saved model and logs, and plotted results after {epoch + 1} epochs!"
            )

        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch + 1}/{num_epochs} - Learning rate: {lr:.6f}  - Training loss: {training_loss:.4f} - Validation loss: {val_loss:.4f}"
        )
    return best_model_save_path

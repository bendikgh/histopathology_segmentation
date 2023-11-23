import time
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm


def plot_losses(training_losses, val_losses, save_path: str):
    plt.plot(training_losses, label="Training loss")
    plt.plot(val_losses, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and validation loss")
    plt.legend()
    plt.savefig(save_path)
    plt.close()


def train(
    num_epochs: int,
    train_dataloader,
    val_dataloader,
    model,
    loss_function,
    optimizer,
    device,
    checkpoint_interval: int = 5,
    break_after_one_iteration: bool = False,
):
    current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    training_losses = []
    val_losses = []
    model.train()
    for epoch in range(num_epochs):
        training_loss = 0
        for images, masks in tqdm(train_dataloader):
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images.permute((0, 3, 1, 2)))
            outputs = outputs.permute((0, 2, 3, 1))
            loss = loss_function(outputs, masks)
            loss.backward()
            optimizer.step()

            training_loss += loss.item()
            if break_after_one_iteration:
                print("Breaking after one iteration")
                break

        training_loss /= len(train_dataloader)
        training_losses.append(training_loss)

        val_loss = 0
        model.eval()
        with torch.no_grad():
            for images, masks in tqdm(val_dataloader):
                images, masks = images.to(device), masks.to(device)

                outputs = model(images.permute((0, 3, 1, 2)))
                outputs = outputs.permute((0, 2, 3, 1))
                loss = loss_function(outputs, masks)

                val_loss += loss.item()
                if break_after_one_iteration:
                    print("Breaking after one iteration")
                    break

        val_loss /= len(val_dataloader)
        val_losses.append(val_loss)

        if (epoch + 1) % checkpoint_interval == 0:
            torch.save(
                model.state_dict(),
                f"outputs/models/{current_time}_deeplabv3plus_cell_only_{epoch + 1}.pth",
            )
            plot_losses(
                training_losses,
                val_losses,
                save_path=f"outputs/plots/{current_time}_deeplabv3plus_cell_only.png",
            )

        print(
            f"Epoch {epoch + 1}/{num_epochs} - Training loss: {training_loss:.4f} - Validation loss: {val_loss:.4f}"
        )
    return training_losses, val_losses

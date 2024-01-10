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

def run_training(
    model,
    train_dataloader,
    optimizer,
    loss_function,
    device,
    break_after_one_iteration: bool = False,
):
    model.train()
    training_loss = 0

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
    model,
    val_dataloader,
    loss_function,
    device,
    break_after_one_iteration: bool = False,
):
    model.eval()
    val_loss = 0
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
    train_dataloader,
    val_dataloader,
    model,
    loss_function,
    optimizer,
    device,
    checkpoint_interval: int = 5,
    break_after_one_iteration: bool = False,
    dropout_rate: float = 0.5,
    backbone: str = "resnet50",
    model_name: str = "cell_only",
    warmup_scheduler = None,
    lr_scheduler = None,
    warmup_epochs: int = 0
):
    learning_rate = optimizer.param_groups[0]["lr"]
    start = time.time()
    current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    training_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):

        lr = optimizer.param_groups[0]["lr"]
        
        training_loss = run_training(
            model=model,
            train_dataloader=train_dataloader,
            optimizer=optimizer,
            loss_function=loss_function,
            device=device,
            break_after_one_iteration=break_after_one_iteration 
        )

        training_losses.append(training_loss)

        if warmup_scheduler and epoch <= warmup_epochs:
            warmup_scheduler.step()
            
        if lr_scheduler and epoch > warmup_epochs:
            lr_scheduler.step()

        val_loss = run_validation(
            model=model,
            val_dataloader=val_dataloader,
            loss_function=loss_function,
            device=device,
            break_after_one_iteration=break_after_one_iteration,
        )
        val_losses.append(val_loss)

        # Plotting results and saving model
        if (epoch + 1) % checkpoint_interval == 0 or (epoch + 1) == num_epochs:
            torch.save(
                model.state_dict(),
                f"outputs/models/{current_time}_deeplabv3plus_{model_name}_lr-{learning_rate}_dropout-{dropout_rate}_backbone-{backbone}_epochs-{epoch + 1}.pth",
            )
            plot_losses(
                training_losses,
                val_losses,
                save_path=f"outputs/plots/{current_time}_deeplabv3plus_{model_name}_lr-{learning_rate}_dropout-{dropout_rate}_backbone-{backbone}.png",
            )
            with open(
                f"outputs/logs/{current_time}_deeplabv3plus_{model_name}_lr-{learning_rate}_dropout-{dropout_rate}_backbone-{backbone}.txt",
                "w",
            ) as file:
                file.write(
                    f"Number of epochs: {epoch + 1}, total time: {time.time() - start:.3f} seconds \n"
                )
                file.write(f"training_losses = {str(training_losses)}\n")
                file.write(f"val_losses = {str(val_losses)}\n")
            print(
                f"Saved model and logs, and plotted results after {epoch + 1} epochs!"
            )

        print(
            f"Epoch {epoch + 1}/{num_epochs} - Learning rate: {lr:.6f}  - Training loss: {training_loss:.4f} - Validation loss: {val_loss:.4f}"
        )
    return training_losses, val_losses

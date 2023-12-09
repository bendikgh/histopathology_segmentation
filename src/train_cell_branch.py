import argparse
import os
import torch
import albumentations as A

from glob import glob
from monai.losses import DiceLoss
from monai.data import ImageDataset
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn.functional import softmax

from deeplabv3.network.modeling import _segm_resnet
from train_utils import train

from torchvision.transforms import ToTensor
from PIL import Image
import numpy as np

# Function for crop and scale tissue image
from utils import crop_and_upscale_tissue, get_metadata
from dataset import CellTissueDataset


# def get_tissue_croped_scaled_tensor(
#     tissue_tensor, image_file, data_path, image_size: int = 1024
# ):
#     data_id = image_file.split("/")[-1].split(".")[0]
#     data_object = get_metadata(data_path)["sample_pairs"][data_id]

#     offset_tensor = (
#         torch.tensor([data_object["patch_x_offset"], data_object["patch_y_offset"]])
#         * image_size
#     )
#     scaling_value = (
#         data_object["cell"]["resized_mpp_x"] / data_object["tissue"]["resized_mpp_x"]
#     )

#     cropped_scaled = crop_and_upscale_tissue(
#         tissue_tensor, offset_tensor, scaling_value
#     )

#     return cropped_scaled


# class CellTissueDataset(ImageDataset):
#     def __init__(self, image_files, seg_files, image_tissue_files, model_tissue, transform=None) -> None:
#         self.image_files = image_files
#         self.seg_files = seg_files
#         self.to_tensor = ToTensor()
#         self.image_tissue_files = image_tissue_files

#         self.model_tissue = model_tissue
#         self.transform = transform


#     def __getitem__(self, idx):
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         data_path = "/cluster/projects/vc/data/mic/open/OCELOT/ocelot_data"

#         # Cell
#         image_path = self.image_files[idx]
#         seg_path = self.seg_files[idx]

#         image = self.to_tensor(Image.open(image_path).convert("RGB"))
#         seg = self.to_tensor(Image.open(seg_path).convert("RGB"))*255

#         # tissue
#         image_path = self.image_tissue_files[idx]
#         image_tissue = self.to_tensor(Image.open(image_path).convert("RGB")).unsqueeze(0)
#         image_tissue = image_tissue.to(device)

#         image_tissue = self.model_tissue(image_tissue)

#         image_tissue = image_tissue.detach().cpu().squeeze(0)

#         # max_values, _ = image_tissue.max(0, keepdim=True)
#         image_tissue = softmax(image_tissue, 0)

#         # Scale and crop
#         image_tissue = get_tissue_croped_scaled_tensor(image_tissue, image_path, data_path)


#         if self.transform:
#             transformed = self.transform(
#                 image=np.array(image.permute((1, 2, 0))),
#                 mask1=np.array(seg.permute((1, 2, 0))),
#                 mask2=np.array(image_tissue.permute((1, 2, 0))),
#             )
#             image = torch.tensor(transformed["image"]).permute((2, 0, 1))
#             seg = torch.tensor(transformed["mask1"]).permute((2, 0, 1))
#             image_tissue = torch.tensor(transformed["mask2"]).permute((2, 0, 1))

#         image = torch.cat((image, image_tissue), dim=0)

#         return image, seg


def main():
    default_epochs = 2
    default_batch_size = 2
    default_data_dir = "ocelot_data"
    default_checkpoint_interval = 5
    default_backbone_model = "resnet50"
    default_dropout_rate = 0.3
    default_learning_rate = 1e-4

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
        "--dropout", type=float, default=default_dropout_rate, help="Dropout rate"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=default_learning_rate,
        help="Learning rate",
    )

    args = parser.parse_args()

    num_epochs = args.epochs
    batch_size = args.batch_size
    data_dir = args.data_dir
    checkpoint_interval = args.checkpoint_interval
    backbone_model = args.backbone
    dropout_rate = args.dropout
    learning_rate = args.learning_rate

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Training with the following parameters:")
    print(f"Data directory: {data_dir}")
    print(f"Number of epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Backbone model: {backbone_model}")
    print(f"Dropout rate: {dropout_rate}")
    print(f"Learning rate: {learning_rate}")
    print(f"Checkpoint interval: {checkpoint_interval}")
    print(f"Device: {device}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")

    data_path = "/cluster/projects/vc/data/mic/open/OCELOT/ocelot_data"

    train_seg_files = glob(
        os.path.join(data_path, "annotations/train/segmented_cell/*")
    )
    train_image_numbers = [
        file_name.split("/")[-1].split(".")[0] for file_name in train_seg_files
    ]
    train_image_files = [
        os.path.join(data_path, "images/train/cell", image_number + ".jpg")
        for image_number in train_image_numbers
    ]

    val_seg_files = glob(os.path.join(data_path, "annotations/val/segmented_cell/*"))
    val_image_numbers = [
        file_name.split("/")[-1].split(".")[0] for file_name in val_seg_files
    ]
    val_image_files = [
        os.path.join(data_path, "images/val/cell", image_number + ".jpg")
        for image_number in val_image_numbers
    ]

    test_seg_files = glob(os.path.join(data_path, "annotations/test/segmented_cell/*"))
    test_image_numbers = [
        file_name.split("/")[-1].split(".")[0] for file_name in test_seg_files
    ]
    test_image_files = [
        os.path.join(data_path, "images/test/cell", image_number + ".jpg")
        for image_number in test_image_numbers
    ]

    # Find the correct files
    train_tissue_seg_files = glob(os.path.join(data_dir, "annotations/train/tissue/*"))

    train_tissue_image_numbers = [
        file_name.split("/")[-1].split(".")[0] for file_name in train_tissue_seg_files
    ]
    train_tissue_image_files = [
        os.path.join(data_dir, "images/train/tissue", image_number + ".jpg")
        for image_number in train_tissue_image_numbers
    ]
    train_tissue_predicted = [
        os.path.join(data_path, "annotations/train/pred_tissue", image_number + ".jpg")
        for image_number in train_tissue_image_numbers
    ]

    val_tissue_seg_files = glob(os.path.join(data_dir, "annotations/val/tissue/*"))
    val_tissue_image_numbers = [
        file_name.split("/")[-1].split(".")[0] for file_name in val_tissue_seg_files
    ]
    val_tissue_image_files = [
        os.path.join(data_dir, "images/val/tissue", image_number + ".jpg")
        for image_number in val_tissue_image_numbers
    ]
    val_tissue_predicted = [
        os.path.join(data_path, "annotations/val/pred_tissue", image_number + ".jpg")
        for image_number in val_tissue_image_numbers
    ]

    # Create dataset and dataloader
    transforms = A.Compose(
        [
            A.GaussianBlur(blur_limit=(3, 7), p=0.5),  # You can adjust the blur limit
            A.GaussNoise(
                var_limit=(0.1, 0.3), p=0.5
            ),  # Adjust var_limit for noise intensity
            A.ColorJitter(brightness=0.2, contrast=0.3, saturation=0.2, hue=0.1, p=1),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
        ],
        additional_targets={"mask1": "mask", "mask2": "mask"},
    )

    model_tissue = _segm_resnet(
        name="deeplabv3plus",
        backbone_name=backbone_model,
        num_classes=3,
        output_stride=8,
        pretrained_backbone=True,
        dropout_rate=dropout_rate,
    )
    model_tissue.to(device)
    model_tissue.load_state_dict(
        torch.load(
            "outputs/models/2023-12-07_22-49-54_deeplabv3plus_cell_only_lr-0.0001_dropout-0.3_backbone-resnet50_epochs-290.pth"
        )
    )
    model_tissue.eval()

    train_cell_tissue_dataset = CellTissueDataset(
        image_files=train_image_files,
        seg_files=train_seg_files,
        image_tissue_files=train_tissue_predicted,
        transform=transforms,
    )
    val_cell_tissue_dataset = CellTissueDataset(
        image_files=val_image_files,
        seg_files=val_seg_files,
        image_tissue_files=val_tissue_predicted,
    )

    train_cell_tissue_dataloader = DataLoader(
        dataset=train_cell_tissue_dataset, batch_size=batch_size, drop_last=True
    )
    val_cell_tissue_dataloader = DataLoader(dataset=val_cell_tissue_dataset)

    # Create model and optimizer
    model_cell = _segm_resnet(
        name="deeplabv3plus",
        backbone_name=backbone_model,
        num_classes=3,
        output_stride=8,
        pretrained_backbone=True,
        dropout_rate=dropout_rate,
        num_channels=6,
    )
    model_cell.to(device)
    model_cell.train()

    loss_function = DiceLoss(softmax=True)
    optimizer = Adam(model_cell.parameters(), lr=learning_rate)

    training_losses, validation_losses = train(
        num_epochs=num_epochs,
        train_dataloader=train_cell_tissue_dataloader,
        val_dataloader=val_cell_tissue_dataloader,
        model=model_cell,
        loss_function=loss_function,
        optimizer=optimizer,
        device=device,
        checkpoint_interval=checkpoint_interval,
        break_after_one_iteration=False,
        dropout_rate=dropout_rate,
        backbone=backbone_model,
        model_name="tissue-cell",
    )


if __name__ == "__main__":
    main()

import os
import torch
import numpy as np

from monai.data import ImageDataset, DataLoader, MetaTensor
from monai.losses import DiceLoss
from torch.optim import Adam

from deeplabv3.network.modeling import _segm_resnet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

model = _segm_resnet(
    name="deeplabv3plus",
    backbone_name="resnet50",
    num_classes=2,
    output_stride=8,
    pretrained_backbone=True,
)
model.to(device)


# Creating the dataset class
class OcelotDataset(ImageDataset):
    @classmethod
    def decode_target(cls, target):
        return target.argmax(1)

    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        label[
            np.logical_or(label == 1, label == 255)
        ] = 0  # Set pixels with label 255 to 1
        label[label == 2] = 1  # Set pixels with label 255 to 1
        return image, label


image_file_path = "ocelot_data/images/train/tissue/"
segmentation_file_path = "ocelot_data/annotations/train/tissue/"


image_files = [
    os.path.join(image_file_path, file_name)
    for file_name in os.listdir(image_file_path)
]
image_files.sort()

segmentation_files = [
    os.path.join(segmentation_file_path, file_name)
    for file_name in os.listdir(segmentation_file_path)
]
segmentation_files.sort()

dataset = OcelotDataset(image_files=image_files, seg_files=segmentation_files)
data_loader = DataLoader(dataset=dataset, batch_size=2)

loss_function = DiceLoss(
    # softmax=True
)
optimizer = Adam(model.parameters(), lr=1e-3)

num_epochs = 1
decode_fn = data_loader.dataset.decode_target

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0

    for inputs, labels in data_loader:
        # Transposing the inputs to fit the expected shape
        inputs_tensor = torch.Tensor(inputs)
        inputs_tensor = inputs_tensor.permute((0, 3, 1, 2))
        inputs = MetaTensor(inputs_tensor, meta=inputs.meta)

        # Continuing with the regular training loop
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        outputs = decode_fn(outputs).to(torch.float32)
        outputs.requires_grad = True

        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        loop_loss = loss.item()
        print(f"Loop loss: {loop_loss}")
        epoch_loss += loop_loss

    epoch_loss /= len(data_loader)
    print(f"Epoch {epoch + 1}, Loss: {epoch_loss}")

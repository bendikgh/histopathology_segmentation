import json
import os
import sys
import torch
import torch.nn.functional as F

sys.path.append(os.getcwd())

from torch import nn
from typing import Union, Optional, Dict
from src.deeplabv3.network.utils import IntermediateLayerGetter
from src.deeplabv3.network.backbone import resnet

from src.deeplabv3.network._deeplab import (
    DeepLabHeadV3Plus,
    DeepLabV3,
)

from src.utils.constants import OCELOT_IMAGE_SIZE, SEGFORMER_ARCHITECTURES
from src.utils.utils import crop_and_resize_tissue_faster

from transformers import (
    SegformerForSemanticSegmentation,
    SegformerConfig,
    SegformerModel,
)


class DeepLabV3plusModel(nn.Module):

    def __init__(
        self,
        backbone_name: str,
        num_classes: int,
        num_channels: int,
        pretrained: bool,
        dropout_rate: float,
    ):
        if num_channels < 3:
            raise ValueError("Number of input channels must be at least 3")

        super().__init__()
        # Assuming an output stride of 8
        aspp_dilate = [12, 24, 36]

        if backbone_name == "resnet34":
            replace_stride_with_dilation = [False, False, False]
            low_level_planes = 64
            inplanes = 512
        else:
            replace_stride_with_dilation = [False, True, True]
            low_level_planes = 256
            inplanes = 2048

        backbone = resnet.__dict__[backbone_name](
            pretrained=pretrained,
            replace_stride_with_dilation=replace_stride_with_dilation,
            dropout_rate=dropout_rate,
        )

        # Updating the first layer to accept the correct number of channels
        input_layer = backbone.conv1
        new_conv1 = nn.Conv2d(
            num_channels,
            input_layer.out_channels,
            kernel_size=input_layer.kernel_size,
            stride=input_layer.stride,
            padding=input_layer.padding,
        )

        # Copying the three first channels from the original weights
        new_conv1.weight.data[:, :3] = input_layer.weight.data
        backbone.conv1 = new_conv1

        return_layers = {"layer4": "out", "layer1": "low_level"}
        classifier = DeepLabHeadV3Plus(
            inplanes, low_level_planes, num_classes, aspp_dilate
        )
        backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.model = DeepLabV3(backbone, classifier)

    def forward(self, x):
        return self.model(x)


class CustomSegformerModel(nn.Module):

    def __init__(
        self,
        backbone_name: str,
        num_classes: int,
        num_channels: int,
        pretrained_dataset: str | None = None,
        output_spatial_shape=OCELOT_IMAGE_SIZE,
    ):
        if num_channels < 3:
            raise ValueError("Number of input channels must be at least 3")

        super().__init__()

        # keeping track of what the model is pretrained on
        self.pretrained_dataset = pretrained_dataset
        self.output_spatial_shape = output_spatial_shape

        # Fetching the parameters for the segformer model
        parameters = SEGFORMER_ARCHITECTURES[backbone_name]

        # Creating model
        configuration = SegformerConfig(
            num_labels=num_classes,
            num_channels=num_channels,
            depths=parameters["depths"],
            hidden_sizes=parameters["hidden_sizes"],
            decoder_hidden_size=parameters["decoder_hidden_size"],
        )

        model = SegformerForSemanticSegmentation(configuration)

        # Loading pretrained weights
        if pretrained_dataset == "ade":
            pretrained = SegformerModel.from_pretrained(
                f"nvidia/segformer-{backbone_name}-finetuned-ade-512-512"
            )
            model.segformer = pretrained
        elif pretrained_dataset == "cityscapes":
            pretrained = SegformerModel.from_pretrained(
                f"nvidia/segformer-{backbone_name}-finetuned-cityscapes-1024-1024"
            )
            model.segformer = pretrained
        elif pretrained_dataset == "imagenet":
            pretrained = SegformerModel.from_pretrained(f"nvidia/mit-{backbone_name}")
            model.segformer = pretrained
        else:
            raise ValueError(f"Invalid pretrained dataset: {pretrained_dataset}")

        if num_channels != 3:
            input_layer = model.segformer.encoder.patch_embeddings[0].proj

            new_input_layer = nn.Conv2d(
                num_channels,
                input_layer.out_channels,
                kernel_size=input_layer.kernel_size,
                stride=input_layer.stride,
                padding=input_layer.padding,
            )

            num_channels_input_layer = input_layer.weight.data.shape[1]

            new_input_layer.weight.data[:, :num_channels_input_layer] = (
                input_layer.weight.data
            )
            new_input_layer.bias.data[:] = input_layer.bias.data

            model.segformer.encoder.patch_embeddings[0].proj = new_input_layer

        self.model = model

    def forward(self, x):
        logits = self.model(x).logits

        # Upscaling the result to the shape of the ground truth
        if logits.shape[1:] != self.output_spatial_shape:
            logits = nn.functional.interpolate(
                logits,
                size=OCELOT_IMAGE_SIZE,  # (height, width) # TODO: this should be self.output_spatial_shape?
                mode="bilinear",
                align_corners=False,
            )

        return logits


def setup_segformer(
    backbone_name: str,
    num_classes: int,
    num_channels: int,
    pretrained_dataset: Optional[str] = None,
):
    parameters = SEGFORMER_ARCHITECTURES[backbone_name]

    # Creating model
    configuration = SegformerConfig(
        num_labels=num_classes,
        num_channels=num_channels,
        depths=parameters["depths"],
        hidden_sizes=parameters["hidden_sizes"],
        decoder_hidden_size=parameters["decoder_hidden_size"],
        output_hidden_states=True,
    )

    model = SegformerForSemanticSegmentation(configuration)

    # Loading pretrained weights
    if pretrained_dataset == "ade":
        pretrained = SegformerModel.from_pretrained(
            f"nvidia/segformer-{backbone_name}-finetuned-ade-512-512"
        )
        model.segformer = pretrained
    elif pretrained_dataset == "cityscapes":
        pretrained = SegformerModel.from_pretrained(
            f"nvidia/segformer-{backbone_name}-finetuned-cityscapes-1024-1024"
        )
        model.segformer = pretrained
    elif pretrained_dataset == "imagenet":
        pretrained = SegformerModel.from_pretrained(f"nvidia/mit-{backbone_name}")
        model.segformer = pretrained
    else:
        raise ValueError(f"Invalid pretrained dataset: {pretrained_dataset}")

    if num_channels != 3:
        input_layer = model.segformer.encoder.patch_embeddings[0].proj

        new_input_layer = nn.Conv2d(
            num_channels,
            input_layer.out_channels,
            kernel_size=input_layer.kernel_size,
            stride=input_layer.stride,
            padding=input_layer.padding,
        )

        num_channels_input_layer = input_layer.weight.data.shape[1]

        new_input_layer.weight.data[:, :num_channels_input_layer] = (
            input_layer.weight.data
        )
        new_input_layer.bias.data[:] = input_layer.bias.data

        model.segformer.encoder.patch_embeddings[0].proj = new_input_layer

    return model


class TissueCellSharingSegformerModel(nn.Module):

    def __init__(
        self,
        backbone_name: str,
        num_classes: int,
        num_channels: int,
        metadata: str,  # TODO: It is not str, right?
        pretrained_dataset: Union[str, None] = None,
        output_spatial_shape=OCELOT_IMAGE_SIZE,
    ):
        if num_channels < 6:
            raise ValueError("Number of input channels must be at least 6")
        if num_channels % 2 != 0:
            raise ValueError("Number of input channels must be even")

        super().__init__()

        # keeping track of what the model is pretrained on
        self.pretrained_dataset = pretrained_dataset
        self.num_channels = num_channels
        self.output_spatial_shape = output_spatial_shape

        # Fetching the parameters for the segformer model

        self.model_tissue = setup_segformer(
            backbone_name=backbone_name,
            num_classes=num_classes,
            # Why do we divide by 2 here?
            # Couldn't we just use the full num_channels and expect the user to input a smaller number?
            # They seem to be equal either way
            num_channels=int(num_channels / 2),
            pretrained_dataset=pretrained_dataset,
        )

        self.model_cell = setup_segformer(
            backbone_name=backbone_name,
            num_classes=num_classes,
            num_channels=int(num_channels / 2),
            pretrained_dataset=pretrained_dataset,
        )

        # Why do this here, as opposed to just using the model directly
        # in the forward() method?
        self.model_cell_encoder = self.model_cell.segformer
        self.model_cell_decoder = self.model_cell.decode_head

        self.metadata = metadata
        self.dim_reduction = None
        self.device = device

    def forward(self, x, pair_id):

        # x.shape: (2, 6, 512, 512) with resize = 512
        # pair_id.shape: (2,), e.g. [100, 63]

        # pair_id is a list of indices that correspond to the metadata, since
        # we process a batch at a time

        # Assuming the three first channels belong to cell

        # TODO: Maybe scary to have to call int() here? Surely that allows for
        # bugs to pass through?
        logits_tissue = self.model_tissue(x[:, int(self.num_channels / 2) :]).logits

        scaled_list = []

        # Go through each image in the batch, crop and resize, and then
        # concatenate them for further processing
        for i in range(len(pair_id)):
            # TODO: Do we fetch the correct metadata for the sample
            meta_pair = self.metadata[pair_id[i]]
            tissue_mpp = meta_pair["tissue"]["resized_mpp_x"]
            cell_mpp = meta_pair["cell"]["resized_mpp_x"]
            x_offset = meta_pair["patch_x_offset"]
            y_offset = meta_pair["patch_y_offset"]

            argmaxed = logits_tissue[i].argmax(dim=0).to(dtype=torch.int)

            scaled_tissue: torch.Tensor = crop_and_resize_tissue_faster(
                image=argmaxed,
                x_offset=x_offset,
                y_offset=y_offset,
            )
            cell_weights = outputs.hidden_states

        logits_tissue = torch.cat(scaled_list, dim=0)

        outputs = self.model_cell_encoder(
            x[:, : int(self.num_channels / 2)], output_hidden_states=True
        )

        # Surely one could do this using outputs.last_hidden_state?
        cell_weights = outputs.hidden_states
        cell_encodings = cell_weights[-1]

        # Flatten cell encodings to match the logits_tissue_flat shape
        cell_encodings_flat = cell_encodings.reshape(cell_encodings.size(0), -1)

        # Flatten logits_tissue (B, C*H*W)
        # (B, H*W*C)

        # tissue: (2, 350)
        # cell: (2, 140)

        # (3, 1024, 1024)
        # (3, 1024, 1024)

        # (2, 490)

        # [[1, 2, 3], [4, 5, 6]]    -> [1, 2, 3, 4, 5, 6]
        # [[1, 2], [3, 4], [5, 6]]  -> [1, 2, 3, 4, 5, 6]

        # [[1, 3],
        #  [2, 5],
        #  [4, 6]]

        # [[1, 2, 3],
        #  [4, 5, 6]]

        logits_tissue_flat = logits_tissue.reshape(logits_tissue.size(0), -1)

        if self.dim_reduction is None:
            self.dim_reduction = nn.Sequential(
                nn.Linear(merged_tensor.shape[1], 4096),
                nn.ReLU(),
                nn.Linear(4096, cell_encodings_flat.shape[1]),
                nn.ReLU(),
            )
            self.dim_reduction.to(self.device)

        # (B, 512, image_size / 32, image_size / 32)

        # Step 3: Concatenate flattened arrays
        merged_tensor = torch.cat((cell_encodings_flat, logits_tissue_flat), dim=1)

        cell_weights = (*cell_weights[: len(cell_weights) - 1], cell_encodings)

        logits_cell = self.model_cell_decoder(cell_weights)

        # Upscaling the result to the shape of the ground truth
        if logits_cell.shape[1:] != self.output_spatial_shape:
            logits_cell = nn.functional.interpolate(
                logits_cell,
                size=OCELOT_IMAGE_SIZE,  # (height, width)
                mode="bilinear",
                align_corners=False,
            )

            logits_tissue = nn.functional.interpolate(
                logits_tissue.unsqueeze(1).float(),
                size=OCELOT_IMAGE_SIZE,  # (height, width)
                mode="nearest",
                # align_corners=False,
            )

        logits_tissue = nn.functional.one_hot(
            logits_tissue.squeeze(1).long(), num_classes=3
        ).permute(0, 3, 1, 2)
        merged_logits = torch.concatenate([logits_cell, logits_tissue], dim=1)

        return merged_logits


class SegformerSharingModel(nn.Module):
    def __init__(
        self,
        backbone_model,
        pretrained_dataset: str,
        input_image_size: int = 1024,
        output_image_size: int = 1024,
    ):
        super().__init__()

        if input_image_size not in [512, 1024]:
            raise ValueError("Invalid image size, must be either 512 or 1024")

        self.input_image_size = (input_image_size, input_image_size)
        self.output_image_dimensions = (output_image_size, output_image_size)

        self.backbone_model = backbone_model
        self.pretrained_dataset = pretrained_dataset

        self.cell_model = setup_segformer(
            backbone_name=self.backbone_model,
            num_classes=3,
            num_channels=6,
            # num_channels=3,
            pretrained_dataset=self.pretrained_dataset,
        )

        self.tissue_model = setup_segformer(
            backbone_name="b1",
            num_classes=3,
            num_channels=3,
            pretrained_dataset=self.pretrained_dataset,
        )

    def forward(self, x: torch.Tensor, offsets: torch.Tensor):
        """
        Assumes that input x has shape (B, 6, H, W), where the first three
        channels in the second dimension correspond to the cell image and
        the three last channels in the second dimension correspond to the
        tissue image

        offsets: (B, 2)
        """
        cell_image = x[:, :3]
        tissue_image = x[:, 3:]

        # Tissue-branch
        tissue_logits = self.tissue_model(tissue_image).logits
        tissue_logits_orig = torch.nn.functional.interpolate(
            tissue_logits,
            size=self.input_image_size,
            mode="bilinear",
            align_corners=False,
        )
        tissue_logits = torch.nn.functional.softmax(tissue_logits_orig, dim=1)
        cropped_tissue_logits = []
        for batch_idx in range(tissue_logits.shape[0]):
            crop = crop_and_resize_tissue_faster(
                tissue_logits[batch_idx],
                x_offset=offsets[batch_idx][0],
                y_offset=offsets[batch_idx][1],
            )
            cropped_tissue_logits.append(crop)
        # Restoring the original shape, but now cropped
        tissue_logits = torch.stack(cropped_tissue_logits)

        # Cell-branch
        model_input = torch.cat((cell_image, tissue_logits), dim=1)
        cell_logits = self.cell_model(model_input).logits
        # cell_logits = self.cell_model(cell_image).logits

        if cell_logits.shape[1:] != self.output_image_dimensions:
            cell_logits = torch.nn.functional.interpolate(
                cell_logits,
                size=self.output_image_dimensions,
                mode="bilinear",
                align_corners=False,
            )

        if tissue_logits_orig.shape[1:] != self.output_image_dimensions:
            tissue_logits_orig = torch.nn.functional.interpolate(
                tissue_logits_orig,
                size=self.output_image_dimensions,
                mode="bilinear",
                align_corners=False,
            )

        return cell_logits, tissue_logits_orig
        # return cell_logits, cell_logits


if __name__ == "__main__":

    backbone_model = "b1"
    pretrained_dataset = "ade"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sharing_model = SegformerSharingModel(
        backbone_model=backbone_model,
        pretrained_dataset=pretrained_dataset,
        output_image_size=512,
    )
    sharing_model.to(device)

    batch_size = 2
    channels = 6
    height = 1024
    width = 1024

    x = torch.ones(batch_size, channels, height, width)
    x = x.to(device)
    offsets = torch.tensor([[0.625, 0.125], [0.125, 0.125]])
    result = sharing_model(x, offsets)
    print(result)

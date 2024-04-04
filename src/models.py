import json
import os
import sys
import torch
import torch.nn.functional as F

sys.path.append(os.getcwd())

from torch import nn
from typing import Union, Optional
from src.deeplabv3.network.utils import IntermediateLayerGetter
from src.deeplabv3.network.backbone import resnet

from src.deeplabv3.network._deeplab import (
    DeepLabHeadV3Plus,
    DeepLabV3,
)

from src.utils.constants import OCELOT_IMAGE_SIZE
from src.utils.utils import crop_and_resize_tissue_patch

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

    segformer_architectures = {
        "b0": {
            "depths": [2, 2, 2, 2],
            "hidden_sizes": [32, 64, 160, 256],
            "decoder_hidden_size": 256,
        },
        "b1": {
            "depths": [2, 2, 2, 2],
            "hidden_sizes": [64, 128, 320, 512],
            "decoder_hidden_size": 256,
        },
        "b2": {
            "depths": [3, 4, 6, 3],
            "hidden_sizes": [64, 128, 320, 512],
            "decoder_hidden_size": 768,
        },
        "b3": {
            "depths": [3, 4, 18, 3],
            "hidden_sizes": [64, 128, 320, 512],
            "decoder_hidden_size": 768,
        },
        "b4": {
            "depths": [3, 8, 27, 3],
            "hidden_sizes": [64, 128, 320, 512],
            "decoder_hidden_size": 768,
        },
        "b5": {
            "depths": [3, 6, 40, 3],
            "hidden_sizes": [64, 128, 320, 512],
            "decoder_hidden_size": 768,
        },
    }

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
        parameters = self.segformer_architectures[backbone_name]

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
                size=OCELOT_IMAGE_SIZE,  # (height, width)
                mode="bilinear",
                align_corners=False,
            )

        return logits


def setup_segformer(
    backbone_name: str,
    num_classes: int,
    num_channels: int,
    parameters,
    pretrained_dataset: Optional[str] = None,
):

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

    segformer_architectures = {
        "b0": {
            "depths": [2, 2, 2, 2],
            "hidden_sizes": [32, 64, 160, 256],
            "decoder_hidden_size": 256,
        },
        "b1": {
            "depths": [2, 2, 2, 2],
            "hidden_sizes": [64, 128, 320, 512],
            "decoder_hidden_size": 256,
        },
        "b2": {
            "depths": [3, 4, 6, 3],
            "hidden_sizes": [64, 128, 320, 512],
            "decoder_hidden_size": 768,
        },
        "b3": {
            "depths": [3, 4, 18, 3],
            "hidden_sizes": [64, 128, 320, 512],
            "decoder_hidden_size": 768,
        },
        "b4": {
            "depths": [3, 8, 27, 3],
            "hidden_sizes": [64, 128, 320, 512],
            "decoder_hidden_size": 768,
        },
        "b5": {
            "depths": [3, 6, 40, 3],
            "hidden_sizes": [64, 128, 320, 512],
            "decoder_hidden_size": 768,
        },
    }

    def __init__(
        self,
        backbone_name: str,
        num_classes: int,
        num_channels: int,
        metadata: str,
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
        parameters = self.segformer_architectures[backbone_name]

        self.model_tissue = setup_segformer(
            backbone_name=backbone_name,
            num_classes=num_classes,
            num_channels=int(num_channels / 2),
            parameters=parameters,
            pretrained_dataset=pretrained_dataset,
        )

        self.model_cell = setup_segformer(
            backbone_name=backbone_name,
            num_classes=num_classes,
            num_channels=int(num_channels / 2),
            parameters=parameters,
            pretrained_dataset=pretrained_dataset,
        )

        self.model_cell_encoder = self.model_cell.segformer
        self.model_cell_decoder = self.model_cell.decode_head

        self.metadata = metadata
        self.dim_reduction = None

    def forward(self, x, pair_id):

        # Assuming the three first channels belong to cell
        logits_tissue = self.model_tissue(x[:, int(self.num_channels / 2) :]).logits

        scaled_list = []

        for i in range(len(pair_id)):
            # TODO: Do we fetch the correct metadata for the sample
            meta_pair = self.metadata[pair_id[i]]
            tissue_mpp = meta_pair["tissue"]["resized_mpp_x"]
            cell_mpp = meta_pair["cell"]["resized_mpp_x"]
            x_offset = meta_pair["patch_x_offset"]
            y_offset = meta_pair["patch_y_offset"]

            argmaxed = logits_tissue[i].argmax(dim=0).to(dtype=torch.int)

            scaled_tissue: torch.Tensor = crop_and_resize_tissue_patch(
                image=argmaxed,
                tissue_mpp=tissue_mpp,
                cell_mpp=cell_mpp,
                x_offset=x_offset,
                y_offset=y_offset,
            )
            scaled_list.append(scaled_tissue.unsqueeze(0))

        logits_tissue = torch.concatenate(scaled_list, dim=0)

        outputs = self.model_cell_encoder(
            x[:, : int(self.num_channels / 2)], output_hidden_states=True
        )
        cell_weights = outputs.hidden_states

        cell_encodings = cell_weights[-1]

        ## Mixing the features

        # Flatten logits_tissue
        logits_tissue_flat = logits_tissue.reshape(
            logits_tissue.size(0), -1
        )  # Flattening

        # Flatten cell encodings to match the logits_tissue_flat shape
        cell_encodings_flat = cell_encodings.reshape(cell_encodings.size(0), -1)

        # Step 3: Concatenate flatten arrays
        merged_tensor = torch.cat((cell_encodings_flat, logits_tissue_flat), dim=1)

        if self.dim_reduction is None:
            self.dim_reduction = nn.Sequential(
                nn.Linear(merged_tensor.shape[1], 4096),
                nn.ReLU(),
                nn.Linear(4096, cell_encodings_flat.shape[1]),
                nn.ReLU(),
            )

        reduced_tensor = self.dim_reduction(merged_tensor)
        cell_encodings = reduced_tensor.view_as(cell_encodings)

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


if __name__ == "__main__":

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = "/cluster/projects/vc/data/mic/open/OCELOT/ocelot_data"

    metadata_path = os.path.join(data_dir, "metadata.json")
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    sharing_model = TissueCellSharingSegformerModel("b1", 3, 6, metadata=list(metadata["sample_pairs"].values()),)
    # sharing_model.to(device)

    batch_size = 2
    channels = 6
    height = 512
    width = 512

    x = torch.ones(batch_size, channels, height, width)
    # x = x.to(device)

    result = sharing_model(x, pair_id=(0,1))
    print(result)

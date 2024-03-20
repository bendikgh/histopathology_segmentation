import os
import sys

sys.path.append(os.getcwd())

from torch import nn
from src.deeplabv3.network.utils import IntermediateLayerGetter
from src.deeplabv3.network.backbone import resnet

# from deeplabv3.network.modeling import _segm_resnet
from src.deeplabv3.network._deeplab import (
    DeepLabHeadV3Plus,
    DeepLabV3,
)

from src.utils.constants import OCELOT_IMAGE_SIZE

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
        pretrained_dataset: str = None,
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
            encoder = SegformerModel.from_pretrained(
                f"nvidia/segformer-{backbone_name}-finetuned-ade-512-512"
            )
            model.segformer = encoder
        elif pretrained_dataset == "cityscapes":
            encoder = SegformerModel.from_pretrained(
                f"nvidia/segformer-{backbone_name}-finetuned-cityscapes-1024-1024"
            )
            model.segformer = encoder

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

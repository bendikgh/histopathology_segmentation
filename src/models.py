import os
import sys
import torch
import torch.nn.functional as F


from torch import nn
from typing import Union, Optional, Dict
from torchvision import transforms
from transformers import (
    SegformerForSemanticSegmentation,
    SegformerConfig,
    SegformerModel,
    Dinov2Model,
    Dinov2PreTrainedModel,
    ViTModel,
    ViTConfig,
)

sys.path.append(os.getcwd())

from src.deeplabv3.network.utils import IntermediateLayerGetter
from src.deeplabv3.network.backbone import resnet
from src.deeplabv3.network._deeplab import (
    DeepLabHeadV3Plus,
    DeepLabV3,
)
from src.utils.constants import OCELOT_IMAGE_SIZE, SEGFORMER_ARCHITECTURES
from src.utils.utils import crop_and_resize_tissue_faster


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

        self.model = setup_segformer(
            backbone_name=backbone_name,
            num_classes=num_classes,
            num_channels=num_channels,
            pretrained_dataset=pretrained_dataset,
        )

    def forward(self, x):
        logits = self.model(x).logits

        # Upscaling the result to the shape of the ground truth
        if logits.shape[1:] != self.output_spatial_shape:
            logits = F.interpolate(
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
            logits_cell = F.interpolate(
                logits_cell,
                size=OCELOT_IMAGE_SIZE,  # (height, width)
                mode="bilinear",
                align_corners=False,
            )

            logits_tissue = F.interpolate(
                logits_tissue.unsqueeze(1).float(),
                size=OCELOT_IMAGE_SIZE,  # (height, width)
                mode="nearest",
                # align_corners=False,
            )

        logits_tissue = F.one_hot(
            logits_tissue.squeeze(1).long(), num_classes=3
        ).permute(0, 3, 1, 2)
        merged_logits = torch.concatenate([logits_cell, logits_tissue], dim=1)

        return merged_logits


class LinearClassifier(torch.nn.Module):
    def __init__(self, in_channels, tokenW=73, tokenH=73, num_labels=1):
        super(LinearClassifier, self).__init__()

        self.in_channels = in_channels
        self.width = tokenW
        self.height = tokenH
        self.classifier = torch.nn.Conv2d(in_channels, num_labels, (1, 1))

    def forward(self, embeddings):
        embeddings = embeddings.reshape(-1, self.height, self.width, self.in_channels)
        embeddings = embeddings.permute(0, 3, 1, 2)

        return self.classifier(embeddings)


class Dinov2ForSemanticSegmentation(Dinov2PreTrainedModel):
    """
    Dinov2 with a linear classifier attached for semantic segmentation.
    """

    def __init__(self, config):
        super().__init__(config)

        self.dinov2 = Dinov2Model(config)
        self.classifier = LinearClassifier(
            config.hidden_size, 73, 73, config.num_labels
        )

        for name, param in self.dinov2.named_parameters():
            if name.startswith("dinov2"):
                param.requires_grad = False

    def forward(
        self, pixel_values, output_hidden_states=False, output_attentions=False
    ):
        outputs = self.dinov2(
            pixel_values,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )

        # get the patch embeddings - so we exclude the CLS token
        patch_embeddings = outputs.last_hidden_state[:, 1:, :]

        # convert to logits and upsample to the size of the pixel values
        logits = self.classifier(patch_embeddings)
        logits = F.interpolate(
            logits, size=pixel_values.shape[2:], mode="bilinear", align_corners=False
        )

        return logits


class SimpleDecoder(nn.Module):
    def __init__(self):
        super(SimpleDecoder, self).__init__()
        self.up1 = nn.ConvTranspose2d(
            768, 384, kernel_size=3, stride=2, padding=1, output_padding=1
        )  # Upsample to 28x28
        self.conv1 = nn.Conv2d(384, 384, kernel_size=3, padding=1)
        self.up2 = nn.ConvTranspose2d(
            384, 192, kernel_size=3, stride=2, padding=1, output_padding=1
        )  # Upsample to 56x56
        self.conv2 = nn.Conv2d(192, 192, kernel_size=3, padding=1)
        self.up3 = nn.ConvTranspose2d(
            192, 96, kernel_size=3, stride=2, padding=1, output_padding=1
        )  # Upsample to 112x112
        self.conv3 = nn.Conv2d(96, 96, kernel_size=3, padding=1)
        self.up4 = nn.ConvTranspose2d(
            96, 48, kernel_size=3, stride=2, padding=1, output_padding=1
        )  # Upsample to 224x224
        self.final_conv = nn.Conv2d(48, 3, kernel_size=1)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = F.relu(self.up1(x))
        x = F.relu(self.conv1(x))
        x = F.relu(self.up2(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.up3(x))
        x = F.relu(self.conv3(x))
        x = self.up4(x)
        x = self.final_conv(x)
        return x


class Deconv2DBlock(nn.Module):
    """
    Deconvolution block with ConvTranspose2d followed by Conv2d, batch-normalisation, ReLU activation and dropout
    this model is copied from https://github.com/Lzy-dot/OCELOT2023/tree/main

    Args:
        in_channels (int): Number of input channels for deconv block
        out_channels (int): Number of output channels for deconv and convolution.
        kernel_size (int, optional): Kernel size for convolution. Defaults to 3.
        dropout (float, optional): Dropout. Defaults to 0.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dropout: float = 0,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=2,
                stride=2,
                padding=0,
                output_padding=0,
            ),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=(kernel_size - 1) // 2,
                groups=out_channels,
            ),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.block(x)


class Conv2DBlock(nn.Module):
    """
    Conv2DBlock with convolution followed by batch-normalisation, ReLU activation and dropout
    This model is copied form https://github.com/Lzy-dot/OCELOT2023/tree/main

    Args:
        in_channels (int): Number of input channels for convolution
        out_channels (int): Number of output channels for convolution
        kernel_size (int, optional): Kernel size for convolution. Defaults to 3.
        dropout (float, optional): Dropout. Defaults to 0.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dropout: float = 0,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=((kernel_size - 1) // 2),
                groups=in_channels,
            ),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.block(x)


class ViTDecoder(nn.Module):
    """
    Decoder which processes the patch embeddings from the transformer blocks.
    The embeddings are transformed and upsampled in a UNet manner.

    This decoder is mostly copied from https://github.com/Lzy-dot/OCELOT2023/tree/main
    """

    def __init__(self, backbone_config, drop_rate=0) -> None:
        super().__init__()

        self.embed_dim = backbone_config.hidden_size  # 768 for vit_base
        self.skip_dim_11 = 512
        self.skip_dim_12 = 256
        self.bottleneck_dim = 512
        self.drop_rate = drop_rate

        # Deocders for the patch embeddings
        self.bottleneck_upsampler = nn.ConvTranspose2d(
            in_channels=self.embed_dim,
            out_channels=self.bottleneck_dim,
            kernel_size=2,
            stride=2,
            padding=0,
            output_padding=0,
        )
        self.decoder3_upsampler = nn.Sequential(
            Conv2DBlock(
                self.bottleneck_dim * 2, self.bottleneck_dim, dropout=self.drop_rate
            ),
            Conv2DBlock(
                self.bottleneck_dim, self.bottleneck_dim, dropout=self.drop_rate
            ),
            Conv2DBlock(
                self.bottleneck_dim, self.bottleneck_dim, dropout=self.drop_rate
            ),
            nn.ConvTranspose2d(
                in_channels=self.bottleneck_dim,
                out_channels=256,
                kernel_size=2,
                stride=2,
                padding=0,
                output_padding=0,
            ),
        )
        self.decoder2_upsampler = nn.Sequential(
            Conv2DBlock(256 * 2, 256, dropout=self.drop_rate),
            Conv2DBlock(256, 256, dropout=self.drop_rate),
            nn.ConvTranspose2d(
                in_channels=256,
                out_channels=128,
                kernel_size=2,
                stride=2,
                padding=0,
                output_padding=0,
            ),
        )
        self.decoder1_upsampler = nn.Sequential(
            Conv2DBlock(128 * 2, 128, dropout=self.drop_rate),
            Conv2DBlock(128, 128, dropout=self.drop_rate),
            nn.ConvTranspose2d(
                in_channels=128,
                out_channels=64,
                kernel_size=2,
                stride=2,
                padding=0,
                output_padding=0,
            ),
        )
        self.decoder0_header = nn.Sequential(
            Conv2DBlock(64 * 2, 64, dropout=self.drop_rate),
            Conv2DBlock(64, 64, dropout=self.drop_rate),
            nn.Conv2d(
                in_channels=64,
                out_channels=3,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
        )

        # Decoder for merging and upsampling patch embeddings
        self.decoder0 = nn.Sequential(
            Conv2DBlock(3, 32, 3, dropout=self.drop_rate),
            Conv2DBlock(32, 64, 3, dropout=self.drop_rate),
        )  # skip connection 0
        self.decoder1 = nn.Sequential(
            Deconv2DBlock(self.embed_dim, self.skip_dim_11, dropout=self.drop_rate),
            Deconv2DBlock(self.skip_dim_11, self.skip_dim_12, dropout=self.drop_rate),
            Deconv2DBlock(self.skip_dim_12, 128, dropout=self.drop_rate),
        )  # skip connection 1
        self.decoder2 = nn.Sequential(
            Deconv2DBlock(self.embed_dim, self.skip_dim_11, dropout=self.drop_rate),
            Deconv2DBlock(self.skip_dim_11, 256, dropout=self.drop_rate),
        )  # skip connection 2
        self.decoder3 = nn.Sequential(
            Deconv2DBlock(self.embed_dim, self.bottleneck_dim, dropout=self.drop_rate)
        )  # skip connection 3

    def forward(
        self,
        z0: torch.Tensor,
        z1: torch.Tensor,
        z2: torch.Tensor,
        z3: torch.Tensor,
        z4: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward upsample branch

        Args:
            z0 (torch.Tensor): Highest skip
            z1 (torch.Tensor): 1. Skip
            z2 (torch.Tensor): 2. Skip
            z3 (torch.Tensor): 3. Skip
            z4 (torch.Tensor): Bottleneck
            branch_decoder (nn.Sequential): Branch decoder network

        Returns:
            torch.Tensor: Branch Output
        """
        b4 = self.bottleneck_upsampler(z4)

        b3 = self.decoder3(z3)
        b3 = self.decoder3_upsampler(torch.cat([b3, b4], dim=1))

        b2 = self.decoder2(z2)
        b2 = self.decoder2_upsampler(torch.cat([b2, b3], dim=1))

        b1 = self.decoder1(z1)
        b1 = self.decoder1_upsampler(torch.cat([b1, b2], dim=1))

        b0 = self.decoder0(z0)
        branch_output = self.decoder0_header(torch.cat([b0, b1], dim=1))

        return branch_output


class ViTUNetModel(torch.nn.Module):
    """
    Vision transformer with UNet structure.
    This model is recreated from https://github.com/Lzy-dot/OCELOT2023/tree/main
    """

    def __init__(
        self,
        pretrained_dataset="owkin/phikon",
        input_spatial_shape=1024,
        output_spatial_shape=1024,
        extract_layers=[3, 6, 9, 12],
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        if len(extract_layers) != 4:
            raise ValueError(
                "The argument extract layers should be a list of length four"
            )

        self.input_spatial_shape = input_spatial_shape
        self.output_spatial_shape = output_spatial_shape

        # Load pretrained weights
        if pretrained_dataset:
            vit_config = ViTConfig.from_pretrained(pretrained_dataset)
            vit_config.image_size = input_spatial_shape
            self.vit_encoder = ViTModel.from_pretrained(
                pretrained_dataset, config=vit_config, ignore_mismatched_sizes=True
            )
            # Interpolate positional embeddings to match the output spatial shape
            self.adjust_positional_embeddings(input_spatial_shape)
        else:
            self.vit_encoder = ViTModel()

        # Which patch embeddings to extract for the decoder
        self.extract_layers = extract_layers

        vit_config = self.vit_encoder.config
        self.decoder = ViTDecoder(vit_config)

        # Freeze backbone/encoder parameters
        # for _, param in self.vit_encoder.named_parameters():
        #     param.requires_grad = False

    def adjust_positional_embeddings(self, new_size):

        # Get the original positional embeddings
        pos_embed = ViTModel.from_pretrained(
            "owkin/phikon"
        ).embeddings.position_embeddings

        # Separate the class token embedding
        class_token_embed = pos_embed[:, 0:1, :]
        patch_embeddings = pos_embed[:, 1:, :]

        # Calculate the new grid size from patch size
        n_patches_side = new_size // self.vit_encoder.config.patch_size
        n_patches = n_patches_side**2

        # Reshape patch embeddings to [1, 14, 14, embedding_dim]
        patch_embeddings = patch_embeddings.reshape(1, -1, 14, 14)

        # Interpolate patch positional embeddings to the new grid size
        new_patch_embed = torch.nn.functional.interpolate(
            patch_embeddings,
            size=(n_patches_side, n_patches_side),
            mode="bilinear",
            align_corners=False,
        )

        # Flatten the interpolated embeddings back to [1, n_patches, embedding_dim]
        new_patch_embed = new_patch_embed.reshape(1, n_patches, -1)

        # Concatenate the class token embedding back
        new_pos_embed = torch.cat([class_token_embed, new_patch_embed], dim=1)

        # Update the model's positional embeddings
        self.vit_encoder.embeddings.position_embeddings = torch.nn.Parameter(
            new_pos_embed.squeeze(0)
        )

    def forward(self, pixel_values):

        outputs = self.vit_encoder(pixel_values, output_hidden_states=True)
        hidden_states = outputs.hidden_states

        patch_size = self.input_spatial_shape // 16

        embeddings = [pixel_values]
        for i in self.extract_layers:
            hidden_state = (
                hidden_states[i][:, 1:]
                .reshape(-1, patch_size, patch_size, 768)
                .permute(0, 3, 1, 2)
            )
            embeddings.append(hidden_state)

        # Decode patch embeddings
        logits = self.decoder(*embeddings)

        if logits.shape[1:] != self.output_spatial_shape:
            logits = F.interpolate(
                logits,
                size=(self.output_spatial_shape, self.output_spatial_shape),
                mode="bilinear",
                align_corners=False,
            )

        return logits


class SegformerJointPred2InputModel(nn.Module):
    def __init__(
        self,
        backbone_model,
        pretrained_dataset: str,
        input_image_size: int = 1024,
        output_image_size: int = 1024,
    ):
        super().__init__()

        if input_image_size not in [224, 512, 1024]:
            raise ValueError("Invalid image size, must be either 512 or 1024")

        self.input_image_size = (input_image_size, input_image_size)
        self.output_image_dimensions = (output_image_size, output_image_size)

        self.backbone_model = backbone_model
        self.pretrained_dataset = pretrained_dataset

        self.cell_model = setup_segformer(
            backbone_name=self.backbone_model,
            num_classes=3,
            num_channels=6,
            pretrained_dataset=self.pretrained_dataset,
        )

        self.tissue_model = setup_segformer(
            backbone_name="b0",
            num_classes=3,
            num_channels=3,
            pretrained_dataset=self.pretrained_dataset,
        )

    def forward(
        self,
        cell_input: torch.Tensor,
        tissue_input: torch.Tensor,
        offsets: torch.Tensor,
    ):
        """
        Performs inference on the inputs with the model. Cell and input images
        might have different dimensions, but the output will be resized to the
        output_image_dimensions. H_c and W_c refer to height and width for
        the cell image, and H_t and W_t refer to height and width for the tissue
        image.

        Args:
            cell_input: Input tensor for the cell model, (B, 3, H_c, W_c)
            tissue_input: Input tensor for the tissue model, (B, 3, H_t, W_t)
            offsets: Offset tensor for the tissue model, (B, 2)

        """

        # Tissue-branch
        tissue_logits = self.tissue_model(tissue_input).logits
        tissue_logits_orig = F.interpolate(
            tissue_logits,
            size=self.input_image_size,
            mode="bilinear",
            align_corners=False,
        )
        tissue_logits = F.softmax(tissue_logits_orig, dim=1)
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
        model_input = torch.cat((cell_input, tissue_logits), dim=1)
        cell_logits = self.cell_model(model_input).logits
        # cell_logits = self.cell_model(cell_image).logits

        if cell_logits.shape[1:] != self.output_image_dimensions:
            cell_logits = F.interpolate(
                cell_logits,
                size=self.output_image_dimensions,
                mode="bilinear",
                align_corners=False,
            )

        if tissue_logits_orig.shape[1:] != self.output_image_dimensions:
            tissue_logits_orig = F.interpolate(
                tissue_logits_orig,
                size=self.output_image_dimensions,
                mode="bilinear",
                align_corners=False,
            )

        return cell_logits, tissue_logits_orig


class SegformerTissueToCellDecoderModel(nn.Module):
    def __init__(
        self,
        backbone_model,
        pretrained_dataset: str,
        drop_rate: float = 0.0,
        input_image_size: int = 1024,
        output_image_size: int = 1024,
    ):
        super().__init__()

        if input_image_size not in [512, 1024]:
            raise ValueError("Invalid image size, must be either 512 or 1024")

        self.drop_rate = drop_rate
        self.input_image_size = (input_image_size, input_image_size)
        self.output_image_dimensions = (output_image_size, output_image_size)

        self.backbone_model = backbone_model
        self.pretrained_dataset = pretrained_dataset

        self.cell_model = setup_segformer(
            backbone_name=self.backbone_model,
            num_classes=3,
            num_channels=3,
            pretrained_dataset=self.pretrained_dataset,
        )

        self.tissue_model = setup_segformer(
            backbone_name="b1",  # TODO: make this into an argument?
            num_classes=3,
            num_channels=3,
            pretrained_dataset=self.pretrained_dataset,
        )

        segformer_info = SEGFORMER_ARCHITECTURES[self.backbone_model]
        hidden_sizes = segformer_info["hidden_sizes"]

        self.model_cell_encoder = self.cell_model.segformer
        self.model_cell_segformer_decoder = self.cell_model.decode_head

        # Convolution layers to adjust tissue logits for merging with cell encodings
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=hidden_sizes[0],
            kernel_size=7,
            stride=4,
            padding=3,
        )
        self.conv2 = nn.Conv2d(
            in_channels=hidden_sizes[0],
            out_channels=hidden_sizes[1],
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.conv3 = nn.Conv2d(
            in_channels=hidden_sizes[1],
            out_channels=hidden_sizes[2],
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.conv4 = nn.Conv2d(
            in_channels=hidden_sizes[2],
            out_channels=hidden_sizes[3],
            kernel_size=3,
            stride=2,
            padding=1,
        )

        # Layer for transforming cell_image + tissue_logits before concat
        self.conv0 = nn.Conv2d(
            in_channels=3,
            out_channels=3,
            kernel_size=7,
            stride=4,
            padding=3,
        )

        # Conv layer sliding each pixel and weighting the channels
        # Outputs the cell predictions
        self.conv_output = nn.Conv2d(in_channels=6, out_channels=3, kernel_size=1)

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
        tissue_logits_orig = F.interpolate(
            tissue_logits,
            size=self.input_image_size,
            mode="bilinear",
            align_corners=False,
        )
        tissue_logits = F.softmax(tissue_logits_orig, dim=1)
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

        # Cell-encoder
        cell_encodings = self.model_cell_encoder(
            cell_image, output_hidden_states=True
        ).hidden_states

        # Extract encodings from different layers in the segformer
        encodings1 = cell_encodings[0]
        encodings2 = cell_encodings[1]
        encodings3 = cell_encodings[2]
        encodings4 = cell_encodings[3]

        # Add tissue_logits to the different encodings.
        # Replicating patch merging within the segformer
        tissue_trans1 = self.conv1(tissue_logits)
        tissue_trans2 = self.conv2(tissue_trans1)
        tissue_trans3 = self.conv3(tissue_trans2)
        tissue_trans4 = self.conv4(tissue_trans3)

        cell_encoding1 = encodings1 + tissue_trans1
        cell_encoding2 = encodings2 + tissue_trans2
        cell_encoding3 = encodings3 + tissue_trans3
        cell_encoding4 = encodings4 + tissue_trans4

        cell_decoder_input = (
            cell_encoding1,
            cell_encoding2,
            cell_encoding3,
            cell_encoding4,
        )

        # Output from the segformer decoder
        cell_segformer_decoder_output = self.model_cell_segformer_decoder(
            cell_decoder_input
        )

        # Extracted info from cell_image and tissue_logits merged
        cell_tissue_info = self.conv0(cell_image + tissue_logits)
        stacked = torch.cat([cell_segformer_decoder_output, cell_tissue_info], dim=1)

        # Conv layer sliding each pixel and weighting the channels
        # Outputs the cell predictions
        cell_logits = self.conv_output(stacked)

        # Interpolation to desired output shape
        if cell_logits.shape[1:] != self.output_image_dimensions:
            cell_logits = F.interpolate(
                cell_logits,
                size=self.output_image_dimensions,
                mode="bilinear",
                align_corners=False,
            )

        if tissue_logits_orig.shape[1:] != self.output_image_dimensions:
            tissue_logits_orig = F.interpolate(
                tissue_logits_orig,
                size=self.output_image_dimensions,
                mode="bilinear",
                align_corners=False,
            )

        return cell_logits, tissue_logits_orig


if __name__ == "__main__":
    batch_size = 2
    channels = 3
    height = 1024
    width = 1024

    model = CustomSegformerModel(
        backbone_name="b1", num_classes=3, num_channels=3, pretrained_dataset="ade"
    )

    x = torch.ones(batch_size, channels, height, width)
    # x = x.to(device)
    offsets = torch.tensor([[0.625, 0.125], [0.125, 0.125]])
    result = model(x)
    print(result)

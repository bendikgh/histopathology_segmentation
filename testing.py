# Fixing automatic autoreload
import os 

# Making sure we are running the code from the root directory
current_directory = os.getcwd()
if current_directory.endswith("notebooks"):
    os.chdir("..")
    print("Changed directory to:", os.getcwd())
else:
    print("Directory was already correct, so did not change.")
import torch
import matplotlib.pyplot as plt

from torch import nn


from src.trainable import SegformerJointPred2InputTrainable
from src.utils.constants import IDUN_OCELOT_DATA_PATH
from ocelot23algo.user.inference import SegformerAdditiveJointPred2DecoderWithTTAModel

from src.utils.utils import (
    get_metadata_dict,
    get_metadata_with_offset,
    get_ocelot_files,
)

from typing import Union, Optional, List

from src.models import (
    CustomSegformerModel,
    DeepLabV3plusModel,
    SegformerJointPred2InputModel,
    SegformerAdditiveJointPred2DecoderModel,
    ViTUNetModel,
)
class SegformerAdditiveJointPred2DecoderTrainable(SegformerJointPred2InputTrainable):

    def create_model(
        self,
        backbone_name: str,
        pretrained: bool,
        device: torch.device,
        model_path: Optional[str] = None,
    ):

        model = SegformerAdditiveJointPred2DecoderModel(
            backbone_model=backbone_name,
            pretrained_dataset=self.pretrained_dataset,
            input_image_size=self.cell_image_input_size,
            output_image_size=1024,
        )
        if model_path is not None:
            model.load_state_dict(torch.load(model_path))
        model.to(device)
        return model

    def create_evaluation_model(self, partition: str):
        metadata = get_metadata_with_offset(
            data_dir=IDUN_OCELOT_DATA_PATH, partition=partition
        )
        cell_transform = self._create_dual_transform(
            normalization=self.normalization, partition=partition, kind="cell"
        )
        tissue_transform = self._create_dual_transform(
            normalization=self.normalization, partition=partition, kind="tissue"
        )
        return SegformerAdditiveJointPred2DecoderWithTTAModel(
            metadata=metadata,
            cell_model=self.model,
            cell_transform=cell_transform,
            tissue_transform=tissue_transform,
        )
normalization = "macenko"
batch_size = 1
pretrained = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
backbone_model = "b3"
pretrained_dataset = "ade"
cell_image_input_size = 1024
tissue_image_input_size = 1024
exclude_bad_images = False
data_dir = IDUN_OCELOT_DATA_PATH
weight_loss = False

trainable = SegformerAdditiveJointPred2DecoderTrainable(
    normalization=normalization,
    batch_size=batch_size,
    pretrained=pretrained,
    device=device,
    backbone_model=backbone_model,
    pretrained_dataset=pretrained_dataset,
    data_dir=data_dir,
    cell_image_input_size=cell_image_input_size,
    tissue_image_input_size=tissue_image_input_size,
    exclude_bad_images=exclude_bad_images,
    weight_loss=weight_loss,
)

best_model_path = "outputs/models/20240528_175128/Segformer_Sharing_backbone-b3_best.pth" #"outputs/models/20240529_005550/Segformer_Sharing_backbone-b3_best.pth" #"outputs/models/20240528_174655/Segformer_Sharing_backbone-b3_best.pth" #"outputs/models/20240528_175128/Segformer_Sharing_backbone-b3_best.pth" #"outputs/models/20240528_133252/Segformer_Sharing_backbone-b3_best.pth" #"outputs/models/20240527_210945/Segformer_Sharing_backbone-b3_best.pth" #"outputs/models/20240526_080329/Segformer_Sharing_backbone-b3_best.pth" #outputs/models/20240518_201826/Segformer_Sharing_backbone-b3_best.pth" #outputs/models/20240519_022221/Segformer_Sharing_backbone-b3_best.pth"

trainable.model = trainable.create_model(
    backbone_name=backbone_model,
    pretrained=pretrained,
    device=trainable.device,
    model_path=best_model_path,
)

trainable.model.eval()
val_evaluation_function = trainable.get_evaluation_function(partition="val")
test_evaluation_function = trainable.get_evaluation_function(partition="test")
val_score = val_evaluation_function()
print(f"val score: {val_score:.4f}")
test_score = test_evaluation_function()
print(f"test score: {test_score:.4f}")
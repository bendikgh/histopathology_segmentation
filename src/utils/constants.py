PYTORCH_STANDARD_IMAGE_SHAPE = (3, 1024, 1024)
NUMPY_STANDARD_IMAGE_SHAPE = (1024, 1024, 3)

IDUN_OCELOT_DATA_PATH = "/cluster/projects/vc/data/mic/open/OCELOT/ocelot_data"
# IDUN_OCELOT_DATA_PATH = "ocelot_data"

DEFAULT_EPOCHS: int = 2
DEFAULT_BATCH_SIZE: int = 2
DEFAULT_DATA_DIR: str = IDUN_OCELOT_DATA_PATH
DEFAULT_CHECKPOINT_INTERVAL: int = 10
DEFAULT_BACKBONE_MODEL: str = "resnet50"
DEFAULT_DROPOUT_RATE: float = 0.3
DEFAULT_LEARNING_RATE: float = 1e-4
DEFAULT_LEARNING_RATE_END: float = 2e-5
DEFAULT_PRETRAINED: int = 1
DEFAULT_WARMUP_EPOCHS: int = 0
DEFAULT_DO_SAVE: int = 0
DEFAULT_DO_EVALUATE: int = 1
DEFAULT_BREAK_AFTER_ONE_ITERATION: int = 1
DEFAULT_NORMALIZATION: str = "off"
DEFAULT_MODEL_ARCHITECTURE = "segformer_cell_only"

SEGFORMER_JOINT_TISSUE_IMAGE_SIZE = 512  # 1024
SEGFORMER_JOINT_CELL_IMAGE_SIZE = 512

MISSING_IMAGE_NUMBERS: list = [8, 42, 53, 217, 392, 558, 570, 586, 589, 609, 615]
MAX_IMAGE_NUMBER: int = 667
DATASET_PARTITION_OFFSETS = {
    "train": 0,
    "val": 400,
    "test": 537,
}

CELL_IMAGE_MEAN: list = [0.75928293, 0.57434749, 0.6941771]
CELL_IMAGE_STD: list = [0.1899926, 0.2419049, 0.18382073]
# TISSUE_IMAGE_MEAN: list = [0.76528257, 0.58330387, 0.69940715]
# TISSUE_IMAGE_STD: list = [0.18308686, 0.23847347, 0.18801605]

OCELOT_IMAGE_SIZE = (1024, 1024)

DEFAULT_TISSUE_MODEL_PATH = "outputs/models/best/20240313_002829_deeplabv3plus-tissue-branch_pretrained-1_lr-1e-04_dropout-0.1_backbone-resnet50_normalization-macenko_id-5_best.pth"

SEGFORMER_ARCHITECTURES = {
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

INDICES_TISSUE_MOST_CANCER = [51, 265, 101, 85, 102, 43, 218, 47, 360, 72, 28, 86, 234, 116, 133, 100, 38, 108, 4, 226, 267, 71, 114, 46, 87, 59, 277, 334, 127, 5, 191, 121, 115, 6, 209, 119, 206, 178, 10, 2, 120, 13, 27, 228, 259, 84, 8, 147, 40, 336, 263, 186, 225, 300, 188, 189, 55, 26, 113, 9, 200, 308, 205, 169, 56, 193, 270, 94, 49, 310, 236, 69, 35, 381, 199, 64, 66, 39, 107, 140, 77, 379, 266, 117, 68, 67, 192, 96, 260, 304, 208, 79, 204, 194, 63, 280, 380, 109, 269, 145]

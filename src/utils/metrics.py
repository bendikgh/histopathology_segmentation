import cv2
import json
import os
import sys
import torch

import albumentations as A
import numpy as np
import scipy.stats as st

sys.path.append(os.getcwd())

from src.utils.constants import (
    DEFAULT_TISSUE_MODEL_PATH,
    IDUN_OCELOT_DATA_PATH,
    DATASET_PARTITION_OFFSETS,
)
from src.utils.utils import get_ground_truth_points
from src.models import DeepLabV3plusModel, CustomSegformerModel

from skimage.feature import peak_local_max
from typing import List, Dict, Any
from tqdm import tqdm

from ocelot23algo.evaluation.eval import (
    _calc_scores,
    _check_validity,
    _convert_format,
    _preprocess_distance_and_confidence,
    CLS_IDX_TO_NAME,
    DISTANCE_CUTOFF,
)
from ocelot23algo.util import gcio

from ocelot23algo.user.inference import (
    SegformerTissueFromFile,
)


def calculate_f1_score(
    model, dataloader, device, micron_radius: float = 3.0, mpp: float = 0.2
):
    """These are the steps for calculating the F1 score, as given in the paper:

    - True positive: If a detected cell is within a valid distance (3 microns) of a
      target cell, then it is considered a TP
    - False positive: If a detected cell does not fulfill the requirements for a
      TP, then it is considered a FP
    - False Negative: If an annotated cell is not detected, then it is counted as a
      False Negative

    """
    model.eval()
    f1_scores = []
    pixel_radius = micron_radius / mpp
    dataset = dataloader.dataset
    batch_size = dataloader.batch_size

    for batch_idx, (image_batch, mask_batch) in enumerate(dataloader):
        image_batch = image_batch.to(device)
        with torch.no_grad():
            output_batch = model(image_batch)

        for idx in range(output_batch.shape[0]):
            output = output_batch[idx]

            image_no = batch_idx * batch_size + idx
            cell_annotation_list = dataset.get_cell_annotation_list(image_no)

            # Preparing output for peak_local_max
            softmaxed = torch.softmax(output, dim=0)
            cells, argmaxed = torch.max(input=softmaxed, dim=0)
            argmaxed = argmaxed.cpu().numpy()
            cells = cells.cpu().numpy()
            peak_points_pred = peak_local_max(
                cells,
                min_distance=20,
                labels=np.logical_or(argmaxed == 1, argmaxed == 2),
                threshold_abs=0.01,
            )

            # # For plotting the predictions and the ground truth on top
            # output = make_prediction_map(output_batch)[0]*255
            # # display results
            # fig, ax = plt.subplots(1, 1, figsize=(8, 3), sharex=True, sharey=True)
            # ax.imshow(output.permute(1, 2, 0))
            # ax.autoscale(False)
            # ax.plot(cell_annotation_list[:, 0], cell_annotation_list[:, 1], 'o', color='black', markersize=0.5)
            # ax.axis('off')
            # ax.set_title('Peak local max')
            # fig.tight_layout()
            # plt.show()

            TP = 0
            FP = 0
            for y, x in peak_points_pred:
                # We check a circle around the point to see if there is a cell in the mask
                # If there is, we count it as a TP
                cell_type = argmaxed[y, x]
                TP_old = TP  # To check if TP changes

                min_distance_squared: float = (pixel_radius + 1) ** 2
                min_distance_cell: Any = -1

                # Calculate distance vector to cell_annotation_list
                if cell_annotation_list.shape[0] > 0:
                    distance_squared: np.ndarray = (
                        x - cell_annotation_list[:, 0]
                    ) ** 2 + (y - cell_annotation_list[:, 1]) ** 2
                    min_distance_squared = np.min(distance_squared)
                    min_distance_cell = np.argmin(distance_squared)

                if min_distance_squared < pixel_radius**2:
                    if cell_annotation_list[min_distance_cell][2] != cell_type:
                        TP += 1
                    cell_annotation_list = np.delete(
                        cell_annotation_list, min_distance_cell, axis=0
                    )

                # If we did not find an annotated cell matching this one, then
                # we count it as a FP
                if TP_old == TP:
                    FP += 1

            FN = len(cell_annotation_list)
            dice_score = (2.0 * TP) / (2 * TP + FP + FN)
            f1_scores.append(dice_score)

    return torch.mean(torch.tensor(f1_scores))


def calculate_point_based_f1(
    ground_truth: List[Dict], predictions: List[Dict], num_images: int
) -> Dict[str, float]:
    _check_validity(predictions)
    _check_validity(ground_truth)

    pred_all, gt_all = _convert_format(predictions, ground_truth, num_images)

    # For each sample, get distance and confidence by comparing prediction and GT
    all_sample_result = _preprocess_distance_and_confidence(pred_all, gt_all)

    # Calculate scores of each class, then get final mF1 score
    scores = {}
    for cls_idx, cls_name in CLS_IDX_TO_NAME.items():
        precision, recall, f1 = _calc_scores(
            all_sample_result, cls_idx, DISTANCE_CUTOFF
        )
        scores[f"Pre/{cls_name}"] = precision
        scores[f"Rec/{cls_name}"] = recall
        scores[f"F1/{cls_name}"] = f1

    scores["mF1"] = sum(
        [scores[f"F1/{cls_name}"] for cls_name in CLS_IDX_TO_NAME.values()]
    ) / len(CLS_IDX_TO_NAME)

    return scores


def get_pointwise_predictions(
    data_dir: str,
    evaluation_model,
    partition: str = "val",
    tissue_file_folder: str = "images/val/tissue_macenko",
    transform=None,
) -> List:
    cell_file_path = os.path.join(data_dir, f"images/{partition}/cell_macenko/")
    tissue_file_path = os.path.join(data_dir, tissue_file_folder)

    # Reading metadata
    metadata_path = os.path.join(data_dir, "metadata.json")
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    metadata = list(metadata["sample_pairs"].values())[
        DATASET_PARTITION_OFFSETS[partition] :
    ]

    loader = gcio.CustomDataLoader(cell_file_path, tissue_file_path)

    predictions = []
    for cell_patch, tissue_patch, pair_id in tqdm(
        loader, desc="Processing samples: ", total=len(loader)
    ):
        cell_classification = evaluation_model(
            cell_patch, tissue_patch, pair_id, transform=transform
        )

        for x, y, class_id, prob in cell_classification:
            predictions.append(
                {
                    "name": f"image_{str(pair_id)}",
                    "point": [int(x), int(y), int(class_id)],
                    "probability": prob,
                }
            )
    return predictions


def predict_and_evaluate(
    evaluation_model,
    partition: str,
    tissue_file_folder: str,
    transform=None,
):
    predictions = get_pointwise_predictions(
        data_dir=IDUN_OCELOT_DATA_PATH,
        evaluation_model=evaluation_model,
        partition=partition,
        tissue_file_folder=tissue_file_folder,
        transform=transform,
    )

    gt_json = get_ground_truth_points(partition=partition)
    num_images = gt_json["num_images"]
    gt_points = gt_json["points"]

    scores = calculate_point_based_f1(
        ground_truth=gt_points,
        predictions=predictions,
        num_images=num_images,
    )
    return scores["mF1"]


def calculate_confidence_interval(results: list, metrics: list[str]) -> None:
    """
    Calculates the mean and 95% confidence interval of specified metrics for each experiment in the results.

    Each experiment in `results` is expected to be a dict containing:
    - "name": String name of the experiment.
    - "results": List of dicts, each containing measurement results for the specified metrics.

    Example of `results` format:
    [
        {
            "name": "Experiment Name",
            "results": [
                {"metric1": value1, "metric2": value2, ... },
                ...
            ],
        },
        ...
    ]

    Args:
    - results (list of dict): A list of experiments, where each experiment is a dict containing
                              the experiment name and its results.
    - metrics (list of str): The metrics for which to calculate the mean and confidence interval.

    Prints:
    - For each metric in each experiment, prints the mean and the half-width of the 95%
      confidence interval.
    """
    for experiment in results:
        result_list: list = experiment["results"]
        print(experiment["name"])

        for metric in metrics:
            scores = np.array([x[metric] for x in result_list])
            scores_ci = st.t.interval(
                0.95, len(scores) - 1, loc=scores.mean(), scale=st.sem(scores)
            )
            print(f"{metric} mean: {scores.mean()*100:.4f} (%)")
            print(
                f"{metric} 95% CI half-width: ±{(scores_ci[1] - scores_ci[0])/2*100:.4f} (%)"
            )
        print()


if __name__ == "__main__":
    partition = "test"
    cell_model_path = "outputs/models/20240323_184831/segformer-tissue-cell_pretrained-1_lr-1e-04_backbone-b3_normalization-macenko_pretrained_dataset-ade_resize-512_id-1_best.pth"
    # "outputs/models/20240323_184831/segformer-tissue-cell_pretrained-1_lr-1e-04_backbone-b3_normalization-macenko_pretrained_dataset-ade_resize-512_id-1_best.pth"

    # Getting the metadata
    metadata_path = os.path.join(IDUN_OCELOT_DATA_PATH, "metadata.json")
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    metadata = list(metadata["sample_pairs"].values())[
        DATASET_PARTITION_OFFSETS[partition] :
    ]

    tissue_file_folder = f"annotations/{partition}/predicted_cropped_tissue"
    resize = 512
    transform = A.Compose(
        [A.Resize(height=resize, width=resize, interpolation=cv2.INTER_NEAREST)],
        additional_targets={"tissue": "image"},
    )

    # Creating model
    cell_model = CustomSegformerModel(
        backbone_name="b3",
        num_classes=3,
        num_channels=6,
    )
    cell_model.load_state_dict(torch.load(cell_model_path))
    evaluation_model = SegformerTissueFromFile(
        metadata=metadata,
        cell_model=cell_model,
        tissue_model_path=None,
    )

    scores = predict_and_evaluate(
        evaluation_model=evaluation_model,
        partition=partition,
        tissue_file_folder=tissue_file_folder,
        transform=transform,
    )

    print(f"Scores: {scores}")

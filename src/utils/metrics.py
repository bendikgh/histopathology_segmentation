import torch
import numpy as np

from skimage.feature import peak_local_max
from torch.nn.functional import interpolate


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
            cells, argmaxed = torch.max(softmaxed, axis=0)
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

                min_distance_squared = (pixel_radius + 1) ** 2
                min_distance_cell = -1

                # Calculate distance vector to cell_annotation_list
                if cell_annotation_list.shape[0] > 0:
                    distance_squared = (x - cell_annotation_list[:, 0]) ** 2 + (
                        y - cell_annotation_list[:, 1]
                    ) ** 2
                    min_distance_squared, min_distance_cell = np.min(
                        distance_squared
                    ), np.argmin(distance_squared)

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


def calculate_f1_score_segformer(
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
            output_batch = model(image_batch).logits

        for idx in range(output_batch.shape[0]):
            output = output_batch[idx]
            output = interpolate(
                output.unsqueeze(0),
                size=mask_batch.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            output = output.squeeze(
                0
            )  # Remove the extra batch dimension that interpolate adds

            image_no = batch_idx * batch_size + idx
            cell_annotation_list = dataset.get_cell_annotation_list(image_no)

            # Preparing output for peak_local_max
            softmaxed = torch.softmax(output, dim=0)
            cells, argmaxed = torch.max(softmaxed, axis=0)
            argmaxed = argmaxed.cpu().numpy()
            cells = cells.cpu().numpy()
            peak_points_pred = peak_local_max(
                cells,
                min_distance=20,
                labels=np.logical_or(argmaxed == 1, argmaxed == 2),
                threshold_abs=0.01,
            )

            TP = 0
            FP = 0
            for y, x in peak_points_pred:
                # We check a circle around the point to see if there is a cell in the mask
                # If there is, we count it as a TP
                cell_type = argmaxed[y, x]
                TP_old = TP  # To check if TP changes

                min_distance_squared = (pixel_radius + 1) ** 2
                min_distance_cell = -1

                # Calculate distance vector to cell_annotation_list
                if cell_annotation_list.shape[0] > 0:
                    distance_squared = (x - cell_annotation_list[:, 0]) ** 2 + (
                        y - cell_annotation_list[:, 1]
                    ) ** 2
                    min_distance_squared, min_distance_cell = np.min(
                        distance_squared
                    ), np.argmin(distance_squared)

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
            f1_score = (2.0 * TP) / (2 * TP + FP + FN)
            f1_scores.append(f1_score)

    return torch.mean(torch.tensor(f1_scores))

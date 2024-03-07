import os
import json
import matplotlib.pyplot as plt
import numpy as np
import cv2

import streamlit as st

from ocelot23algo.evaluation.eval import _check_validity, _convert_format

plt.style.use('dark_background')


def create_scatter_plot(ax, x, y, cls):
    ax.scatter(x[cls == 1], y[cls == 1], c='red', label='Class 1')  # Points with class 1 in red
    ax.scatter(x[cls == 2], y[cls == 2], c='green', label='Class 2')  # Points with class 2 in green
    ax.legend()

def create_image_plot(ax, image):
    ax.imshow(image)
    ax.axis('off')

def create_grid_plot(images, data_list, grid_dims, titles=None):
    """
    Creates a grid of plots that can include both scatter plots and image plots.
    Allows specifying titles for each subplot.

    Parameters:
    - images: List of np.ndarray images to be plotted. Use None for scatter plots.
    - data_list: List of tuples, each containing (x, y, cls) for each dataset to plot. Use None for image plots.
    - grid_dims: Tuple of (rows, cols) specifying the grid layout dimensions.
    - titles: Optional list of titles for each subplot.
    """
    nrows, ncols = grid_dims
    fig, axs = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))

    # Ensure axs is a 2D array for consistent indexing
    axs = axs.reshape(nrows, ncols)

    for i, image in enumerate(images):
        row, col = divmod(i, ncols)
        create_image_plot(axs[row, col], image)
        if titles and i < len(titles):
            axs[row, col].set_title(titles[i])

    for i, (x, y, cls) in enumerate(data_list, start=len(images)):
        row, col = divmod(i, ncols)
        create_scatter_plot(axs[row, col], x, y, cls)
        if titles and i < len(titles):
            axs[row, col].set_title(titles[i])

    plt.tight_layout()
    return fig


def initialize():
    
    DATA_DIR = "/cluster/projects/vc/data/mic/open/OCELOT/ocelot_data"
    
    cell_path = os.path.join(DATA_DIR, f"images/{st.session_state.partition}/cell/")
    tissue_path = os.path.join(DATA_DIR, f"annotations/{st.session_state.partition}/pred_tissue/")

    
    tissue_ending = ".png"

    cell_patches = sorted(
        [os.path.join(cell_path, f) for f in os.listdir(cell_path) if ".jpg" in f],
        key=lambda x: int(x.split("/")[-1].split(".")[0]),
    )
    tissue_patches = sorted(
        [
            os.path.join(tissue_path, f)
            for f in os.listdir(tissue_path)
            if tissue_ending in f
        ],
        key=lambda x: int(x.split("/")[-1].split(".")[0]),
    )


    algorithm_output_path = (
        f"{os.getcwd()}/eval_outputs/cell_classification_{st.session_state.partition}.json"
    )
    with open(algorithm_output_path, "r") as f:
        pred_json = json.load(f)["points"]

    # Path where GT is stored
    gt_path = f"{os.getcwd()}/eval_outputs/cell_gt_{st.session_state.partition}.json"
    with open(gt_path, "r") as f:
        data = json.load(f)
        gt_json = data["points"]
        num_images = data["num_images"]

    # Check the validity (e.g. type) of algorithm output
    _check_validity(pred_json)
    _check_validity(gt_json)

    # Convert the format of GT and pred for easy score computation
    pred_all, gt_all = _convert_format(pred_json, gt_json, num_images)

    prediction_all = []
    label_points_all = []

    for i in range(len(pred_all)):
        # Get the correct patch
        prediction_all.append(np.array(pred_all[i]))
        label_points_all.append(np.array(gt_all[i]))

    st.session_state.prediction_all = prediction_all
    st.session_state.label_points_all = label_points_all

    st.session_state.cell_patches = cell_patches
    st.session_state.tissue_patches = tissue_patches

    st.session_state['initialized'] = True



def main():

    st.title("Evaluation plots")

    partitions = ["val", "test"]

    if "image_index" not in st.session_state:
        st.session_state.partition = "val"

    st.session_state.partition = st.selectbox("Select partition", partitions, on_change=initialize)   

    # Init neccessary structures only on start up
    if 'initialized' not in st.session_state:
        initialize()

    ## Get the cell predictions and cell labels


    number_of_images = len(st.session_state.prediction_all)
    
    if "image_index" not in st.session_state:
        st.session_state.image_index = 0
    
    chosen_index = st.selectbox(
        "Choose an image number:",
        list(range(number_of_images)),
        index=st.session_state.image_index
    )
    st.session_state.image_index = chosen_index

    def button_next_callback():
        st.session_state.image_index += 1

    def button_past_callback():
        st.session_state.image_index -= 1

    st.button("Next", on_click=button_next_callback)
    st.button("Past", on_click=button_past_callback)

    # Get the correct patch
    prediction_points = st.session_state.prediction_all[st.session_state.image_index]
    label_points = st.session_state.label_points_all[st.session_state.image_index]

    ## Get cell predictions and cell labels
    x_pred = prediction_points[:, 0]
    y_pred = prediction_points[:, 1]
    cls_pred = prediction_points[:, 2]

    x_label = label_points[:, 0]
    y_label = label_points[:, 1]
    cls_label = label_points[:, 2]

    data_for_plot = [(x_pred, y_pred, cls_pred), (x_label, y_label, cls_label)]

    ## Get the tissue predictions and cell input image

    cell_patch = cv2.imread(st.session_state.cell_patches[st.session_state.image_index])
    tissue_patch = cv2.imread(st.session_state.tissue_patches[st.session_state.image_index])*255

    cell_patch: np.ndarray = cv2.cvtColor(cell_patch, cv2.COLOR_BGR2RGB)
    tissue_patch: np.ndarray = cv2.cvtColor(tissue_patch, cv2.COLOR_BGR2RGB)

    images = [cell_patch, tissue_patch]

    fig = create_grid_plot(images, data_for_plot, (2, 2), titles=["Cell patch", "Tissue patch", "Cell-predictions", "Cell-labels"])
    st.write(fig)


if __name__ == "__main__":
    main()

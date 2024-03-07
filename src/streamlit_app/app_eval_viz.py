import os
import json
import matplotlib.pyplot as plt
import numpy as np
import pkgutil

import streamlit as st

from ocelot23algo.evaluation.eval import _check_validity, _convert_format


def create_scatter_plot(ax, x, y, cls):
    ax.scatter(x[cls == 1], y[cls == 1], c='red', label='Class 1')  # Points with class 1 in red
    ax.scatter(x[cls == 2], y[cls == 2], c='green', label='Class 2')  # Points with class 2 in green
    ax.legend()

def create_grid_plot(data_list, grid_dims, titles=None):
    """
    Creates a grid of scatter plots using create_scatter_plot for each subplot.
    Allows specifying titles for each subplot.

    Parameters:
    - data_list: List of tuples, each containing (x, y, cls) for each dataset to plot.
    - grid_dims: Tuple of (rows, cols) specifying the grid layout dimensions.
    - titles: Optional list of titles for each subplot.
    """
    nrows, ncols = grid_dims
    fig, axs = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    plt.style.use('dark_background')

    # Ensure axs is a 2D array for consistent indexing
    axs = axs.reshape(nrows, ncols)

    for i, (x, y, cls) in enumerate(data_list):
        row, col = divmod(i, ncols)
        create_scatter_plot(axs[row, col], x, y, cls)
        if titles and i < len(titles):
            axs[row, col].set_title(titles[i])

    plt.tight_layout()
    return fig


def initialize():    

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

    x_pred = prediction_points[:, 0]
    y_pred = prediction_points[:, 1]
    cls_pred = prediction_points[:, 2]

    x_label = label_points[:, 0]
    y_label = label_points[:, 1]
    cls_label = label_points[:, 2]

    plotting_list = [(x_pred, y_pred, cls_pred), (x_label, y_label, cls_label)]
    fig = create_grid_plot(plotting_list, (1, 2), titles=["Cell-predictions", "Cell-labels"])
    st.write(fig)

    ## Get the tissue predictions and tissue labels


if __name__ == "__main__":
    main()

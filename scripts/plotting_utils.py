#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""plotting_utils.py
This module contains all the plotting utilities for the repository.

TODO:
    - see how tracing to patient can be implemented based on indices of matrix
"""

__author__ = "Philip Hartout"
__email__ = "philip.hartout@protonmail.com"


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.figure_factory as ff
import dotenv

# Shape of a patch
SHAPE = (1, 30, 36, 30)
HOMOLOGY_DIMENSIONS = (0, 1, 2)
DOTENV_KEY2VAL = dotenv.dotenv_values()
N_JOBS = 6  # Set number of workers when parallel processing is useful.


def make_3d_scatterplot(point_cloud, title):
    """Shows 3D scatterplot based on a point cloud.

    Args:
        point_cloud (np.ndarray): array of shape (n_samples, 3) determining
        the 3D coordinates of each of the points.
        title (str): title of the plot

    Returns:
        None
    """
    df = pd.DataFrame(point_cloud).rename(columns={0: "x", 1: "y", 2: "z"})

    x, y, z = df["x"].values, df["y"].values, df["z"].values
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="markers",
                marker=dict(
                    size=5,
                    color=x,  # set color to an array/list of desired values
                    colorscale="Viridis",  # choose a colorscale
                    opacity=0.8,
                ),
            )
        ]
    )

    # tight layout
    fig.update_layout(margin=dict(l=0, r=0, b=0), title=title)
    fig.show()


def multi_slice_viewer(volume):
    """Multislide viewser as implemented from
    https://www.datacamp.com/community/tutorials/matplotlib-3d-volumetric-data
    """
    remove_keymap_conflicts({"j", "k"})
    fig, ax = plt.subplots()
    ax.volume = volume
    ax.index = volume.shape[0] // 2
    ax.imshow(volume[ax.index])
    fig.canvas.mpl_connect("key_press_event", process_key)


def process_key(event):
    """Multislide viewser as implemented from
        https://www.datacamp.com/community/tutorials/matplotlib-3d-volumetric
        -data
    """
    fig = event.canvas.figure
    ax = fig.axes[0]
    if event.key == "j":
        previous_slice(ax)
    elif event.key == "k":
        next_slice(ax)
    fig.canvas.draw()


def previous_slice(ax):
    """Multislide viewser as implemented from
        https://www.datacamp.com/community/tutorials/matplotlib-3d-volumetric
        -data
    """
    volume = ax.volume
    ax.index = (ax.index - 1) % volume.shape[0]  # wrap around using %
    ax.images[0].set_array(volume[ax.index])


def next_slice(ax):
    """Multislide viewser as implemented from
        https://www.datacamp.com/community/tutorials/matplotlib-3d-volumetric
        -data
    """
    volume = ax.volume
    ax.index = (ax.index + 1) % volume.shape[0]
    ax.images[0].set_array(volume[ax.index])


def remove_keymap_conflicts(new_keys_set):
    """Multislide viewser as implemented from
        https://www.datacamp.com/community/tutorials/matplotlib-3d-volumetric
        -data
    """
    for distanceprop in plt.rcParams:
        if prop.startswith("keymap."):
            keys = plt.rcParams[prop]
            remove_list = set(keys) & new_keys_set
            for key in remove_list:
                keys.remove(key)

def plot_distance_matrix(
    X_distance, title=None, file_prefix=None,
):
    """Plots and saves the distance matrix for each of the homology dimensions

    Args:
        X_distance (np.ndarray): array containing the distance data in each
            of the homology dimensions
        title (str): title of the plot
        file_prefix: prefix of the file, useful when calling the function on
            different data in a row.

    Returns:
        None
    """

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8, 16))
    plt.suptitle(title)
    for i, ax in enumerate(axes.flat):
        im = ax.imshow(X_distance[:, :, i], cmap="Blues")

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    plt.savefig(
        DOTENV_KEY2VAL["GEN_FIGURES_DIR"]
        + file_prefix
        + "_distance_matrix.png",
    )



def distplot(vectors, group_labels, title=None):
    """Get log transformed and non log-transformed distplot of the vectors
    entered.

    Args:
        vectors (list): list of vectors to plot in the same plot
        group_labels (list): list of labels to accompany the vectors. (must
        be the same length as vectors)
        title (str): title of the plot.

    Returns:
        None
    """
    """list_of_distance_vectors must be structured in such a way that each list
    of vectors can be plotted individually for each distance."""

    fig = ff.create_distplot(
        [np.log1p(vector) for vector in vectors], group_labels, bin_size=0.1
    ).update_layout(title=title + " log transformed")
    fig.show()
    fig.write_html(
        DOTENV_KEY2VAL["GEN_FIGURES_DIR"]
        + "distplot_"
        + title
        + "_distance_vectors_log_transformed.html"
    )
    fig = ff.create_distplot(
        [vector for vector in vectors], group_labels, bin_size=0.1
    ).update_layout(title=title + " not log transformed")
    fig.show()
    fig.write_html(
        DOTENV_KEY2VAL["GEN_FIGURES_DIR"]
        + "distplot_"
        + title
        + "_distance_vectors_not_transformed.html"
    )

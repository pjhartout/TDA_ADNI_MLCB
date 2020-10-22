#!/usr/bin/env python

"""utils.py

Utils module for the functions used in the exploring/ directory of this project.

Functions included:
    - data loader
    - make a 3D scatterplot
    - Vietoris-Rips filtration
    - cubical persistence
    - erosion filtration

TODO:
    - add docstrings
"""

__author__ = "Philip Hartout"
__email__ = "philip.hartout@protonmail.com"


import nibabel
import nibabel as nib  # Useful to load data

import nilearn
from nilearn import datasets
from nilearn.regions import RegionExtractor
from nilearn import plotting
from nilearn.image import index_img
from nilearn.plotting import find_xyz_cut_coords
from nilearn.input_data import NiftiMapsMasker
from nilearn.connectome import ConnectivityMeasure
from nilearn.regions import RegionExtractor
from nilearn import plotting
from nilearn.image import index_img
from nilearn import datasets
from nilearn import plotting

import matplotlib.pyplot as plt


from pathlib import Path
import dotenv


import numpy as np
import networkx as nx
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

import gtda
from gtda.images import ImageToPointCloud, ErosionFiltration
from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import (
    PersistenceEntropy,
    PersistenceLandscape,
    PersistenceImage,
)
from gtda.pipeline import Pipeline
from gtda.plotting import plot_diagram, plot_point_cloud, plot_heatmap
from gtda.homology import CubicalPersistence
from gtda.diagrams import (
    Scaler,
    Filtering,
    PersistenceEntropy,
    BettiCurve,
    PairwiseDistance,
)

import os
import tempfile
from urllib.request import urlretrieve
import zipfile

from sklearn.linear_model import LogisticRegression
from skimage import io

import glob
import json
import dotenv

# Shape of a patch
SHAPE = (1, 30, 36, 30)
DOTENV_KEY2VAL = dotenv.dotenv_values()


def make_3d_scatterplot(point_cloud, title):
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
    remove_keymap_conflicts({"j", "k"})
    fig, ax = plt.subplots()
    ax.volume = volume
    ax.index = volume.shape[0] // 2
    ax.imshow(volume[ax.index])
    fig.canvas.mpl_connect("key_press_event", process_key)


def process_key(event):
    fig = event.canvas.figure
    ax = fig.axes[0]
    if event.key == "j":
        previous_slice(ax)
    elif event.key == "k":
        next_slice(ax)
    fig.canvas.draw()


def previous_slice(ax):
    volume = ax.volume
    ax.index = (ax.index - 1) % volume.shape[0]  # wrap around using %
    ax.images[0].set_array(volume[ax.index])


def next_slice(ax):
    volume = ax.volume
    ax.index = (ax.index + 1) % volume.shape[0]
    ax.images[0].set_array(volume[ax.index])


def remove_keymap_conflicts(new_keys_set):
    for prop in plt.rcParams:
        if prop.startswith("keymap."):
            keys = plt.rcParams[prop]
            remove_list = set(keys) & new_keys_set
            for key in remove_list:
                keys.remove(key)


def prepare_image(directory, threshold):
    patch = np.load(directory)
    patch.shape
    binarized_patch_ad = np.where(patch > threshold, 1, 0)
    return binarized_patch_ad, patch


def prepare_point_cloud(patch_binarized):
    point_cloud_tranformer = gtda.images.ImageToPointCloud()
    patch_pc = point_cloud_tranformer.fit_transform(
        patch_binarized.reshape(SHAPE)
    )
    return patch_pc


def prepare_point_cloud(patch_binarized):
    point_cloud_tranformer = gtda.images.ImageToPointCloud()
    patch_pc = point_cloud_tranformer.fit_transform(
        patch_binarized.reshape(SHAPE)
    )
    return patch_pc


def vr_persistent_homology(patch_pc):
    homology_dimensions = (0, 1, 2)
    VR = VietorisRipsPersistence(
        metric="euclidean",
        max_edge_length=5,
        homology_dimensions=homology_dimensions,
        n_jobs=4,
    )
    diagrams_VietorisRips = VR.fit_transform(np.asarray(patch_pc))
    VR.plot(diagrams_VietorisRips).show()
    BC = BettiCurve()
    X_betti_curves = BC.fit_transform(diagrams_VietorisRips)
    BC.plot(X_betti_curves).show()
    return diagrams_VietorisRips


def cubical_persistence(
    images, title, plot_diagrams=False, betti_curves=False
):
    homology_dimensions = (0, 1, 2)
    cp = CubicalPersistence(
        homology_dimensions=homology_dimensions,
        coeff=2,
        periodic_dimensions=None,
        infinity_values=None,
        reduced_homology=True,
        n_jobs=4,
    )
    diagrams_cubical_persistence = cp.fit_transform(images)
    sc = Scaler(metric="bottleneck")
    scaled_diagrams_cubical_persistence = sc.fit_transform(
        diagrams_cubical_persistence
    )
    if plot_diagrams:
        fig = cp.plot(scaled_diagrams_cubical_persistence)
        fig.update_layout(title=title)
    if betti_curves:
        BC = BettiCurve()
        X_betti_curves = BC.fit_transform(scaled_diagrams_cubical_persistence)
        fig = BC.plot(X_betti_curves)
        fig.update_layout(title=title)
        fig.show()
    print(f"Computed CP for {title}")
    return scaled_diagrams_cubical_persistence


def erosion_filtration(img):
    ef = ErosionFiltration(n_iterations=None, n_jobs=4)
    diagrams_erosion = ef.fit_transformp(img)
    # BC = BettiCurve()
    # X_betti_curves = BC.fit_transform(diagrams_Erosion)
    # BC.plot(X_betti_curves).show()
    return diagrams_erosion


def persistence_landscape(persistence_diagram, title):
    pl = PersistenceLandscape(n_layers=1, n_bins=100, n_jobs=4)
    persistence_landscape = pl.fit_transform(persistence_diagram)
    fig = pl.plot(persistence_landscape)
    fig.update_layout(
        title=title,
        xaxis_title="Filtration parameter value",
        yaxis_title="Persistence",
    ).show()
    return persistence_landscape


def persistence_image(persistence_diagram, sigma, title):
    pl = PersistenceImage(
        sigma=sigma, n_bins=100, weight_function=None, n_jobs=4
    )
    persistence_image = pl.fit_transform(persistence_diagram)
    fig = pl.plot(persistence_image)
    fig.update_layout(title=title).show()
    return persistence_image


def get_arrays_from_dir(directory, filelist):
    filelist = [directory + file for file in filelist]
    images = []
    for arr in filelist:
        try:
            images.append(np.load(arr))
        except FileNotFoundError:
            print(
                f"Patient {arr} had no corresponding array available (no "
                f"MRI was performed at the time of diagnosis)"
            )
    return np.asarray(images)


def get_earliest_available_diagnosis(path_to_diags):
    """Gets diagnosis at first available timepoint"""
    cn_patients = []
    mci_patients = []
    ad_patients = []
    with open(path_to_diags) as f:
        diagnoses = json.load(f)

    for patient in list(diagnoses.keys()):
        if diagnoses[patient][list(diagnoses[patient].keys())[0]] == "CN":
            cn_patients.append(format_patient(patient, diagnoses))
        elif diagnoses[patient][list(diagnoses[patient].keys())[0]] == "MCI":
            mci_patients.append(format_patient(patient, diagnoses))
        elif diagnoses[patient][list(diagnoses[patient].keys())[0]] == "AD":
            ad_patients.append(format_patient(patient, diagnoses))
    return cn_patients, mci_patients, ad_patients


def format_patient(patient, diagnoses):
    patient = patient + "-" + list(diagnoses[patient].keys())[0] + "-MNI.npy"
    return patient.replace("-ses", "")


def compute_distance_matrix(
    diagrams,
    metric,
    metric_params,
    plot_distance_matrix=False,
    title=None,
    file_prefix=None,
):
    PD = PairwiseDistance(
        metric=metric, metric_params=metric_params, order=None, n_jobs=4
    )
    X_distance = PD.fit_transform(diagrams)
    if plot_distance_matrix:
        fig = make_subplots(
            rows=1,
            cols=3,
            subplot_titles=(
                "Distance between PDs for H_0",
                "Distance between PDs for H_1",
                "Distance between PDs for H_2",
            ),
        )
        fig.add_trace(
            go.Heatmap(
                z=X_distance[:, :, 0],
                colorbar_x=1 / 3 - 0.05,
                colorbar_thickness=10,
                colorscale="Greens",
            ),
            1,
            1,
        )
        fig.add_trace(
            go.Heatmap(
                z=X_distance[:, :, 1],
                colorbar_x=2 / 3 - 0.025,
                colorbar_thickness=10,
                colorscale="Blues",
            ),
            1,
            2,
        )
        fig.add_trace(
            go.Heatmap(
                z=X_distance[:, :, 2],
                colorbar_x=1,
                colorbar_thickness=10,
                colorscale="Oranges",
            ),
            1,
            3,
        )
        fig.update_layout(title=title)
        fig.write_html(
            DOTENV_KEY2VAL["GEN_FIGURES_DIR"]
            + file_prefix
            + "_distance_matrix.html",
        )
        fig.show()
    return X_distance


def evaluate_distance_functions(
    diagrams,
    list_of_distance_functions,
    plot_distance_matrix=False,
    file_prefix=None,
):
    distance_matrices = []
    if "landscape" in list_of_distance_functions:
        distance_matrix = compute_distance_matrix(
            diagrams,
            metric="landscape",
            metric_params={"p": 2, "n_layers": 5, "n_bins": 1000},
            plot_distance_matrix=True,
            title=file_prefix + "Landscape distance matrix between PDs",
            file_prefix=file_prefix + "_landscape_distance",
        )
        distance_matrices.append(distance_matrix)

    if "wasserstein" in list_of_distance_functions:
        distance_matrix = compute_distance_matrix(
            diagrams,
            metric="wasserstein",
            metric_params={"p": 2, "delta": 0.1},
            plot_distance_matrix=plot_distance_matrix,
            title=file_prefix + "Wasserstein distance matrix between PDs",
            file_prefix=file_prefix + "_wasserstein",
        )
        distance_matrices.append(distance_matrix)

    if "betti" in list_of_distance_functions:
        distance_matrix = compute_distance_matrix(
            diagrams,
            metric="betti",
            metric_params={"p": 2, "n_bins": 1000},
            plot_distance_matrix=plot_distance_matrix,
            title=file_prefix + "Betti distance matrix between PDs",
            file_prefix=file_prefix + "_betti",
        )
        distance_matrices.append(distance_matrix)

    if "silhouette" in list_of_distance_functions:
        distance_matrix = compute_distance_matrix(
            diagrams,
            metric="silhouette",
            metric_params={"p": 2, "power": 1, "n_bins": 1000},
            plot_distance_matrix=plot_distance_matrix,
            title=file_prefix + "Silhouette distance matrix between PDs",
            file_prefix=file_prefix + "_silhouette",
        )
        distance_matrices.append(distance_matrix)

    if "heat" in list_of_distance_functions:
        distance_matrix = compute_distance_matrix(
            diagrams,
            metric="heat",
            metric_params={"p": 2, "sigma": 0.1, "n_bins": 1000},
            plot_distance_matrix=plot_distance_matrix,
            title=file_prefix + "Heat distance matrix between PDs",
            file_prefix=file_prefix + "_heat",
        )
        distance_matrices.append(distance_matrix)

    if "persistence_image" in list_of_distance_functions:
        distance_matrix = compute_distance_matrix(
            diagrams,
            metric="persistence_image",
            metric_params={
                "p": 2,
                "sigma": 0.1,
                "n_bins": 1000,
                "weight_function": None,
            },
            plot_distance_matrix=plot_distance_matrix,
            title=file_prefix
            + "Persistence image distance matrix between PDs",
            file_prefix=file_prefix + "_persistence_image",
        )
        distance_matrices.append(distance_matrix)
    print(
        f"Computed distance matrices for {list_of_distance_functions} for"
        f" {file_prefix}"
    )
    return distance_matrices


def get_distance_vector_from_matrices(distance_matrices):
    distance_vectors = []
    for distance_matrix in distance_matrices:
        upper_triangular_distance_matrix = np.triu(distance_matrix)
        distance_vectors.append(
            upper_triangular_distance_matrix[
                upper_triangular_distance_matrix > 0
            ]
        )
    return distance_vectors


def compute_distplot(vectors, group_labels, title=None):
    """list_of_distance_vectors must be structured in such a way that each list
    of vectors can be plotted individually for each distance."""
    fig = ff.create_distplot(
        [np.log(vector) for vector in vectors], group_labels, bin_size=0.1
    ).update_layout(title=title)
    fig.show()
    fig.write_html(
        DOTENV_KEY2VAL["GEN_FIGURES_DIR"]
        + "distplot_"
        + title
        + "_distance_vectors.html"
    )

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

import nighres

import gtda
from gtda.images import ImageToPointCloud, ErosionFiltration
from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import PersistenceEntropy
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


SHAPE = (1, 30, 36, 30)


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
        n_jobs=8,
    )
    diagrams_VietorisRips = VR.fit_transform(np.asarray(patch_pc))
    VR.plot(diagrams_VietorisRips).show()
    BC = BettiCurve()
    X_betti_curves = BC.fit_transform(diagrams_VietorisRips)
    BC.plot(X_betti_curves).show()


def cubical_persistence(patch_cn):
    homology_dimensions = (0, 1, 2)
    cp = CubicalPersistence(
        homology_dimensions=homology_dimensions,
        coeff=2,
        periodic_dimensions=None,
        infinity_values=None,
        reduced_homology=True,
        n_jobs=None,
    )
    diagrams_CubicalPersistence = cp.fit_transform_plot(
        patch_cn.reshape(SHAPE)
    )
    BC = BettiCurve()
    X_betti_curves = BC.fit_transform(diagrams_CubicalPersistence)
    BC.plot(X_betti_curves).show()


def erosion_filtration(patch_cn):
    ef = ErosionFiltration(n_iterations=None, n_jobs=None)
    diagrams_Erosion = ef.fit_transform_plot(patch_cn)
    BC = BettiCurve()
    X_betti_curves = BC.fit_transform(diagrams_Erosion)
    BC.plot(X_betti_curves).show()

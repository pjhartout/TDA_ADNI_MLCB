#!/usr/bin/env python
# coding: utf-8
"""mapper_patches.py

This is an implementation of mapper for the patches.

TODO:
    - add docstrings
    - make sure PEP standards are met.
    - check if the parameters can be tweaked for higher dimensions of the PCA.
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
import seaborn as sns

import numpy as np
import networkx as nx
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go
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
from gtda.mapper import make_mapper_pipeline
from gtda.mapper import (
    CubicalCover,
    make_mapper_pipeline,
    Projection,
    plot_static_mapper_graph,
    plot_interactive_mapper_graph,
)


import os
import tempfile
from urllib.request import urlretrieve
import zipfile

from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

from skimage import io

import glob
import json

# Import utils library
import utils


DOTENV_KEY2VAL = dotenv.dotenv_values()


def main():
    directory = DOTENV_KEY2VAL["DATA_DIR"]
    image_dir = directory + "/patch_92/"
    diagnosis_json = "collected_diagnoses_complete.json"

    (
        cn_patients,
        mci_patients,
        ad_patients,
    ) = utils.get_earliest_available_diagnosis(directory + diagnosis_json)
    images_all = utils.get_arrays_from_dir(
        image_dir, cn_patients + mci_patients + ad_patients
    )

    cn_patient_list = [
        1 for patient in range(len(cn_patients) - 1)
    ]  # substracting one due to unfound MRI for one CN patient
    mci_patient_list = [2 for patient in range(len(mci_patients))]
    ad_patient_list = [3 for patient in range(len(ad_patients))]

    labels = np.array(
        cn_patient_list + mci_patient_list + ad_patient_list
    ).reshape(-1, 1)

    images = []
    for image in images_all:
        images.append(image.flatten())
    images_all = np.asarray(images)
    pca = PCA(n_components=440)
    pca.fit(images_all)

    fig, ax0 = plt.subplots(nrows=1, sharex=True, figsize=(6, 6))
    ax0.plot(
        np.arange(1, pca.n_components_ + 1),
        pca.explained_variance_ratio_,
        "+",
        linewidth=2,
    )
    ax0.set_ylabel("PCA explained variance ratio")
    ax0.legend(prop=dict(size=12))
    plt.savefig(DOTENV_KEY2VAL["GEN_FIGURES_DIR"] + "elbow_plot.png")

    n_components = 3
    pca = PCA(n_components=n_components)
    images_all_projected = pca.fit_transform(images_all)

    images_all_projected = np.append(images_all_projected, labels, axis=1)

    mapper_pipeline = make_mapper_pipeline(
        filter_func=Projection(columns=[index for index in range(2)]),
        cover=CubicalCover(n_intervals=10, overlap_frac=0.25),
        clusterer=DBSCAN(eps=0.5, min_samples=5),
        verbose=True,
        n_jobs=4,
    )

    fig = plot_static_mapper_graph(
        mapper_pipeline,
        images_all_projected,
        layout_dim=3,
        color_by_columns_dropdown=True,
    )

    fig.write_html(
        DOTENV_KEY2VAL["GEN_FIGURES_DIR"]
        + "mapper_2_dimensional_reduction.html"
    )

    images_all_projected = pd.DataFrame(images_all_projected)
    fig = px.scatter_3d(
        images_all_projected,
        x=0,
        y=1,
        z=2,
        color=3,
        title="3D scatterplot of the PCA of the image data",
    )
    fig.write_html(
        DOTENV_KEY2VAL["GEN_FIGURES_DIR"] + "scatterplot_pca_3d.html"
    )


if __name__ == "__main__":
    main()

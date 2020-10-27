#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""distance_between_pds.py

This script assesses the distance between persistence images for CN, MCI and
AD patients using various pd representations and distance functions.

TODO:
    - add docstrings
    - make sure PEP standards are met.
    - remove unused imports
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
import seaborn as sns

from pathlib import Path
import dotenv


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

import os
import tempfile
from urllib.request import urlretrieve
import zipfile

from sklearn.linear_model import LogisticRegression
from skimage import io

import glob
import json

# Import utils library
import utils


DOTENV_KEY2VAL = dotenv.dotenv_values()


def main():

    directory = DOTENV_KEY2VAL["DATA_DIR"]
    image_dir = directory + "/patch_91/"
    diagnosis_json = "collected_diagnoses_complete.json"

    if not os.path.exists(DOTENV_KEY2VAL["GEN_FIGURES_DIR"]):
        print(f"Creating {DOTENV_KEY2VAL['GEN_FIGURES_DIR']}")
        os.mkdir(DOTENV_KEY2VAL["GEN_FIGURES_DIR"])

    # First we get all the diagnoses
    (
        cn_patients,
        mci_patients,
        ad_patients,
    ) = utils.get_earliest_available_diagnosis(directory + diagnosis_json)

    images_cn = utils.get_arrays_from_dir(image_dir, cn_patients)
    images_mci = utils.get_arrays_from_dir(image_dir, mci_patients)
    images_ad = utils.get_arrays_from_dir(image_dir, ad_patients)

    # Then we compute the PD on each image.
    diagrams_cn = utils.cubical_persistence(
        images_cn,
        "Patch 92 of CN patients",
        plot_diagrams=False,
        betti_curves=False,
    )
    diagrams_mci = utils.cubical_persistence(
        images_mci,
        "Patch 92 of MCI Patient",
        plot_diagrams=False,
        betti_curves=False,
    )
    diagrams_ad = utils.cubical_persistence(
        images_ad,
        "Patch 92 of AD Patient",
        plot_diagrams=False,
        betti_curves=False,
    )
    distances_to_evaluate = [
        "wasserstein",
        "betti",
        "landscape",
        "silhouette",
        # "heat",
        # "persistence_image",
    ]
    # Then we compute the distance between the PDs.
    distance_matrices_cn = utils.evaluate_distance_functions(
        diagrams_cn,
        distances_to_evaluate,
        plot_distance_matrix=True,
        file_prefix="CN ",
    )
    distance_matrices_mci = utils.evaluate_distance_functions(
        diagrams_mci,
        distances_to_evaluate,
        plot_distance_matrix=True,
        file_prefix="MCI ",
    )
    distance_matrices_ad = utils.evaluate_distance_functions(
        diagrams_ad,
        distances_to_evaluate,
        plot_distance_matrix=True,
        file_prefix="AD ",
    )

    # We can plot the different distances between them.
    dist_vectors_cn = utils.get_distance_vectors_from_matrices(
        distance_matrices_cn
    )
    dist_vectors_mci = utils.get_distance_vectors_from_matrices(
        distance_matrices_mci
    )
    dist_vectors_ad = utils.get_distance_vectors_from_matrices(
        distance_matrices_ad
    )
    group_labels = ["CN", "MCI", "AD"]
    within_group_comparisons = pd.DataFrame()
    for index, vectors in enumerate(
        zip(dist_vectors_cn, dist_vectors_mci, dist_vectors_ad)
    ):
        dist_data = pd.DataFrame()
        for i, distance_vector in enumerate(vectors):
            # Now we loop through each of the diagnoses
            diag_data = (
                pd.DataFrame(distance_vector).T.melt()
                # .replace(to_replace=range(len(vectors)), value=group_labels)
            )
            diag_data["diagnosis"] = group_labels[i]
            dist_data = dist_data.append(diag_data)
        dist_data["distance"] = distances_to_evaluate[index]
        within_group_comparisons = within_group_comparisons.append(dist_data)
    within_group_comparisons = within_group_comparisons.rename(columns={
        "variable":"homology dimension"})
    within_group_comparisons.to_csv(
            DOTENV_KEY2VAL["GEN_DATA_DIR"]
            + "data_"
            + distances_to_evaluate[index]
            + ".png"
    )

    # We can also conduct the same analysis to uncover heterogeneity between
    # all images regardless of diagnosis
    images_all = utils.get_arrays_from_dir(
        image_dir, cn_patients + mci_patients + ad_patients
    )
    diagrams_all = utils.cubical_persistence(
        images_all,
        "Patch 92 of all patients",
        plot_diagrams=False,
        betti_curves=False,
    )
    distance_matrices_all = utils.evaluate_distance_functions(
        diagrams_all,
        distances_to_evaluate,
        plot_distance_matrix=True,
        file_prefix="All ",
    )
    dist_vectors_all = utils.get_distance_vectors_from_matrices(
        distance_matrices_all
    )
    for index, vectors in enumerate(dist_vectors_all):
        # Loop through distances
        all_groups = pd.DataFrame()
        for i, distance_vector in enumerate(vectors):
            dist_data = pd.DataFrame(vectors).T.melt()
            dist_data["distance"] = distances_to_evaluate[index]
            dist_data.append(dist_data)
    dist_data.to_csv(
            DOTENV_KEY2VAL["GEN_DATA_DIR"]
            + "data_"
            + distances_to_evaluate[index]
            + ".png"
    )


if __name__ == "__main__":
    main()

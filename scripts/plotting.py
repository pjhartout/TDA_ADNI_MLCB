#!/usr/bin/env python

"""plotting.py

This script aims to produce all the figures in this repository.

"""

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
import tqdm

import numpy as np
import networkx as nx
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go

import gtda
from gtda.images import ImageToPointCloud, ErosionFiltration
from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import PersistenceEntropy, PersistenceLandscape
from gtda.pipeline import Pipeline
from gtda.plotting import plot_diagram, plot_point_cloud, plot_heatmap
from gtda.homology import CubicalPersistence
from gtda.diagrams import (
    Scaler,
    Filtering,
    PersistenceEntropy,
    PersistenceImage,
    HeatKernel,
    Silhouette,
    BettiCurve,
    PairwiseDistance,
    PersistenceEntropy,
    PersistenceLandscape,
)

import os
import tempfile
from urllib.request import urlretrieve
import zipfile

from sklearn.linear_model import LogisticRegression
from skimage import io
from plotly.subplots import make_subplots
import sys
import utils


DOTENV_KEY2VAL = dotenv.dotenv_values()
N_JOBS = -1


def generate_sample_representations(paths_to_patches, labels):
    for i, path in enumerate(paths_to_patches):
        patch = np.load(path)

        cp = CubicalPersistence(
            homology_dimensions=(0, 1, 2),
            coeff=2,
            periodic_dimensions=None,
            infinity_values=None,
            reduced_homology=True,
            n_jobs=N_JOBS,
        )
        diagrams_cubical_persistence = cp.fit_transform(
            patch.reshape(1, 30, 36, 30)
        )
        cp.plot(diagrams_cubical_persistence).update_layout(
            title=f"Persistence diagram of a {labels[i]} patient"
        ).write_image(f"../figures/persistence_diagram_cn.png")

        representations = {
            PersistenceLandscape(n_layers=1, n_bins=100, n_jobs=N_JOBS),
            BettiCurve(),
            PersistenceImage(sigma=0.05, n_bins=100, n_jobs=-1),
            HeatKernel(sigma=0.1, n_bins=100, n_jobs=N_JOBS),
            Silhouette(power=1.0, n_bins=100, n_jobs=None),
        }
        representation_names = [
            "Persistence landscape",
            "Betti curve",
            "Persistence image",
            "Heat kernel",
            "Silhouette",
        ]
        for j, rep in tqdm.tqdm(enumerate(representations)):
            vectorial_representation = rep.fit_transform(
                diagrams_cubical_persistence
            )
            rep.plot(vectorial_representation).update_layout(
                title="Persistence landscape of a CN patient"
            ).write_image(
                f"../figures/{representation_names[j].replace(' ', '_')}_{labels[i]}.png"
            )
        print(f"Done plotting {labels[i]} sample")


def main():
    # First, we want to generate a typical representation of the data for each
    # diagnostic category

    generate_sample_representations(
        [
            "../data/patch_91/sub-ADNI002S0295-M00-MNI.npy",
            "../data/patch_91/sub-ADNI128S0225-M48-MNI.npy",
            "../data/patch_91/sub-ADNI128S0227-M48-MNI.npy",
        ],
        ["CN", "MCI", "AD"],
    )


if __name__ == "__main__":
    main()

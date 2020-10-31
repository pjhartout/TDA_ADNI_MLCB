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
from tqdm import tqdm

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
import zipfile
import seaborn as sns
import sys
import utils
import textwrap

DOTENV_KEY2VAL = dotenv.dotenv_values()
N_JOBS = 1
HOMOLOGY_DIMENSIONS = (0, 1, 2)
SAMPLE_REP = False
DISTPLOT_PD_DISTANCES = False
EVOLUTION_TIME_SERIES = True


def generate_sample_representations(paths_to_patches, labels):
    sample_rep_dir = DOTENV_KEY2VAL["GEN_FIGURES_DIR"] + "/sample_rep/"
    try:
        os.mkdir(sample_rep_dir)
    except OSError:
        print("Creation of the directory %s failed" % sample_rep_dir)
    else:
        print("Successfully created the directory %s " % sample_rep_dir)
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
        ).write_image(sample_rep_dir + f"persistence_diagram_cn.png")

        representation_names = [
            "Persistence landscape",
            "Betti curve",
            "Persistence image",
            "Heat kernel",
            "Silhouette",
        ]

        for j, rep in enumerate(representation_names):
            # Have not found a better way of doing this yet.
            if rep == "Persistence landscape":
                rep = PersistenceLandscape(
                    n_layers=1, n_bins=100, n_jobs=N_JOBS
                )
            elif rep == "Betti curve":
                rep = BettiCurve()
            elif rep == "Persistence image":
                rep = PersistenceImage(sigma=0.05, n_bins=100, n_jobs=N_JOBS)
            elif rep == "Heat kernel":
                rep = HeatKernel(sigma=0.1, n_bins=100, n_jobs=N_JOBS)
            elif rep == "Silhouette":
                rep = Silhouette(power=1.0, n_bins=100, n_jobs=N_JOBS)

            vectorial_representation = rep.fit_transform(
                diagrams_cubical_persistence
            )

            if representation_names[j] in ["Persistence image"]:
                for image in range(vectorial_representation.shape[1]):
                    plt.imshow(
                        vectorial_representation[0:, image, :, :].reshape(
                            100, 100
                        )
                    )
                    plt.title(
                        f"{representation_names[j]} representation of a "
                        f"{labels[i]} patient in h_{image}"
                    )
                    plt.savefig(
                        sample_rep_dir
                        + f"{representation_names[j].replace(' ', '_')}"
                          f"_{labels[i]}_h_{image}.png"
                    )
            else:
                rep.plot(vectorial_representation).update_layout(
                    title=f"{representation_names[j]} representation of a"
                          f" {labels[i]} patient"
                ).write_image(
                    sample_rep_dir
                    + f"{representation_names[j].replace(' ', '_')}"
                      f"_{labels[i]}.png"
                )
        print(f"Done plotting {labels[i]} sample")


def generate_displot_of_pd_distances(path_to_pd_pairwise_distances):
    """
    Generates distplot for each of the diagnostic categories. The data should
    have been generated by the distance_between_pds.py script.
    """
    data = pd.read_csv(path_to_pd_pairwise_distances).drop(
        columns="Unnamed: 0"
    )
    distance_pd_dir = DOTENV_KEY2VAL["GEN_FIGURES_DIR"] + "/pd_distances"

    try:
        os.mkdir(distance_pd_dir)
    except OSError:
        print("Creation of the directory %s failed" % distance_pd_dir)
    else:
        print("Successfully created the directory %s " % distance_pd_dir)

    for distance in data["distance"].unique():
        dist_data = data.loc[data["distance"] == distance]
        for homology_dimension in data["homology dimension"].unique():
            dist_data_hom = dist_data.loc[
                dist_data["homology dimension"] == homology_dimension
            ]
            fig = sns.displot(
                data=dist_data_hom, x="value", hue="diagnosis", kind="ecdf"
            )
            plt.subplots_adjust(top=0.85)
            plt.title(
                "\n".join(
                    textwrap.wrap(
                        f"Empirical cumulative distribution function of the"
                        f" {distance} distances among diagnostic categories "
                        f"for H_{homology_dimension}",
                        50,
                    )
                )
            )
            plt.savefig(
                f"../figures/pd_distances/ecdf_{distance}_h"
                f"_{homology_dimension}.png"
            )

            fig = sns.displot(
                data=dist_data_hom, x="value", hue="diagnosis", kind="kde"
            )
            plt.subplots_adjust(top=0.85)
            plt.title(
                "\n".join(
                    textwrap.wrap(
                        f"Kernel density estimation function of the "
                        f"{distance} distances among diagnostic categories for "
                        f"H_{homology_dimension}",
                        50,
                    )
                )
            )
            plt.savefig(
                f"../figures/pd_distances/kde_{distance}_h"
                f"_{homology_dimension}.png"
            )

            fig = sns.displot(
                data=dist_data_hom,
                x="value",
                hue="diagnosis",
                multiple="stack",
            )
            plt.subplots_adjust(top=0.85)
            plt.title(
                "\n".join(
                    textwrap.wrap(
                        f"Stacked histogram of the {distance} distances "
                        f"among diagnostic categories for H"
                        f"_{homology_dimension}",
                        50,
                    )
                )
            )
            plt.savefig(
                f"../figures/pd_distances/stacked_{distance}_h"
                f"_{homology_dimension}.png"
            )
            plt.close("all")


def plot_evolution_time_series(path_to_distance_matrices):
    """
    This takes the distance matrix, extract the first column and plots it as
    a time series for each patients for which this has been computed in
    scripts/patient_evolution.py
    """
    metrics = [
        "wasserstein",
        "betti",
        "landscape",
        "silhouette",
        "heat",
        "persistence_image",
    ]
    for metric in metrics:
        # patients here means the collection of pairwise distances for each
        # patient

        patients = pd.DataFrame()
        for root, dirs, files in os.walk(path_to_distance_matrices):
            # loop through files to find relevant ones for metric
            for file in files:
                if metric in file:
                    X_distance = np.load(path_to_distance_matrices + file)
                    for h_dim in HOMOLOGY_DIMENSIONS:
                        sample = pd.DataFrame(X_distance[0, :, h_dim]).melt()
                        sample["metric"] = metric
                        sample["homology_dimension"] = h_dim
                        sample["file"] = "_".join(file.split("_", 6)[4:])
                        sample = sample.drop(columns=["variable"])
                        patients = patients.append(sample)
        patients = patients.reset_index()
        for h_dim in patients["homology_dimension"].unique():
            hdim_patients = patients.loc[
                patients["homology_dimension"] == h_dim
            ]
            sns.lineplot(x="index", y="value", hue="file", data=hdim_patients)
            plt.title(
                f"{metric.replace('_', ' ').capitalize()} distance from baseline for "
                f"patients over time in {h_dim}."
            )
            plt.savefig(
                DOTENV_KEY2VAL["GEN_FIGURES_DIR"] + "/temporal_evolution/" + metric + str(h_dim) + ".png"
            )
            plt.close("all")


def main():
    # First, we want to generate a typical representation of the data for each
    # diagnostic category
    utils.make_dir(DOTENV_KEY2VAL["GEN_FIGURES_DIR"])

    if SAMPLE_REP:
        generate_sample_representations(
            [
                "../data/patch_91/sub-ADNI002S0295-M00-MNI.npy",
                "../data/patch_91/sub-ADNI128S0225-M48-MNI.npy",
                "../data/patch_91/sub-ADNI128S0227-M48-MNI.npy",
            ],
            ["CN", "MCI", "AD"],
        )
    if DISTPLOT_PD_DISTANCES:
        generate_displot_of_pd_distances(
            "../generated_data/data_patients_within_group.csv"
        )

    if EVOLUTION_TIME_SERIES:
        plot_evolution_time_series(
            DOTENV_KEY2VAL["GEN_DATA_DIR"] + "/temporal_evolution/"
        )


if __name__ == "__main__":
    main()

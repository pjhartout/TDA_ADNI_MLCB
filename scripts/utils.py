#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""utils.py

Utils module for the functions used in the exploring/ directory of this project.

TODO:
    - see how tracing to patient can be implemented based on indices of matrix
"""

__author__ = "Philip Hartout"
__email__ = "philip.hartout@protonmail.com"


import matplotlib.pyplot as plt
from textwrap import wrap

import numpy as np
import pandas as pd
import seaborn as sns

import plotly.graph_objects as go
import plotly.figure_factory as ff

import gtda
from gtda.images import ErosionFiltration
from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import (
    PersistenceLandscape,
    PersistenceImage,
)
from gtda.homology import CubicalPersistence
from gtda.diagrams import (
    Scaler,
    BettiCurve,
    PairwiseDistance,
)

import json
import dotenv
import os

# Shape of a patch
SHAPE = (1, 30, 36, 30)
HOMOLOGY_DIMENSIONS = (0, 1, 2)
DOTENV_KEY2VAL = dotenv.dotenv_values()
N_JOBS = 6  # Set number of workers when parallel processing is useful.

def get_arrays_from_dir(directory, filelist):
    """This function gets the appropriate images given a list of files in one
    array.

    Args:
        directory (str): directory where filelist is located
        filelist (list): list of files in directory to be loaded

    Returns:
        np.ndarray: numpy array of snape (n_sampels, n_length, n_width,
        homology_dim) of images.
    """

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
    return np.stack(images)



def prepare_image(directory, threshold):
    patch = np.load(directory)
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
        n_jobs=N_JOBS,
    )
    diagrams_VietorisRips = VR.fit_transform(np.asarray(patch_pc))
    VR.plot(diagrams_VietorisRips).show()
    BC = BettiCurve()
    X_betti_curves = BC.fit_transform(diagrams_VietorisRips)
    BC.plot(X_betti_curves).show()
    return diagrams_VietorisRips


def cubical_persistence(
    images, title, plot_diagrams=False, betti_curves=False, scaled=False
):
    homology_dimensions = (0, 1, 2)
    cp = CubicalPersistence(
        homology_dimensions=homology_dimensions,
        coeff=2,
        periodic_dimensions=None,
        infinity_values=None,
        reduced_homology=True,
        n_jobs=N_JOBS,
    )
    diagrams_cubical_persistence = cp.fit_transform(images)
    if scaled:
        sc = Scaler(metric="bottleneck")
        scaled_diagrams_cubical_persistence = sc.fit_transform(
            diagrams_cubical_persistence
        )
    else:
        scaled_diagrams_cubical_persistence = diagrams_cubical_persistence

    if plot_diagrams:
        fig = cp.plot(scaled_diagrams_cubical_persistence)
        fig.update_layout(title=title)
        fig.show()
    if betti_curves:
        BC = BettiCurve()
        X_betti_curves = BC.fit_transform(scaled_diagrams_cubical_persistence)
        fig = BC.plot(X_betti_curves)
        fig.update_layout(title=title)
        fig.show()
    print(f"Computed CP for {title}")
    return scaled_diagrams_cubical_persistence


def erosion_filtration(img):
    ef = ErosionFiltration(n_iterations=None, n_jobs=N_JOBS)
    diagrams_erosion = ef.fit_transformp(img)
    # BC = BettiCurve()
    # X_betti_curves = BC.fit_transform(diagrams_Erosion)
    # BC.plot(X_betti_curves).show()
    return diagrams_erosion


def persistence_landscape(persistence_diagram, title):
    pl = PersistenceLandscape(n_layers=1, n_bins=100, n_jobs=N_JOBS)
    persistence_landscape = pl.fit_transform(persistence_diagram)
    fig = pl.plot(persistence_landscape)
    fig.update_layout(
        title=title,
        xaxis_title="Filtration parameter value",
        yaxis_title="Persistence",
    ).show()
    return persistence_landscape


def persistence_image(persistence_diagram, sigma, title):
    pi = PersistenceImage(
        sigma=sigma, n_bins=100, weight_function=None, n_jobs=N_JOBS
    )
    persistence_image = pi.fit_transform(persistence_diagram)
    fig = pi.plot(persistence_image)
    fig.update_layout(title=title).show()
    return persistence_image


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

def get_all_available_diagnoses(path_to_diags):
    """Gets diagnosis at all available timepoint"""
    cn_images = []
    mci_images = []
    ad_images = []
    with open(path_to_diags) as f:
        diagnoses = json.load(f)
    counts = 0
    for patient in list(diagnoses.keys()):
        for timepoint in list(diagnoses[patient].keys()):
            if diagnoses[patient][timepoint] == "CN":
                counts = counts + 1
                cn_images.append(format_patient_timepoint(patient, timepoint))
            elif diagnoses[patient][timepoint] == "MCI":
                mci_images.append(format_patient_timepoint(patient, timepoint))
                counts = counts + 1
            elif diagnoses[patient][timepoint] == "AD":
                ad_images.append(format_patient_timepoint(patient, timepoint))
                counts = counts + 1
            else:
                print(f"Unknown diagnosis ({diagnoses[patient][timepoint]}) "
                      f"specified for patient {patient}")
    return cn_images, mci_images, ad_images


def format_patient(patient, diagnoses):
    patient = patient + "-" + list(diagnoses[patient].keys())[0] + "-MNI.npy"
    return patient.replace("-ses", "")


def format_patient_timepoint(patient, timepoint):
    patient = patient + "-" + timepoint + "-MNI.npy"
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
        metric=metric, metric_params=metric_params, order=None, n_jobs=N_JOBS
    )
    X_distance = PD.fit_transform(diagrams)
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


def get_distance_vectors_from_matrices(distance_matrices):
    """returns the upper triangular matrices of the distance matrices in each
    homology dimension."""
    distance_vectors = []
    for distance_matrix in distance_matrices:
        vector_for_distance_metric = []
        for i in HOMOLOGY_DIMENSIONS:
            vector_for_distance_metric.append(
                distance_matrix[:, :, i][
                    np.triu_indices(len(distance_matrix[:, :, i]), k=1)
                ]
            )
        distance_vectors.append(vector_for_distance_metric)
    return distance_vectors


def plot_density_plots(metric):
    data = (
        pd.read_csv(f"../generated_data/data_{metric}.csv")
        .drop(columns="Unnamed: 0")
        .dropna()
        .rename(columns={"value": "distance"})
    )
    sns.displot(
        data=data,
        x="distance",
        hue="variable",
        multiple="stack",
        height=6,
        aspect=0.7,
    )
    plt.subplots_adjust(top=0.85)
    plt.title(
        "\n".join(
            wrap(
                f"Distribution of the {metric} distances "
                f"among diagnostic categories",
                50,
            )
        )
    )
    plt.savefig(DOTENV_KEY2VAL["GEN_FIGURES_DIR"] + metric + "_histogram.png",)


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

def make_dir(directory):
    """Makes directory and handles errors
    """
    try:
        os.mkdir(directory)
    except OSError:
        print("Creation of the directory %s failed" % directory)
    else:
        print("Successfully created the directory %s " % directory)

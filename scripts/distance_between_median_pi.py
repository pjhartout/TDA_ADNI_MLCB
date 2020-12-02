#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""distance_between_median_pi.py

This script computes the average distance between the persistence images.

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
    Scaler,
    BettiCurve,
    PairwiseDistance,
)
from gtda.homology import CubicalPersistence

import json
import dotenv
import os
import utils
from scipy.spatial import distance

DOTENV_KEY2VAL = dotenv.dotenv_values()
N_JOBS = -1

def compute_average_pl(patient_list, image_dir):
    file_names = []
    images = []
    for patient in patient_list:
        try:
            images.append(np.load(image_dir + patient))
            file_names.append(patient)
        except FileNotFoundError:
            print(f"{patient} not found, skipping.")
    images = np.stack(images, axis=0)
    pds = utils.cubical_persistence(
        images, None, plot_diagrams=False, betti_curves=False, scaled=False
    )
    pi = PersistenceImage(
        sigma=0.001, n_bins=100, weight_function=None, n_jobs=N_JOBS
    )
    pis = pi.fit_transform(pds)
    return pis, np.median(pis, axis=0), file_names


def compute_pi_distance(
    pis, average_pi, p, dest_dir, file_names, patient_category
):
    diffs = []
    for pl in range(pis.shape[0]):
        # Loop through each patient
        patient_dist_from_avg = []
        for h_dim in range(pis.shape[1]):
            # Loop through each dimension
            patient_dist_from_avg.append(
                distance.minkowski(
                    pis[pl, h_dim, :, :].flatten(),
                    average_pi[h_dim, :, :].flatten(),
                    p,
                )
            )
        diffs.append(patient_dist_from_avg)
    diffs = np.array(diffs)
    # with open(dest_dir + ".npy", "wb") as f:
    #     np.save(f, diffs)
    diffs = pd.DataFrame(diffs, columns=["H_0", "H_1", "H_2"])
    file_names = np.array(file_names)
    outliers = pd.DataFrame()
    # Select patients who are outliers
    for col in diffs.columns:
        outliers[col] = list(
            file_names[
                np.array(diffs.nlargest(140, columns=col).index)
            ]
        )
    outliers.to_csv(dest_dir + f"outliers_{patient_category}.csv", index=False)
    diffs.index = file_names
    diffs.to_csv(dest_dir + f"distance_from_median_pi_{patient_category}.csv",
                 index=True)


def compute_category(patient_list, patient_category, image_dir, dest_dir):
    # Compute average
    pis, average_pi, file_names = compute_average_pl(patient_list, image_dir)
    with open(dest_dir + f"median_pi_{patient_category}.npy", "wb") as f:
         np.save(f, average_pi)

    # Compute distance to average
    compute_pi_distance(
        pis,
        average_pi,
        1,
        dest_dir,
        file_names,
        patient_category,
    )


def main():
    directory = DOTENV_KEY2VAL["DATA_DIR"]
    image_dir = directory + "/patch_91/"
    diagnosis_json = (
        DOTENV_KEY2VAL["DATA_DIR"] + "collected_diagnoses_complete.json"
    )
    gen_data_dir = DOTENV_KEY2VAL["GEN_DATA_DIR"] + \
                   "/distance_from_median_image/"
    utils.make_dir(gen_data_dir)
    (
        cn_patients,
        mci_patients,
        ad_patients,
    ) = utils.get_all_available_diagnoses(diagnosis_json)
    compute_category(cn_patients, "CN", image_dir, gen_data_dir)
    compute_category(mci_patients, "MCI", image_dir, gen_data_dir)
    compute_category(ad_patients, "AD", image_dir, gen_data_dir)


if __name__ == "__main__":
    main()

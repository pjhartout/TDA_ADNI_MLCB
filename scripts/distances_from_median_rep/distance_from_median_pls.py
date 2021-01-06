#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""distance_from_median_pls.py

This script investigates the L1 distance between each of the images to the
median vectorial representation of a persistence image within a diagnostic
category.

TODO:

"""

__author__ = "Philip Hartout"
__email__ = "philip.hartout@protonmail.com"

import numpy as np
import pandas as pd
import os
import dotenv
import json
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial import distance
from gtda.homology import CubicalPersistence
from gtda.diagrams import (
    PersistenceLandscape,
    PersistenceImage,
    PairwiseDistance,
)

DOTENV_KEY2VAL = dotenv.dotenv_values()
N_JOBS = -1


def format_patient(patient, diagnoses):
    patient = patient + "-" + list(diagnoses[patient].keys())[0] + "-MNI.npy"
    return patient.replace("-ses", "")


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


def make_dir(directory):
    """Makes directory and handles errors"""
    try:
        os.mkdir(directory)
    except OSError:
        print("Creation of the directory %s failed" % directory)
    else:
        print("Successfully created the directory %s " % directory)


def persistence_landscape(persistence_diagram):
    pl = PersistenceLandscape(n_layers=1, n_bins=100, n_jobs=N_JOBS)
    persistence_landscape = pl.fit_transform(persistence_diagram)
    return persistence_landscape


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

    return diagrams_cubical_persistence


def compute_median_pl(patient_list, image_dir):
    file_names = []
    images = []
    for patient in patient_list:
        try:
            images.append(np.load(image_dir + patient))
            file_names.append(patient)
        except FileNotFoundError:
            print(f"{patient} not found, skipping.")
    images = np.stack(images, axis=0)
    pls = cubical_persistence(
        images, None, plot_diagrams=False, betti_curves=False, scaled=False
    )
    pls = persistence_landscape(pls)
    return pls, np.median(pls, axis=0), file_names


def compute_pl_distance(
    pls, median_pl, p, dest_dir, file_names, patient_category
):
    diffs = []
    for pl in range(pls.shape[0]):
        # Loop through each patient
        patient_dist_from_avg = []
        for h_dim in range(pls.shape[1]):
            # Loop through each dimension
            patient_dist_from_avg.append(
                distance.minkowski(
                    pls[pl, h_dim, :].flatten(),
                    median_pl[h_dim, :].flatten(),
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
            file_names[np.array(diffs.nlargest(140, columns=col).index)]
        )
    outliers.to_csv(dest_dir + f"outliers_{patient_category}.csv", index=False)
    diffs.index = file_names
    diffs.to_csv(
        dest_dir + f"distance_from_median_pl_{patient_category}.csv",
        index=True,
    )


def compute_category(patient_list, patient_category, image_dir, dest_dir):
    # Compute median
    pls, median_pl, file_names = compute_median_pl(patient_list, image_dir)
    with open(dest_dir + f"median_pl_{patient_category}.npy", "wb") as f:
        np.save(f, median_pl)

    # Compute distance to median
    compute_pl_distance(
        pls,
        median_pl,
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
    gen_data_dir = DOTENV_KEY2VAL["GEN_DATA_DIR"] + "/distance_from_median_pl/"
    make_dir(gen_data_dir)
    (
        cn_patients,
        mci_patients,
        ad_patients,
    ) = get_earliest_available_diagnosis(diagnosis_json)
    compute_category(cn_patients, "CN", image_dir, gen_data_dir)
    compute_category(mci_patients, "MCI", image_dir, gen_data_dir)
    compute_category(ad_patients, "AD", image_dir, gen_data_dir)


if __name__ == "__main__":
    main()

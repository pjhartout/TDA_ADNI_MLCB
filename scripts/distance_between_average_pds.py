#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""distance_between_average_pds.py
This script investigates the "distance" between each of the images to the
average vectorial representation of a persistence image within a diagnostic
category.

TODO:

"""

__author__ = "Philip Hartout"
__email__ = "philip.hartout@protonmail.com"

import numpy as np
import pandas as pd
import os
import dotenv
import utils
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial import distance
from gtda.diagrams import PairwiseDistance

DOTENV_KEY2VAL = dotenv.dotenv_values()


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
    pis = utils.cubical_persistence(
        images, None, plot_diagrams=False, betti_curves=False, scaled=False
    )
    pls = utils.persistence_landscape(pis)
    return pls, np.mean(pls, axis=0), file_names


def compute_pl_distance(
    pls, average_pl, p, dest_dir, file_names, patient_category
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
                    average_pl[h_dim, :].flatten(),
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
                np.array(diffs.nlargest(378, columns=col).index)
            ]
        )
    outliers.to_csv(dest_dir + f"outliers_{patient_category}.csv", index=False)
    diffs.to_csv(dest_dir + f"distance_from_average_pl_{patient_category}.csv",
                 index=False)


def compute_category(patient_list, patient_category, image_dir, dest_dir):
    # Compute average
    pls, average_pl, file_names = compute_average_pl(patient_list, image_dir)
    with open(dest_dir + f"average_pl_{patient_category}.npy", "wb") as f:
         np.save(f, average_pl)

    # Compute distance to average
    compute_pl_distance(
        pls,
        average_pl,
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
    gen_data_dir = DOTENV_KEY2VAL["GEN_DATA_DIR"] + "/distance_from_average/"
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

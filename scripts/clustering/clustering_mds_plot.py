#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""filename.py

This uses the features from the persistence images to create an MDS plot.

"""

import os
import json
import dotenv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.manifold import MDS

DOTENV_KEY2VAL = dotenv.dotenv_values()
N_JOBS = -1
rcParams["figure.figsize"] = 20, 20


def format_patient_timepoint(patient, timepoint):
    patient = patient + "-" + timepoint + "-MNI.npy"
    return patient.replace("-ses", "")


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


def get_all_available_diagnoses(path_to_diags):
    """Gets diagnosis at all available timepoint"""
    cn_images = []
    mci_images = []
    ad_images = []
    with open(path_to_diags) as f:
        diagnoses = json.load(f)
    counts = 0
    unknown = list()
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
                print(
                    f"Unknown diagnosis ({diagnoses[patient][timepoint]}) "
                    f"specified for patient {patient}"
                )

                unknown.append(format_patient_timepoint(patient, timepoint))
    print(counts)
    return cn_images, mci_images, ad_images, unknown


def main():
    diagnosis_json = (
        DOTENV_KEY2VAL["DATA_DIR"] + "collected_diagnoses_complete.json"
    )
    pi_dir = DOTENV_KEY2VAL["DATA_DIR"] + "/patch_91_persistence_images/"
    (
        cn_patients,
        mci_patients,
        ad_patients,
        unknown,
    ) = get_all_available_diagnoses(diagnosis_json)

    cn_images = get_arrays_from_dir(pi_dir, cn_patients)
    mci_images = get_arrays_from_dir(pi_dir, mci_patients)
    ad_images = get_arrays_from_dir(pi_dir, ad_patients)

    all_images = np.vstack([cn_images, mci_images, ad_images])
    print("Finished preprocessing")
    print(f"There are {len(cn_patients)} CN patients")
    print(f"There are {len(mci_patients)} MCI patients")
    print(f"There are {len(ad_patients)} AD patients")

    print("Fitting MDS embedding")
    embedding = MDS(n_components=2, metric=True, n_jobs=N_JOBS)
    embedding.fit_transform(all_images.reshape(-1, 30000))

    # Play with index.
    cn_idx = len(cn_patients) + 1  # (10) -> 0:11
    mci_idx = cn_idx + len(mci_patients) + 1  # (10) -> 11:21

    print("Plotting")
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.scatter(
        all_images[:cn_idx, 0],
        all_images[:cn_idx, 1],
        c="b",
        marker="s",
        label="CN",
    )
    ax1.scatter(
        all_images[cn_idx:mci_idx, 0],
        all_images[cn_idx:mci_idx, 1],
        c="r",
        marker="o",
        label="MCI",
    )
    ax1.scatter(
        all_images[mci_idx:, 0],
        all_images[mci_idx:, 1],
        c="g",
        marker="+",
        label="AD",
    )
    plt.legend(loc="upper right")
    plt.savefig(DOTENV_KEY2VAL["GEN_FIGURES_DIR"] + "mds_plot.png")


if __name__ == "__main__":
    main()

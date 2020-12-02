#!/usr/bin/env python

import utils
import matplotlib.pyplot as plt
import dotenv

import numpy as np
import pandas as pd

from gtda.homology import CubicalPersistence
from gtda.diagrams import (
    PersistenceImage,
    HeatKernel,
    Silhouette,
    BettiCurve,
    PersistenceLandscape,
)
from scipy.stats import kurtosis, skew
import os
import seaborn as sns
import textwrap
import json
import utils
from plotly.subplots import make_subplots
from plotly.graph_objects import Figure, Scatter
from scipy.stats import mannwhitneyu
import itertools
from scipy.spatial import distance

# Global vars
DOTENV_KEY2VAL = dotenv.dotenv_values()
HOMOLOGY_DIMENSIONS = (0, 1, 2)
N_JOBS = -1

directory = DOTENV_KEY2VAL["DATA_DIR"]
image_dir = directory + "/patch_91/"


def main():

    # Import average images
    median_cn = np.load(
        "../generated_data/distance_from_median_image/median_pi_CN.npy"
    )
    median_mci = np.load(
        "../generated_data/distance_from_median_image/median_pi_MCI.npy"
    )
    median_ad = np.load(
        "../generated_data/distance_from_median_image/median_pi_AD.npy"
    )
    # median_mci = np.load("../generated_data/distance_from_average_image/average_pi_MCI.npy")
    # median_cn = np.load("../generated_data/distance_from_average_image/average_pi_CN.npy")

    diagnosis_json = (
        DOTENV_KEY2VAL["DATA_DIR"] + "collected_diagnoses_complete.json"
    )
    (
        cn_patients,
        mci_patients,
        ad_patients,
        unknown,
    ) = utils.get_all_available_diagnoses(diagnosis_json)

    patients = list(itertools.chain(*[cn_patients, mci_patients, ad_patients]))

    file_names = list()
    images = list()
    unfound = list()

    for patient in patients:
        try:
            images.append(np.load(image_dir + patient))
            file_names.append(patient)
        except FileNotFoundError:
            unfound.append(patient)
            # if patient in cn_patients:
            #     cn_patients.remove(patient)
            # elif patient in mci_patients:
            #     mci_patients.remove(patient)
            # elif patient in ad_patients:
            #     ad_patients.remove(patient)
            print(f"{patient} not found, skipping.")

    images = np.stack(images)

    cp = CubicalPersistence(
        homology_dimensions=HOMOLOGY_DIMENSIONS,
        coeff=2,
        periodic_dimensions=None,
        infinity_values=None,
        reduced_homology=True,
        n_jobs=N_JOBS,
    )
    diagrams_cubical_persistence = cp.fit_transform(images)
    pi = PersistenceImage()
    pis = pi.fit_transform(diagrams_cubical_persistence)

    distances = list()
    for pi in range(pis.shape[0]):
        distances_patient = list()
        for diagnosis in [median_cn, median_mci, median_ad]:
            distances_diagnosis = list()
            for h_dim in HOMOLOGY_DIMENSIONS:
                distances_diagnosis.append(
                    distance.minkowski(
                        pis[pi, h_dim, :].flatten(),
                        diagnosis[h_dim, :].flatten(),
                        1,
                    )
                )
            distances_patient.append(distances_diagnosis)
        distances.append(itertools.chain(*distances_patient))
    distances = pd.DataFrame(
        distances,
        columns=[
            "CN_H_0",
            "CN_H_1",
            "CN_H_2",
            "MCI_H_0",
            "MCI_H_1",
            "MCI_H_2",
            "AD_H_0",
            "AD_H_1",
            "AD_H_2",
        ],
        index=[
            patient
            for patient in patients
            if patient not in unknown
            if patient not in unfound
        ],
    )
    pd.DataFrame(cn_patients).to_csv("CN_patients_processed.csv")
    pd.DataFrame(mci_patients).to_csv(
        "MCI_patients_processed.csv"
    )
    pd.DataFrame(ad_patients).to_csv("AD_patients_processed.csv")

    exported_distances = (
        DOTENV_KEY2VAL["GEN_DATA_DIR"]
        + "../generated_data/distance_from_median_image/"
        + "L_1_distances_to_mutliple_diagnostic_medians.csv"
    )
    distances.to_csv(exported_distances)


if __name__ == "__main__":
    main()

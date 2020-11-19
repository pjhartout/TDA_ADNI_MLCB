#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""compare_distribution_misclassified_vs_dist_normally_classified.py
This script aims to compare the distribution of samples that are misclassified
with the distribution of distance values from the average persistence landscape
"""

__author__ = "Philip Hartout"
__email__ = "philip.hartout@protonmail.com"

import dotenv
import random
import datetime
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from itertools import islice
from sklearn.model_selection import train_test_split, GroupKFold
import pydot
import shutil
import utils
import json


DOTENV_KEY2VAL = dotenv.dotenv_values()
HOMOLOGY_DIMENSIONS = (0, 1, 2)


def main():
    # fetch list of misclassified samples
    dest_dir_images = utils.make_dir(DOTENV_KEY2VAL["GEN_FIGURES_DIR"] + \
                      "/significance_topological_outliers/")
    misclassification = list(
        map(
            lambda x: str(x[0]),
            pd.read_csv(
                DOTENV_KEY2VAL["GEN_DATA_DIR"] + "misclassification.csv",
                usecols=[1],
            ).values,
        )
    )
    misclassification = [i + ".npy" for i in misclassification]
    # fetch associated distances from the distance_from_average matrices
    distance_files = [
        DOTENV_KEY2VAL["GEN_DATA_DIR"] + "/distance_from_average/distance_from_average_pl_CN.csv",
        DOTENV_KEY2VAL["GEN_DATA_DIR"] + "/distance_from_average/distance_from_average_pl_AD.csv",
    ]
    patient_types = [
        "CN",
        "AD",
    ]
    for distances, patient_type in zip(distance_files, patient_types):
        distances = pd.read_csv(distances)
        distances.columns = ["index", "H_0", "H_1", "H_2"]
        distances["misclassified"] = distances.isin({"index": misclassification})["index"]
        distances = distances.set_index("index")
        for i in HOMOLOGY_DIMENSIONS:
            sns.displot(distances, x=f"H_{i}", hue="misclassified", kind="kde",
                        fill=True)
            plt.savefig(
                DOTENV_KEY2VAL["GEN_FIGURES_DIR"]
                + "/significance_topological_outliers/"
                + f"distribution_distance_misclassified"
                  f"_{patient_type}_H"
                  f"_{i}.png"
            )
            plt.close("all")
    # We now look at the proportion of patients changing
    # diagnosis in the dataset vs. the proportion of patients who don't.

    with open(
        DOTENV_KEY2VAL["DATA_DIR"] + "collected_diagnoses_complete.json"
    ) as f:
        diagnoses = json.load(f)
    for patient in diagnoses.keys():
        diagnoses[patient] = [str(x) for x in diagnoses[patient].values()  if x != "nan"]

    for patient in diagnoses.keys():
        if len(
            set([str(x) for x in diagnoses[patient] if x != "nan"])
        ) > 2:
            print(diagnoses[patient])
            print(patient)
        diagnoses[patient] = len(
            set([str(x) for x in diagnoses[patient] if x != "nan"])
        )

    df = pd.read_csv(DOTENV_KEY2VAL["GEN_DATA_DIR"] + "misclassification.csv", usecols=[1])
    list_of_patients = df["0"].to_list()
    list_of_patients = [i.split("-")[1:2][0] for i in list_of_patients]
    list_of_patients = set(list_of_patients)
    list_of_diag_lengths = []
    for i in list(list_of_patients):
        list_of_diag_lengths.append(diagnoses["sub-" + i])

    all_diags = []
    for diagnosis in list(diagnoses.keys()):
        all_diags.append(diagnoses[diagnosis])
    print(f"Poportions of diagnoses for misclassified patients "
          f"{pd.DataFrame(list_of_diag_lengths).value_counts()/len(list_of_diag_lengths)}")
    print(f"Counts: {pd.DataFrame(list_of_diag_lengths).value_counts()}")
    print(f"Poportions of diagnoses for all patients"
          f" {pd.DataFrame(all_diags).value_counts()/len(all_diags)}")
    print(f"Counts: {pd.DataFrame(all_diags).value_counts()}")

if __name__ == "__main__":
    main()

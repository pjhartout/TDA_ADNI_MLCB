#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""patient_evolution.py

This script aims to track the topological evolution of a patient over time.
"""

__author__ = "Philip Hartout"
__email__ = "philip.hartout@protonmail.com"


import numpy as np
import json
import dotenv
import collections
import seaborn as sns
from gtda.homology import CubicalPersistence
from gtda.diagrams import PairwiseDistance
import utils


SHAPE = (1, 30, 36, 30)
HOMOLOGY_DIMENSIONS = (0, 1, 2)
DOTENV_KEY2VAL = dotenv.dotenv_values()
N_JOBS = -1  # Set number of workers when parallel processing is useful.


def main():
    path_to_diags = "../data/collected_diagnoses_complete.json"
    patients = ["sub-ADNI005S0223"]
    progr = ["cn_mci_ad"]
    with open(path_to_diags) as f:
        diagnoses = json.load(f)

    # Sort diagnoses key
    diagnoses = collections.OrderedDict(sorted(diagnoses.items()))

    # Where the data comes from
    data_dir = DOTENV_KEY2VAL["DATA_DIR"] + "/patch_91/"

    # Where the figures are saved
    temporal_progression_dir = "/temporal_evolution/"
    utils.make_dir(
        DOTENV_KEY2VAL["GEN_FIGURES_DIR"] + temporal_progression_dir
    )

    # Where the resulting distance matrices are saved.
    time_series_dir = "/temporal_evolution/"
    utils.make_dir(DOTENV_KEY2VAL["GEN_DATA_DIR"] + temporal_progression_dir)

    # If we want to process multiple patients, we just throw them in a loop.
    for i, patient in enumerate(patients):
        print(
            "Processing longitudinal data for "
            + patient
            + " with progression pattern "
            + progr[i]
        )
        patches = []
        for mri in diagnoses[patient]:
            try:
                patches.append(
                    np.load(
                        data_dir
                        + patient
                        + mri.replace("ses", "")
                        + "-MNI.npy"
                    )
                )
            except FileNotFoundError:
                print(
                    data_dir
                    + patient
                    + mri.replace("ses", "")
                    + "-MNI.npy"
                    + " not found"
                )
        # Stacking enables multiprocessing
        patches = np.stack(patches)

        cp = CubicalPersistence(
            homology_dimensions=HOMOLOGY_DIMENSIONS,
            coeff=2,
            periodic_dimensions=None,
            infinity_values=None,
            reduced_homology=True,
            n_jobs=-1,
        )
        diagrams_cubical_persistence = cp.fit_transform(patches)

        pl_dist = PairwiseDistance(
            metric="landscape", metric_params=None, order=None, n_jobs=-1
        )
        X_distance = pl_dist.fit_transform(
            diagrams_cubical_persistence
        )
        with open(
            DOTENV_KEY2VAL["GEN_DATA_DIR"]
            + time_series_dir
            + f"distance_data_patient_{patient}_{progr[i]}_landscape.npy",
            "wb",
        ) as f:
            np.save(f, X_distance)


if __name__ == "__main__":
    main()

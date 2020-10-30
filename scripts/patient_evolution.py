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

from gtda.homology import CubicalPersistence
from gtda.diagrams import PairwiseDistance
from tqdm import tqdm
import utils

SHAPE = (1, 30, 36, 30)
HOMOLOGY_DIMENSIONS = (0, 1, 2)
DOTENV_KEY2VAL = dotenv.dotenv_values()
N_JOBS = -1  # Set number of workers when parallel processing is useful.


def main():
    path_to_diags = "../data/collected_diagnoses_complete.json"
    patients = ["sub-ADNI029S0878"]
    progr = ["mci_ad"]
    with open(path_to_diags) as f:
        diagnoses = json.load(f)

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

        metrics = [
            "wasserstein",
            "betti",
            "landscape",
            "silhouette",
            "heat",
            "persistence_image",
        ]
        for metric in metrics:
            print(metric)
            pairwise_distance = PairwiseDistance(
                metric=metric, metric_params=None, order=None, n_jobs=-1
            )
            X_distance = pairwise_distance.fit_transform(
                diagrams_cubical_persistence
            )
            utils.plot_distance_matrix(
                X_distance,
                title="Progression of a patient over time of patient patient",
                file_prefix=temporal_progression_dir
                + f"time_progression_patient_{patient}_{progr[i]}_{metric}",
            )
            with open(
                DOTENV_KEY2VAL["GEN_DATA_DIR"]
                + time_series_dir
                + f"distance_data_patient_{patient}_{progr[i]}_{metric}.npy",
                "wb",
            ) as f:
                np.save(f, X_distance)


if __name__ == "__main__":
    main()

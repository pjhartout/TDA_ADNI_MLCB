#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""stitch_images.py

This script stitches the patches from the same patient together.
The whole brain is obtained by 6x6x6 patches
"""

__author__ = "Philip Hartout"
__email__ = "philip.hartout@protonmail.com"


import os
import numpy as np
import dotenv
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from tqdm import tqdm

DOTENV_KEY2VAL = dotenv.dotenv_values()
N_JOBS = -1
DATA_PATCH_DIR = (
    DOTENV_KEY2VAL["DATA_DIR"]
    + "/data_patches_brain_extraction-complete_scaled_1e-6_999/"
)


def reconstruct_brain_patient(patient):
    list_of_patches = list()
    for patch in range(216):
        list_of_patches.append(
            np.load(DATA_PATCH_DIR + str(patch) + "/" + patient)
        )

    reconstructed = np.zeros((180, 216, 180))
    ids = np.zeros((6, 6, 6))
    for patch in range(216):
        patchid2 = int(
            patch
            - np.floor(patch / 36) * 36
            - np.floor((patch - np.floor(patch / 36) * 36) / 6) * 6
        )
        patchid1 = int(np.floor((patch - np.floor(patch / 36) * 36) / 6))
        patchid0 = int(np.floor(patch / 36))

        ids[int(patchid0), int(patchid1), int(patchid2)] = patch

        reconstructed[
            patchid0 * 30 : patchid0 * 30 + 30,
            patchid1 * 36 : patchid1 * 36 + 36,
            patchid2 * 30 : patchid2 * 30 + 30,
        ] = list_of_patches[patch]

    with open(
        f"{DOTENV_KEY2VAL['DATA_DIR']}/reconstructed/{patient}", "wb"
    ) as f:
        np.save(f, reconstructed)


def main():
    """Main function"""

    # The dirs are set up as /patch/patient so we have to look at the first one
    # to get patient ids
    patients = os.listdir(DATA_PATCH_DIR + "/0")
    # Don't process if exists
    for patient in patients:
        if os.path.exists(
            f"{DOTENV_KEY2VAL['DATA_DIR']}/reconstructed/{patient}.npy"
        ):
            patients.remove(patient)
    for patient in tqdm(patients):
        reconstruct_brain_patient(patient)


if __name__ == "__main__":
    main()

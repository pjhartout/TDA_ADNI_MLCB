#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""global_persistence_image_generator.py

This script generates the persistence images for the entire brain of a patient.
The resulting image can then be used for classification purposes.

The script to generate global persistence has a different architecture than the other
persistence image generation script.

TODO:
    -
"""


__author__ = "Philip Hartout"
__email__ = "philip.hartout@protonmail.com"

import dotenv
import os
import numpy as np
from tqdm import tqdm
from gtda.homology import CubicalPersistence
from gtda.diagrams import PersistenceImage
from joblib import Parallel, delayed
import gc

DOTENV_KEY2VAL = dotenv.dotenv_values()
N_JOBS = 6
HOMOLOGY_DIMENSIONS = (0, 1, 2)
N_BINS = 1000


def get_all_available_patches(path_to_raw_images):
    """Obtains all array data in given path_to_raw_images and tracks loaded files.

    Args:
        path_to_raw_image (str): path to the patches of interest

    Returns:

    """
    files_loaded = []
    for root, dirs, files in os.walk(path_to_raw_images):
        for fil in files:
            if fil.endswith(".npy"):
                files_loaded.append(fil)
    return files_loaded


def get_cubical_persistence(patch):
    cp = CubicalPersistence(
        homology_dimensions=HOMOLOGY_DIMENSIONS,
        coeff=2,
        periodic_dimensions=None,
        infinity_values=None,
        reduced_homology=True,
        n_jobs=N_JOBS,
    )
    diagrams_cubical_persistence = cp.fit_transform(patch)
    return diagrams_cubical_persistence


def get_persistence_images(persistence_diagram):
    pi = PersistenceImage(
        sigma=0.001, n_bins=N_BINS, weight_function=None, n_jobs=N_JOBS
    )
    return pi.fit_transform(persistence_diagram)


def process_image(image_name, path_to_raw_images, path_to_persistence_images):
    patch = np.load(path_to_raw_images + image_name).reshape(1, 180, 216, 180)
    persistence_diagram = get_cubical_persistence(patch)
    persistence_image = get_persistence_images(persistence_diagram)

    with open(path_to_persistence_images + image_name, "wb") as f:
        np.save(f, persistence_image)
        gc.collect()


def main():
    path_to_raw_images = DOTENV_KEY2VAL["DATA_DIR"] + "reconstructed/"
    path_to_persistence_images = (
        DOTENV_KEY2VAL["DATA_DIR"] + "/global_persistence_images/"
    )

    if not os.path.exists(path_to_persistence_images):
        os.mkdir(path_to_persistence_images)

    file_names = get_all_available_patches(path_to_raw_images)

    print("Removing images for which the PI has already been computed")
    processed_files = os.listdir(path_to_persistence_images)
    for image in tqdm(processed_files):
        file_names.remove(image)

    print(f"There arex {len(file_names)} files to process.")
    Parallel(n_jobs=N_JOBS, verbose=10)(
        delayed(process_image)(
            image_name, path_to_raw_images, path_to_persistence_images
        )
        for image_name in file_names
    )
    print("Persistence images for whole-brain images completed")


if __name__ == "__main__":
    main()

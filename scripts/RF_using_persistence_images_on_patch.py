#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""CNN_using_persistence_images_on_patch.py

The aim of this script is to perform the training of a CNN using persistence
images as a input. This script is heavily inspired from this script
https://github.com/BorgwardtLab/ADNI_MRI_Analysis/blob/mixed_CNN/mixed_CNN/run_Sarah.py.
"""

__author__ = "Philip Hartout"
__email__ = "philip.hartout@protonmail.com"

import dotenv

import datetime
import os

import numpy as np

from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

from sklearn.metrics import fbeta_score, make_scorer
from sklearn.covariance import EllipticEnvelope
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import (
    ExtraTreesRegressor,
    IsolationForest,
    RandomForestClassifier,
)
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import (
    RobustScaler,
    QuantileTransformer,
    PowerTransformer,
)
from sklearn.feature_selection import SelectKBest, f_regression, SelectFdr
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    WhiteKernel,
    ExpSineSquared,
    RBF,
    CompoundKernel,
    ConstantKernel,
    DotProduct,
    Exponentiation,
    Hyperparameter,
    Kernel,
    Matern,
    PairwiseKernel,
    Product,
    RationalQuadratic,
    Sum,
    WhiteKernel,
)
from sklearn.datasets import make_friedman2
from sklearn.metrics import balanced_accuracy_score
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier

import shutil
import utils

DOTENV_KEY2VAL = dotenv.dotenv_values()


def get_arrays_from_dir(directory, filelist):
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


def main():
    # This defines where we are in the machien
    diagnosis_json = (
        DOTENV_KEY2VAL["DATA_DIR"] + "/collected_diagnoses_complete.json"
    )
    persistence_image_location = (
        DOTENV_KEY2VAL["DATA_DIR"] + "/patch_91_persistence_images/"
    )

    ############################################################################
    #  Data loading and processing
    ############################################################################

    # First, we load relevant images (only baseline image for now)
    (
        cn_patients,
        mci_patients,
        ad_patients,
    ) = utils.get_earliest_available_diagnosis(diagnosis_json)

    # For now we only get images for CN and MCI patients
    images_cn = get_arrays_from_dir(persistence_image_location, cn_patients)
    images_ad = get_arrays_from_dir(persistence_image_location, ad_patients)
    # Concatenate both arrays
    labels = np.array(
        [0 for i in range(images_cn.shape[0])]  # Labels for CN is 0
        + [1 for i in range(images_ad.shape[0])]  # Labels for AD is 2
    ).T
    images = np.vstack((images_ad, images_cn))

    X_train, X_test, y_train, y_test = train_test_split(
        images.reshape(images.shape[0], 100 * 100 * 3),
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    ############################################################################
    #  Model definition
    ############################################################################

    params = {
        "n_estimators": [200],
        "criterion": ["gini"],
        "max_depth": [None],
        "min_samples_split": [2],
        "min_samples_leaf": [1],
        "min_weight_fraction_leaf": [0.0],
        "max_features": ["auto"],
        "max_leaf_nodes": [None],
        "min_impurity_decrease": [0.0],
        "min_impurity_split": [None],
        "bootstrap": [True],
        "oob_score": [False],
        "n_jobs": [None],
        "random_state": [42],
        "verbose": [0],
        "warm_start": [False],
        "class_weight": [None],
        "ccp_alpha": [0.0],
        "max_samples": [None],
    }

    search = GridSearchCV(
        RandomForestClassifier(),
        param_grid=params,
        cv=5,
        scoring=make_scorer(balanced_accuracy_score),
        n_jobs=-1,
        verbose=2,
    )

    ############################################################################
    #  Model definition
    ############################################################################

    search.fit(X_train, y_train)

    ############################################################################
    #  Model evaluation
    ############################################################################

    y_pred = search.best_estimator_.predict(X_test)

    print(
        f"Best model training set cross validated score using RF is"
        f" {search.best_score_}"
    )
    print(f"Best performance is achieved using " f"{search.best_params_}")
    print(
        f"Best test set score using RF is {balanced_accuracy_score(y_test, y_pred)}"
    )


if __name__ == "__main__":
    main()

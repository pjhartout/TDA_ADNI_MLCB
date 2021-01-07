#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""CNN_using_persistence_images_on_patch.py

The aim of this script is to perform the training of a CNN using persistence
images as a input. This script is inspired from this script:
BorgwardtLab/ADNI_MRI_Analysis/blob/mixed_CNN/mixed_CNN/run_Sarah.py

To get real time information into the model training and structure, run
$ tensorboard --logdir logs/fit

once this script has been started.


NOTES:
    - One loaded, the "big" 100x100x3 images aren't that big (>400MB in RAM) so
      NO GENERATOR NEEDED

"""

__author__ = "Philip Hartout"
__email__ = "philip.hartout@protonmail.com"

import dotenv
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, make_scorer

DOTENV_KEY2VAL = dotenv.dotenv_values()
N_BINS = 100
N_LAYERS = 50
################################################################################
#  Functions
################################################################################

persistence_landscape_location = (
    DOTENV_KEY2VAL["DATA_DIR"] + "/patch_91_persistence_landscapes/"
)
partitions_location = DOTENV_KEY2VAL["DATA_DIR"] + "/partitions/"
diagnosis_json = (
    DOTENV_KEY2VAL["DATA_DIR"] + "/collected_diagnoses_complete.json"
)


def get_partitions(partitions_location):
    partition = []
    labels = []
    for root, dirs, files in os.walk(partitions_location):
        for file in files:
            if file.split("_")[0] == "partition":
                partition.append(
                    np.load(
                        partitions_location + file, allow_pickle=True
                    ).item()
                )
            elif file.split("_")[0] == "labels":
                labels.append(
                    np.load(
                        partitions_location + file, allow_pickle=True
                    ).item()
                )
            else:
                print(f"File {file} is neither partition nor labels file")
    return partition, labels


################################################################################
#  Main
################################################################################
def main():
    ############################################################################
    #  Data loading and processing
    ############################################################################
    inits = 1
    partitions, labels = get_partitions(partitions_location)
    histories = []
    for partition, label in zip(partitions, labels):
        for i in range(inits):
            # Make sure there aren't the same patients in train and test
            X_train_lst = []
            y_train_lst = []
            for landscape in partition["train"]:
                X_train_lst.append(
                    np.load(persistence_landscape_location + landscape + ".npy")
                )
                y_train_lst.append(label[landscape])

                X_train, y_train = (
                    np.stack(X_train_lst, axis=0).reshape(
                        len(X_train_lst), N_BINS, N_LAYERS, 3
                    ),
                    np.vstack(y_train_lst),
                )
            X_val_lst = []
            y_val_lst = []
            for landscape in partition["validation"]:
                X_val_lst.append(
                    np.load(persistence_landscape_location + landscape + ".npy")
                )
                y_val_lst.append(label[landscape])

                X_val, y_val = (
                    np.stack(X_val_lst, axis=0).reshape(
                        len(X_val_lst), N_BINS, N_LAYERS, 3
                    ),
                    np.vstack(y_val_lst),
                )

            ####################################################################
            #  Model definition
            ####################################################################
            model = RandomForestClassifier()
            params = {
                "n_estimators": [300],
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
                cv=10,
                scoring=make_scorer(accuracy_score),
                n_jobs=-1,
            )
            search.fit(X_train.reshape(len(X_train),-1), y_train.ravel())
            y_val_pred = search.best_estimator_.predict(X_val.reshape(len(
                X_val),-1))
            print(
                f"Best model training set cross validated score using RF is"
                f" {search.best_score_}"
            )
            print(
                f"Best performance is achieved using " f"{search.best_params_}")
            print(
                f"Best test set score using RF is "
                f"{accuracy_score(y_val, y_val_pred)}"
            )

if __name__ == "__main__":
    main()

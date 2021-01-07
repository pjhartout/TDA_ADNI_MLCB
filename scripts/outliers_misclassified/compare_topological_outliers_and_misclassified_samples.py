#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""compare_topological_outliers_and_misclassified_samples.py

Compares samples in both lists and look at overlap.
"""

__author__ = "Philip Hartout"
__email__ = "philip.hartout@protonmail.com"

import dotenv
import random
import datetime
import os
import numpy as np
import pandas as pd
from itertools import islice
import pydot
import shutil
import utils
from itertools import chain

DOTENV_KEY2VAL = dotenv.dotenv_values()


def main():
    gen_data_dir = DOTENV_KEY2VAL["GEN_DATA_DIR"] + "/distance_from_average/"
    list_of_df = []
    for root, dirs, files in os.walk(gen_data_dir, topdown=False):
        for file in files:
            if "MCI" not in file and "outliers" in file:
                df = pd.read_csv(gen_data_dir + file)
                df = df.values
                list_of_df.append(list(df.reshape(df.shape[0] * 3, 1)))
    outliers = set(
        list(
            map(
                lambda x: x.replace(".npy", ""),
                list(map(str, list(chain(*list_of_df[0])))),
            )
        )
    )
    misclassification = set(
        list(
            map(
                lambda x: str(x[0]),
                pd.read_csv(
                    DOTENV_KEY2VAL["GEN_DATA_DIR"] + "misclassification.csv",
                    usecols=[1],
                ).values,
            )
        )
    )
    misclassification_in_outliers = []
    for i in misclassification:
        misclassification_in_outliers.append(i in outliers)
    outliers_in_misclassification = pd.DataFrame(misclassification_in_outliers)
    print(
        f"Of the {len(misclassification)} miscalssified samples,"
        f" {int(outliers_in_misclassification.value_counts()[1])} samples "
        f"where found to be in the 90th percentile of the representations "
        f"that were the most divergent from the average persistence "
        f"landscape. This set of topological outliers includes"
        f" {len(outliers)} samples in total."
    )


if __name__ == "__main__":
    main()

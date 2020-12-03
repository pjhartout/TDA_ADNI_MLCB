#!/usr/bin/env python
# coding: utf-8

"""clustering_plotting.py

This script explores of the distances obtained from the comparisons of each pi
to all other median PIs for each diagnostic category.
"""


import utils
import matplotlib.pyplot as plt
from matplotlib import rcParams
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
from plotly.subplots import make_subplots
from plotly.graph_objects import Figure, Scatter
from scipy.stats import mannwhitneyu
import plotly.express as px
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.decomposition import PCA

HOMOLOGY_DIMENSIONS = (0, 1, 2)
DOTENV_KEY2VAL = dotenv.dotenv_values()
SCALE = 10
rcParams["figure.figsize"] = 20, 20


def format_tex(float_number):
    """Format large numbers in latex """
    exponent = np.floor(np.log10(float_number))
    mantissa = float_number / 10 ** exponent
    mantissa_format = str(mantissa)[0:3]
    return "${0}\times10^{{{1}}}$".format(mantissa_format, str(int(exponent)))


def main():
    diagnosis_json = (
        DOTENV_KEY2VAL["DATA_DIR"] + "collected_diagnoses_complete.json"
    )
    (
        cn_patients,
        mci_patients,
        ad_patients,
        unknown,
    ) = utils.get_all_available_diagnoses(diagnosis_json)

    distance_data = pd.read_csv(
        "../generated_data/distance_from_median_image/"
        + "L_1_distances_to_mutliple_diagnostic_medians.csv"
    ).rename(columns={"Unnamed: 0": "patients"})

    distance_stats_df = pd.DataFrame()
    distance_data_diags = pd.DataFrame()
    for patient_type, list_of_patients in zip(
        ["CN", "MCI", "AD"], [cn_patients, mci_patients, ad_patients]
    ):
        distance_patient_type = distance_data[
            distance_data["patients"].isin(list_of_patients)
        ]
        distance_data[distance_data["patients"].isin(list_of_patients)][
            "diag"
        ] = patient_type
        for h_dim in HOMOLOGY_DIMENSIONS:
            distance_patient_type_hdim = distance_patient_type[
                f"{patient_type}_H_{h_dim}"
            ]
            # Displot of the distance for a particular patient category
            # sns.displot(distance_patient_type_hdim, color=CMAP_DIAG[patient_type])
            # plt.xlim(
            #     min(distance_data),
            #     max(distance_data),
            # )
            # plt.show()

            # Create dict with stats for the distribution of a given patient category
            distance_stats_dict = dict()
            distance_stats_dict["Mean"] = np.mean(distance_patient_type_hdim)
            distance_stats_dict["Median"] = np.median(
                distance_patient_type_hdim
            )
            distance_stats_dict["Standard deviation"] = np.std(
                distance_patient_type_hdim
            )
            distance_stats_dict["Q3"] = np.quantile(
                distance_patient_type_hdim, 0.75
            )
            distance_stats_dict["Max"] = np.max(distance_patient_type_hdim)
            #            distance_stats_dict["kurtosis"] = kurtosis(distance_data)
            distance_stats_dict["Skewness"] = skew(distance_patient_type_hdim)
            distance_stats_df_entry = pd.DataFrame.from_dict(
                distance_stats_dict, orient="index"
            )
            distance_stats_df_entry.columns = [f"{patient_type} $H_{h_dim}$"]
            distance_stats_df = distance_stats_df.append(
                distance_stats_df_entry.T
            )
            # Append diagnosis
        distance_patient_type["diag"] = patient_type
        distance_patient_type = distance_patient_type[["patients", "diag"]]
        distance_data_diags = distance_data_diags.append(distance_patient_type)

    distance_stats_df = distance_stats_df.applymap(lambda x: format_tex(x))
    # distance_stats_df.to_latex(
    #     DOTENV_KEY2VAL["GEN_DATA_DIR"]
    #     + "output_distance_statistics_persistence_image.tex",
    #     float_format="{:0.2f}".format,
    #     escape=False,
    # )
    scaler = StandardScaler()
    patients = distance_data["patients"]
    columns = distance_data.columns[1:]
    distance_data = scaler.fit_transform(distance_data.set_index("patients"))
    # distance_data = distance_data.apply(
    #     lambda x: np.log10(x) if np.issubdtype(x.dtype, np.number) else x
    # )
    distance_data = pd.DataFrame(
        distance_data, index=patients, columns=columns
    )
    distance_data = distance_data.merge(distance_data_diags, on="patients")
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(distance_data[["CN_H_2", "AD_H_2"]])
    pca_data = pd.DataFrame(
        pca_data, index=patients, columns=["PCA_1", "PCA_2"]
    )
    pca_data = pca_data.merge(distance_data_diags, on="patients")
    sns.scatterplot(
        data=pca_data, x="PCA_1", y="PCA_2", hue="diag", sizes=(2, 2)
    )
    plt.savefig(
        DOTENV_KEY2VAL["GEN_FIGURES_DIR"] + "cluster_CN_H_2_AD_H_2_PCA.png",
        dpi=300,
        bbox_inches="tight",
    )


if __name__ == "__main__":
    main()

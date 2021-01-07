#!/usr/bin/env python

"""plotting.py

This script aims to produce all the figures in this repository.
"""

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
import argparse
from plotly.subplots import make_subplots
from plotly.graph_objects import Figure, Scatter
from scipy.stats import mannwhitneyu, shapiro, anderson

# Global variables that should not be changed
DOTENV_KEY2VAL = dotenv.dotenv_values()
N_JOBS = 1
HOMOLOGY_DIMENSIONS = (0, 1, 2)

# Global variables determining whether plots should be produced
# Could have been implemented using CLI args as well.
HOMOLOGY_CMAP = {0: "red", 1: "green", 2: "purple"}
SAMPLE_REP = False
# DISTPLOT_PD_DISTANCES = False
MEDIAN_PL = False
MEDIAN_PI = True
# AVERAGE_PL_MULTI = False
PLOT_DISTANCE_FROM_MEDIAN_PL = True
PLOT_DISTANCE_FROM_MEDIAN_PI = True
PATIENT_EVOLUTION = False
PATIENT_EVOLUTION_AVERAGE = False
DIVERGENCE_BETWEEN_PDS = False
SCALE = 5  # resolution of exported images
VEC_SIZE = 100
N_LAYERS = 1


def make_dir(directory):
    """Makes directory and handles errors"""
    try:
        os.mkdir(directory)
    except OSError:
        print("Creation of the directory %s failed" % directory)
    else:
        print("Successfully created the directory %s " % directory)


def get_earliest_available_diagnosis(path_to_diags):
    """Gets diagnosis at first available timepoint"""
    cn_patients = []
    mci_patients = []
    ad_patients = []
    with open(path_to_diags) as f:
        diagnoses = json.load(f)

    for patient in list(diagnoses.keys()):
        if diagnoses[patient][list(diagnoses[patient].keys())[0]] == "CN":
            cn_patients.append(format_patient(patient, diagnoses))
        elif diagnoses[patient][list(diagnoses[patient].keys())[0]] == "MCI":
            mci_patients.append(format_patient(patient, diagnoses))
        elif diagnoses[patient][list(diagnoses[patient].keys())[0]] == "AD":
            ad_patients.append(format_patient(patient, diagnoses))
    return cn_patients, mci_patients, ad_patients


def format_tex_numbers(float_number):
    """Surrounds $float_number$. Do not use if format_tex is used."""
    return f"${np.round(float_number, 3)}$"


def format_tex(float_number):
    """Format large numbers in latex """
    exponent = np.floor(np.log10(float_number))
    mantissa = float_number / 10 ** exponent
    mantissa_format = str(mantissa)[0:3]
    return "${0}\times10^{{{1}}}$".format(mantissa_format, str(int(exponent)))


def generate_sample_representations(paths_to_patches, labels):
    sample_rep_dir = DOTENV_KEY2VAL["GEN_FIGURES_DIR"] + "/sample_rep/"
    try:
        os.mkdir(sample_rep_dir)
    except OSError:
        print("Creation of the directory %s failed" % sample_rep_dir)
    else:
        print("Successfully created the directory %s " % sample_rep_dir)
    for i, path in enumerate(paths_to_patches):
        patch = np.load(path)

        cp = CubicalPersistence(
            homology_dimensions=(0, 1, 2),
            coeff=2,
            periodic_dimensions=None,
            infinity_values=None,
            reduced_homology=True,
            n_jobs=N_JOBS,
        )
        diagrams_cubical_persistence = cp.fit_transform(
            patch.reshape(1, 30, 36, 30)
        )
        for h_dim in HOMOLOGY_DIMENSIONS:
            cp.plot(
                diagrams_cubical_persistence,
                homology_dimensions=[h_dim],
            ).update_traces(
                marker=dict(size=10, color=HOMOLOGY_CMAP[h_dim]),
            ).write_image(
                sample_rep_dir
                + f"persistence_diagram_{labels[i]}_H_{h_dim}.png",
                scale=SCALE,
            )

        representation_names = [
            "Persistence landscape",
            "Betti curve",
            "Persistence image",
            "Heat kernel",
            "Silhouette",
        ]

        for j, rep in enumerate(representation_names):
            # Have not found a better way of doing this yet.
            if rep == "Persistence landscape":
                rep = PersistenceLandscape(
                    n_layers=N_LAYERS, n_bins=VEC_SIZE, n_jobs=N_JOBS
                )
            elif rep == "Betti curve":
                rep = BettiCurve()
            elif rep == "Persistence image":
                rep = PersistenceImage(
                    sigma=0.001, n_bins=VEC_SIZE, n_jobs=N_JOBS
                )
            elif rep == "Heat kernel":
                rep = HeatKernel(sigma=0.001, n_bins=VEC_SIZE, n_jobs=N_JOBS)
            elif rep == "Silhouette":
                rep = Silhouette(power=1.0, n_bins=VEC_SIZE, n_jobs=N_JOBS)

            vectorial_representation = rep.fit_transform(
                diagrams_cubical_persistence
            )

            if representation_names[j] in ["Persistence image", "Heat kernel"]:
                for h_dim in range(vectorial_representation.shape[1]):
                    plt.imshow(
                        vectorial_representation[0:, h_dim, :, :].reshape(
                            VEC_SIZE, VEC_SIZE
                        ),
                        cmap=(HOMOLOGY_CMAP[h_dim] + "s").capitalize(),
                    )
                    # plt.title(
                    #     f"{representation_names[j]} representation of a "
                    #     f"{labels[i]} patient in h_{image}"
                    # )
                    plt.savefig(
                        sample_rep_dir
                        + f"{representation_names[j].replace(' ', '_')}"
                        f"_{labels[i]}_h_{h_dim}.png",
                        bbox_inches="tight",
                    )
            else:
                rep.plot(vectorial_representation).update_layout(
                    title=None,
                    margin=dict(l=0, r=0, b=0, t=0, pad=4),
                ).write_image(
                    sample_rep_dir
                    + f"{representation_names[j].replace(' ', '_')}"
                    f"_{labels[i]}.png",
                    scale=SCALE,
                )
        print(f"Done plotting {labels[i]} sample")


def generate_displot_of_pd_distances(path_to_pd_pairwise_distances):
    """
    Generates distplot for each of the diagnostic categories. The data should
    have been generated by the distance_between_pds.py script.
    """
    data = pd.read_csv(path_to_pd_pairwise_distances).drop(
        columns="Unnamed: 0"
    )
    distance_pd_dir = DOTENV_KEY2VAL["GEN_FIGURES_DIR"] + "/pd_distances"

    try:
        os.mkdir(distance_pd_dir)
    except OSError:
        print("Creation of the directory %s failed" % distance_pd_dir)
    else:
        print("Successfully created the directory %s " % distance_pd_dir)

    for distance in data["distance"].unique():
        dist_data = data.loc[data["distance"] == distance]
        for homology_dimension in data["homology dimension"].unique():
            dist_data_hom = dist_data.loc[
                dist_data["homology dimension"] == homology_dimension
            ]
            fig = sns.displot(
                data=dist_data_hom, x="value", hue="diagnosis", kind="ecdf"
            )
            plt.subplots_adjust(top=0.85)
            plt.title(
                "\n".join(
                    textwrap.wrap(
                        f"Empirical cumulative distribution function of the"
                        f" {distance} distances among diagnostic categories "
                        f"for H_{homology_dimension}",
                        50,
                    )
                )
            )
            plt.savefig(
                f"../figures/pd_distances/ecdf_{distance}_h"
                f"_{homology_dimension}.png",
                bbox_inches="tight",
            )

            fig = sns.displot(
                data=dist_data_hom, x="value", hue="diagnosis", kind="kde"
            )
            plt.subplots_adjust(top=0.85)
            plt.title(
                "\n".join(
                    textwrap.wrap(
                        f"Kernel density estimation function of the "
                        f"{distance} distances among diagnostic categories for "
                        f"H_{homology_dimension}",
                        50,
                    )
                )
            )
            plt.savefig(
                f"../figures/pd_distances/kde_{distance}_h"
                f"_{homology_dimension}.png",
                bbox_inches="tight",
            )

            fig = sns.displot(
                data=dist_data_hom,
                x="value",
                hue="diagnosis",
                multiple="stack",
            )
            plt.subplots_adjust(top=0.85)
            plt.title(
                "\n".join(
                    textwrap.wrap(
                        f"Stacked histogram of the {distance} distances "
                        f"among diagnostic categories for H"
                        f"_{homology_dimension}",
                        50,
                    )
                )
            )
            plt.savefig(
                f"../figures/pd_distances/stacked_{distance}_h"
                f"_{homology_dimension}.png",
                bbox_inches="tight",
            )
            plt.close("all")


def plot_evolution_time_series(path_to_distance_matrices):
    """
    This takes the distance matrix, extract the first column and plots it as
    a time series for each patients for which this has been computed in
    scripts/patient_evolution.py
    """
    metrics = [
        "wasserstein",
        "betti",
        "landscape",
        "silhouette",
        "heat",
        "persistence_image",
    ]
    for metric in metrics:
        # patients here means the collection of pairwise distances for each
        # patient

        patients = pd.DataFrame()
        for root, dirs, files in os.walk(path_to_distance_matrices):
            # loop through files to find relevant ones for metric
            for file in files:
                if metric in file:
                    X_distance = np.load(path_to_distance_matrices + file)
                    for h_dim in HOMOLOGY_DIMENSIONS:
                        sample = pd.DataFrame(X_distance[0, :, h_dim]).melt()
                        sample["metric"] = metric
                        sample["homology_dimension"] = h_dim
                        sample["file"] = "_".join(file.split("_", 6)[4:])
                        sample = sample.drop(columns=["variable"])
                        patients = patients.append(sample)
        patients = patients.reset_index()
        for h_dim in patients["homology_dimension"].unique():
            hdim_patients = patients.loc[
                patients["homology_dimension"] == h_dim
            ]
            sns.lineplot(x="index", y="value", hue="file", data=hdim_patients)
            plt.title(
                f"{metric.replace('_', ' ').capitalize()} distance from baseline"
                f" for patients over time in {h_dim}."
            )
            plt.savefig(
                DOTENV_KEY2VAL["GEN_FIGURES_DIR"]
                + "/temporal_evolution/"
                + metric
                + str(h_dim)
                + ".png",
                bbox_inches="tight",
            )
            plt.close("all")


def plot_deviation_from_avg_pl(path_to_distance_matrices, figures):
    """
    This takes the distance matrix, extract the first column and plots it as
    a time series for each patients for which this has been computed in
    scripts/patient_evolution.py
    """
    make_dir(figures + "/distance_from_avg/")
    for root, dirs, files in os.walk(path_to_distance_matrices):
        for file in files:
            if "landscape_difference" in file:
                distances = np.load(path_to_distance_matrices + file)
                distances = pd.DataFrame(
                    distances, columns=["H_0", "H_1", "H_2"]
                ).melt()
                sns.displot(
                    data=distances,
                    x="value",
                    hue="variable",
                    kind="kde",
                    fill=True,
                )
                patient_type = file.split("_")[0]
                plt.title(
                    f"distance from the average persistence landscape"
                    f"representation for {patient_type}."
                )
                plt.savefig(
                    figures + "distribution_distance_from_avg_{}.png",
                    bbox_inches="tight",
                )


def plot_median_persistence_landscapes(image_dir, patient_types):
    """
    This function plots the median persistence landscape of the diagnostic
    categories
    """
    for i, pl in enumerate(image_dir):
        pl = np.load(DOTENV_KEY2VAL["GEN_DATA_DIR"] + pl)
        ax = pd.DataFrame(pl).T.plot()
        ax.set(ylim=(0, 0.17))
        ax.legend(["$H_0$", "$H_1$", "$H_2$"])
        plt.savefig(
            DOTENV_KEY2VAL["GEN_FIGURES_DIR"]
            + "/median_pls/"
            + f"median_pl_{patient_types[i]}_rep.png",
            bbox_inches="tight",
        )
        plt.close("all")


def plot_median_persistence_image(image_dir, patient_types):
    """
    This function plots the median  persistence landscape of the diagnostic
    categories and plots them
    """
    for i, pi in enumerate(image_dir):
        pi = np.load(DOTENV_KEY2VAL["GEN_DATA_DIR"] + pi)
        for h_dim in HOMOLOGY_DIMENSIONS:
            plt.imshow(pi[h_dim, :, :], cmap="Blues")
            plt.savefig(
                DOTENV_KEY2VAL["GEN_FIGURES_DIR"]
                + "/median_pis/"
                + f"median_pi_{patient_types[i]}_h_{h_dim}_rep.png",
                bbox_inches="tight",
            )
            plt.close("all")


def plot_distance_from_median_pl(distance_files, patient_types):
    print(f"-------- Distance from median PL --------")
    distance_stats_df = pd.DataFrame()
    for distances, patient_type in zip(distance_files, patient_types):
        distances = pd.read_csv(DOTENV_KEY2VAL["GEN_DATA_DIR"] + distances)
        distances = distances.set_index("Unnamed: 0")
        for i in range(len(HOMOLOGY_DIMENSIONS)):
            distance_data = distances.iloc[:, i]
            print(f"-------- {patient_type} H_{i} --------")
            distance_stats_dict = dict()
            distance_stats_dict["Mean"] = np.mean(distance_data)
            distance_stats_dict["Median"] = np.median(distance_data)
            distance_stats_dict["Standard deviation"] = np.std(distance_data)
            distance_stats_dict["Q3"] = np.quantile(distance_data, 0.75)
            distance_stats_dict["Max"] = np.max(distance_data)
            #            distance_stats_dict["kurtosis"] = kurtosis(distance_data)
            distance_stats_dict["Skewness"] = skew(distance_data)
            distance_stats_dict["Shapiro-Wilk test"] = shapiro(
                distance_data
            ).pvalue
            print(f"Shapiro-Wilk: {shapiro(distance_data)}")
            distance_stats_df_entry = pd.DataFrame.from_dict(
                distance_stats_dict, orient="index"
            )
            distance_stats_df_entry.columns = [f"{patient_type} $H_{i}$"]
            distance_stats_df = distance_stats_df.append(
                distance_stats_df_entry.T
            )
            ax = sns.displot(distance_data, kde=True)
            # ax.set(ylim=(0, 1))  # Finetuned to the data
            ax.set(xlim=(0, 12))
            plt.savefig(
                DOTENV_KEY2VAL["GEN_FIGURES_DIR"]
                + "/median_pls/"
                + f"median_pl_{patient_type}_H_{i}_displot.png",
                bbox_inches="tight",
            )
    test_results = pd.DataFrame(
        distance_stats_df["Shapiro-Wilk test"]
    ).applymap(lambda x: format_tex(x))
    stats = distance_stats_df[
        ["Mean", "Median", "Standard deviation", "Q3", "Max", "Skewness"]
    ].applymap(lambda x: format_tex_numbers(x))
    distance_stats_df = stats.join(test_results)
    print(test_results)
    distance_stats_df.to_latex(
        DOTENV_KEY2VAL["GEN_DATA_DIR"]
        + "distance_from_median_pl_statistics.tex",
        float_format="{:0.2f}".format,
        escape=False,
    )


def plot_distance_from_median_pi(distance_files, patient_types):
    print(f"-------- Distance from median PI --------")
    distance_stats_df = pd.DataFrame()
    for distances, patient_type in zip(distance_files, patient_types):
        distances = pd.read_csv(DOTENV_KEY2VAL["GEN_DATA_DIR"] + distances)
        distances = distances.set_index("Unnamed: 0")
        for i in range(len(HOMOLOGY_DIMENSIONS)):
            distance_data = distances.iloc[:, i]
            print(f"-------- {patient_type} H_{i} --------")
            distance_stats_dict = dict()
            distance_stats_dict["Mean"] = np.mean(distance_data)
            distance_stats_dict["Median"] = np.median(distance_data)
            distance_stats_dict["Standard deviation"] = np.std(distance_data)
            distance_stats_dict["Q3"] = np.quantile(distance_data, 0.75)
            distance_stats_dict["Max"] = np.max(distance_data)
            #            distance_stats_dict["kurtosis"] = kurtosis(distance_data)
            distance_stats_dict["Skewness"] = skew(distance_data)
            distance_stats_dict["Shapiro-Wilk test"] = shapiro(
                distance_data
            ).pvalue
            print(f"Shapiro-Wilk: {shapiro(distance_data)}")
            distance_stats_df_entry = pd.DataFrame.from_dict(
                distance_stats_dict, orient="index"
            )
            distance_stats_df_entry.columns = [f"{patient_type} $H_{i}$"]
            distance_stats_df = distance_stats_df.append(
                distance_stats_df_entry.T
            )
            ax = sns.displot(distance_data, kde=True)
            # ax.set(ylim=(0, 1))  # Finetuned to the data
            ax.set(xlim=(0, 14000000))
            plt.savefig(
                DOTENV_KEY2VAL["GEN_FIGURES_DIR"]
                + "/median_pis/"
                + f"median_pi_{patient_type}_H_{i}_displot.png",
                bbox_inches="tight",
            )
    test_results = pd.DataFrame(
        distance_stats_df["Shapiro-Wilk test"]
    ).applymap(lambda x: format_tex_numbers(x))
    stats = distance_stats_df[
        ["Mean", "Median", "Standard deviation", "Q3", "Max", "Skewness"]
    ].applymap(lambda x: format_tex(x))
    distance_stats_df = stats.join(test_results)
    print(test_results)
    distance_stats_df.to_latex(
        DOTENV_KEY2VAL["GEN_DATA_DIR"]
        + "distance_from_median_pi_statistics.tex",
        float_format="{:0.2f}".format,
        escape=False,
    )


def plot_patient_evolution(generated_distance_data):
    """
    Function to deal with output of patient_evolution.py

    Args:
        generated_distance_data:

    Returns:

    """
    X_distance = []
    available_timepoints = []
    for root, dirs, files in os.walk(generated_distance_data):
        for file in files:
            X_distance.append(
                np.load(generated_distance_data + file, allow_pickle=True)
            )
            # Search for the available labels
            patient_id = file.split("_")[3]
            timepoints_for_patient = []
            for root, dirs, files in os.walk(DOTENV_KEY2VAL["DATA_DIR"]):
                for file in files:
                    if patient_id in file:
                        timepoints_for_patient.append(file.split("-")[2])
            timepoints_for_patient.sort()
            available_timepoints.append(timepoints_for_patient)

    v_min = min(np.min(distance) for distance in X_distance)
    v_max = max(np.max(distance) for distance in X_distance)
    for root, dirs, files in os.walk(generated_distance_data):
        for i, file in enumerate(files):
            X_distance = np.load(
                generated_distance_data + file, allow_pickle=True
            )
            for j in HOMOLOGY_DIMENSIONS:
                ax = sns.heatmap(
                    X_distance[:, :, j], vmin=v_min, vmax=v_max, cmap="YlOrBr"
                )
                # ax.set_xticklabels(available_timepoints[i])
                # ax.set_yticklabels(available_timepoints[i])
                plt.savefig(
                    DOTENV_KEY2VAL["GEN_FIGURES_DIR"]
                    + "/temporal_evolution/"
                    + f"persistence_image_distance_for_"
                    f"{file.split('_')[5]}_h_{j}.png",
                    bbox_inches="tight",
                )
                plt.close("all")


def plot_patient_evolution_average(generated_distance_data):
    # Get whether or not a patient has "switched diagnosis"
    with open(
        DOTENV_KEY2VAL["DATA_DIR"] + "collected_diagnoses_complete.json"
    ) as f:
        diagnoses = json.load(f)
    for patient in diagnoses.keys():
        diagnoses[patient] = [str(x) for x in diagnoses[patient].values()]

    for patient in diagnoses.keys():
        diagnoses[patient] = len(
            set([str(x) for x in diagnoses[patient] if x != "nan"])
        )

    distance_data = pd.DataFrame()
    distances_to_evaluate = [
        # "bottleneck",
        # "wasserstein",
        # "betti",
        # "landscape",
        # "silhouette",
        # "heat",
        "persistence_image",
    ]
    for distance in distances_to_evaluate:
        for root, dirs, files in os.walk(generated_distance_data):
            for patient_file in files:
                if (
                    "patient_evolution_distance_data_patient" in patient_file
                    and distance in patient_file
                ):
                    data_patient = np.load(
                        generated_distance_data + patient_file
                    )
                    patient_id = patient_file.split("_")[5:6][0]
                    data_patient_processed = pd.DataFrame()
                    data_patient_processed["index"] = [patient_id]
                    data_patient_processed = data_patient_processed.set_index(
                        "index"
                    )
                    if diagnoses[patient_id] >= 2:
                        data_patient_processed["diagnosis_changed"] = True
                    else:
                        data_patient_processed["diagnosis_changed"] = False
                    for i in HOMOLOGY_DIMENSIONS:
                        data_patient_processed[f"H_{i}"] = np.nanmean(
                            data_patient[:, :, i]
                        )
                    distance_data = distance_data.append(
                        data_patient_processed
                    )

        for i in HOMOLOGY_DIMENSIONS:
            # Loop through files and then decide
            legend_name = "Diagnosis change"
            ax = sns.displot(
                distance_data,
                x=f"H_{i}",
                hue="diagnosis_changed",
                # bins=50,
                kind="kde",
                fill=True,
            )
            ax._legend.set_title(legend_name)
            # ax.set_title("Title")
            plt.savefig(
                DOTENV_KEY2VAL["GEN_FIGURES_DIR"]
                + "/temporal_evolution/"
                + f"{distance}_H_{i}_dist_diag_change.png",
                bbox_inches="tight",
            )
            plt.close("all")
            # Mann-Whitney U test
            x = distance_data[f"H_{i}"].loc[
                distance_data["diagnosis_changed"] == True
            ]
            y = distance_data[f"H_{i}"].loc[
                distance_data["diagnosis_changed"] == False
            ]
            if distance in ["wasserstein", "bottleneck", "landscape"]:
                print(
                    f"For {distance} in H_{i}, the Mann-Whitney U yields"
                    f" {mannwhitneyu(x,y).pvalue}"
                )


def main():
    # First, we want to generate a typical representation of the data for each
    # diagnostic category
    make_dir(DOTENV_KEY2VAL["GEN_FIGURES_DIR"])

    if SAMPLE_REP:
        generate_sample_representations(
            [
                "../data/patch_91/sub-ADNI002S0295-M00-MNI.npy",
                "../data/patch_91/sub-ADNI128S0225-M48-MNI.npy",
                "../data/patch_91/sub-ADNI128S0227-M48-MNI.npy",
            ],
            ["CN", "MCI", "AD"],
        )

    if MEDIAN_PL:
        make_dir(DOTENV_KEY2VAL["GEN_FIGURES_DIR"] + "/median_pls/")
        plot_median_persistence_landscapes(
            [
                "/distance_from_average/average_pl_CN.npy",
                "/distance_from_average/average_pl_MCI.npy",
                "/distance_from_average/average_pl_AD.npy",
            ],
            ["CN", "MCI", "AD"],
        )

    if MEDIAN_PI:
        make_dir(DOTENV_KEY2VAL["GEN_FIGURES_DIR"] + "/median_pis/")
        plot_median_persistence_image(
            [
                "/distance_from_median_pi/median_pi_CN.npy",
                "/distance_from_median_pi/median_pi_MCI.npy",
                "/distance_from_median_pi/median_pi_AD.npy",
            ],
            ["CN", "MCI", "AD"],
        )

    if PLOT_DISTANCE_FROM_MEDIAN_PL:
        plot_distance_from_median_pl(
            [
                "/distance_from_median_pl/distance_from_median_pl_CN.csv",
                "/distance_from_median_pl/distance_from_median_pl_MCI.csv",
                "/distance_from_median_pl/distance_from_median_pl_AD.csv",
            ],
            ["CN", "MCI", "AD"],
        )

    if PLOT_DISTANCE_FROM_MEDIAN_PI:
        plot_distance_from_median_pi(
            [
                "/distance_from_median_pi/distance_from_median_pi_CN.csv",
                "/distance_from_median_pi/distance_from_median_pi_MCI.csv",
                "/distance_from_median_pi/distance_from_median_pi_AD.csv",
            ],
            ["CN", "MCI", "AD"],
        )

    if PATIENT_EVOLUTION:
        plot_patient_evolution(
            DOTENV_KEY2VAL["GEN_DATA_DIR"] + "/temporal_evolution/"
        )

    if PATIENT_EVOLUTION_AVERAGE:
        plot_patient_evolution_average(
            DOTENV_KEY2VAL["GEN_DATA_DIR"] + "/temporal_evolution/"
        )

    if DIVERGENCE_BETWEEN_PDS:
        plot_deviation_from_avg_pl(
            DOTENV_KEY2VAL["GEN_DATA_DIR"] + "/distance_from_average/",
            DOTENV_KEY2VAL["GEN_FIGURES_DIR"],
        )


if __name__ == "__main__":
    main()

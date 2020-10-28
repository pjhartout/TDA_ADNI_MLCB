#!/usr/bin/env python
# coding: utf-8

"""mapper_patches.py
This is a very handwavy implementation of the mapper pipeline for patch data.

TODO:
    - check if the parameters can be tweaked for higher dimensions of the PCA.
    - Are there other useful knobs to turn? Use persistence data as input?
"""

__author__ = "Philip Hartout"
__email__ = "philip.hartout@protonmail.com"

import matplotlib.pyplot as plt


import dotenv

import numpy as np
import pandas as pd

import plotly.express as px

from gtda.mapper import (
    CubicalCover,
    make_mapper_pipeline,
    Projection,
    plot_static_mapper_graph,
)

from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import OneHotEncoder

# Import utils library
import utils


DOTENV_KEY2VAL = dotenv.dotenv_values()


def main():
    directory = DOTENV_KEY2VAL["DATA_DIR"]
    image_dir = directory + "/patch_92/"
    diagnosis_json = "collected_diagnoses_complete.json"

    (
        cn_patients,
        mci_patients,
        ad_patients,
    ) = utils.get_earliest_available_diagnosis(directory + diagnosis_json)
    images_all = utils.get_arrays_from_dir(
        image_dir, cn_patients + mci_patients + ad_patients
    )

    cn_patient_list = [
        1 for patient in range(len(cn_patients) - 1)
    ]  # substracting one due to unfound MRI for one CN patient
    mci_patient_list = [2 for patient in range(len(mci_patients))]
    ad_patient_list = [3 for patient in range(len(ad_patients))]

    diags = np.array(
        cn_patient_list + mci_patient_list + ad_patient_list
    ).reshape(-1, 1)
    ohe = OneHotEncoder()
    labels = ohe.fit_transform(diags).toarray()

    images = []
    for image in images_all:
        images.append(image.flatten())
    images_all = np.asarray(images)
    pca = PCA(n_components=440)
    pca.fit(images_all)

    fig, ax0 = plt.subplots(nrows=1, sharex=True, figsize=(6, 6))
    ax0.plot(
        np.arange(1, pca.n_components_ + 1),
        pca.explained_variance_ratio_,
        "+",
        linewidth=2,
    )
    ax0.set_ylabel("PCA explained variance ratio")
    ax0.legend(prop=dict(size=12))
    plt.savefig(DOTENV_KEY2VAL["GEN_FIGURES_DIR"] + "elbow_plot.png")

    n_components = 3
    pca = PCA(n_components=n_components)
    images_all_projected = pca.fit_transform(images_all)

    images_all_projected = np.append(images_all_projected, labels, axis=1)

    mapper_pipeline = make_mapper_pipeline(
        filter_func=Projection(columns=[index for index in range(2)]),
        cover=CubicalCover(n_intervals=10, overlap_frac=0.25),
        clusterer=DBSCAN(eps=0.5, min_samples=5),
        verbose=True,
        n_jobs=4,
    )
    plotly_params = {"node_trace": {"marker_colorscale": "Blues"}}
    fig = plot_static_mapper_graph(
        mapper_pipeline,
        images_all_projected,
        layout_dim=3,
        color_by_columns_dropdown=True,
        plotly_params=plotly_params,
    )

    fig.write_html(
        DOTENV_KEY2VAL["GEN_FIGURES_DIR"]
        + "mapper_2_dimensional_reduction.html"
    )

    images_all_projected = pd.DataFrame(images_all_projected)
    fig = px.scatter_3d(
        images_all_projected,
        x=0,
        y=1,
        z=2,
        color=3,
        title="3D scatterplot of the PCA of the image data",
    )
    fig.write_html(
        DOTENV_KEY2VAL["GEN_FIGURES_DIR"] + "scatterplot_pca_3d.html"
    )


if __name__ == "__main__":
    main()

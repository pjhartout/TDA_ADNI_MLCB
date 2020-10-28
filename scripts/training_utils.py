#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""training_utils.py

The aim of this script is to provide a set of utilities functions to the
CNN_using_persistence_images_on_patch.py script.
"""
__author__ = "Philip Hartout"
__email__ = "philip.hartout@protonmail.com"

import nibabel
import nibabel as nib  # Useful to load data

import nilearn
from nilearn import datasets
from nilearn.regions import RegionExtractor
from nilearn import plotting
from nilearn.image import index_img
from nilearn.plotting import find_xyz_cut_coords
from nilearn.input_data import NiftiMapsMasker
from nilearn.connectome import ConnectivityMeasure
from nilearn.regions import RegionExtractor
from nilearn import plotting
from nilearn.image import index_img
from nilearn import datasets
from nilearn import plotting

import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
import dotenv


import numpy as np
import networkx as nx
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff

import gtda
from gtda.images import ImageToPointCloud, ErosionFiltration
from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import (
    PersistenceEntropy,
    PersistenceLandscape,
    PersistenceImage,
)
from gtda.pipeline import Pipeline
from gtda.plotting import plot_diagram, plot_point_cloud, plot_heatmap
from gtda.homology import CubicalPersistence
from gtda.diagrams import (
    Scaler,
    Filtering,
    PersistenceEntropy,
    BettiCurve,
    PairwiseDistance,
)

import os
import tempfile
from urllib.request import urlretrieve
import zipfile

from sklearn.linear_model import LogisticRegression
from skimage import io

import glob
import json

# Import utils library
import utils

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

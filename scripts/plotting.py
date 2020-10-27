#!/usr/bin/env python

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


from pathlib import Path
import dotenv


import numpy as np
import networkx as nx
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go

import gtda
from gtda.images import ImageToPointCloud, ErosionFiltration
from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import PersistenceEntropy, PersistenceLandscape
from gtda.pipeline import Pipeline
from gtda.plotting import plot_diagram, plot_point_cloud, plot_heatmap
from gtda.homology import CubicalPersistence
from gtda.diagrams import (
    Scaler,
    Filtering,
    PersistenceEntropy,
    PersistenceImage,
    HeatKernel,
    Silhouette,
    BettiCurve,
    PairwiseDistance,
    PersistenceEntropy,
    PersistenceLandscape
)

import os
import tempfile
from urllib.request import urlretrieve
import zipfile

from sklearn.linear_model import LogisticRegression
from skimage import io
from plotly.subplots import make_subplots


DOTENV_KEY2VAL = dotenv.dotenv_values()
import sys
sys.path.insert(1, '/home/pjh/Documents/Git/TDA_ADNI_MLCB/scripts/')
import utils
distances_to_evaluate = [
        "wasserstein",
        "betti",
        "landscape",
        "silhouette",
        "heat",
        "persistence_image",
]
for i in distances_to_evaluate:
    utils.plot_density_plots(i)

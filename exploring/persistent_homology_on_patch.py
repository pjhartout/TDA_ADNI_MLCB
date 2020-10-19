#!/usr/bin/env python
import nibabel

from nilearn import datasets
from nilearn.regions import RegionExtractor
from nilearn import plotting
from nilearn.image import index_img
from nilearn.plotting import find_xyz_cut_coords
import matplotlib.pyplot as plt
import nibabel as nib # Useful to load data
import nilearn
from nilearn import datasets
from nilearn import plotting
from pathlib import Path
import dotenv
import gtda
import gtda.images
import numpy as np
import networkx as nx
import pandas as pd
from nilearn.input_data import NiftiMapsMasker
from nilearn.connectome import ConnectivityMeasure
from nilearn.regions import RegionExtractor
from nilearn import plotting
from nilearn.image import index_img
from nilearn.plotting import find_xyz_cut_coords

import plotly.express as px
import plotly.graph_objects as go

from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import PersistenceEntropy
from gtda.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from gtda.plotting import plot_diagram, plot_point_cloud, plot_heatmap
from gtda.homology import CubicalPersistence
from gtda.diagrams import Scaler, Filtering, PersistenceEntropy, BettiCurve, PairwiseDistance

import os
import tempfile
from urllib.request import urlretrieve
import zipfile

from skimage import io

patch = np.load("../data/cropped/sub-ADNI002S0295-M00-MNI.npy")

binarized_patch = np.where(patch>0.25, 1, 0)

shape = (1, 30, 36, 30)
point_cloud_tranformer = gtda.images.ImageToPointCloud()
patch_pc = point_cloud_tranformer.fit_transform(binarized_patch.reshape(shape))

homology_dimensions = (0, 1, 2)
VR = VietorisRipsPersistence(metric="euclidean", max_edge_length=5, homology_dimensions=homology_dimensions, n_jobs=8)
diagrams_data = VR.fit_transform(np.asarray(patch_pc))
BC = BettiCurve()

# X_betti_curves = BC.fit_transform(diagrams_VietorisRips)
# BC.plot(X_betti_curves)

with open('diagrams_data.npy', 'wb') as f:
    np.save(f, diagrams_data)

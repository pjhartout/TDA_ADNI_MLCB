#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""CNN_using_persistence_images_on_patch.py

The aim of this script is to perform the training of a CNN using persistence
images as a input. This script is heavily inspired from this script
https://github.com/BorgwardtLab/ADNI_MRI_Analysis/blob/mixed_CNN/mixed_CNN/run_Sarah.py.

To get real time information into the model training and structure, run
$ tensorboard --logdir logs/fit

once this script has been started.
"""

__author__ = "Philip Hartout"
__email__ = "philip.hartout@protonmail.com"

import dotenv

import datetime
import os

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.model_selection import train_test_split

import shutil
import utils

DOTENV_KEY2VAL = dotenv.dotenv_values()
tf.random.set_seed(42)


def make_model(input_shape, num_classes):
    """Makes a keras model.

    Args:
        input_shape (tuple): input shape of the neural network
        num_classes (int): number of classes involved

    Returns:
        keral.Model: model ready to be trained
    """
    inputs = keras.Input(shape=input_shape)

    x = layers.Conv2D(100, kernel_size=5, padding="valid", activation="relu")(
        inputs
    )
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalMaxPooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inputs, outputs)


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
    images_cn = utils.get_arrays_from_dir(persistence_image_location, cn_patients)
    images_ad = utils.get_arrays_from_dir(persistence_image_location, ad_patients)
    # Concatenate both arrays
    labels = np.array(
        [0 for i in range(images_cn.shape[0])]  # Labels for CN is 0
        + [1 for i in range(images_ad.shape[0])]  # Labels for AD is 2
    ).T
    images = np.vstack((images_ad, images_cn))

    X_train, X_test, y_train, y_test = train_test_split(
        images.reshape(images.shape[0], 100, 100, 3),
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )
    train_dataset = tf.data.Dataset.from_tensors(
        (tf.convert_to_tensor(X_train), tf.convert_to_tensor(y_train))
    )
    test_dataset = tf.data.Dataset.from_tensors(
        (tf.convert_to_tensor(X_test), tf.convert_to_tensor(y_test))
    )

    ############################################################################
    #  Model definition
    ############################################################################

    model = make_model(input_shape=(100, 100, 3), num_classes=2)
    keras.utils.plot_model(model, show_shapes=True)

    ############################################################################
    #  Model definition
    ############################################################################

    epochs = 300

    tensorboard_logs = "logs/fit"
    if os.path.exists(tensorboard_logs):
        shutil.rmtree(tensorboard_logs)

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    callbacks = [
        # keras.callbacks.ModelCheckpoint("save_at_{epoch}.h5"),
        tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    ]
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=0.05,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07,
            amsgrad=False,
        ),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        train_dataset,
        epochs=epochs,
        callbacks=callbacks,
        validation_data=test_dataset,
    )

    ############################################################################
    #  Model evaluation
    ############################################################################

    # Mosly already included into the traing procedure.

if __name__ == "__main__":
    main()

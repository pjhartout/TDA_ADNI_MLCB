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
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.model_selection import train_test_split

# Import utils library
import utils
import training_utils


DOTENV_KEY2VAL = dotenv.dotenv_values()


def make_model(input_shape, num_classes):
    """ This implements the Xception network architecture
    """
    inputs = keras.Input(shape=input_shape)

    # Entry block
    x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [128, 256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    if num_classes == 2:
        activation = "sigmoid"
        units = 1
    else:
        activation = "softmax"
        units = num_classes

    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(units, activation=activation)(x)
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
    images_cn = training_utils.get_arrays_from_dir(
        persistence_image_location, cn_patients
    )
    images_ad = training_utils.get_arrays_from_dir(
        persistence_image_location, ad_patients
    )
    # Concatenate both arrays
    labels = np.array(
        [0 for i in range(images_cn.shape[0])]  # Labels for CN is 0
        + [2 for i in range(images_ad.shape[0])]  # Labels for AD is 2
    ).T
    images = np.vstack((images_ad, images_cn))

    X_train, X_test, y_train, y_test = train_test_split(
        images.reshape(images.shape[0],100,100,3), labels, test_size=0.2,
        random_state=42, stratify=labels,
    )
    # X_train =
    # X_test =
    # y_train =
    # y_test =
    train_dataset = tf.data.Dataset.from_tensors(
        (tf.convert_to_tensor(X_train), tf.convert_to_tensor(y_train))
    )
    test_dataset = tf.data.Dataset.from_tensors(
        (tf.convert_to_tensor(X_test), tf.convert_to_tensor(y_test))
    )

    ############################################################################
    #  Model definition
    ############################################################################

    model = make_model(input_shape=(100,100,3), num_classes=2)
    keras.utils.plot_model(model, show_shapes=True)

    ############################################################################
    #  Model definition
    ############################################################################

    epochs = 50

    callbacks = [
        keras.callbacks.ModelCheckpoint("save_at_{epoch}.h5"),
    ]
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
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


if __name__ == "__main__":
    main()

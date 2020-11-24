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
import random
import datetime
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from itertools import islice
from sklearn.model_selection import train_test_split, GroupKFold
import pydot
import shutil
import utils


print(tf.test.gpu_device_name())
print("DEVICE INFO")

DOTENV_KEY2VAL = dotenv.dotenv_values()
tf.random.set_seed(42)
N_BINS = 100

################################################################################
#  Functions
################################################################################

persistence_image_location = (
    DOTENV_KEY2VAL["DATA_DIR"] + "/patch_91_persistence_images/"
)
partitions_location = DOTENV_KEY2VAL["DATA_DIR"] + "/partitions/"
diagnosis_json = (
    DOTENV_KEY2VAL["DATA_DIR"] + "/collected_diagnoses_complete.json"
)


def take(n, iterable):
    "Return first n items of the iterable as a list"
    return dict(islice(iterable, n))


def cleaning(image_labels):
    """Quick sanity check to check if files exist prior to declaring in
    generator"""
    print("Performing sanity check for existence of all files.")
    image_labels_cleaned = image_labels.copy()
    counts = 0
    for image in image_labels.keys():
        try:
            X = np.load(persistence_image_location + image)
        except FileNotFoundError:
            counts = counts + 1
            del image_labels_cleaned[image]
    print(f"{counts} images not found, removed from data labels.")
    return image_labels_cleaned


def make_model(input_shape):
    """Makes a keras model.

    Args:
        input_shape (tuple): input shape of the neural network
        num_classes (int): number of classes involved

    Returns:
        keral.Model: model ready to be trained
    """
    inputs = keras.Input(shape=input_shape)

    tower_1 = layers.Conv2D(2, 4, padding="same", activation="relu")(
        inputs[:, :, :, 0:1]
    )
    tower_1 = layers.BatchNormalization()(tower_1)
    tower_1 = layers.MaxPooling2D()(tower_1)

    tower_2 = layers.Conv2D(2, 4, padding="same", activation="relu")(
        inputs[:, :, :, 1:2]
    )
    tower_2 = layers.BatchNormalization()(tower_2)
    tower_2 = layers.MaxPooling2D()(tower_2)

    tower_3 = layers.Conv2D(2, 4, padding="same", activation="relu")(
        inputs[:, :, :, 2:]
    )
    tower_3 = layers.BatchNormalization()(tower_3)
    tower_3 = layers.MaxPooling2D()(tower_3)

    merged = layers.concatenate([tower_1, tower_2, tower_3], axis=1)
    merged = layers.Flatten()(merged)
    x = layers.Dense(500, activation="relu")(merged)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(500, activation="relu")(merged)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inputs, outputs)


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
    inits = 3
    partitions, labels = get_partitions(partitions_location)
    histories = []
    for partition, label in zip(partitions, labels):
        for i in range(inits):
            # Make sure there aren't the same patients in train and test
            X_train_lst = []
            y_train_lst = []
            for image in partition["train"]:
                X_train_lst.append(
                    np.load(persistence_image_location + image + ".npy")
                )
                y_train_lst.append(label[image])

                X_train, y_train = (
                    np.stack(X_train_lst, axis=0).reshape(
                        len(X_train_lst), N_BINS, N_BINS, 3
                    ),
                    np.vstack(y_train_lst),
                )
            X_val_lst = []
            y_val_lst = []
            for image in partition["validation"]:
                X_val_lst.append(
                    np.load(persistence_image_location + image + ".npy")
                )
                y_val_lst.append(label[image])

                X_val, y_val = (
                    np.stack(X_val_lst, axis=0).reshape(
                        len(X_val_lst), N_BINS, N_BINS, 3
                    ),
                    np.vstack(y_val_lst),
                )

            ####################################################################
            #  Model definition
            ####################################################################

            model = make_model(input_shape=(N_BINS, N_BINS, 3))
            tf.keras.utils.plot_model(
                model,
                to_file="model.png",
                show_shapes=True,
                show_layer_names=True,
                rankdir="TB",
                expand_nested=True,
                dpi=96,
            )

            ####################################################################
            #  Model training
            ####################################################################

            epochs = 100

            tensorboard_logs = "logs/fit"
            if os.path.exists(tensorboard_logs):
                shutil.rmtree(tensorboard_logs)

            log_dir = "logs/fit/" + datetime.datetime.now().strftime(
                "%Y%m%d-%H%M%S"
            )
            callbacks = [
                tf.keras.callbacks.TensorBoard(
                    log_dir=log_dir, histogram_freq=1
                ),
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    min_delta=0.00001,
                    patience=10,
                    verbose=0,
                    mode="auto",
                    baseline=None,
                    restore_best_weights=True,
                ),
            ]
            lr = keras.optimizers.schedules.ExponentialDecay(
                0.01, decay_steps=30, decay_rate=0.6, staircase=True
            )
            model.compile(
                optimizer=keras.optimizers.Adam(
                    learning_rate=lr,
                    beta_1=0.9,
                    beta_2=0.999,
                    epsilon=1e-07,
                    amsgrad=False,
                ),
                loss="binary_crossentropy",
                metrics=[
                    keras.metrics.BinaryAccuracy(name="accuracy"),
                    keras.metrics.Precision(name="precision"),
                    keras.metrics.Recall(name="recall"),
                    keras.metrics.AUC(name="auc"),
                ],
                # run_eagerly=True,
            )
            history = model.fit(
                X_train,
                y_train,
                epochs=epochs,
                callbacks=callbacks,
                batch_size=16,
                validation_data=(X_val, y_val),
            )
            histories.append(history)
    ############################################################################
    #  Model evaluation
    ############################################################################
    # Mosly already included into the training procedure.
    last_acc = []
    last_val_acc = []
    last_val_prec = []
    last_val_rec = []
    last_val_auc = []
    for hist in histories:
        last_acc.append(hist.history["accuracy"][-1])
        last_val_acc.append(hist.history["val_accuracy"][-1])
        last_val_prec.append(hist.history["val_precision"][-1])
        last_val_rec.append(hist.history["val_recall"][-1])
        last_val_auc.append(hist.history["val_auc"][-1])
    print(
        f"The mean training accuracy over the folds is {np.mean(last_acc)}, pm {np.std(last_acc)}"
    )
    print(
        f"The mean validation accuracy over the folds is {np.mean(last_val_acc)}, pm {np.std(last_val_acc)}"
    )
    print(
        f"The mean validation precision over the folds is {np.mean(last_val_prec)}, pm {np.std(last_val_prec)}"
    )
    print(
        f"The mean validation recall over the folds is {np.mean(last_val_rec)}, pm {np.std(last_val_rec)}"
    )
    print(
        f"The mean validation auc over the folds is {np.mean(last_val_auc)}, pm {np.std(last_val_auc)}"
    )

    ############################################################################
    #  Model evaluation
    ############################################################################
    # Here we actually extract the id of the samples that are misclassified
    y_pred = model.predict(X_train)
    difference = np.round(y_train - y_pred)
    index = np.nonzero(difference)
    y_pred = model.predict(X_val)
    difference = np.round(y_val - y_pred)
    index_2 = np.nonzero(difference)
    df_misclassified_train = pd.DataFrame(
        np.array(partitions[0]["train"])[index[0]]
    )
    df_misclassified_val = pd.DataFrame(
        np.array(partitions[0]["validation"])[index_2[0]]
    )
    df_misclassified = pd.concat(
        [df_misclassified_train, df_misclassified_val]
    )
    df_misclassified.to_csv(
        DOTENV_KEY2VAL["GEN_DATA_DIR"] + "misclassification.csv"
    )


if __name__ == "__main__":
    main()

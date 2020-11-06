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

# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)
# gpus = tf.config.experimental.list_physical_devices("GPU")
# if gpus:
#     try:
#         # Currently, memory growth needs to be the same across GPUs
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#         logical_gpus = tf.config.experimental.list_logical_devices("GPU")
#         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#     except RuntimeError as e:
#         # Memory growth must be set before GPUs have been initialized
#         print(e)


# partition = {}
# partition["validation"] = list(take(100, image_labels.items()).keys())
# partition["train"] = [
#     x for x in image_labels if x not in partition["validation"]
# ]
#
# batch_size = 64
#
#
# def create_batch(lst, size):
#     batch_list = [
#         lst[i * size : (i + 1) * size]
#         for i in range((len(lst) + size - 1) // size)
#     ]
#     del batch_list[-1]
#     return batch_list
#
#
# partition["train"] = create_batch(partition["train"], batch_size)
# partition["validation"] = create_batch(partition["validation"], batch_size)
#
#
# def train_gen():
#     for batch in partition["train"]:
#         X = np.empty((batch_size, 100, 100, 3))
#         y = np.empty((batch_size, 1))
#         Xs = []
#         ys = []
#         for image in batch:
#             Xs.append(
#                 np.load(persistence_image_location + image).reshape(
#                     1, 100, 100, 3
#                 )
#             )
#             ys.append(image_labels[image])
#
#         yield np.vstack(Xs), np.vstack(ys)
#
#
# def val_gen():
#     for batch in partition["validation"]:
#         X = np.empty((batch_size, 100, 100, 3))
#         y = np.empty((batch_size, 1))
#         Xs = []
#         ys = []
#         for image in batch:
#             Xs.append(
#                 np.load(persistence_image_location + image).reshape(
#                     1, 100, 100, 3
#                 )
#             )
#             ys.append(image_labels[image])
#
#         yield np.vstack(Xs), np.vstack(ys)


################################################################################
#  Classes
################################################################################


# class DataGenerator(keras.utils.Sequence):
#     "Generates data for Keras"
#     """
#     https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
#     """
#
#     def __init__(
#         self,
#         list_IDs,
#         labels,
#         batch_size=32,
#         dim=(100, 100, 3),
#         n_classes=2,
#         shuffle=True,
#     ):
#         "Initialization"
#         self.dim = dim
#         self.batch_size = batch_size
#         self.labels = labels
#         self.list_IDs = list_IDs
#         self.n_classes = n_classes
#         self.shuffle = shuffle
#         self.on_epoch_end()
#
#     def __len__(self):
#         "Denotes the number of batches per epoch"
#         return int(np.floor(len(self.list_IDs) / self.batch_size))
#
#     def __getitem__(self, index):
#         "Generate one batch of data"
#         # Generate indexes of the batch
#         indexes = self.indexes[
#             index * self.batch_size : (index + 1) * self.batch_size
#         ]
#
#         # Find list of IDs
#         list_IDs_temp = [self.list_IDs[k] for k in indexes]
#
#         # Generate data
#         X, y = self.__data_generation(list_IDs_temp)
#
#         yield X, y
#
#     def on_epoch_end(self):
#         "Updates indexes after each epoch"
#         self.indexes = np.arange(len(self.list_IDs))
#         if self.shuffle == True:
#             np.random.shuffle(self.indexes)
#
#     def __data_generation(self, list_IDs_temp):
#         "Generates data containing batch_size samples"  # X : (n_samples, *dim, n_channels)
#         # Initialization
#         X = np.empty((self.batch_size, *self.dim))
#         y = np.empty((self.batch_size), dtype=int)
#
#         # Generate data
#         for i, ID in enumerate(list_IDs_temp):
#             # Store sample
#             X[i,] = np.load(persistence_image_location + ID).reshape(100, 100, 3)
#             # Store class
#             y[i] = self.labels[ID]
#
#         return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
#

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
    x_conv_result = []
    for i in range(inputs.shape[3]):
        if i != inputs.shape[3]:
            x = layers.Conv2D(
                5,
                kernel_size=10,
                strides=(5, 5),
                padding="valid",
                activation="relu",
            )(inputs[:, :, :, i : i + 1])
        else:
            x = layers.Conv2D(
                5,
                kernel_size=10,
                strides=(5, 5),
                padding="valid",
                activation="relu",
            )(inputs[:, :, :, i:])
        x = layers.BatchNormalization()(x)
        x_conv_result.append(
            layers.GlobalMaxPooling2D()(tf.reshape(x, (-1, -1, 100, 100)))
        )

    x = layers.Dense(600, activation="relu")(tf.concat(x_conv_result, 1))
    outputs = layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inputs, outputs)


def make_simpler_model(input_shape):
    """Makes a keras model.

    Args:
        input_shape (tuple): input shape of the neural network
        num_classes (int): number of classes involved

    Returns:
        keral.Model: model ready to be trained
    """
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(
        20, kernel_size=10, strides=(1, 1), padding="valid", activation="relu"
    )(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalMaxPooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(100, activation="relu")(x)
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

    # First, we load relevant images (only baseline image for now)
    # (
    #     cn_patients,
    #     mci_patients,
    #     ad_patients,
    # ) = utils.get_all_available_diagnoses(diagnosis_json)

    # image_names = cn_patients + ad_patients
    # raw_labels = np.array(
    #     [0 for i in range(len(cn_patients))]  # Labels for CN is 0
    #     + [1 for i in range(len(ad_patients))]  # Labels for AD is 1
    # )
    # ratio_cn_ad = np.bincount(raw_labels)[1] / np.bincount(raw_labels)[0]
    # print(f"Ratio of AD/CN patients is {ratio_cn_ad}")

    # image_labels = {}
    # for i, j in zip(image_names, list(raw_labels)):
    #     image_labels[i] = j
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
                        len(X_train_lst), 100, 100, 3
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
                        len(X_val_lst), 100, 100, 3
                    ),
                    np.vstack(y_val_lst),
                )

            # group_kfold = GroupKFold(n_splits=2)

            # groups = list(map(lambda x: str(x.split("-")[1:2][0]), image_labels.keys()))
            # mapping = {}
            # for i, patient in enumerate(set(groups)):
            #     mapping[patient] = i
            # groups = [mapping[patient] for patient in groups]

            # for train_index, test_index in group_kfold.split(
            #     X, y, groups
            # ):
            #     X_train, X_test = X[train_index,:,:,:], X[test_index,:,:,:]
            #     y_train, y_test = y[train_index], y[test_index]

            # X_train, X_test, y_train, y_test = train_test_split(
            #     ,
            #     ,
            #     test_size=100,
            #     random_state=42,
            # )
            # list(training_generator.take(2).as_numpy_iterator())

            # validation_generator = DataGenerator(
            #     partition["validation"], image_labels, **params
            # )

            # next(train_data_generator)
            ############################################################################
            #  Model definition
            ############################################################################

            model = make_simpler_model(input_shape=(100, 100, 3))
            keras.utils.plot_model(model, show_shapes=True)

            ############################################################################
            #  Model training
            ############################################################################

            epochs = 100

            tensorboard_logs = "logs/fit"
            if os.path.exists(tensorboard_logs):
                shutil.rmtree(tensorboard_logs)

            log_dir = "logs/fit/" + datetime.datetime.now().strftime(
                "%Y%m%d-%H%M%S"
            )
            callbacks = [
                # keras.callbacks.ModelCheckpoint("save_at_{epoch}.h5"),
                tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1),
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    min_delta=0.001,
                    patience=2,
                    verbose=0,
                    mode="auto",
                    baseline=None,
                    restore_best_weights=False,
                ),
            ]
            lr = keras.optimizers.schedules.ExponentialDecay(
                0.01, decay_steps=100, decay_rate=0.6, staircase=True
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
                    keras.metrics.TruePositives(name="tp"),
                    keras.metrics.FalsePositives(name="fp"),
                    keras.metrics.TrueNegatives(name="tn"),
                    keras.metrics.FalseNegatives(name="fn"),
                    keras.metrics.Precision(name="precision"),
                    keras.metrics.Recall(name="recall"),
                    keras.metrics.AUC(name="auc"),
                ],
                # run_eagerly=True,
            )
            history =  model.fit(
                X_train,
                y_train,
                epochs=epochs,
                callbacks=callbacks,
                batch_size=32,
                validation_data=(X_val, y_val),
            )
            histories.append(history)
            ############################################################################
            #  Model evaluation
            ############################################################################
            # Mosly already included into the training procedure.
    # Stepping back out of the partitions loop
    last_acc = []
    for hist in histories:
        last_acc.append(hist.history["accuracy"][-1])
    print(f"The mean validation accuracy over the  folds is {np.mean(last_acc)}")
    ############################################################################
    #  Model evaluation
    ############################################################################
    # Here we actually extract the id of the samples that are misclassified
    y_pred = model.predict(X_train)
    difference = np.round(y_train - y_pred)
    index = np.nonzero(difference)
    df_misclassified = pd.DataFrame(
        np.array(partitions[0]["train"])[index[0]]
    )
    df_misclassified.to_csv(
        DOTENV_KEY2VAL["GEN_DATA_DIR"] + "misclassification.csv"
    )


if __name__ == "__main__":
    main()

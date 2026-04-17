"""
cnn_classifier.py
Neural network classifier.

Copyright(C) 2021-2024 by
Trey V. Wenger; tvwenger@gmail.com

GNU General Public License v3 (GNU GPLv3)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published
by the Free Software Foundation, either version 3 of the License,
or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

Trey Wenger - October 2021
"""

import os
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from keras import models, layers, callbacks
from keras.optimizers import Adam
from keras.losses import CategoricalCrossentropy
from scipy.optimize import minimize

from igps_kdar.utils import calc_ce, predict_calibrated


def build_cnn(
    input_shape,
    n_outputs,
    filter_sizes,
    kernel_sizes,
    dense_layers,
    dropout_frac=None,
    init_lr=0.01,
):
    """
    Build a new CNN model.

    Inputs:
        input_shape :: tuple of integers
            Shape of input data
        n_outputs :: integer
            Number of output categories
        filter_sizes :: list of integers
            The filter size of each convolution layer
        kernel_sizes :: list of integers
            Each element is one convolutional layer with the given size.
            The size of this list must match that of filter_sizes
        dense_layers :: list of intengers
            Each element is one dense layer with the given size
        dropout_frac :: scalar
            If not None, add dropout layers with this fraction in (0, 1) between
            each dense layer.
        init_lr :: scalar
            Initial learning rate

    Returns: model
        model :: tensorflow.keras.models.Model
            The CNN model
    """
    if len(filter_sizes) != len(kernel_sizes):
        raise ValueError("filter_sizes and kernel_sizes mismatch")

    model = models.Sequential()

    # Convolution layers
    for i, (filters, kernel_size) in enumerate(zip(filter_sizes, kernel_sizes)):
        kwargs = {}
        # pass input_shape to first layer
        if i == 0:
            kwargs["input_shape"] = input_shape
        model.add(
            layers.Conv1D(
                filters,
                int(kernel_size),
                strides=1,
                padding="same",
                activation="relu",
                **kwargs,
            )
        )
        model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Flatten())

    # dense layers and dropout
    for size in dense_layers:
        model.add(layers.Dense(size, activation="relu"))
        if dropout_frac is not None:
            model.add(layers.Dropout(dropout_frac))

    # Output layer
    model.add(layers.Dense(n_outputs, activation="softmax"))

    # compile
    opt = Adam(learning_rate=init_lr)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    return model


def build_resnet(
    input_shape,
    n_outputs,
    num_blocks,
    kernel_sizes,
    batch_norm=False,
    init_lr=0.01,
):
    """
    Build a new ResNet model.

    Inputs:
        input_shape :: tuple of integers
            Shape of input data
        n_outputs :: integer
            Number of output categories
        num_blocks :: integer
            Number of residual blocks
        kernel_sizes :: list of integers
            Each element is one convolutional layer with the given size
        batch_norm :: boolean
            If True, add batch normalization after every layer
        init_lr :: scalar
            Initial learning rate

    Returns: model
        model :: tensorflow.keras.models.Model
            The ResNet model
    """
    # build the ResNet
    input_layer = layers.Input(shape=input_shape)
    starting_filters = input_shape[-1]

    # Add the residual blocks
    last_block = input_layer
    for i in range(num_blocks):
        # double filters each block
        filters = 2**i * starting_filters

        # Convolution layers
        last_layer = last_block
        for j, kernel_size in enumerate(kernel_sizes):
            conv = layers.Conv1D(filters, int(kernel_size), padding="same")(last_layer)
            if batch_norm:
                conv = layers.BatchNormalization()(conv)

            # skip activation on last layer
            if j < len(kernel_sizes) - 1:
                conv = layers.Activation("relu")(conv)
            last_layer = conv

        # Shortcut
        if i > 0:
            shortcut = layers.Conv1D(filters, 1, padding="same")(last_block)
            if batch_norm:
                shortcut = layers.BatchNormalization()(shortcut)
        elif batch_norm:
            shortcut = layers.BatchNormalization()(last_block)
        else:
            shortcut = last_block
        output = layers.add([shortcut, last_layer])

        # Final activation for this block
        output = layers.Activation("relu")(output)
        last_block = output

    # Add the dense layer
    gmp_layer = layers.GlobalMaxPooling1D()(last_layer)
    output_layer = layers.Dense(n_outputs, activation="softmax")(gmp_layer)

    # compile
    model = models.Model(inputs=input_layer, outputs=output_layer)
    opt = Adam(learning_rate=init_lr)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    return model


class CNNClassifier:
    """
    Convolutional neural network classifier.
    """

    def __init__(
        self,
        fname,
        model=None,
        verbose=False,
    ):
        """
        Initialize a new classifier.

        Inputs:
            fname :: string
                HD5 file where CNN model data are stored.
            model :: keras.Model
                Compiled keras model.
            verbose :: boolean
                If True, print verbose information

        Returns: classifier
            classifier :: Classifier
                A new Classifier instance
        """
        self.fname = fname
        self.model = model
        self.verbose = verbose

        if not os.path.exists(fname):
            if model is None:
                raise ValueError(f"No model found in {fname}, must supply model")
            self.model = model
        elif os.path.exists(fname):
            if model is not None:
                raise ValueError(f"Will not overwrite model in {fname}")
            self.model = models.load_model(fname)

        if self.verbose:
            self.model.summary()

        # add callbacks
        self.callbacks = [
            callbacks.ReduceLROnPlateau(factor=0.5, patience=10, min_lr=1.0e-6),
            callbacks.EarlyStopping(patience=20, start_from_epoch=10),
            callbacks.ModelCheckpoint(filepath=self.fname, save_best_only=True),
        ]

    def fit(self, trainX, trainy, valX, valy, batch_size=128, epochs=100):
        """
        Train the classifier.

        Inputs:
            trainX :: ND-array of scalars
                The training feature set
            trainy :: 1-D array of scalars
                Train training label set
            valX :: nd-array of scalars
                The validation feature set
            valy :: 1-D array of scalars
                The validation label set
            batch_size :: integer
                Batch size
            epochs :: integer
                Number of epochs

        Returns: Nothing
        """
        self.model.fit(
            trainX,
            trainy,
            batch_size=batch_size,
            epochs=epochs,
            verbose=self.verbose,
            validation_data=(valX, valy),
            callbacks=self.callbacks,
        )

    def score(self, testX, testy):
        """
        Calculate the accuracy of the classifier for some test data.

        Inputs:
            testX :: nd-array of scalars
                The test feature set
            testy :: 1-D array of scalars
                The test label set

        Returns:
            score :: scalar
                the accuracy
        """
        proby = self.model.predict(testX)
        predy = tf.one_hot(tf.argmax(proby, axis=1), depth=proby.shape[1]).numpy()
        score = np.all(testy == predy, axis=1).mean()
        return score

    def calibrate(self, valX, valy, calplot=None, nbins=10):
        """
        Calibrate model probabilities by determining the temperature that minimizes
        the loss for the validation set.

        Inputs:
            valX :: nd-array of scalars
                The validation feature set. Must be the same used during training.
            valy :: 1-D array of scalars
                The validation label set. Must be the same used during training.
            calplot :: string
                If not None, plot the binned accuracy vs. confidence before and
                after calibration.
            nbins :: integer
                Number of probability bins.

        Returns:
            temp :: scalar
                Temperature that calibrates probabilities
        """
        cce = CategoricalCrossentropy()
        proby = self.model.predict(valX)

        old_loss = cce(valy, proby).numpy()
        old_ece, old_mce, bins, old_bin_pop, old_bin_accuracy = calc_ce(
            proby, valy, nbins=nbins
        )

        def loss(temp):
            scaled_proby = predict_calibrated(proby, temperature=temp)
            return cce(valy, scaled_proby).numpy()

        res = minimize(loss, 1.0, bounds=[(0.0, np.inf)])
        temp = res.x[0]

        new_proby = predict_calibrated(proby, temperature=temp)
        new_loss = cce(valy, new_proby).numpy()
        new_ece, new_mce, bins, new_bin_pop, new_bin_accuracy = calc_ce(
            new_proby, valy, nbins=nbins
        )

        if calplot is not None:
            fig, (ax1, ax2) = plt.subplots(2, figsize=(5, 8))
            bin_width = bins[1] - bins[0]
            bin_center = bins[:-1] + (bins[1:] - bins[:-1]) / 2
            ax1.bar(
                bins[:-1],
                old_bin_accuracy,
                width=bin_width,
                color="gray",
                align="edge",
                edgecolor="k",
                linewidth=0.1,
            )
            ax1.plot([0, 1], [0, 1], "k--")
            for x, num in zip(bin_center, old_bin_pop):
                ax1.text(
                    x,
                    0.05,
                    num,
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
            ax1.text(
                0.05,
                0.95,
                f"ECE = {100*old_ece:.1f}%\n"
                + f"MCE = {100*old_mce:.1f}%\n"
                + f"Loss = {old_loss:.3f}",
                ha="left",
                va="top",
                bbox=dict(boxstyle="round", alpha=0.8, color="white"),
            )
            ax2.bar(
                bins[:-1],
                new_bin_accuracy,
                width=bin_width,
                color="gray",
                align="edge",
                edgecolor="k",
                linewidth=0.1,
            )
            for x, num in zip(bin_center, new_bin_pop):
                ax2.text(
                    x,
                    0.05,
                    num,
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
            ax2.text(
                0.05,
                0.95,
                f"ECE = {100*new_ece:.1f}%\n"
                + f"MCE = {100*new_mce:.1f}%\n"
                + f"Loss = {new_loss:.3f}\n"
                + f"T = {temp:.3f}",
                ha="left",
                va="top",
                bbox=dict(boxstyle="round", alpha=0.8, color="white"),
            )
            ax2.plot([0, 1], [0, 1], "k--")
            ax1.grid(False)
            ax2.grid(False)
            ax1.set_xlim(0, 1)
            ax2.set_xlim(0, 1)
            ax1.set_ylim(0, 1)
            ax2.set_ylim(0, 1)
            ax1.set_xlabel("Confidence")
            ax2.set_xlabel("Probability")
            ax1.set_ylabel("Accuracy")
            ax2.set_ylabel("Accuracy")
            fig.tight_layout()
            fig.savefig(calplot, bbox_inches="tight")
            plt.close(fig)
        return temp

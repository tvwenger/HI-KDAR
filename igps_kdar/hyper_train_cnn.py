"""
hyper_train_cnn.py
Determine the optimal hyperparameters for a convolutional neural network (CNN)
to resolve the kinematic distance ambiguity using HI absorption spectra.

Copyright(C) 2024 by
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

Trey Wenger - April 2024
"""

import argparse

import numpy as np

from keras import callbacks
import kerastuner as kt
from kerastuner.tuners import Hyperband

from igps_kdar import __version__
from igps_kdar.cnn_classifier import build_cnn, build_resnet
from igps_kdar.train_cnn import load_data


def hypertrain_cnn(
    datafiles,
    model_dir="./",
    model_name="hypertrain",
    model_type="cnn",
    tangent=False,
    test_frac=0.2,
    val_frac=0.2,
    num_filters=[3, 5],
    base_filter_size=[16, 32],
    num_dense=[1, 2],
    base_dense_size=[128, 512],
    equal_kernel_size=[True, False],
    regularize=[True],
    dropout_frac=[0.2],
    num_blocks=[2, 3],
    num_layers=[3, 5],
    init_lr=[1.0e-2, 1.0e-3],
    batch_size=[16, 64],
    max_epochs=100,
    seed=1234,
    verbose=False,
):
    """
    Tune the CNN/Resnet hyperparameters.

    Inputs:
        datafiles :: list of strings
            Files containing HI absorption data
        model_type :: string
            One of "cnn" or "resnet".
        tangent :: boolean
            If True, train on tangent labels (in addition to N/F) for real data.
        test_frac :: scalar
            Fraction of data reserved for testing.
        val_frac :: scalar
            Fraction of non-testing data reserved for validation.
        num_filters :: list of integers
            CNN only: Number of filters to tune.
        base_filter_size :: list of integers
            CNN only: Base filter size to tune.
            Filter sizes are base_filter_size * 2 ** np.arange(num_filters)
        num_dense :: list of integers
            CNN only: Number of dense layers to tune.
        base_dense_size :: list of integers
            CNN only: Base dense layer size to tune.
            Dense layer sizes are base_dense_size / 2 ** np.arange(num_filters)
        equal_kernel_size :: list of boolean
            Resnet or CNN: Flag to use equal kernel sizes, to tune.
            If True, each kernel size is 3.
            If False, kernel size is 3 + 2 * np.arange(num_filters)
        regularize :: list of boolean
            Resnet or CNN: Flag to use regularization (dropout or pooling), to tune.
        dropout_frac :: list of scalars
            CNN only: Dropout fraction (0 - 1) to tune.
        num_blocks :: list of integers
            Resnet only: Number of residual blocks to be tuned.
        num_layers :: list of integers
            Resnet only: Number of convolutional layers per block to be tuned.
        init_lr :: list of scalars
            Initial learning rate to be tuned.
        batch_size :: list of scalars
            Batch size to be tuned.
        max_epochs :: integer
            Maximum number of epochs
        seed :: integer
            Random seed
        verbose :: boolean
            If True, print info

    Returns: Nothing
    """
    # Random state
    rng = np.random.RandomState(seed=seed)

    if verbose:
        print("Loading data...")
    (
        data_scaler,
        data_train,
        data_val,
        data_test,
        trainX,
        trainy,
        valX,
        valy,
        testX,
        testy,
    ) = load_data(
        datafiles, rng, test_frac=test_frac, val_frac=val_frac, tangent=tangent
    )
    if verbose:
        print(f"{len(data_train)} spectra in training set")
        print(f"{len(data_val)} spectra in validation set")
        print(f"{len(data_test)} spectra in test set")

    # initialize tuner
    input_shape = trainX.shape[1:]
    n_outputs = trainy.shape[1]

    class HyperModel(kt.HyperModel):
        def build(self, hp):
            init_lr_in = hp.Choice("init_lr", init_lr)

            if model_type == "cnn":
                num_filters_in = hp.Choice("num_filters", num_filters)
                base_filter_size_in = hp.Choice("base_filter_size", base_filter_size)
                # double filter size every other filter. e.g., 32, 32, 64, 64, 128, 128, 256
                filter_sizes_in = base_filter_size_in * 2 ** (
                    np.arange(num_filters_in) // 2
                )
                num_dense_in = hp.Choice("num_dense", num_dense)
                base_dense_size_in = hp.Choice("base_dense_size", base_dense_size)
                # halve dense layer size each layer. e.g., 1024, 512, 256
                dense_layers_in = base_dense_size_in / 2 ** np.arange(num_dense_in)
                equal_kernel_size_in = hp.Choice("equal_kernel_size", equal_kernel_size)
                if equal_kernel_size_in:
                    # fixed kernel size 3
                    kernel_sizes_in = [3] * num_filters_in
                else:
                    # increase kernel size by 2 each filter. e.g., 3, 5, 7, 9
                    kernel_sizes_in = list(3 + 2 * np.arange(num_filters_in))
                regularize_in = hp.Choice("regularize", regularize)
                dropout_frac_in = hp.Choice("dropout_frac", dropout_frac)
                return build_cnn(
                    input_shape,
                    n_outputs,
                    filter_sizes_in,
                    kernel_sizes_in,
                    dense_layers_in,
                    dropout_frac=dropout_frac_in if regularize_in else None,
                    init_lr=init_lr_in,
                )
            else:
                num_blocks_in = hp.Choice("num_blocks", num_blocks)
                num_layers_in = hp.Choice("num_layers", num_layers)
                equal_kernel_size_in = hp.Choice("equal_kernel_size", equal_kernel_size)
                if equal_kernel_size_in:
                    # fixed kernel size 3
                    kernel_sizes_in = [3] * num_layers_in
                else:
                    # increase kernel size by 2 each filter. e.g., 3, 5, 7, 9
                    kernel_sizes_in = list(3 + 2 * np.arange(num_layers_in))
                regularize_in = hp.Choice("regularize", regularize)
                return build_resnet(
                    input_shape,
                    n_outputs,
                    num_blocks_in,
                    kernel_sizes_in,
                    batch_norm=regularize_in,
                    init_lr=init_lr_in,
                )

        def fit(self, hp, model, *args, **kwargs):
            batch_size_in = hp.Choice("batch_size", batch_size)
            return model.fit(
                *args,
                batch_size=batch_size_in,
                **kwargs,
            )

    tuner = Hyperband(
        HyperModel(),
        objective="val_accuracy",
        max_epochs=max_epochs,
        factor=2,
        directory=model_dir,
        project_name=model_name,
    )
    cbs = [
        callbacks.ReduceLROnPlateau(factor=0.5, patience=10, min_lr=1.0e-6),
        callbacks.EarlyStopping(patience=20, start_from_epoch=10),
    ]
    tuner.search(
        trainX,
        trainy,
        verbose=verbose,
        validation_data=(valX, valy),
        callbacks=cbs,
    )
    print(tuner.results_summary())


def main():
    parser = argparse.ArgumentParser(
        description="Tune CNN hyperparameters a CNN",
        prog="hyper_train_cnn.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=__version__,
    )
    parser.add_argument(
        "--datafiles",
        type=str,
        nargs="+",
        required=True,
        help="Path to input data pickle files",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./",
        help="Directory where training results are saved",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="hypertraining",
        help="Sub-directory where training results are saved",
    )
    parser.add_argument(
        "--tangent",
        action="store_true",
        default=False,
        help="Also train on tangent labels",
    )
    parser.add_argument(
        "--test_frac",
        type=float,
        default=0.2,
        help="Fraction of data reserved for testing",
    )
    parser.add_argument(
        "--val_frac",
        type=float,
        default=0.2,
        help="Fraction of non-testing data reserved for validation",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="cnn",
        help="cnn or resnet",
    )
    parser.add_argument(
        "--num_filters",
        type=int,
        nargs="+",
        default=[3, 5],
        help="CNN only: Number of filters to tune",
    )
    parser.add_argument(
        "--base_filter_size",
        type=int,
        nargs="+",
        default=[16, 32],
        help="CNN only: Base filter size to tune",
    )
    parser.add_argument(
        "--num_dense",
        type=int,
        nargs="+",
        default=[1, 2],
        help="CNN only: Number of dense layers to tune",
    )
    parser.add_argument(
        "--base_dense_size",
        type=int,
        nargs="+",
        default=[128, 512],
        help="CNN only: Base dense layer size",
    )
    parser.add_argument(
        "--equal_kernel_size",
        type=bool,
        nargs="+",
        default=[True, False],
        help="CNN or Resnet: Equal kernel size flag to tune",
    )
    parser.add_argument(
        "--regularize",
        type=bool,
        nargs="+",
        default=[True],
        help="CNN or Resnet: Regularization flag to tune",
    )
    parser.add_argument(
        "--dropout_frac",
        type=float,
        nargs="+",
        default=[0.2],
        help="CNN only: Dropout fraction to tune",
    )
    parser.add_argument(
        "--num_blocks",
        type=int,
        nargs="+",
        default=[2, 3],
        help="Resnet only: Number of residual blocks to tune",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        nargs="+",
        default=[2, 3],
        help="Resnet only: Number of conv layers per block to tune",
    )
    parser.add_argument(
        "--init_lr",
        type=float,
        nargs="+",
        default=[1.0e-2, 1.0e-3],
        help="Initial learning rate to tune",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        nargs="+",
        default=[16, 64],
        help="Batch size to tune",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=100,
        help="Maximum number of epochs",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Print info",
    )
    args = parser.parse_args()
    hypertrain_cnn(**vars(args))


if __name__ == "__main__":
    main()

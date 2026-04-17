"""
train_cnn.py
Train a convolutional neural network (CNN) to resolve the
kinematic distance ambiguity using HI absorption spectra.

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
import argparse
import pickle
import time

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from igps_kdar import __version__
from igps_kdar.cnn_classifier import build_cnn, build_resnet, CNNClassifier
from igps_kdar.utils import smooth_regrid_spec, RobustScaler


def merge_final_data(dataset, merged_data, datafiles):
    """
    This is a silly function that merges the final data needed for training
    into the format
    [datafile_0_spec, datafile_0_rms, datafile_1_spec, datafile_1_rms, ..., ID]

    Inputs:
        dataset :: pandas.DataFrame
            "best_kdar" dataframe for this dataset (i.e., either train, test, or val)
        merged_data :: pandas.DataFrame
            Full merged dataset
        datafiles :: list of strings
            List of datafiles. Training data follow this order.

    Returns:
        X :: nd-array of scalars
            Feature set
        y :: nd-array of scalars
            Label set
    """
    X = np.zeros(
        (len(dataset), len(merged_data.loc[0, "prep_spectrum"]), 2 * len(datafiles) + 1)
    )
    for i, (_, row) in enumerate(dataset.iterrows()):
        for j, datafile in enumerate(datafiles):
            X[i, :, 2 * j] = merged_data.loc[
                (merged_data["gname"] == row["gname"])
                * (merged_data["datafile"] == datafile),
                "prep_spectrum",
            ].to_numpy()[0]
            X[i, :, 2 * j + 1] = merged_data.loc[
                (merged_data["gname"] == row["gname"])
                * (merged_data["datafile"] == datafile),
                "prep_rms",
            ].to_numpy()[0]
        X[i, :, -1] = merged_data.loc[
            (merged_data["gname"] == row["gname"]),
            "prep_id",
        ].to_numpy()[0]
    y = pd.get_dummies(dataset["merged_kdar"])
    return X, y


def load_data(datafiles, rng, test_frac=0.2, val_frac=0.2, tangent=False):
    """
    Load and prepare the data.

    Inputs:
        datafiles :: list of string
            Path to data pickle files. Order of data in output datasets is
            [datafile_0_spec, datafile_0_rms, datafile_1_spec, datafile_1_rms, ..., ID]
        rng :: np.RandomState
            RNG state
        test_frac :: scalar
            Fraction of data reserved for testing.
        val_frac :: scalar
            Fraction of non-testing data reserved for validation.
        tangent :: boolean
            If True, also load data labeled "T"

    Returns:
        data_scaler :: RobustScaler
            Data scaler
        data_train, data_val, data_test :: pd.DataFrame
            Training, validation, and test datasets
        trainX, trainy :: ND-arrays of scalars
            Training features and labels
        valX, valy :: ND-arrays of scalars
            Validation features and labels
        testX, testy :: ND-arrays of scalars
            Test features and labels
    """
    merged_data = None
    for datafile in datafiles:
        print(f"Loading data from {datafile}")

        # Load the data
        with open(datafile, "rb") as f:
            data = pickle.load(f)
        data["datafile"] = datafile

        # Keep best QF per gname
        print(f"{len(data)} entries in datafile")
        data = (
            data.sort_values(by=["kdar_manual_qf"])
            .drop_duplicates("gname", keep="first")
            .sort_values("gname")
            .reset_index(drop=True)
        )
        print(f"{len(data)} entries remain after dropping duplicates")

        for kdar in ["N", "F"]:
            print(f"{np.sum((data['kdar_manual'] == kdar))} assigned {kdar}")
            for qf in ["A", "B", "C"]:
                print(
                    f"{np.sum((data['kdar_manual'] == kdar)*(data['kdar_manual_qf'] == qf))} -> QF {qf}"
                )
        print(f"{np.sum(data['kdar_manual'] == 'T')} assigned T")
        print(f"{np.sum(data['kdar_manual'].isna())} un-assigned")

        # Prepare the dataset
        if merged_data is None:
            merged_data = prepare_data(data)
        else:
            merged_data = pd.concat(
                [merged_data, prepare_data(data)], ignore_index=True
            )

    # Assign KDARs: Select best QF, then most common KDAR
    # sort by QF, get highest QF per gname
    best_kdar = (
        merged_data.sort_values(by=["kdar_manual_qf"])
        .drop_duplicates("gname", keep="first")
        .sort_values("gname")[["gname", "kdar_manual_qf"]]
    )

    # Get most common KDAR with that label
    best_kdar = best_kdar.assign(merged_kdar=[None] * len(best_kdar))
    for idx, row in best_kdar.iterrows():
        kdars = merged_data.loc[
            (merged_data["gname"] == row["gname"])
            & (merged_data["kdar_manual_qf"] == row["kdar_manual_qf"])
        ]["kdar_manual"]
        if len(kdars) == 0:
            continue
        kdar_mode = kdars.mode()[0]
        if not np.all(kdar_mode == kdars):
            print(f"KDAR mismatch: {row['gname']}. Using {kdar_mode}")
        best_kdar.loc[idx, "merged_kdar"] = kdar_mode

    # Drop tangent if necessary
    if not tangent:
        best_kdar.loc[best_kdar["merged_kdar"] == "T"] = None
    best_kdar = best_kdar[~best_kdar["merged_kdar"].isna()]

    # Get training, validation, and test set gnames and labels
    # statify on KDAR+QF
    best_kdar["strat_label"] = best_kdar["merged_kdar"] + best_kdar["kdar_manual_qf"]
    data_train_val, data_test = train_test_split(
        best_kdar,
        test_size=test_frac,
        stratify=best_kdar["strat_label"],
        random_state=rng,
    )
    data_train, data_val = train_test_split(
        data_train_val,
        test_size=val_frac,
        stratify=data_train_val["strat_label"],
        random_state=rng,
    )

    for dataset, datalabel in zip(
        [best_kdar, data_train, data_val, data_test],
        ["all", "training", "validation", "testing"],
    ):
        print(f"Dataset: {datalabel}")
        labels = dataset["merged_kdar"].unique()
        labels.sort()
        for kdar in labels:
            qfs = dataset["kdar_manual_qf"].unique()
            qfs.sort()
            num_qf = [
                (
                    (dataset["merged_kdar"] == kdar) * (dataset["kdar_manual_qf"] == qf)
                ).sum()
                for qf in qfs
            ]
            print(f"KDAR {kdar}: {np.sum(num_qf)}")
            print(f"QFs ({qfs}): {num_qf}")

    # extract features and labels
    # order: [spec_1, rms_1, spec_2, rms_2, ..., id]
    trainX, trainy = merge_final_data(data_train, merged_data, datafiles)
    valX, valy = merge_final_data(data_val, merged_data, datafiles)
    testX, testy = merge_final_data(data_test, merged_data, datafiles)

    # Normalize data, except "ID" axis
    channels = [i for i in range(trainX.shape[-1] - 1)]
    data_scaler = RobustScaler()
    data_scaler.fit(trainX)
    trainX = data_scaler.transform(trainX, channels=channels)
    valX = data_scaler.transform(valX, channels=channels)
    testX = data_scaler.transform(testX, channels=channels)

    return (
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
    )


def prepare_data(data):
    """
    Prepare a dataset by
    (1) Reversing the spectrum for fourth quadrant sources, resampling
    to a velocity axis between -100 and 200 km/s, saving as "prep_spectrum"
    (2) Reversing the rms spectrum for fourth quadrant sources, resampling
    to a velocity axis between -100 and 200 km/s, saving as "prep_rms"
    (3) Creating the "ID" feature spectrum, saving as "prep_id", where
    ID = -1 for velocity < 0
       = velocity/rrl_velocity - 1 for 0 < velocity < rrl_velocity
       = (velocity - rrl_velocity) / (tp_velocity - rrl_velocity) for rrl_velocity < 0 < tp_velocity
       = 1 for velocity > tp_velocity

    Inputs:
        data :: pandas.DataFrame
            Dataset

    Returns:
        newdata :: pandas.DataFrame
            Dataset with new columns added
    """
    new_velocity = np.arange(-100.0, 201.0, 1.0)

    data = data.assign(
        prep_velocity=[new_velocity] * len(data),
        prep_spectrum=[np.zeros_like(new_velocity)] * len(data),
        prep_rms=[np.zeros_like(new_velocity)] * len(data),
        prep_id=[np.zeros_like(new_velocity)] * len(data),
    )
    for idx, row in data.iterrows():
        # reverse order in 4th quadrant
        old_velocity = row["velocity"]
        prep_spectrum = row["spectrum"]
        prep_rms = row["rms"]
        rrl_velocity = row["rrl_velocity"]
        tp_velocity = row["tp_velocity"]
        if row["glong"] > 270.0:
            old_velocity = -1.0 * old_velocity[::-1]
            prep_spectrum = prep_spectrum[::-1]
            prep_rms = prep_rms[::-1]
            rrl_velocity *= -1.0
            tp_velocity *= -1.0

        # resample
        prep_spectrum = smooth_regrid_spec(prep_spectrum, old_velocity, new_velocity)
        prep_rms = smooth_regrid_spec(prep_rms, old_velocity, new_velocity)

        # catch NaN, replace with zeros
        isnan = np.isnan(prep_spectrum) + np.isnan(prep_rms)
        prep_spectrum[isnan] = 0.0
        prep_rms[isnan] = 0.0

        prep_id = np.zeros_like(new_velocity)
        prep_id[new_velocity <= 0.0] = -1.0
        good = (new_velocity > 0.0) * (new_velocity <= rrl_velocity)
        prep_id[good] = new_velocity[good] / rrl_velocity - 1.0
        good = (new_velocity > rrl_velocity) * (new_velocity < tp_velocity)
        prep_id[good] = (new_velocity[good] - rrl_velocity) / (
            tp_velocity - rrl_velocity
        )
        prep_id[new_velocity >= tp_velocity] = 1.0
        data.at[idx, "prep_spectrum"] = prep_spectrum
        data.at[idx, "prep_rms"] = prep_rms
        data.at[idx, "prep_id"] = prep_id
    return data


def train_cnn(
    datafiles,
    modelfile,
    scalerfile,
    restart=False,
    model_type="cnn",
    tangent=False,
    test_frac=0.2,
    val_frac=0.2,
    filter_sizes=[3],
    dense_layers=[9],
    num_blocks=3,
    kernel_sizes=[3],
    regularize=False,
    dropout_frac=0.2,
    init_lr=0.01,
    batch_size=128,
    epochs=100,
    seed=1234,
    calplot=None,
    verbose=False,
):
    """
    Create, train, and test a convolutional neural network to resolve
    the kinematic distance ambiguity using HI absorption spectra.

    Inputs:
        datafiles :: list of string
            Path to data pickle files.
        modelfile :: string
            Filepath where CNN model is saved. If it doesn't exist, a new
            model is created. Otherwise, the existing model is used.
        scalerfile :: string
            Filepath where data scaler is pickled. Always overwritten.
        restart :: boolean
            If True, delete existing model (if present) and start a new one
        model_type :: string
            One of "cnn" or "resnet".
        tangent :: boolean
            If True, train on tangent labels (in addition to N/F) for real data.
        test_frac :: scalar
            Fraction of data reserved for testing.
        val_frac :: scalar
            Fraction of non-testing data reserved for validation.
        filter_sizes :: list of integers
            CNN only: The filter size of each convolution layer
        dense_layers :: list of intengers
            CNN only: The size of each dense layer
        num_blocks :: integer
            Resnet only: Number of residual blocks
        kernel_sizes :: list of integers
            CNN and Resnet: The kernel size of each convolutional layer
            For CNN, the size of this list must match that of filter_sizes
        regularize :: boolean
            If True, add dropout (CNN) or batch normalization (Resnet)
        dropout_frac :: scalar
            CNN only: Dropout fraction in (0, 1)
        init_lr :: scalar
            Initial learning rate
        batch_size :: integer
            Batch size
        epochs :: integer
            Number of epochs
        seed :: integer
            Random seed
        calplot:: string
            If not None, plot the binned accuracy vs. confidence before and
            after calibration.
        verbose :: boolean
            If True, print info

    Returns: Nothing
    """
    # Random state
    rng = np.random.RandomState(seed=seed)

    if restart and os.path.exists(modelfile):
        if verbose:
            print(f"Removing {modelfile}")
        os.remove(modelfile)

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
        print(f"{len(data_train)} spectra in training set (Order: {trainy.keys()})")
        print(f"{len(data_val)} spectra in validation set (Order: {valy.keys()})")
        print(f"{len(data_test)} spectra in test set (Order: {testy.keys()})")
    with open(scalerfile, "wb") as f:
        pickle.dump(data_scaler, f)

    # initialize classifier
    input_shape = trainX.shape[1:]
    n_outputs = trainy.shape[1]

    if model_type == "cnn":
        model = build_cnn(
            input_shape,
            n_outputs,
            filter_sizes,
            kernel_sizes,
            dense_layers,
            dropout_frac=dropout_frac if regularize else None,
            init_lr=init_lr,
        )
    elif model_type == "resnet":
        model = build_resnet(
            input_shape,
            n_outputs,
            num_blocks,
            kernel_sizes,
            batch_norm=regularize,
            init_lr=init_lr,
        )
    else:
        raise ValueError(f"Invalid model type: {model_type}")

    classifier = CNNClassifier(
        modelfile,
        model=model,
        verbose=verbose,
    )

    # train classifier
    start = time.time()
    classifier.fit(trainX, trainy, valX, valy, batch_size=batch_size, epochs=epochs)
    runtime = time.time() - start

    # score test data
    score = classifier.score(testX, testy)
    if verbose:
        print(f"Test data score: {score:.2f}")
        print(f"Training time: {runtime:.1f} seconds")

    # calibrate on validation set
    if verbose:
        print("Calibrating probabilities...")
    temp = classifier.calibrate(valX, valy, calplot=calplot)
    if verbose:
        print(f"Calibrated temperature: {temp:.1f}")


def main():
    parser = argparse.ArgumentParser(
        description="Train a CNN to resolve the kinematic distance ambiguity",
        prog="train_cnn.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=__version__,
    )
    parser.add_argument(
        "modelfile",
        type=str,
        help="Path to Keras model",
    )
    parser.add_argument(
        "scalerfile",
        type=str,
        help="Path to data scaler pickle output",
    )
    parser.add_argument(
        "--datafiles",
        type=str,
        nargs="+",
        required=True,
        help="Path to input data pickle files",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        default=False,
        help="Delete existing model (if present) and restart",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="cnn",
        help="cnn or resnet",
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
        "--filter_sizes",
        type=int,
        nargs="+",
        default=[3],
        help="CNN only: Filter sizes",
    )
    parser.add_argument(
        "--dense_layers",
        type=int,
        nargs="+",
        default=[9],
        help="CNN only: size of each dense layer",
    )
    parser.add_argument(
        "--num_blocks",
        type=int,
        default=3,
        help="Resnet only: number of residual blocks",
    )
    parser.add_argument(
        "--kernel_sizes",
        type=int,
        nargs="+",
        default=[3],
        help="CNN or Resnet: Kernel sizes",
    )
    parser.add_argument(
        "--regularize",
        action="store_true",
        default=False,
        help="Add dropout (CNN) or batch normalization (Resnet)",
    )
    parser.add_argument(
        "--dropout_frac",
        type=float,
        default=0.2,
        help="CNN: dropout fraction",
    )
    parser.add_argument(
        "--init_lr",
        type=float,
        default=0.01,
        help="Initial learning rate",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Training batch size",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1000,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed",
    )
    parser.add_argument(
        "--calplot",
        type=str,
        default=None,
        help="Reliability plot filename",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Print info",
    )
    args = parser.parse_args()
    train_cnn(**vars(args))


if __name__ == "__main__":
    main()

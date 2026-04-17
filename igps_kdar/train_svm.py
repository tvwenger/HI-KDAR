"""
train_cvm.py
Train a support vector machine (SVM) to resolve the
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

import argparse
import pickle
import time

import numpy as np
from statistics import mode

from sklearn.model_selection import train_test_split

from igps_kdar import __version__
from igps_kdar.utils import RobustScaler
from igps_kdar.svm_classifier import SVMClassifier


def extract_final_data(dataset, datafiles):
    """
    This is a silly function that extracts the relevant columns from the dataframe
    and prepares them for training.

    Inputs:
        dataset :: pandas.DataFrame
            "merged_data" dataframe for this dataset (i.e., either train, test, or val)
        datafiles :: list of strings
            List of datafiles. Training data follow this order.

    Returns:
        X :: nd-array of scalars
            Feature set
        y :: nd-array of scalars
            Label set
        channels :: list of integers
            Channels that should be normalized (i.e., not fractions)
    """
    features = []
    channels = []
    for i in range(len(datafiles)):
        features += [
            f"near_frac_chans_{i}",
            f"near_max_snr_{i}",
            f"near_int_snr_{i}",
            f"far_frac_chans_{i}",
            f"far_max_snr_{i}",
            f"far_int_snr_{i}",
        ]
        channels += [6 * i + 1, 6 * i + 2, 6 * i + 4, 6 * i + 5]
    X = dataset[features].to_numpy()
    y = dataset["merged_kdar"]
    return X, y, channels


def load_data(datafiles, rng, test_frac=0.2, tangent=False):
    """
    Load and prepare the data.

    Inputs:
        datafiles :: list of string
            Path to data pickle files. Order of data in output datasets is
            [datafile_0, datafile_1, ...]
        rng :: np.RandomState
            RNG state
        test_frac :: scalar
            Fraction of data reserved for testing.
        tangent :: boolean
            If True, also load data labeled "T"

    Returns:
        data_scaler :: RobustScaler
            Data scaler
        data_train, data_test :: pd.DataFrame
            Training and test datasets
        trainX, trainy :: ND-arrays of scalars
            Training features and labels
        testX, testy :: ND-arrays of scalars
            Test features and labels
    """
    merged_data = None
    for i, datafile in enumerate(datafiles):
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
            merged_data = data.add_suffix(f"_{i}")
        else:
            merged_data = merged_data.merge(
                data.add_suffix(f"_{i}"),
                how="inner",
                left_on="gname_0",
                right_on=f"gname_{i}",
            )

    # Assign KDARs: Select best QF, then most common KDAR
    best_kdar_qfs = np.sort(
        merged_data[[f"kdar_manual_qf_{i}" for i in range(len(datafiles))]]
        .fillna("X")
        .values,
        axis=1,
    )[:, 0]
    best_kdar_qfs[best_kdar_qfs == "X"] = np.nan

    # Get most common KDAR with that label
    merged_data = merged_data.assign(merged_kdar=[None] * len(best_kdar_qfs))
    merged_data = merged_data.assign(merged_kdar_qf=best_kdar_qfs)
    for idx, row in merged_data.iterrows():
        kdars = np.array(
            [
                row[f"kdar_manual_{i}"]
                for i in range(len(datafiles))
                if row[f"kdar_manual_qf_{i}"] == row["merged_kdar_qf"]
            ]
        )
        if len(kdars) == 0:
            continue
        kdar_mode = mode(kdars)
        if not np.all(kdar_mode == kdars):
            print(f"KDAR mismatch: {row['gname_0']}. Using {kdar_mode}")
        merged_data.loc[idx, "merged_kdar"] = kdar_mode

    # Drop tangent if necessary
    if not tangent:
        merged_data.loc[merged_data["merged_kdar"] == "T"] = None
    merged_data = merged_data[~merged_data["merged_kdar"].isna()]

    # Get training, validation, and test set gnames and labels
    # statify on KDAR+QF
    merged_data["strat_label"] = (
        merged_data["merged_kdar"] + merged_data["merged_kdar_qf"]
    )
    data_train, data_test = train_test_split(
        merged_data,
        test_size=test_frac,
        stratify=merged_data["strat_label"],
        random_state=rng,
    )

    for dataset, datalabel in zip(
        [merged_data, data_train, data_test],
        ["all", "training", "testing"],
    ):
        print(f"Dataset: {datalabel}")
        labels = dataset["merged_kdar"].unique()
        labels.sort()
        for kdar in labels:
            qfs = dataset["merged_kdar_qf"].unique()
            qfs.sort()
            num_qf = [
                (
                    (dataset["merged_kdar"] == kdar) * (dataset["merged_kdar_qf"] == qf)
                ).sum()
                for qf in qfs
            ]
            print(f"KDAR {kdar}: {np.sum(num_qf)}")
            print(f"QFs ({qfs}): {num_qf}")

    # extract features and labels
    # order: [datafile_0, datafile_1, ...]
    trainX, trainy, channels = extract_final_data(data_train, datafiles)
    testX, testy, channels = extract_final_data(data_test, datafiles)

    # Normalize data, except channel fraction columns
    data_scaler = RobustScaler()
    data_scaler.fit(trainX)
    trainX = data_scaler.transform(trainX, channels=channels)
    testX = data_scaler.transform(testX, channels=channels)

    return (
        data_scaler,
        data_train,
        data_test,
        trainX,
        trainy,
        testX,
        testy,
    )


def train_svm(
    datafiles,
    scalerfile,
    tangent=False,
    test_frac=0.2,
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

    if verbose:
        print("Loading data...")
    (
        data_scaler,
        data_train,
        data_test,
        trainX,
        trainy,
        testX,
        testy,
    ) = load_data(datafiles, rng, test_frac=test_frac, tangent=tangent)
    if verbose:
        print(f"{len(data_train)} spectra in training set")
        print(f"{len(data_test)} spectra in test set")
    with open(scalerfile, "wb") as f:
        pickle.dump(data_scaler, f)

    # train classifier
    start = time.time()
    classifier = SVMClassifier(kernel="linear")
    classifier.fit(trainX, trainy)
    runtime = time.time() - start

    # score test data
    score = classifier.score(testX, testy)
    if verbose:
        print(f"Test data score: {score:.2f}")
        print(f"Training time: {runtime:.1f} seconds")

    # calibrate on validation set
    if verbose:
        print("Calibrating probabilities...")
    temp = classifier.calibrate(trainX, trainy, calplot=calplot)
    if verbose:
        print(f"Calibrated temperature: {temp:.1f}")


def main():
    parser = argparse.ArgumentParser(
        description="Train a SVM to resolve the kinematic distance ambiguity",
        prog="train_svm.py",
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
    train_svm(**vars(args))


if __name__ == "__main__":
    main()

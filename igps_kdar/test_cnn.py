"""
test_cnn.py
Compare CNN predictions to manual/true labels.

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

Trey Wenger - April 2024
"""

import argparse
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patches

import tensorflow as tf
from keras import models

from igps_kdar import __version__
from igps_kdar.train_cnn import prepare_data, merge_final_data
from igps_kdar.utils import predict_calibrated


def load_data(datafiles, scalerfile, tangent=False):
    """
    Load and prepare the data.

    Inputs:
        datafiles :: list of string
            Path to data pickle files. Order of data in output datasets is
            [datafile_0_spec, datafile_0_rms, datafile_1_spec, datafile_1_rms, ..., ID]
        scalerfile :: string
            Filepath where data scaler is pickled.
        tangent :: boolean
            If True, also load data labeled "T"

    Returns:
        data :: pd.DataFrame
            Test dataset
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

    # extract features and labels
    # order: [spec_1, rms_1, spec_2, rms_2, ..., id]
    testX, testy = merge_final_data(best_kdar, merged_data, datafiles)

    # Load the data scaler
    with open(scalerfile, "rb") as f:
        data_scaler = pickle.load(f)

    # Normalize data, except "ID" axis
    testX = data_scaler.transform(testX, channels=[0, 1])

    return best_kdar, testX, testy


def test_cnn(
    datafiles,
    modelfile,
    scalerfile,
    temperature=1.0,
    tangent=False,
    prefix="figures/",
    prob_fname=None,
    verbose=False,
):
    """
    Compare CNN predictions to manual/true labels.

    Inputs:
        datafiles :: list of string
            Files containing HI absorption data
        modelfile :: string
            Filepath where CNN model is saved.
        scalerfile :: string
            Filepath where data scaler is pickled.
        temperature :: scalar
            Temperature used to calibrate probabilities.
        tangent :: boolean
            If True, include tangent sources in test
        prefix :: string
            Figures are saved to {prefix}_*.pdf
        prob_fname :: string
            If not None, save assigned probabilities to this text file
        verbose :: boolean
            If True, print info

    Returns: Nothing
    """
    if verbose:
        print("Loading data...")
    data_test, testX, testy = load_data(datafiles, scalerfile, tangent=tangent)

    if verbose:
        print("Loading model...")
    model = models.load_model(modelfile)

    if verbose:
        print("Predicting...")
    confy = model.predict(testX)
    proby = predict_calibrated(confy, temperature=temperature)

    # Labels for model: either ["F", "N"] or ["F", "N", "T"]
    # Here we adapt all possibilities to a common ["F", "N", "T"]
    pred_tangent = True
    if proby.shape[1] == 2:
        # This is ["F", "N"]
        # add a column of zeros for tangent
        proby = np.concatenate([proby, np.zeros_like(proby[:, 0:1])], axis=1)

        # And we drop tangent from labels since we can't predict it
        testy.iloc[:, 2] = 0
        pred_tangent = False

    # Add probabilities to dataframe and save
    if prob_fname is not None:
        data_test = data_test.assign(prob_F=proby[:, 0])
        data_test = data_test.assign(prob_N=proby[:, 1])
        data_test = data_test.assign(prob_T=proby[:, 2])
        data_test.to_string(
            prob_fname,
            float_format="{:.6f}".format,
        )

    predy = (
        tf.one_hot(tf.argmax(proby, axis=1), depth=proby.shape[1]).numpy().astype(int)
    )
    assigned_probs = (predy * proby).sum(axis=1)

    # Calculate statistics, breakdown by QF
    labels = ["F", "N"]
    if pred_tangent:
        labels += ["T"]
    qfs = ["A", "B", "C"]
    for qf in qfs:
        print(f"Testing QF {qf}")
        is_qf = data_test["merged_kdar"].isin(labels) * (
            data_test["kdar_manual_qf"] == qf
        )
        is_right = (predy[is_qf] == testy[is_qf]).all(axis=1)
        print(
            f"Matching labels: {is_right.sum()}/{is_qf.sum()} "
            + f"({100*is_right.sum()/is_qf.sum():.2f}%)"
        )
        is_wrong = (predy[is_qf] != testy[is_qf]).any(axis=1)
        print(
            f"Incorrect labels: {is_wrong.sum()}/{is_qf.sum()} "
            + f"({100*is_wrong.sum()/is_qf.sum():.2f}%)"
        )
        for gname, prob in zip(
            data_test[is_qf][is_wrong]["gname"], assigned_probs[is_qf][is_wrong]
        ):
            print(f"{gname} Prob: {100*prob:.2f}%")
        print()
        for i, label in enumerate(labels):
            is_label = testy[label].to_numpy().astype(bool)
            if is_label[is_qf].sum() == 0:
                print(f"No data labeled {label} with QF {qf}")
                continue

            # Breakdown by sep from TP
            tp_buffers = [None]
            for tp_buffer in tp_buffers:
                not_tp = np.ones(len(data_test)).astype(bool)
                if tp_buffer is not None:
                    not_tp = (
                        data_test["rrl_velocity"] < data_test["tp_velocity"] - tp_buffer
                    )
                    print(f"Excluding V_RRL > V_TP + {tp_buffer}")
                good = is_qf * is_label * not_tp
                is_right_label = good * (predy[:, i] == testy[label])
                print(
                    f"Matching {label}: {is_right_label.sum()}/{good.sum()} "
                    + f"({100*is_right_label.sum()/good.sum():.2f}%)"
                )
                is_wrong_label = good * (predy[:, i] != testy[label])
                print(
                    f"Incorrect {label}: {is_wrong_label.sum()}/{good.sum()} "
                    + f"({100*is_wrong_label.sum()/good.sum():.2f}%)"
                )
                print(data_test[is_wrong_label]["gname"])
                bad = is_qf * (~is_label) * not_tp
                is_false_label = bad * predy[:, i].astype(bool)
                print(
                    f"False positive {label}: {is_false_label.sum()}/{bad.sum()} "
                    + f"({100*is_false_label.sum()/bad.sum():.2f}%)"
                )
                print(data_test[is_false_label]["gname"])
                print()

    # plot number of right, wrong, unlabeled vs. probability
    is_NF = testy.iloc[:, 0:2].any(axis=1)
    is_NF_right = is_NF * (predy == testy).all(axis=1)
    is_NF_wrong = is_NF * (predy != testy).any(axis=1)
    is_T = testy.iloc[:, 2].astype(bool)
    is_T_right = is_T * (predy == testy).all(axis=1)
    is_T_wrong = is_T * (predy != testy).any(axis=1)
    prob_NF_right = [
        proby[is_NF_right * (data_test["kdar_manual_qf"] == qf)].max(axis=1)
        for qf in ["A", "B", "C"]
    ]
    prob_NF_wrong = [
        proby[is_NF_wrong * (data_test["kdar_manual_qf"] == qf)].max(axis=1)
        for qf in ["A", "B", "C"]
    ]
    prob_T_right = proby[is_T_right].max(axis=1)
    prob_T_wrong = proby[is_T_wrong].max(axis=1)
    prob_unlabeled = proby[~(is_NF + is_T)].max(axis=1)

    cm = plt.get_cmap("viridis")
    colors_right = cm(np.linspace(0.0, 0.1, len(prob_NF_right)))
    colors_wrong = cm(np.linspace(0.4, 0.5, len(prob_NF_wrong)))
    colors_tangent = cm(np.linspace(0.8, 0.9, 2))
    colors_unlabeled = cm(1.0)
    right_patch = patches.Patch(color=colors_right[0], label="Correct")
    wrong_patch = patches.Patch(color=colors_wrong[0], label="Incorrect")
    tangent_patch = patches.Patch(color=colors_tangent[0], label="Tangent")
    unlabeled_patch = patches.Patch(color=colors_unlabeled, label="Unlabeled")

    fig, ax = plt.subplots()
    bin_edges = np.arange(0.0, 1.01, 0.1)
    bin_edges[-1] += 0.001  # include 1.0
    probs = prob_NF_right + prob_NF_wrong + [prob_T_right, prob_T_wrong, prob_unlabeled]
    colors = np.concatenate(
        [colors_right, colors_wrong, colors_tangent, [colors_unlabeled]]
    )
    ax.hist(
        probs,
        bin_edges,
        stacked=True,
        color=colors,
        edgecolor="k",
        linewidth=0.1,
    )
    ax.legend(
        loc="upper left",
        handles=[right_patch, wrong_patch, tangent_patch, unlabeled_patch],
    )
    ax.set_xlim(0.0, 1.0)
    ax.grid(False)
    # ax.set_yscale("log")
    ax.set_xlabel("Probability")
    ax.set_ylabel("Number")
    fig.tight_layout()
    fig.savefig(f"{prefix}_prob_number.pdf", bbox_inches="tight")
    plt.close(fig)

    # plot reliability excl. unlabeled
    fig, ax = plt.subplots()
    is_NF_right_binned = np.array(
        [np.histogram(prob, bins=bin_edges)[0] for prob in prob_NF_right]
    )
    is_NF_wrong_binned = np.array(
        [np.histogram(prob, bins=bin_edges)[0] for prob in prob_NF_wrong]
    )
    is_T_right_binned = np.histogram(prob_T_right, bins=bin_edges)[0]
    is_T_wrong_binned = np.histogram(prob_T_wrong, bins=bin_edges)[0]
    probs = np.concatenate(
        [is_NF_right_binned, is_NF_wrong_binned, [is_T_right_binned, is_T_wrong_binned]]
    )
    total_binned = np.sum(probs, axis=0)
    bottom = np.zeros(len(total_binned))
    for prob, color in zip(probs, colors):
        ax.bar(
            bin_edges[:-1],
            prob / total_binned,
            width=bin_edges[1] - bin_edges[0],
            color=color,
            align="edge",
            bottom=bottom,
            edgecolor="k",
            linewidth=0.1,
        )
        bottom += prob / total_binned
    ax.legend(loc="upper left", handles=[right_patch, wrong_patch, tangent_patch])
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(False)
    ax.set_xlabel("Probability")
    ax.set_ylabel("Fraction")
    fig.tight_layout()
    fig.savefig(f"{prefix}_prob_fraction.pdf", bbox_inches="tight")
    plt.close(fig)

    # plot reliability incl. unlabeled and tangent
    fig, ax = plt.subplots()
    is_unlabeled_binned = np.histogram(prob_unlabeled, bins=bin_edges)[0]
    probs = np.concatenate(
        [
            is_NF_right_binned,
            is_NF_wrong_binned,
            [is_T_right_binned, is_T_wrong_binned, is_unlabeled_binned],
        ]
    )
    total_binned = np.sum(probs, axis=0)
    bottom = np.zeros(len(total_binned))
    for prob, color in zip(probs, colors):
        ax.bar(
            bin_edges[:-1],
            prob / total_binned,
            width=bin_edges[1] - bin_edges[0],
            color=color,
            align="edge",
            bottom=bottom,
            edgecolor="k",
            linewidth=0.1,
        )
        bottom += prob / total_binned
    ax.legend(
        loc="upper left",
        handles=[right_patch, wrong_patch, tangent_patch, unlabeled_patch],
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(False)
    ax.set_xlabel("Probability")
    ax.set_ylabel("Fraction")
    fig.tight_layout()
    fig.savefig(f"{prefix}_prob_fraction_unlabeled.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Test CNN",
        prog="test_cnn.py",
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
        help="Path to input data pickle file",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature to calibrate probabilities",
    )
    parser.add_argument(
        "--tangent",
        action="store_true",
        default=False,
        help="Include tangent sources in test",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="figures/",
        help="Figures are saved to {prefix}_*.pdf",
    )
    parser.add_argument(
        "--prob_fname",
        type=str,
        default=None,
        help="Save probabilities to this text file",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Print info",
    )
    args = parser.parse_args()
    test_cnn(**vars(args))


if __name__ == "__main__":
    main()

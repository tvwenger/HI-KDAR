"""
classify_jones.py
Use a set of criteria to classify absorption spectra. Follows the
traditional method of Jones & Dickey (2012).

Copyright(C) 2022-2024 by
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

Trey Wenger - March 2022
Trey Wenger - January 2024 - Polish and 1.0 release
"""

import argparse
import pickle
import numpy as np

from . import __version__


def classify_criteria(
    datafile,
    rrl_buffer=25.0,
    tp_buffer=25.0,
    tp_rrl_buffer=10.0,
    min_snr=3.0,
    outfile="criteria_labeled.pkl",
):
    """
    Classify HI absorption spectra using a set of criteria, following Jones & Dickey (2012):
    - Case I: RRL within tp_buffer km/s of tangent point (TP) - assign to TP
    - Case II: RRL within tp_buffer+tp_rrl_buffer km/s of TP - assign NO KDAR
    - Case III: Sum from RRL+rrl_buffer to TP
        - if median tau rms > max_rms_tau - we cannot consider this since our spectra are not normalized
        - instead: if Sum(0, TP)/Noise < min_snr (no evidence for absorption anywhere) - assign NO KDAR
        - if Sum(RRL+rrl_buffer, TP)/Noise < 1 (no evidence for absorption) - assign NEAR KDAR
        - if Sum(RRL+rrl_buffer, TP)/Noise > min_snr) (significant absorption) - assign FAR KDAR
        - Otherwise (i.e., 1 < signal/noise < 3; uncertain absorption) - assign NO KDAR

    Inputs:
        datafile :: string
            File containing IGPS HI absorption data
        rrl_buffer :: float (km/s)
            Start summatation after RRL velocity + rrl_buffer
        tp_buffer :: float (km/s)
            Assign to TP if RRL velocity closer than tp_buffer to TP velocity
        tp_rrl_buffer :: float (km/s)
            Assign no KDAR if RRl velocity within (tp_buffer + tp_rrl_buffer) of
            TP velocity
        min_snr :: scalar
            Require absorption exceeding this signal-to-noise ratio to assign FAR
        outfile :: string
            Where the labeled datafile is saved

    Returns:
        Nothing
    """
    # Read data
    with open(datafile, "rb") as f:
        data = pickle.load(f)

    # Assign KDAR
    data = data.assign(kdar_criteria=["?"] * len(data))
    data = data.assign(kdar_criteria_reason=[""] * len(data))
    for idx, row in data.iterrows():
        is_QI = row["glong"] < 90.0
        is_QIV = row["glong"] > 270.0

        # Outer Galaxy
        if (
            (is_QI and row["rrl_velocity"] < 0.0)
            or (is_QIV and row["rrl_velocity"] > 0.0)
            or not (is_QI or is_QIV)
        ):
            data.loc[idx, "kdar_criteria"] = None
            data.loc[idx, "kdar_criteria_reason"] = "Outer Galaxy"
            continue

        # Case I: RRL within {tp_buffer} of TP - assign to TP
        if (is_QI and row["rrl_velocity"] > row["tp_velocity"] - tp_buffer) or (
            is_QIV and row["rrl_velocity"] < row["tp_velocity"] + tp_buffer
        ):
            data.loc[idx, "kdar_criteria"] = "T"
            data.loc[idx, "kdar_criteria_reason"] = "TP"
            continue

        # Case II: RRL within {tp_buffer}+{rrl_buffer} of TP - assign NO KDAR
        if (
            is_QI
            and row["rrl_velocity"] > row["tp_velocity"] - tp_buffer - tp_rrl_buffer
        ) or (
            is_QIV
            and row["rrl_velocity"] < row["tp_velocity"] + tp_buffer + tp_rrl_buffer
        ):
            data.loc[idx, "kdar_criteria"] = None
            data.loc[idx, "kdar_criteria_reason"] = "Close to TP"
            continue

        # Calculate absorption signal over entire spectrum
        if is_QI:
            start_vel = 0.0
            end_vel = row["tp_velocity"]
        else:
            start_vel = row["tp_velocity"]
            end_vel = 0.0
        start_idx = np.argmin(np.abs(row["velocity"] - start_vel))
        end_idx = np.argmin(np.abs(row["velocity"] - end_vel))
        spec_signal = np.sum(row["spectrum"][start_idx : end_idx + 1])
        spec_noise = np.sqrt(np.sum(row["rms"][start_idx : end_idx + 1] ** 2.0))

        # Case IIIa: Sum(0, VT)/Noise < min_snr (no evidence for absorption anywhere) - assign NO KDAR
        if spec_signal / spec_noise < min_snr:
            data.loc[idx, "kdar_criteria"] = None
            data.loc[idx, "kdar_criteria_reason"] = "No Significant Absorption"
            continue

        # Calculate absorption signal over far part of spectrum
        if is_QI:
            start_vel = row["rrl_velocity"] + rrl_buffer
            end_vel = row["tp_velocity"]
        else:
            start_vel = row["tp_velocity"]
            end_vel = row["rrl_velocity"] - rrl_buffer
        start_idx = np.argmin(np.abs(row["velocity"] - start_vel))
        end_idx = np.argmin(np.abs(row["velocity"] - end_vel))
        spec_signal = np.sum(row["spectrum"][start_idx : end_idx + 1])
        spec_noise = np.sqrt(np.sum(row["rms"][start_idx : end_idx + 1] ** 2.0))

        # Case IIIb: Sum(RRL+rrl_buffer, VT)/Noise < 1 (no evidence for far absorption) - assign NEAR KDAR
        if spec_signal / spec_noise < 1.0:
            data.loc[idx, "kdar_criteria"] = "N"
            data.loc[idx, "kdar_criteria_reason"] = "No Far Absorption"
            continue

        # Case IIIc: Sum(RRL+rrl_buffer, VT)/Noise > min_snr (significant absorption) - assign FAR KDAR
        if spec_signal / spec_noise > min_snr:
            data.loc[idx, "kdar_criteria"] = "F"
            data.loc[idx, "kdar_criteria_reason"] = "Significant Far Absorption"
            continue

        # Case IIId: Otherwise - assign NO KDAR
        data.loc[idx, "kdar_criteria"] = None
        data.loc[idx, "kdar_criteria_reason"] = "Questionable Far Absorption"

    # Save to disk
    with open(outfile, "wb") as f:
        pickle.dump(data, f)


def main():
    parser = argparse.ArgumentParser(
        description="Classify IGPS HI absorption spectra using Jones & Dickey criteria",
        prog="classify_jones.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=__version__,
    )
    parser.add_argument(
        "datafile",
        type=str,
        help="Spectra datafile",
    )
    parser.add_argument(
        "--rrl_buffer",
        type=float,
        default=25.0,
        help="RRL velocity buffer (km/s)",
    )
    parser.add_argument(
        "--tp_buffer",
        type=float,
        default=25.0,
        help="Tangent point velocity buffer (km/s)",
    )
    parser.add_argument(
        "--tp_rrl_buffer",
        type=float,
        default=10.0,
        help="Exclusion buffer between TP buffer and RRL (km/s)",
    )
    parser.add_argument(
        "--min_snr",
        type=float,
        default=3.0,
        help="Signal-to-noise ratio threshold",
    )
    parser.add_argument(
        "--outfile",
        type=str,
        default="criteria_labeled.pkl",
        help="Output pickle file with labels",
    )
    args = parser.parse_args()
    classify_criteria(**vars(args))


if __name__ == "__main__":
    main()

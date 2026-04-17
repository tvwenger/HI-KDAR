"""
classify_kolpak.py
Use a set of criteria to classify absorption spectra. Follows the
traditional method of Kolpak et al. (2003).

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
    min_snr=3.0,
    outfile="criteria_labeled.pkl",
):
    """
    Classify HI absorption spectra using a set of criteria, following Kolpak et al. (2003):
    Calculate VA: maximum velocity of significant absorption, and compare to
    VR: RRL velocity and VT: TP velocity
    - Case I: VT - VR < 0.0: assign TP
    - Case II: VT - VA > 15.0: assign NEAR QF A
    - Case III: VT - VA > min(15.0, 2*(VT-VR)): assign NEAR QF B
    - Case IV: VT - VA < min(15.0, -15 + (VT-VR)): assign FAR QF A
    - Case V: Otherwise: assign FAR QF B

    Inputs:
        datafile :: string
            File containing IGPS HI absorption data
        min_snr :: scalar
            Identify "significant" absorption about this signal-to-noise ratio
            threshold
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
    data = data.assign(kdar_criteria_qf=[""] * len(data))
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

        # Case I: VT - VR < 0.0: TP
        if (is_QI and row["tp_velocity"] - row["rrl_velocity"] < 0.0) or (
            is_QIV and row["tp_velocity"] - row["rrl_velocity"] > 0.0
        ):
            data.loc[idx, "kdar_criteria"] = "T"
            data.loc[idx, "kdar_criteria_qf"] = "A"
            continue

        snr = row["spectrum"] / row["rms"]
        abs_channels = np.where(snr > min_snr)[0]

        # No absorption
        if len(abs_channels) == 0:
            data.loc[idx, "kdar_criteria"] = None
            data.loc[idx, "kdar_criteria_qf"] = "No absorption"
            continue

        if is_QI:
            VA = row["velocity"][abs_channels[-1]]
        elif is_QIV:
            VA = row["velocity"][abs_channels[0]]

        # Case II: VT - VA > 15.0: NEAR QF A
        if (is_QI and row["tp_velocity"] - VA > 15.0) or (
            is_QIV and row["tp_velocity"] - VA < -15.0
        ):
            data.loc[idx, "kdar_criteria"] = "N"
            data.loc[idx, "kdar_criteria_qf"] = "A"
            continue

        # Case III: VT - VA > min(15.0, 2*(VT-VR)): NEAR QF B
        if (
            is_QI
            and row["tp_velocity"] - VA
            > np.min([15.0, 2.0 * (row["tp_velocity"] - row["rrl_velocity"])])
        ) or (
            is_QIV
            and row["tp_velocity"] - VA
            < np.max([-15.0, 2.0 * (row["tp_velocity"] - row["rrl_velocity"])])
        ):
            data.loc[idx, "kdar_criteria"] = "N"
            data.loc[idx, "kdar_criteria_qf"] = "B"
            continue

        # Case IV: VT - VA < min(15.0, -15 + (VT-VR)): FAR QF A
        if (
            is_QI
            and row["tp_velocity"] - VA
            < np.min([15.0, -15.0 + row["tp_velocity"] - row["rrl_velocity"]])
        ) or (
            is_QIV
            and row["tp_velocity"] - VA
            > np.max([-15.0, 15.0 - row["tp_velocity"] + row["rrl_velocity"]])
        ):
            data.loc[idx, "kdar_criteria"] = "F"
            data.loc[idx, "kdar_criteria_qf"] = "A"
            continue

        # Case V: Otherwise
        data.loc[idx, "kdar_criteria"] = "F"
        data.loc[idx, "kdar_criteria_qf"] = "B"

    # Save to disk
    with open(outfile, "wb") as f:
        pickle.dump(data, f)


def main():
    parser = argparse.ArgumentParser(
        description="Classify IGPS HI absorption spectra using Jones & Dickey criteria",
        prog="classify_kolpak.py",
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

"""
generate_labels.py
Plot HI absorption spectra, generate labels for manual editing.

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
"""

import os
import pickle
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from astropy.io import fits
from astropy.wcs import WCS

from . import __version__


def absorption_stats(row, buffer=25.0, min_snr=3.0):
    """
    Generate absorption statistics for a given spectrum.

    Inputs:
        row :: pandas.Series
            Relevant HI absorption data
        buffer :: float (km/s)
            Near summation 0 to RRL velocity
            Far summation RRL velocity + buffer to TP velocity + buffer
        min_snr :: scalar
            SNR threshold for absorption

    Returns: output
        output :: Dictionary
            With keys:
                near_chans, far_chans :: integers
                    Number of near and far channels
                near_abs_chans, far_abs_chans :: integers
                    Number of near and far channels with absorption
                near_frac_chans, far_frac_chans :: integers
                    Fraction of near and far channels with absorption
                near_max_snr, far_max_snr :: scalars
                    Peak signal-to-noise ratio
                near_int_snr, far_int_snr :: scalars
                    Integrated signal-to-noise ratio
                last_abs_vlsr :: scalar
                    VLSR of last channel with significant absorption
    """
    output = {}

    # Identify absorption signals over relevant regions of spectrum
    is_QI = row["glong"] < 90.0
    is_QIV = row["glong"] > 270.0
    if is_QI:
        near_start_vel = 0.0
        near_end_vel = row["rrl_velocity"]
        far_start_vel = row["rrl_velocity"] + buffer
        far_end_vel = row["tp_velocity"] + buffer
        all_start_vel = near_start_vel
        all_end_vel = far_end_vel

    elif is_QIV:
        near_start_vel = row["rrl_velocity"]
        near_end_vel = 0.0
        far_start_vel = row["tp_velocity"] - buffer
        far_end_vel = row["rrl_velocity"] - buffer
        all_start_vel = far_start_vel
        all_end_vel = near_end_vel
    else:
        raise ValueError("Outer Galaxy")

    near_start_idx = np.argmin(np.abs(row["velocity"] - near_start_vel))
    near_end_idx = np.argmin(np.abs(row["velocity"] - near_end_vel))
    far_start_idx = np.argmin(np.abs(row["velocity"] - far_start_vel))
    far_end_idx = np.argmin(np.abs(row["velocity"] - far_end_vel))
    all_start_idx = np.argmin(np.abs(row["velocity"] - all_start_vel))
    all_end_idx = np.argmin(np.abs(row["velocity"] - all_end_vel))

    # Near stats
    output["near_chans"] = near_end_idx + 1 - near_start_idx
    near_signal = row["spectrum"][near_start_idx : near_end_idx + 1]
    near_noise = row["rms"][near_start_idx : near_end_idx + 1]
    near_abs_channels = np.where(near_signal / near_noise > min_snr)[0]
    output["near_abs_chans"] = len(near_abs_channels)
    output["near_frac_chans"] = output["near_abs_chans"] / output["near_chans"]
    if len(near_signal) > 0:
        output["near_max_snr"] = np.max(near_signal / near_noise)
        output["near_int_snr"] = np.sum(near_signal) / np.sqrt(np.sum(near_noise**2.0))
        output["near_peak_chan"] = near_start_idx + np.argmax(near_signal / near_noise)
    else:
        output["near_max_snr"] = 0.0
        output["near_int_snr"] = 0.0
        output["near_peak_chan"] = 0

    # Far stats, catch VRRL > VTP
    num_far_chans = far_end_idx + 1 - far_start_idx
    if num_far_chans < 1:
        output["far_chans"] = 0
        output["far_abs_chans"] = 0
        output["far_frac_chans"] = 0.0
        output["far_max_snr"] = 0.0
        output["far_int_snr"] = 0.0
        output["far_peak_chan"] = -1
    else:
        output["far_chans"] = num_far_chans
        far_signal = row["spectrum"][far_start_idx : far_end_idx + 1]
        far_noise = row["rms"][far_start_idx : far_end_idx + 1]
        far_abs_channels = np.where(far_signal / far_noise > min_snr)[0]
        output["far_abs_chans"] = len(far_abs_channels)
        output["far_frac_chans"] = output["far_abs_chans"] / output["far_chans"]
        if len(far_signal) > 0:
            output["far_max_snr"] = np.max(far_signal / far_noise)
            output["far_int_snr"] = np.sum(far_signal) / np.sqrt(np.sum(far_noise**2.0))
            output["far_peak_chan"] = far_start_idx + np.argmax(far_signal / far_noise)
        else:
            output["far_max_snr"] = 0.0
            output["far_int_snr"] = 0.0
            output["far_peak_chan"] = -1

    all_signal = row["spectrum"][all_start_idx : all_end_idx + 1]
    all_noise = row["rms"][all_start_idx : all_end_idx + 1]
    all_abs_channels = np.where(all_signal / all_noise > min_snr)[0]
    if len(all_abs_channels) == 0:
        output["last_abs_vlsr"] = 0.0
    elif is_QI:
        output["last_abs_vlsr"] = row["velocity"][all_start_idx + all_abs_channels[-1]]
    elif is_QIV:
        output["last_abs_vlsr"] = row["velocity"][all_start_idx + all_abs_channels[0]]
    else:
        raise ValueError("Outer Galaxy")

    return output


def plot(
    idx,
    row,
    cont_image,
    cont_filtered_image,
    line_image,
    line_filtered_image,
    fname,
):
    """
    Generate plots

    Inputs:
        idx :: integer
            Row index
        row :: pandas.Series
            Relevant HI absorption data
        cont_image, cont_filtered_image, line_image, line_filtered_image :: fits.HDU
            Continuum and line images
        fname :: string
            Filename for saved plot

    Returns: Nothing
    """
    # generate figure and clear axis if necessary
    fig = plt.figure(figsize=(28, 12))
    gs = GridSpec(
        2, 4, height_ratios=[2, 1], width_ratios=[1, 1, 1, 1], hspace=0.0, wspace=0.6
    )

    # plot continuum image
    wcs = WCS(cont_image.header).celestial
    cont_data = cont_image.data
    # remove Stokes and velocity
    while len(cont_data.shape) > 2:
        cont_data = cont_data[0]
    ax = plt.subplot(gs[0, 0], projection=wcs)
    cax = ax.imshow(
        cont_data,
        origin="lower",
        interpolation="none",
        cmap="inferno",
        vmax=row["tcont"],
    )
    ax.plot(cont_image.header["LOCX"], cont_image.header["LOCY"], "k+")
    ax.set_xlabel("Galactic Longitude (deg)")
    ax.set_ylabel("Galactic Latitude (deg)")
    ax.coords[0].set_major_formatter("d.dd")
    ax.coords[1].set_major_formatter("d.dd")
    label = f"{idx}\n"
    label += f"{row['gname'].replace('-', '$-$')}\n"
    label += f"{row['dataset']}\n"
    label += f"$T_{{\\rm cont}}$ = {row['tcont']:.2f} K\n"
    label += f"$\\widetilde{{T_{{\\rm cont}}}}$ = {row['tcont_filtered']:.3f}"
    ax.text(
        0.05,
        0.95,
        label,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, pad=0.5),
    )
    # add colorbar
    cbar = fig.colorbar(cax, fraction=0.046, pad=0.04)
    cbar.set_label(r"Continuum $T_B$ (K)")

    # plot filtered continuum image
    wcs = WCS(cont_filtered_image.header).celestial
    cont_data = cont_filtered_image.data
    # remove Stokes and velocity
    while len(cont_data.shape) > 2:
        cont_data = cont_data[0]
    ax = plt.subplot(gs[0, 1], projection=wcs)
    cax = ax.imshow(
        cont_data,
        origin="lower",
        interpolation="none",
        cmap="inferno",
        vmax=row["tcont_filtered"],
    )
    ax.plot(cont_image.header["LOCX"], cont_image.header["LOCY"], "k+")
    ax.set_xlabel("Galactic Longitude (deg)")
    ax.set_ylabel("Galactic Latitude (deg)")
    ax.coords[0].set_major_formatter("d.dd")
    ax.coords[1].set_major_formatter("d.dd")
    # add colorbar
    cbar = fig.colorbar(cax, fraction=0.046, pad=0.04)
    cbar.set_label(r"Filtered Continuum $T_B$ (K)")

    # Get channel image at peak absorption SNR
    line_data = line_image.data
    # remove stokes
    while len(line_data.shape) > 3:
        line_data = line_data[0]
    if row["kdar_manual"] == "N" or row["near_max_snr"] > row["far_max_snr"]:
        peak_chan = row["near_peak_chan"]
    else:
        peak_chan = row["near_peak_chan"]
    line_data_peak = line_data[peak_chan]

    # plot line image
    wcs = WCS(line_image.header).celestial
    ax = plt.subplot(gs[0, 2], projection=wcs)
    cax = ax.imshow(
        line_data_peak,
        origin="lower",
        interpolation="none",
        cmap="inferno",
    )
    ax.plot(cont_image.header["LOCX"], cont_image.header["LOCY"], "k+")
    ax.set_xlabel("Galactic Longitude (deg)")
    ax.set_ylabel("Galactic Latitude (deg)")
    ax.coords[0].set_major_formatter("d.dd")
    ax.coords[1].set_major_formatter("d.dd")
    label = f"WISE KDAR: {row['kdar_wise']}\n"
    label += f"Near Max SNR = {row['near_max_snr']:.1f}\n"
    label += f"Near Int. SNR = {row['near_int_snr']:.1f}\n"
    label += f"Far Max SNR = {row['far_max_snr']:.1f}\n"
    label += f"Far Int. SNR = {row['far_int_snr']:.1f}"
    ax.text(
        0.05,
        0.95,
        label,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, pad=0.5),
    )
    # add colorbar
    cbar = fig.colorbar(cax, fraction=0.046, pad=0.04)
    cbar.set_label("Peak HI $T_B$ (K)")

    # Get filtered channel image at peak absorption SNR
    line_data = line_filtered_image.data
    # remove stokes
    while len(line_data.shape) > 3:
        line_data = line_data[0]
    line_data_peak = line_data[peak_chan]
    vmin = -row["spectrum"][peak_chan]

    # plot filtered line image
    wcs = WCS(line_filtered_image.header).celestial
    ax = plt.subplot(gs[0, 3], projection=wcs)
    cax = ax.imshow(
        line_data_peak,
        origin="lower",
        interpolation="none",
        cmap="inferno",
        vmin=vmin,
        vmax=1.5 * row["med_rms"],
    )
    ax.plot(cont_image.header["LOCX"], cont_image.header["LOCY"], "k+")
    ax.set_xlabel("Galactic Longitude (deg)")
    ax.set_ylabel("Galactic Latitude (deg)")
    ax.coords[0].set_major_formatter("d.dd")
    ax.coords[1].set_major_formatter("d.dd")
    # add colorbar
    cbar = fig.colorbar(cax, fraction=0.046, pad=0.04)
    cbar.set_label("Peak Filtered HI $T_B$ (K)")

    # plot spectrum
    ax = plt.subplot(gs[1, :])
    ax.axhline(0.0, color="k")
    ax.plot(row["velocity"], row["spectrum"], "k-", linewidth=2)
    ax.plot(row["velocity"], row["rms"], "k-", linewidth=0.5)
    ax.plot(row["velocity"], -row["rms"], "k-", linewidth=0.5)
    ax.axvline(row["rrl_velocity"], color="k", linewidth=2, label="RRL")
    ax.axvline(
        row["tp_velocity"], color="k", linestyle="dashed", linewidth=2, label="TP"
    )
    ax.axvline(row["velocity"][peak_chan], color="r", linewidth=2, label="Image")
    ax.set_xlabel(r"$v_{\rm LSR}$ km s$^{-1}$")
    ax.set_ylabel("Filtered Absorption")
    ax.text(
        0.02,
        0.92,
        f"{row['kdar_manual']} ({row['kdar_manual_qf']})",
        va="top",
        ha="left",
        transform=ax.transAxes,
        fontsize=24,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )
    ax.legend(loc="upper right")

    # set sensible xrange
    if row["glong"] < 90.0:
        xmin = -75.0
        xmax = np.min([np.max(row["velocity"]), row["tp_velocity"] + 50.0])
    elif row["glong"] > 270.0:
        xmin = np.max([np.min(row["velocity"]), row["tp_velocity"] - 50.0])
        xmax = 75.0
    else:
        raise ValueError("Outer Galaxy")
    ax.set_xlim(xmin, xmax)
    start_idx = np.argmin(np.abs(row["velocity"] - xmin))
    end_idx = np.argmin(np.abs(row["velocity"] - xmax))

    # set sensible yrange
    ymin = np.min(row["spectrum"][start_idx : end_idx + 1])
    ymax = np.max(row["spectrum"][start_idx : end_idx + 1])
    yrange = ymax - ymin
    ymin = ymin - 0.1 * yrange
    ymax = ymax + 0.1 * yrange
    ax.set_ylim(ymin, ymax)

    # fig.tight_layout()
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)


def generate_labels(
    datafile,
    buffer=25.0,
    int_snr_threshold=3.0,
    max_snr_threshold=5.0,
    labelfile="manual_labels.csv",
    imagedir="/data/igps_hi",
    outfile="manual_labeled.pkl",
    plotfile="labeled_plots.tex",
):
    """
    Plot HI absorption spectra, create datafiles with HI absorption statistics and
    labeled KDARs.

    Inputs:
        datafile :: string
            HI absorption data file
        buffer :: string (km/s)
            "Near" channels between 0 and RRL + buffer
            "Far" channels between RRL + buffer and TP + buffer
            Assign to TP for sources within buffer of TP
        int_snr_threshold :: scalar
        max_snr_threshold :: scalar
            Only consider sources with
            integrated signal-to-noise ratio exceeding int_snr_threshold or
            peak signal-to-noise ratio exceeding max_snr_threshold
        labelfile :: string
            File containing manually-labeled data. If present, then these KDARs
            will be assigned into outfile. If not present, then it will be created
            and populated with WISE KDARs.
        imagedir :: string
            Path to IGPS images. Plots will be stored there.
        outfile :: string
            Output pickle file with manual labels.
        plotfile :: string
            Filename for compiled LaTeX document

    Returns: Nothing
    """
    # Storage for plot fnames
    plot_fnames = []

    # Read data
    with open(datafile, "rb") as f:
        data = pickle.load(f)

    # Calculate statistics, populate KDARs
    data = data.assign(
        kdar_manual=[""] * len(data),
        kdar_manual_qf=[""] * len(data),
        tcont=[np.nan] * len(data),
        med_rms=[np.nan] * len(data),
        near_chans=[np.nan] * len(data),
        near_abs_chans=[np.nan] * len(data),
        near_peak_chan=[0] * len(data),
        near_max_snr=[np.nan] * len(data),
        near_int_snr=[np.nan] * len(data),
        last_abs_vlsr=[np.nan] * len(data),
        far_chans=[np.nan] * len(data),
        far_abs_chans=[np.nan] * len(data),
        far_peak_chan=[0] * len(data),
        far_max_snr=[np.nan] * len(data),
        far_int_snr=[np.nan] * len(data),
    )
    data["kdar_manual"] = None
    data["kdar_manual_qf"] = None

    # load assigned QF data
    labeldata = None
    if os.path.exists(labelfile):
        print(f"Loading manual labels from {labelfile}")
        labeldata = pd.read_table(labelfile, delim_whitespace=True)

        # merge manual KDARs
        data["kdar_manual"] = labeldata["kdar_manual"]
        data["kdar_manual_qf"] = labeldata["kdar_manual_qf"]

    for idx, row in data.iterrows():
        print(f"Processing: {idx} / {len(data)}", end="\r")
        is_QI = row["glong"] < 90.0
        is_QIV = row["glong"] > 270.0

        # Load images
        cont_fname = f"{imagedir}/{row['gname']}_{row['dataset']}_cont.fits"
        cont_filtered_fname = (
            f"{imagedir}/{row['gname']}_{row['dataset']}_cont_filtered.fits"
        )
        line_fname = f"{imagedir}/{row['gname']}_{row['dataset']}_line.fits"
        line_filtered_fname = (
            f"{imagedir}/{row['gname']}_{row['dataset']}_line_filtered.fits"
        )
        cont_image = fits.open(cont_fname)[0]
        cont_filtered_image = fits.open(cont_filtered_fname)[0]
        line_image = fits.open(line_fname)[0]
        line_filtered_image = fits.open(line_filtered_fname)[0]

        # Update stats
        data.loc[idx, "tcont"] = cont_image.header["SRCTB"]
        data.loc[idx, "tcont_filtered"] = cont_filtered_image.header["SRCTB"]
        data.loc[idx, "med_rms"] = np.median(row["rms"])

        # Get absorption stats
        output = absorption_stats(row, buffer=buffer)
        for key in output.keys():
            data.loc[idx, key] = output[key]

        # Skip plotting close to TP. QF A if > TP, QF B if within buffer
        if (is_QI and row["rrl_velocity"] > row["tp_velocity"] - buffer) or (
            is_QIV and row["rrl_velocity"] < row["tp_velocity"] + buffer
        ):
            data.loc[idx, "kdar_manual"] = "T"
            data.loc[idx, "kdar_manual_qf"] = "B"
            if (is_QI and row["rrl_velocity"] > row["tp_velocity"]) or (
                is_QIV and row["rrl_velocity"] < row["tp_velocity"]
            ):
                data.loc[idx, "kdar_manual_qf"] = "A"
            continue

        # Skip plotting no significant absorptoin
        if row["int_snr"] < int_snr_threshold and row["max_snr"] < max_snr_threshold:
            continue

        fname = cont_fname.replace("_cont.fits", ".pdf")
        plot(
            idx,
            data.loc[idx],
            cont_image,
            cont_filtered_image,
            line_image,
            line_filtered_image,
            fname,
        )
        plot_fnames.append(fname)
    print()

    # Save to disk
    with open(outfile, "wb") as f:
        pickle.dump(data, f)

    # Save file for manual labels
    if labeldata is None:
        print(f"Assign manual labels here: {labelfile}")
        good = (data["kdar_manual"] != "T") * (
            (data["int_snr"] > int_snr_threshold)
            + (data["max_snr"] > max_snr_threshold)
        )
        data.loc[good].to_string(
            labelfile,
            columns=[
                "gname",
                "dataset",
                "rrl_velocity",
                "tp_velocity",
                "max_snr",
                "int_snr",
                "kdar_wise",
                "kdar_manual",
                "kdar_manual_qf",
            ],
            float_format="{:.1f}".format,
        )

    # Save plots
    with open(plotfile, "w") as f:
        f.write(r"\documentclass{article}" + "\n")
        f.write(r"\usepackage{graphicx}" + "\n")
        f.write(
            r"\usepackage[paperwidth=28in,paperheight=12in,margin=0.1in]{geometry}"
            + "\n"
        )
        f.write(r"\begin{document}" + "\n")
        for fname in plot_fnames:
            f.write(r"\begin{figure}" + "\n")
            f.write(r"\centering" + "\n")
            f.write(
                r"\includegraphics[width=\textwidth]{{"
                + fname.replace(".pdf", "")
                + "}.pdf}\n"
            )
            f.write(r"\end{figure}" + "\n")
            f.write(r"\clearpage" + "\n")
        f.write(r"\end{document}")
    # Compile plots
    os.system(f"pdflatex -interaction=batchmode {plotfile}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot IGPS HI absorption spectra, derive statistics, generate label file",
        prog="generate_labels.py",
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
        help="HI absorption data file",
    )
    parser.add_argument(
        "--buffer",
        type=float,
        default=25.0,
        help="Velocity buffer (km/s)",
    )
    parser.add_argument(
        "--int_snr_threshold",
        type=float,
        default=3.0,
        help="Integrated ignal-to-noise ratio threshold",
    )
    parser.add_argument(
        "--max_snr_threshold",
        type=float,
        default=5.0,
        help="Peak signal-to-noise ratio threshold",
    )
    parser.add_argument(
        "--labelfile",
        type=str,
        default="manual_labels.csv",
        help="Label file including assigned QFs",
    )
    parser.add_argument(
        "--imagedir",
        type=str,
        default="/data/igps_hi",
        help="Directory where IGPS data are saved",
    )
    parser.add_argument(
        "--outfile",
        type=str,
        default="manual_labeled.pkl",
        help="Output data file with labels",
    )
    parser.add_argument(
        "--plotfile",
        type=str,
        default="labeled_plots.tex",
        help="LaTeX file for plot compilation",
    )
    args = parser.parse_args()
    generate_labels(**vars(args))


if __name__ == "__main__":
    main()

"""
generate_spectra.py
Generate IGPS continuum images, filtered HI data cubes, and HI absorption spectra
for all WISE Catalog nebulae.

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
Trey Wenger - March 2024 - v1.0
"""

import os
import pickle
import argparse
import glob
import sqlite3

import pandas as pd
import numpy as np

from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.wcs import WCS

from igps_kdar import __version__


def load_dataset(dataset, datadir):
    """
    Load SGPS, VGPS, or CGPS datasets.

    Inputs:
        dataset :: string
            One of "sgps", "vgps", or "cgps"
        datadir :: string
            Where the data are saved. Data should be in sub directories
            like "{datadir}/sgps", "{datadir}/vgps", "{datadir}/cgps"

    Returns:
        datasets :: list of dictionaries
            Dictionaries containing relevant dataset information
    """
    datasets = []

    # get data files
    if dataset == "sgps":
        line_fnames = glob.glob(os.path.join(datadir, "sgps/*_line.fits"))
        cont_fnames = [
            line_fname.replace("_line.fits", "_cont_interp.fits")
            for line_fname in line_fnames
        ]
    elif dataset == "vgps":
        cont_fnames = glob.glob(os.path.join(datadir, "vgps/*_cont.Tb.fits"))
        line_fnames = [cont_fname.replace("_cont", "") for cont_fname in cont_fnames]
    elif dataset == "cgps":
        cont_fnames = glob.glob(os.path.join(datadir, "cgps/*_1420_MHz_I_image.fits"))
        line_fnames = [
            cont_fname.replace("1420_MHz_I", "HI_line") for cont_fname in cont_fnames
        ]

    for line_fname, cont_fname in zip(line_fnames, cont_fnames):
        print(cont_fname, line_fname)
        # load headers to check that files exist
        cont_header = fits.getheader(cont_fname)
        line_header = fits.getheader(line_fname)

        # fix headers, load WCS
        if dataset == "vgps":
            cont_header["CUNIT3"] = "m/s"
            line_header["CUNIT3"] = "m/s"
        if dataset == "cgps":
            cont_header["CDELT4"] = 1.0
            line_header["CDELT4"] = 1.0
        cont_wcs = WCS(cont_header).celestial
        line_wcs = WCS(line_header).celestial

        # generate VLSR axis
        vlsr = line_header["CRVAL3"] + line_header["CDELT3"] * (
            np.arange(line_header["NAXIS3"]) + 1 - line_header["CRPIX3"]
        )

        # check orientation of VLSR axis
        reverse = False
        if vlsr[1] - vlsr[0] < 0.0:
            vlsr = vlsr[::-1]
            reverse = True

        # get unique filename label
        label = dataset.upper()
        if dataset == "vgps":
            label += f"_{line_fname.split('/')[-1].replace('.Tb.fits', '')}"
        elif dataset == "cgps":
            label += f"_{line_fname.split('/')[-1].replace('_HI_line_image.fits', '')}"
        elif dataset == "sgps":
            label += f"_{line_fname.split('/')[-1].replace('_line.fits', '')}"

        # image footprints. Offset by one pixel to account for edge effects
        cont_footprint = cont_wcs.calc_footprint() * u.deg
        cont_xlo = min(cont_footprint[:2, 0])
        cont_xhi = max(cont_footprint[2:, 0])
        cont_ylo = max(cont_footprint[0, 1], cont_footprint[3, 1])
        cont_yhi = min(cont_footprint[1, 1], cont_footprint[2, 1])
        cont_footprint = [
            cont_xlo + cont_wcs.wcs.cdelt[0] * u.deg,
            cont_ylo + cont_wcs.wcs.cdelt[1] * u.deg,
            cont_xhi - cont_wcs.wcs.cdelt[0] * u.deg,
            cont_yhi - cont_wcs.wcs.cdelt[1] * u.deg,
        ]

        line_footprint = line_wcs.calc_footprint() * u.deg
        line_xlo = min(line_footprint[:2, 0])
        line_xhi = max(line_footprint[2:, 0])
        line_ylo = max(line_footprint[0, 1], line_footprint[3, 1])
        line_yhi = min(line_footprint[1, 1], line_footprint[2, 1])
        line_footprint = [
            line_xlo + line_wcs.wcs.cdelt[0] * u.deg,
            line_ylo + line_wcs.wcs.cdelt[1] * u.deg,
            line_xhi - line_wcs.wcs.cdelt[0] * u.deg,
            line_yhi - line_wcs.wcs.cdelt[1] * u.deg,
        ]

        datasets.append(
            {
                "dataset": dataset,
                "cont_header": cont_header,
                "line_header": line_header,
                "cont_fname": cont_fname,
                "line_fname": line_fname,
                "cont_wcs": cont_wcs,
                "line_wcs": line_wcs,
                "cont_footprint": cont_footprint,
                "line_footprint": line_footprint,
                "vlsr": vlsr / 1000.0,  # km/s
                "reverse": reverse,
                "label": label,
            }
        )
    return datasets


def load_wise_catalog(db):
    """
    Read the WISE Catalog database and return relevant data

    Inputs:
        db :: string
            Filepath to database

    Returns:
        wise_data :: pandas.DataFrame
            Relevant data
    """
    # Get WISE Catalog data
    with sqlite3.connect(db) as conn:
        cur = conn.cursor()
        cur.execute("PRAGMA foreign_keys = ON")
        cur.execute(
            """
            SELECT cat.gname, cat.radius, cat.glong, cat.glat, det.vlsr,
            det.e_vlsr, dist.vlsr_tangent, dist.vlsr_tangent_err_neg,
            dist.vlsr_tangent_err_pos, cat.kdar, cat.dist_method,
            cat.dist_author, det.lines, det.telescope, det.author
            FROM Detections det
            INNER JOIN CatalogDetections catdet on catdet.detection_id = det.id
            INNER JOIN Catalog cat on catdet.catalog_id = cat.id
            INNER JOIN Distances_Reid2019 dist ON cat.id = dist.catalog_id
            WHERE det.source = "WISE Catalog" AND det.component IS NULL
            """
        )
        wise_data = pd.DataFrame(
            cur.fetchall(), columns=[desc[0] for desc in cur.description]
        )
        print(f"Found {len(wise_data)} WISE Catalog detections")
    return wise_data


def get_data(dataset):
    """
    Read and return line and continuum data

    Inputs:
        dataset :: dictionary
            The dataset dictionary to be read

    Returns:
        cont_data :: 2-D array of scalars
            The continuum image
        line_data :: 3-D array of scalars
            The line data cube
    """
    cont_hdu = fits.open(dataset["cont_fname"], ignore_blank=True)[0]
    line_hdu = fits.open(dataset["line_fname"], ignore_blank=True)[0]

    # convert SGPS continuum data to brightness temperature
    if dataset["dataset"] == "sgps":
        freq = dataset["line_header"]["RESTFREQ"] / 1.0e9
        bmin = dataset["cont_header"]["BMIN"]
        bmaj = dataset["cont_header"]["BMAJ"]
        factor = 1.222e6 / (freq**2.0 * 3600.0**2.0 * bmin * bmaj)
        cont_hdu.header["BUNIT"] = "K"
        cont_hdu.data *= factor

    # reverse VLSR axis if necessary
    if dataset["reverse"]:
        line_hdu.data = line_hdu.data.T[:, :, ::-1].T

    # remove Stokes axis
    cont_data = cont_hdu.data
    while len(cont_data.shape) > 2:
        cont_data = cont_data[0]
    line_data = line_hdu.data
    while len(line_data.shape) > 3:
        line_data = line_data[0]

    return cont_data.T, line_data.T


def matched_cutout(coord, dataset, imsize):
    """
    Generate a matched continuum image and line cube cutout. Return
    HDU with updated WCS parameters.

    Inputs:
        coord :: astropy.coordinates.SkyCoord
            Coordinate of source
        dataset :: dictionary
            The relevant dataset dictionary
        imsize :: scalar
            The maximum sub-image size (degrees)

    Returns:
        cont_cutout :: 2-D array of scalars
            Continuum image cutout
        line_cutout :: 2-D array of scalars
            Line image cutout
        cutout_wcs :: wcs.WCS
            WCS for cutout images
    """
    # Cutout sub-image boundaries
    imsize = imsize * u.deg
    xlo = coord.l + imsize / 2.0
    xhi = coord.l - imsize / 2.0
    ylo = coord.b - imsize / 2.0
    yhi = coord.b + imsize / 2.0

    # get cutout edges
    xlo = min(dataset["cont_footprint"][0], dataset["line_footprint"][0], xlo)
    xhi = max(dataset["cont_footprint"][2], dataset["line_footprint"][2], xhi)
    ylo = max(dataset["cont_footprint"][1], dataset["line_footprint"][1], ylo)
    yhi = min(dataset["cont_footprint"][3], dataset["line_footprint"][3], yhi)
    start = SkyCoord(xlo, ylo, frame="galactic")
    end = SkyCoord(xhi, yhi, frame="galactic")

    # get edge indices
    cont_ylo_idx, cont_xlo_idx = dataset["cont_wcs"].world_to_array_index(start)
    cont_yhi_idx, cont_xhi_idx = dataset["cont_wcs"].world_to_array_index(end)
    line_ylo_idx, line_xlo_idx = dataset["line_wcs"].world_to_array_index(start)
    line_yhi_idx, line_xhi_idx = dataset["line_wcs"].world_to_array_index(end)

    # get sub-images, update WCS
    cont_data, line_data = get_data(dataset)
    cont_cutout = cont_data[cont_xlo_idx:cont_xhi_idx, cont_ylo_idx:cont_yhi_idx]
    line_cutout = line_data[line_xlo_idx:line_xhi_idx, line_ylo_idx:line_yhi_idx]
    cutout_wcs = dataset["cont_wcs"][
        cont_ylo_idx:cont_yhi_idx, cont_xlo_idx:cont_xhi_idx
    ]
    return cont_cutout, line_cutout, cutout_wcs


def extract_spectrum(coord, dataset, imsize, scale_filter):
    """
    Apply spatial filter and extract absorption spectrum.

    Inputs:
        coord :: astropy.coordinates.SkyCoord
            Coordinate of source
        dataset :: dictionary
            The relevant dataset dictionary
        imsize :: scalar
            The maximum sub-image size (degrees)
        scale_filter :: scalar
            The spatial filter FWHM (arcmin)

    Returns:
        cont_image :: 2-D array of scalars
            Continuum image
        cont_image_filtered :: 2-D array of scalars
            Filtered continuum image
        line_image :: 3-D array of scalars
            Line data cube
        line_image_filtered :: 3-D array of scalars
            Filtered line data cube
        wcs :: wcs.WCS
            WCS for cutout images
        cont_tb :: scalar
            Continuum brightness temperature (K)
        cont_tb_filtered :: scalar
            Filtered continuum brightness temperature (K)
        source_pix :: 2-element list of scalars
            The array location of the source in cutout image
        on_spec :: 1-D array of scalars
            Absorption spectrum
            = -(filtered (on) - continuum) / continuum
            = 1 - exp(-tau)
        rms :: 1-D array of scalars
            Estimated spectral rms
    """
    # load data
    cont_image, line_image, wcs = matched_cutout(coord, dataset, imsize)

    # ensure cont and line image are same size
    if cont_image.shape != line_image.shape[0:2]:
        print(wcs)
        raise ValueError("Shape mismatch!")

    # replace NaNs
    cont_image[np.isnan(cont_image)] = 0.0
    line_image[np.isnan(line_image)] = 0.0

    # replace bad data
    bad = np.abs(line_image) > 1.0e3
    line_image[bad] = 0.0

    # location of source in cutout
    source_pix = wcs.world_to_array_index(coord)[::-1]

    # continuum brightness temperature (K)
    cont_tb = cont_image[source_pix]
    if cont_tb <= 0.0:
        return None

    # fft to visibility plane, shift so zero-frequency is centered
    line_vis = np.fft.fftshift(np.fft.fft2(line_image, axes=(0, 1)))
    cont_vis = np.fft.fftshift(np.fft.fft2(cont_image))

    # get axis frequencies (deg-1)
    freq_x = np.fft.fftshift(
        np.fft.fftfreq(line_image.shape[0], dataset["line_header"]["CDELT1"])
    )
    freq_y = np.fft.fftshift(
        np.fft.fftfreq(line_image.shape[1], dataset["line_header"]["CDELT2"])
    )
    freq_x, freq_y = np.meshgrid(freq_x, freq_y, indexing="ij")
    freq = np.sqrt(freq_x**2.0 + freq_y**2.0)

    # generate gaussian filter
    spatial_scale = np.divide(60.0, freq, where=(np.abs(freq) > 0.0))  # arcmin
    spatial_scale[freq == 0.0] = imsize * 60.0  # arcmin
    gaussian_filter = np.exp(-4.0 * np.log(2.0) * (spatial_scale / scale_filter) ** 2.0)

    # filter large spatial scales (arcmin)
    line_vis = line_vis * gaussian_filter[..., np.newaxis]
    cont_vis = cont_vis * gaussian_filter

    # ifft to image plane
    line_image_filtered = np.real(np.fft.ifft2(np.fft.ifftshift(line_vis), axes=(0, 1)))
    cont_image_filtered = np.real(np.fft.ifft2(np.fft.ifftshift(cont_vis)))

    # filtered continuum brightness temperature (K)
    cont_tb_filtered = cont_image_filtered[source_pix]

    # extract filtered "on" spectrum
    on_spec = line_image_filtered[source_pix[0], source_pix[1], :].copy()

    # mask source
    masked_line_image = line_image_filtered.copy()
    masked_line_image[
        source_pix[0] - 3 : source_pix[0] + 4,
        source_pix[1] - 3 : source_pix[1] + 4,
        :,
    ] = np.nan

    # mask edges
    masked_line_image[:3, :, :] = np.nan
    masked_line_image[-3:, :, :] = np.nan
    masked_line_image[:, :3, :] = np.nan
    masked_line_image[:, -3:, :] = np.nan

    # estimate rms across filtered image
    med = np.nanmedian(masked_line_image, axis=(0, 1))
    rms = 1.4826 * np.nanmedian(
        np.abs(masked_line_image - med),
        axis=(0, 1),
    )
    return (
        cont_image,
        cont_image_filtered,
        line_image,
        line_image_filtered,
        wcs,
        cont_tb,
        cont_tb_filtered,
        source_pix,
        on_spec,
        rms,
    )


def generate_spectra(
    db,
    datadir="/data",
    imsize=0.5,
    scale_filter=10.0,
    imagedir="data",
    outfile="wise_hi_absorption.pkl",
):
    """
    Generate HI absorption spectra for all WISE catalog sources.

    Inputs:
        db :: string
            Filepath to WISE Catalog database
        datadir :: string
            Where the data are saved. Data should be in sub directories
            like "{datadir}/sgps", "{datadir}/vgps", "{datadir}/cgps"
        imsize :: scalar
            Sub-image size (degrees)
        scale_filter :: scalar
            Spatial filter FWHM (arcmin)
        imagedir :: string
            Directory where images are stored. Convention is
                {imagedir}/{wise_name}_{dataset}_cont.fits
                {imagedir}/{wise_name}_{dataset}_line_filtered.fits
        outfile :: string
            Filename where spectra are stored

    Returns: Nothing
    """
    # create spectra directory
    if not os.path.exists(imagedir):
        os.mkdir(imagedir)

    # get WISE data
    wise_data = load_wise_catalog(db)

    # load images
    image_datasets = []
    image_datasets += load_dataset("vgps", datadir)
    image_datasets += load_dataset("cgps", datadir)
    image_datasets += load_dataset("sgps", datadir)

    # storage for spectra
    data = []

    # Loop over WISE detections
    for i, wise in wise_data.iterrows():
        # skip outer galaxy
        if wise["glong"] > 90.0 and wise["glong"] < 270.0:
            continue
        if wise["glong"] < 90.0 and wise["vlsr"] < 0.0:
            continue
        if wise["glong"] > 270.0 and wise["vlsr"] > 0.0:
            continue

        print(f"{i+1}/{len(wise_data)} {wise['gname']}")

        # Loop over datasets
        for dataset in image_datasets:
            # skip if source is within three pixels of the edge of the dataset
            coord = SkyCoord(
                wise["glong"] * u.deg, wise["glat"] * u.deg, frame="galactic"
            )
            cont_footprint = dataset["cont_footprint"]
            line_footprint = dataset["line_footprint"]
            lpix = dataset["cont_wcs"].wcs.cdelt[0] * u.deg
            bpix = dataset["cont_wcs"].wcs.cdelt[1] * u.deg
            if (
                coord.l > cont_footprint[0] + 3 * lpix
                or coord.l < cont_footprint[2] - 3 * lpix
                or coord.b < cont_footprint[1] + 3 * bpix
                or coord.b > cont_footprint[3] - 3 * bpix
                or coord.l > line_footprint[0] + 3 * lpix
                or coord.l < line_footprint[2] - 3 * lpix
                or coord.b < line_footprint[1] + 3 * bpix
                or coord.b > line_footprint[3] - 3 * bpix
            ):
                continue

            # get spectra
            output = extract_spectrum(coord, dataset, imsize, scale_filter)
            if output is None:
                continue
            (
                cont_image,
                cont_image_filtered,
                line_image,
                line_image_filtered,
                wcs,
                cont_tb,
                cont_tb_filtered,
                source_pix,
                on_spec,
                rms,
            ) = output

            # save continuum image
            cont_hdu = fits.PrimaryHDU()
            cont_hdu.data = cont_image.T
            cont_hdu.header = wcs.to_header()
            cont_hdu.header["OBJECT"] = wise["gname"]
            cont_hdu.header["SRCTB"] = cont_tb
            cont_hdu.header["LOCX"] = source_pix[0]
            cont_hdu.header["LOCY"] = source_pix[1]
            fname = f"{imagedir}/{wise['gname']}_{dataset['label']}_cont.fits"
            cont_hdu.writeto(fname, overwrite=True, output_verify="silentfix")

            # save filtered continuum image
            cont_hdu = fits.PrimaryHDU()
            cont_hdu.data = cont_image_filtered.T
            cont_hdu.header = wcs.to_header()
            cont_hdu.header["OBJECT"] = wise["gname"]
            cont_hdu.header["SRCTB"] = cont_tb_filtered
            cont_hdu.header["LOCX"] = source_pix[0]
            cont_hdu.header["LOCY"] = source_pix[1]
            fname = f"{imagedir}/{wise['gname']}_{dataset['label']}_cont_filtered.fits"
            cont_hdu.writeto(fname, overwrite=True, output_verify="silentfix")

            # save line image
            wcs = wcs.sub([1, 2, 0])
            wcs.wcs.ctype[2] = "VRAD"
            wcs.wcs.crpix[2] = 1
            wcs.wcs.crval[2] = dataset["vlsr"][0]
            wcs.wcs.cdelt[2] = dataset["vlsr"][1] - dataset["vlsr"][0]
            line_hdu = fits.PrimaryHDU()
            line_hdu.data = line_image.T
            line_hdu.header = wcs.to_header()
            line_hdu.header["OBJECT"] = wise["gname"]
            line_hdu.header["BUNIT"] = "K"
            line_hdu.header["SRCTB"] = cont_tb
            line_hdu.header["LOCX"] = source_pix[0]
            line_hdu.header["LOCY"] = source_pix[1]
            fname = f"{imagedir}/{wise['gname']}_{dataset['label']}_line.fits"
            line_hdu.writeto(fname, overwrite=True, output_verify="silentfix")

            # save filtered line image
            line_hdu = fits.PrimaryHDU()
            line_hdu.data = line_image_filtered.T
            line_hdu.header = wcs.to_header()
            line_hdu.header["OBJECT"] = wise["gname"]
            line_hdu.header["BUNIT"] = "K"
            line_hdu.header["SRCTB"] = cont_tb_filtered
            line_hdu.header["LOCX"] = source_pix[0]
            line_hdu.header["LOCY"] = source_pix[1]
            fname = f"{imagedir}/{wise['gname']}_{dataset['label']}_line_filtered.fits"
            line_hdu.writeto(fname, overwrite=True, output_verify="silentfix")

            # absorption stats between 0 and TP
            spectrum = -1.0 * on_spec  # Positive = absorption

            if wise["glong"] < 90.0:
                start_idx = np.argmin(np.abs(dataset["vlsr"] - 0.0))
                end_idx = np.argmin(np.abs(dataset["vlsr"] - wise["vlsr_tangent"]))
            elif wise["glong"] > 270.0:
                start_idx = np.argmin(np.abs(dataset["vlsr"] - wise["vlsr_tangent"]))
                end_idx = np.argmin(np.abs(dataset["vlsr"] - 0.0))
            max_snr = np.max(
                spectrum[start_idx : end_idx + 1] / rms[start_idx : end_idx + 1]
            )
            int_snr = np.sum(spectrum[start_idx : end_idx + 1]) / np.sqrt(
                np.sum(rms[start_idx : end_idx + 1] ** 2.0)
            )
            data.append(
                {
                    "gname": wise["gname"],
                    "glong": wise["glong"],
                    "glat": wise["glat"],
                    "radius": wise["radius"],
                    "dataset": dataset["label"],
                    "tp_velocity": wise["vlsr_tangent"],
                    "rrl_velocity": wise["vlsr"],
                    "kdar_wise": wise["kdar"],
                    "max_snr": max_snr,
                    "int_snr": int_snr,
                    "rms": rms,
                    "velocity": dataset["vlsr"],
                    "spectrum": spectrum,
                }
            )
    data = pd.DataFrame(data)
    with open(outfile, "wb") as f:
        pickle.dump(data, f)


def main():
    parser = argparse.ArgumentParser(
        description="Generate IGPS HI absorption spectra for all WISE Catalog sources",
        prog="generate_spectra.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=__version__,
    )
    parser.add_argument(
        "db",
        type=str,
        help="WISE Catalog database filepath",
    )
    parser.add_argument(
        "outfile",
        type=str,
        help="Where spectra data are saved",
    )
    parser.add_argument(
        "--datadir",
        type=str,
        default="/data",
        help="Directory where IGPS data are saved",
    )
    parser.add_argument(
        "--imsize",
        type=float,
        default=0.5,
        help="Sub-image size (deg)",
    )
    parser.add_argument(
        "--scale_filter",
        type=float,
        default=10.0,
        help="Spatial filter FWHM (arcmin)",
    )
    parser.add_argument(
        "--imagedir",
        type=str,
        default="data",
        help="Output directory for images",
    )
    args = parser.parse_args()
    generate_spectra(**vars(args))


if __name__ == "__main__":
    main()

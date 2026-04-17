# HI-KDAR
Resolve the kinematic distance ambiguity (KDA) using HI absorption observations
from the International Galactic Plane Survey (IGPS). See Wenger et al. (in prep.).

## Installation

```bash
conda create -n kdar -c conda-forge python==3.11 ipython
conda activate kdar
python -m pip install .
```

## Synthetic Spectra

The program `igps-kdar-simulate-spectra` executes the `main` function of `igps_kdar/simulate_spectra.py`.
This function creates synthetic HI absorption spectra which are used to train and test the various KDAR methods.

```bash
igps-kdar-simulate-spectra --help
```

## Usage

The program `igps-kdar-generate-spectra` executes the `main` function of `igps_kdar/generate_spectra.py`.
This function creates IGPS continuum images and spatially filtered HI spectra towards all WISE Catalog HII regions.

```bash
igps-kdar-generate-spectra --help
```

```
usage: generate_spectra.py [-h] [--version] [--datadir DATADIR] [--imsize IMSIZE] [--scale_filter SCALE_FILTER]
                           [--outdir OUTDIR]
                           db

Generate IGPS HI absorption spectra for all WISE Catalog sources

positional arguments:
  db                    WISE Catalog database filepath

options:
  -h, --help            show this help message and exit
  --version             show program's version number and exit
  --datadir DATADIR     Directory where IGPS data are saved (default: /data)
  --imsize IMSIZE       Sub-image size (deg) (default: 0.5)
  --scale_filter SCALE_FILTER
                        Spatial filter size (arcmin) (default: 10.0)
  --outdir OUTDIR       Output directory for data products (default: data)
```


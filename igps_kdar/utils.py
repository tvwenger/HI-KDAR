"""
utils.py
Package utilities.

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

import numpy as np

import tensorflow as tf


class RobustScaler:
    """
    Data scaling based on the inner-quartile range.
    """

    def __init__(self):
        """
        Initialize a new RobustScaler.

        Inputs: None
        Returns: Nothing
        """
        self.range = None
        self.mean = None

    def fit(self, X):
        """
        Determine scale factors per channel.

        Inputs:
            X :: ND-array of scalars
                Data. Channel axis must be last.

        Returns: Nothing
        """
        axis = tuple(range(len(X.shape)))[:-1]
        iqr = np.nanpercentile(X, [25.0, 75.0], axis=axis)
        self.range = iqr[1] - iqr[0]
        self.mean = np.nanmean(X, axis=axis)

    def transform(self, X, channels=None):
        """
        Apply transformation to some data.

        Inputs:
            X :: ND-array of scalars
                Data to be transformed. Channel axis must be last.
            channels :: iterable of integers
                Channels to transform. None == all channels.

        Returns:
            newX :: ND-array of scalars
                Transformed data
        """
        if self.mean is None:
            raise ValueError("RobustScaler not yet fit!")
        if channels is None:
            channels = range(X.shape[-1])
        newX = X.copy()
        for channel in channels:
            newX[..., channel] = (X[..., channel] - self.mean[channel]) / self.range[
                channel
            ]
        return newX


def smooth_regrid_spec(data, old_velocity_axis, new_velocity_axis):
    """
    Smooth and re-grid a spectrum to a new velocity axis using sinc interpolation.

    Inputs:
        data :: 1-D array of scalars
            Spectrum
        old_velocity_axis :: 1-D array of scalars
            Current velocity axis
        new_velocity_axis :: 1-D array of scalars
            Regridded velocity axis

    Returns: newdata
        newdata :: 1-D array of scalars
            Regridded spectrum
    """
    old_res = old_velocity_axis[1] - old_velocity_axis[0]
    new_res = new_velocity_axis[1] - new_velocity_axis[0]
    if new_res < old_res:
        raise ValueError("Cannot smooth to a finer resolution!")

    # construct sinc weights
    sinc_wts = np.sinc((new_velocity_axis[:, None] - old_velocity_axis) / new_res)
    # normalize weights
    sinc_wts = (sinc_wts.T / np.sum(sinc_wts, axis=1)).T

    # catch out of bounds
    out = (new_velocity_axis < old_velocity_axis[0]) + (
        new_velocity_axis > old_velocity_axis[-1]
    )
    sinc_wts[out] = np.nan

    # catch NaNs
    isnan = np.isnan(data)
    data[isnan] = 0.0
    nan_weights = np.ones_like(data)
    nan_weights[isnan] = 0.0

    # apply, handle NaNs
    new_data = np.nansum(sinc_wts * data, axis=1)
    new_nan_weights = np.nansum(sinc_wts * nan_weights, axis=1)
    new_nan_weights[new_nan_weights < 0.5] = np.nan
    new_data = new_data / new_nan_weights

    return new_data


def calc_ce(proby, truey, nbins=10):
    """
    Calculate the expected and maximum calibration error.

    Inputs:
        proby :: 2-D array of scalars
            Predicted probabilities
        truey :: 2-D array of scalars
            True labels
        nbins :: integer
            Number of bins

    Returns:
        ece :: scalar
            Expected calibration error
        mce :: scalar
            Maximum calibration error
        bins :: 1-D array of scalars
            Bin edges
        bin_pop :: 1-D array of integers
            Population size of each bin
        bin_accuracy :: 1-D array of scalars
            Accuracy in each bin
    """
    predy = tf.one_hot(tf.argmax(proby, axis=1), depth=proby.shape[1]).numpy()
    accurate = np.all(truey == predy, axis=1)
    max_proby = np.max(proby, axis=1)

    bins = np.linspace(0.0, 1.0, nbins + 1)
    bins[-1] += 0.001  # include 1.0 in last bin
    bin_pop = np.zeros(nbins, dtype=int)
    bin_avg_prob = np.zeros(nbins, dtype=float)
    bin_accuracy = np.zeros(nbins, dtype=float)

    for i in range(nbins):
        good = (max_proby >= bins[i]) * (max_proby < bins[i + 1])
        bin_pop[i] = np.sum(good)
        if bin_pop[i] > 0:
            bin_avg_prob[i] = np.mean(max_proby[good])
            bin_accuracy[i] = np.sum(accurate[good]) / bin_pop[i]

    ece = np.average(np.abs(bin_avg_prob - bin_accuracy), weights=bin_pop)
    mce = np.max(np.abs(bin_avg_prob - bin_accuracy))

    return ece, mce, bins, bin_pop, bin_accuracy


def predict_calibrated(confy, temperature=1.0):
    """
    Return calibrated model confidence probabilities.

    Inputs:
        confy :: nd-array of scalars
            The uncalibrated model confidence.
        temperature :: scalar
            Temperature to calibrate

    Returns:
        proby :: nd-array of scalars
            Calibrated confidence probabilities
    """
    scaled_logits = np.log(confy) / temperature
    return tf.nn.softmax(scaled_logits, axis=1).numpy()

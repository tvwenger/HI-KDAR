"""
svm_classifier.py
Support vector machine classifier.

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
import pandas as pd
import matplotlib.pyplot as plt

from keras.losses import CategoricalCrossentropy
from scipy.optimize import minimize
from sklearn.svm import SVC

from igps_kdar.utils import calc_ce, predict_calibrated


class SVMClassifier:
    """
    Support vector machine (SVM) classifier.
    """

    def __init__(self, **kwargs):
        """
        Initialize a new classifier.

        Inputs:
            **kwargs :: passed to SVN

        Returns: classifier
            classifier :: SVMClassifier
                A new SVMClassifier instance
        """
        super().__init__()
        self.svc = SVC(probability=True, **kwargs)

    def fit(self, trainX, trainy):
        """
        Train the classifier.

        Inputs:
            trainX :: ND-array of scalars
                The training feature set
            trainy :: 1-D array of scalars
                Train training label set

        Returns: Nothing
        """
        self.svc.fit(
            trainX,
            trainy,
        )

    def predict(self, X):
        """
        Predict on features. Returns "one hot" prediction.

        Inputs:
            X :: nd-array of scalars
                Features

        Returns:
            predy :: 1-D array
                Predicted labels
        """
        return self.svc.predict(X)

    def predict_prob(self, X):
        """
        Predict on features. Returns prediction probabilities.

        Inputs:
            X :: nd-array of scalars
                Features

        Returns:
            proby :: nd-array
                Predicted probabilities
        """
        log_proby = self.svc.predict_log_proba(X)
        return np.exp(log_proby)

    def score(self, testX, testy):
        """
        Calculate the accuracy of the classifier for some test data.

        Inputs:
            testX :: nd-array of scalars
                The test feature set
            testy :: 1-D array of scalars
                The test label set

        Returns:
            score :: scalar
                the accuracy
        """
        predy = self.predict(testX)
        score = (testy == predy).mean()
        return score

    def calibrate(self, valX, valy, calplot=None, nbins=10):
        """
        Calibrate model probabilities by determining the temperature that minimizes
        the loss for the validation set.

        Inputs:
            valX :: nd-array of scalars
                The validation feature set. Must be the same used during training.
            valy :: 1-D array of scalars
                The validation label set. Must be the same used during training.
            calplot :: string
                If not None, plot the binned accuracy vs. confidence before and
                after calibration.
            nbins :: integer
                Number of probability bins.

        Returns:
            temp :: scalar
                Temperature that calibrates probabilities
        """
        # unpack labels
        valy = pd.get_dummies(valy)[self.svc.classes_].to_numpy().astype(int)

        cce = CategoricalCrossentropy()
        proby = self.predict_prob(valX)

        old_loss = cce(valy, proby).numpy()
        old_ece, old_mce, bins, old_bin_pop, old_bin_accuracy = calc_ce(
            proby, valy, nbins=nbins
        )

        def loss(temp):
            scaled_proby = predict_calibrated(proby, temperature=temp)
            return cce(valy, scaled_proby).numpy()

        res = minimize(loss, 1.0, bounds=[(0.0, np.inf)])
        temp = res.x[0]

        new_proby = predict_calibrated(proby, temperature=temp)
        new_loss = cce(valy, new_proby).numpy()
        new_ece, new_mce, bins, new_bin_pop, new_bin_accuracy = calc_ce(
            new_proby, valy, nbins=nbins
        )

        if calplot is not None:
            fig, (ax1, ax2) = plt.subplots(2, figsize=(5, 8))
            bin_width = bins[1] - bins[0]
            bin_center = bins[:-1] + (bins[1:] - bins[:-1]) / 2
            ax1.bar(
                bins[:-1],
                old_bin_accuracy,
                width=bin_width,
                color="gray",
                align="edge",
                edgecolor="k",
                linewidth=0.1,
            )
            ax1.plot([0, 1], [0, 1], "k--")
            for x, num in zip(bin_center, old_bin_pop):
                ax1.text(
                    x,
                    0.05,
                    num,
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
            ax1.text(
                0.05,
                0.95,
                f"ECE = {100*old_ece:.1f}%\n"
                + f"MCE = {100*old_mce:.1f}%\n"
                + f"Loss = {old_loss:.3f}",
                ha="left",
                va="top",
                bbox=dict(boxstyle="round", alpha=0.8, color="white"),
            )
            ax2.bar(
                bins[:-1],
                new_bin_accuracy,
                width=bin_width,
                color="gray",
                align="edge",
                edgecolor="k",
                linewidth=0.1,
            )
            for x, num in zip(bin_center, new_bin_pop):
                ax2.text(
                    x,
                    0.05,
                    num,
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
            ax2.text(
                0.05,
                0.95,
                f"ECE = {100*new_ece:.1f}%\n"
                + f"MCE = {100*new_mce:.1f}%\n"
                + f"Loss = {new_loss:.3f}\n"
                + f"T = {temp:.3f}",
                ha="left",
                va="top",
                bbox=dict(boxstyle="round", alpha=0.8, color="white"),
            )
            ax2.plot([0, 1], [0, 1], "k--")
            ax1.grid(False)
            ax2.grid(False)
            ax1.set_xlim(0, 1)
            ax2.set_xlim(0, 1)
            ax1.set_ylim(0, 1)
            ax2.set_ylim(0, 1)
            ax1.set_xlabel("Confidence")
            ax2.set_xlabel("Probability")
            ax1.set_ylabel("Accuracy")
            ax2.set_ylabel("Accuracy")
            fig.tight_layout()
            fig.savefig(calplot, bbox_inches="tight")
            plt.close(fig)
        return temp

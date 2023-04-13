#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
The pre-processing functions for continuous-valued features.
"""
from sklearn import preprocessing
from .base import TransformBase


class Standard(TransformBase):
    """
    Standard normalization, i.e., zero mean and unit variance.
    """

    def __init__(self):
        super().__init__()
        self.scaler = preprocessing.StandardScaler(with_mean=True, with_std=True)

    def fit(self, x):
        self.scaler.fit(x)
        return self

    def transform(self, x):
        return self.scaler.transform(x)

    def invert(self, x):
        return self.scaler.inverse_transform(x)


class MinMax(TransformBase):
    """
    Rescales the values to the range [0, 1].
    """

    def __init__(self):
        super().__init__()
        self.scaler = preprocessing.MinMaxScaler()

    def fit(self, x):
        self.scaler.fit(x)
        return self

    def transform(self, x):
        return self.scaler.transform(x)

    def invert(self, x):
        return self.scaler.inverse_transform(x)


class Scale(TransformBase):
    """
    Rescales the values to values * ratio.
    """

    def __init__(self, ratio=1.0):
        super().__init__()
        assert ratio != 0, "The ratio cannot be zero."
        self.ratio = ratio

    def fit(self, x):
        return self

    def transform(self, x):
        return x * self.ratio

    def invert(self, x):
        return x / self.ratio

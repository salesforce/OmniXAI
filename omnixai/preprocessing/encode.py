#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
The pre-processing functions for categorical and continuous-valued features.
"""
import warnings

warnings.filterwarnings("ignore", ".*bins whose width are too small.*")

import pandas as pd
from sklearn import preprocessing
from .base import TransformBase


class KBins(TransformBase):
    """
    Discretizes continuous values into bins.
    """

    def __init__(self, n_bins, **kwargs):
        super().__init__()
        self.encoder = preprocessing.KBinsDiscretizer(n_bins=n_bins, encode="ordinal", **kwargs)

    def fit(self, x):
        if isinstance(x, pd.DataFrame):
            x = x.values
        self.encoder.fit(x)
        return self

    def transform(self, x):
        if isinstance(x, pd.DataFrame):
            x = x.values
        return self.encoder.transform(x)

    def invert(self, x):
        return self.encoder.inverse_transform(x)


class OneHot(TransformBase):
    """
    One-hot encoding for categorical values.
    """

    def __init__(self, drop=None, **kwargs):
        super().__init__()
        if drop is None:
            self.encoder = preprocessing.OneHotEncoder(handle_unknown="ignore", **kwargs)
        else:
            self.encoder = preprocessing.OneHotEncoder(drop=drop, **kwargs)

    def fit(self, x):
        if isinstance(x, pd.DataFrame):
            x = x.values
        self.encoder.fit(x)
        return self

    def transform(self, x):
        if isinstance(x, pd.DataFrame):
            x = x.values
        return self.encoder.transform(x).toarray()

    def invert(self, x):
        return self.encoder.inverse_transform(x)

    @property
    def categories(self):
        """
        Returns the categories for each feature.
        """
        return self.encoder.categories_

    def get_feature_names(self, input_features=None):
        """
        Returns the feature names in the transformed data.
        """
        return self.encoder.get_feature_names(input_features)


class Ordinal(TransformBase):
    """
    Ordinal encoding for categorical values.
    """

    def __init__(self):
        super().__init__()
        self.encoder = preprocessing.OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

    def fit(self, x):
        if isinstance(x, pd.DataFrame):
            x = x.values
        self.encoder.fit(x)
        return self

    def transform(self, x):
        if isinstance(x, pd.DataFrame):
            x = x.values
        return self.encoder.transform(x)

    def invert(self, x):
        return self.encoder.inverse_transform(x)

    @property
    def categories(self):
        """
        Returns the categories for each feature.
        """
        return self.encoder.categories_


class LabelEncoder(TransformBase):
    """
    Ordinal encoding for targets/labels.
    """

    def __init__(self):
        super().__init__()
        self.encoder = preprocessing.LabelEncoder()

    def fit(self, x):
        if isinstance(x, pd.DataFrame):
            x = x.values.flatten()
        self.encoder.fit(x)
        return self

    def transform(self, x):
        if isinstance(x, pd.DataFrame):
            x = x.values.flatten()
        return self.encoder.transform(x)

    def invert(self, x):
        return self.encoder.inverse_transform(x.astype(int))

    @property
    def categories(self):
        """
        Returns the class labels.
        """
        return self.encoder.classes_

#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
The base class for all the transforms.
"""
import pandas as pd
from abc import abstractmethod
from ..utils.misc import AutodocABCMeta


class TransformBase(metaclass=AutodocABCMeta):
    """
    Abstract base class for a data pre-processing transform.
    """

    def __init__(self):
        pass

    @abstractmethod
    def fit(self, x):
        """
        Estimates the parameters of the transform.

        :param x: The data for estimating the parameters.
        :return: The current instance.
        """
        raise NotImplementedError

    @abstractmethod
    def transform(self, x):
        """
        Applies the transform to the input data.

        :param x: The data on which to apply the transform.
        :return: The transformed data.
        """
        raise NotImplementedError

    @abstractmethod
    def invert(self, x):
        """
        Applies the inverse transform to the input data.

        :param x: The data on which to apply the inverse transform.
        :return: The inverse transformed data.
        """
        raise NotImplementedError


class Identity(TransformBase):
    """
    Identity transformation.
    """

    def fit(self, x):
        return self

    def transform(self, x):
        if isinstance(x, (pd.DataFrame, pd.Series)):
            x = x.to_numpy()
        return x

    def invert(self, x):
        return x

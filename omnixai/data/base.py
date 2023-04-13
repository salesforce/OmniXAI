#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
The base class for all data types.
"""
from abc import abstractmethod
from ..utils.misc import AutodocABCMeta


class Data(metaclass=AutodocABCMeta):
    """
    Abstract base class for differet data types.
    """

    def __init__(self):
        pass

    @property
    @abstractmethod
    def data_type(self):
        """
        :return: A string indicates the data type, e.g., tabular, image, text or time series
        """
        raise NotImplementedError

    @abstractmethod
    def values(self):
        """
        :return: The raw values of the data object.
        """
        raise NotImplementedError

    @abstractmethod
    def num_samples(self):
        """
        :return: The number samples in the dataset.
        """
        raise NotImplementedError

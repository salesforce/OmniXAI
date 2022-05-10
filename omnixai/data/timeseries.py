#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
The class for time series data.
"""
import numpy as np
import pandas as pd
from typing import Union, List
from .base import Data


class Timeseries(Data):
    """
    This class represents a univariate/multivariate time series dataset, e.g. a batch of time series.
    The dataset may contain a batch of time series whose metric values are stored in a numpy array
    with shape `(batch_size, timestamps, num_variables)`. If there is only one time series, `batch_size`
    is 1.
    """

    data_type = "timeseries"

    def __init__(
            self,
            data: np.ndarray,
            timestamps: List = None,
            variable_names: List = None
    ):
        """
        :param data: A numpy array contains one or a batch of time series. If it has one time series only,
            the shape of ``data`` is `(timestamps, num_variables)`. If it has a batch of time series, the
            shape of ``data`` is `(batch_size, timestamps, num_variables)`.
        :param timestamps: If ``data`` has one time series, ``timestamps`` is a list of the corresponding
            timestamps. If ``data`` has a batch of time series, ``timestamps`` is a batch of lists of the
            timestamps.
        :param variable_names: A list of metric/variable names in time series data.
        """
        super().__init__()
        assert len(data.shape) in [2, 3], \
            "The shape of data should be either (timestamps, num_variables) " \
            "or (batch_size, timestamps, num_variables)"
        if len(data.shape) == 2:
            data = np.expand_dims(data, axis=0)
            if timestamps is not None:
                assert len(timestamps) == data.shape[1], \
                    f"The numbers of timestamps in `data` and `timestamps` don't match, " \
                    f"{data.shape[1]} != {len(timestamps)}"
                timestamps = [timestamps]
        else:
            if timestamps is not None:
                assert len(timestamps) == data.shape[0], \
                    f"The batch size in `data` and `timestamps` don't match, " \
                    f"{data.shape[0]} != {len(timestamps)}"
                assert len(timestamps[0]) == data.shape[1], \
                    f"The numbers of timestamps in `data` and `timestamps` don't match, " \
                    f"{data.shape[1]} != {len(timestamps[0])}"

        if variable_names is not None:
            assert data.shape[-1] == len(variable_names), \
                f"The number of variables in `data` and `variable_names` don't match, " \
                f"{data.shape[-1]} != {len(variable_names)}"

        self.data = data
        if timestamps is None:
            timestamps = [list(range(data.shape[1])) for _ in range(data.shape[0])]
        self.timestamps = np.array(timestamps)
        if variable_names is None:
            variable_names = list(range(data.shape[-1]))
        self.variable_names = variable_names

    def __len__(self):
        """
        Returns the batch_size. Call `ts_len` if the length of time series is needed.
        """
        return len(self.data)

    def __repr__(self):
        return repr(self.to_pd())

    def __getitem__(self, i: Union[int, slice, list]):
        """
        Gets a subset of the batched time series given the indices.

        :param i: An integer index or slice.
        :return: A subset of the batched time series.
        :rtype: Timeseries
        """
        return Timeseries(
            data=self.data[i],
            timestamps=self.timestamps[i],
            variable_names=self.variable_names
        )

    @property
    def ts_len(self):
        """
        Returns the length of the time series.
        """
        return self.data.shape[1]

    @property
    def batch_size(self):
        """
        Returns the batch_size.
        """
        return self.data.shape[0]

    @property
    def shape(self) -> tuple:
        """
        Returns the raw data shape, e.g., (batch_size, timestamps, num_variables).
        If there is only one time series, `batch_size` is 1.

        :return: A tuple for the raw data shape.
        :rtype: tuple
        """
        return self.data.shape

    @property
    def values(self) -> np.ndarray:
        """
        Returns the raw values of the data object.

        :return: A numpy array of the data object.
        :rtype: np.ndarray
        """
        return self.data

    @property
    def columns(self) -> List:
        """
        Gets the metric/variable names.

        :return: The list of the metric/variable names.
        :rtype: List
        """
        return self.variable_names

    @property
    def index(self) -> List:
        """
        Gets the timestamps.

        :return: A list of timestamps.
        :rtype: pd.DatetimeIndex
        """
        return self.timestamps

    def to_pd(self) -> Union[pd.DataFrame, List[pd.DataFrame]]:
        """
        Converts `Timeseries` to `pd.DataFrame`.

        :return: A pandas dataframe or a batch of pandas dataframes representing the time series.
        :rtype: pd.DataFrame
        """
        dfs = []
        for i, x in enumerate(self.data):
            df = pd.DataFrame(x, columns=self.variable_names)
            df.index = self.timestamps[i]
            df.index.name = "timestamp"
            dfs.append(df)
        return dfs if len(dfs) > 1 else dfs[0]

    def to_numpy(self, copy=True) -> np.ndarray:
        """
        Converts `Timeseries` to `np.ndarray`.

        :param copy: `True` if it returns a data copy, or `False` otherwise.
        :return: A numpy ndarray representing the time series.
        :rtype: np.ndarray
        """
        return self.data.copy() if copy else self.data

    def copy(self):
        """
        Returns a copy of the time series instance.

        :return: The copied time series instance.
        :rtype: Timeseries
        """
        return Timeseries(
            data=self.data.copy(),
            timestamps=self.timestamps.copy(),
            variable_names=self.variable_names.copy()
        )

    @classmethod
    def from_pd(cls, df):
        """
        Creates a `Timeseries` instance from one or multiple pandas dataframes.
        `df` is either one pandas dataframe or a list of pandas dataframes.
        The index of each dataframe should contain the timestamps.

        :return A `Timeseries` instance.
        :rtype: Timeseries
        """
        if isinstance(df, pd.DataFrame):
            return cls(
                data=df.values,
                timestamps=list(df.index.values),
                variable_names=list(df.columns)
            )
        elif isinstance(df, (list, tuple)):
            return cls(
                data=np.stack([d.values for d in df]),
                timestamps=[list(d.index.values) for d in df],
                variable_names=list(df[0].columns)
            )
        else:
            raise ValueError(f"`df` can only be `pd.DataFrame` or "
                             f"a list of `pd.DataFrame` instead of {type(df)}")

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
from typing import Union, List, Dict
from .base import Data


class Timeseries(Data):
    """
    This class represents a univariate/multivariate time series dataset.
    The dataset contains a time series whose metric values are stored in a numpy array
    with shape `(timestamps, num_variables)`.
    """

    data_type = "timeseries"

    def __init__(
            self,
            data: np.ndarray,
            timestamps: List = None,
            variable_names: List = None
    ):
        """
        :param data: A numpy array containing a time series. The shape of ``data`` is `(timestamps, num_variables)`.
        :param timestamps: A list of timestamps.
        :param variable_names: A list of metric/variable names.
        """
        super().__init__()
        assert len(data.shape) in [1, 2], \
            "The shape of data should be (timestamps, num_variables)."
        if len(data.shape) == 1:
            data = np.expand_dims(data, axis=-1)

        if timestamps is None:
            timestamps = list(range(data.shape[0]))
        assert len(timestamps) == data.shape[0], \
            f"The numbers of timestamps in `data` and `timestamps` don't match, " \
            f"{data.shape[0]} != {len(timestamps)}"

        if variable_names is not None:
            assert data.shape[-1] == len(variable_names), \
                f"The number of variables in `data` and `variable_names` don't match, " \
                f"{data.shape[-1]} != {len(variable_names)}"
        else:
            variable_names = list(range(data.shape[-1]))

        self.data = pd.DataFrame(
            data,
            columns=variable_names,
            index=timestamps
        )

    def __len__(self):
        """
        Returns the length of the time series.
        """
        return self.data.shape[0]

    def __repr__(self):
        return repr(self.to_pd())

    def __getitem__(self, i: Union[int, slice, list]):
        """
        Gets a subset of the time series given the indices.

        :param i: An integer index or slice.
        :return: A subset of the time series.
        :rtype: Timeseries
        """
        return Timeseries.from_pd(self.data.iloc[i])

    @property
    def ts_len(self) -> int:
        """
        Returns the length of the time series.
        """
        return self.data.shape[0]

    @property
    def shape(self) -> tuple:
        """
        Returns the raw data shape, e.g., (timestamps, num_variables).

        :return: A tuple for the raw data shape.
        :rtype: tuple
        """
        return self.data.shape

    @property
    def values(self) -> np.ndarray:
        """
        Returns the raw values of the data object.

        :return: A numpy array of the data object.
        """
        return self.data.values

    @property
    def columns(self) -> List:
        """
        Gets the metric/variable names.

        :return: The list of the metric/variable names.
        """
        return list(self.data.columns)

    @property
    def index(self) -> List:
        """
        Gets the timestamps.

        :return: A list of timestamps.
        """
        return list(self.data.index)

    def to_pd(self, copy=True) -> Union[pd.DataFrame, List[pd.DataFrame]]:
        """
        Converts `Timeseries` to `pd.DataFrame`.

        :return: A pandas dataframe representing the time series.
        """
        return self.data.copy() if copy else self.data

    def to_numpy(self, copy=True) -> np.ndarray:
        """
        Converts `Timeseries` to `np.ndarray`.

        :param copy: `True` if it returns a data copy, or `False` otherwise.
        :return: A numpy ndarray representing the time series.
        """
        return self.data.values.copy() if copy else self.data.values

    def copy(self):
        """
        Returns a copy of the time series instance.

        :return: The copied time series instance.
        :rtype: Timeseries
        """
        return Timeseries.from_pd(self.data.copy())

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
        else:
            raise ValueError(f"`df` can only be `pd.DataFrame` or "
                             f"a list of `pd.DataFrame` instead of {type(df)}")

    @staticmethod
    def get_timestamp_info(df) -> Dict:
        """
        Returns a dict containing timestamp information, e.g., timestamp index name, timestamp values.

        :param df: The time-series in a pandas dataframe.
        :return: The timestamp information.
        """
        timestamps = df.index.values
        info = {
            "name": df.index.name,
        }
        if isinstance(timestamps[0], (np.int32, np.int64, np.float32, np.float64)):
            values = timestamps.copy()
        elif isinstance(timestamps[0], np.datetime64):
            values = timestamps.astype(np.int64) / (10 ** 9)
        else:
            values = [hash(t) for t in timestamps]
        info["ts2val"] = {ts: val for ts, val in zip(timestamps, values)}
        info["val2ts"] = {val: ts for ts, val in zip(timestamps, values)}
        return info

    @staticmethod
    def reset_timestamp_index(df, timestamp_info) -> pd.DataFrame:
        """
        Moves the timestamp index to a column and converts timestamps into floats.

        :param df: The time-series in a pandas dataframe.
        :param timestamp_info: The timestamp information.
        :return: The converted time-series dataframe.
        """
        d = timestamp_info["ts2val"]
        new_df = pd.DataFrame(df.values, columns=df.columns)
        new_df["@timestamp"] = [d[i] for i in df.index.values]
        return new_df

    @staticmethod
    def restore_timestamp_index(df, timestamp_info) -> pd.DataFrame:
        """
        Moves the timestamp column to the index and converts the floats back to timestamps.

        :param df: The time-series in a pandas dataframe.
        :param timestamp_info: The timestamp information.
        :return: The original time-series dataframe.
        """
        x = df.values[:, :-1]
        d = timestamp_info["val2ts"]
        timestamps = [d[v] for v in df["@timestamp"].values]
        df = pd.DataFrame(x, columns=df.columns[:-1], index=timestamps)
        df.index.name = timestamp_info["name"]
        return df

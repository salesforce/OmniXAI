#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
The class for tabular data.
"""
import numpy as np
import pandas as pd
from typing import Union, List, Dict, Sequence
from .base import Data


class Tabular(Data):
    """
    The class represents a tabular dataset that may contain categorical features,
    continuous-valued features and targets/labels (optional).
    """

    data_type = "tabular"

    def __init__(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        feature_columns: List = None,
        categorical_columns: List = None,
        target_column: Union[str, int] = None,
    ):
        """
        :param data: A pandas dataframe or a numpy array containing the raw data. `data` should have the
            shape `(num_samples, num_features)`.
        :param feature_columns: The feature column names. When ``feature_columns`` is None, ``feature_columns``
            will be the column names in the pandas dataframe or the indices in the numpy array.
        :param categorical_columns: A list of categorical feature names, e.g., a subset of feature column names in
            a pandas dataframe. If ``data`` is a numpy array and ``feature_columns = None``, ``categorical_columns``
            should be the indices of categorical features.
        :param target_column: The target/label column name. Set ``target_column`` to None if there is no
            target column.
        """
        super().__init__()
        assert isinstance(data, (pd.DataFrame, np.ndarray)), "data must be a pandas dataframe or a numpy array."
        # Rename the columns according to ``feature_columns``
        if feature_columns is not None:
            data = pd.DataFrame(data, columns=feature_columns)

        if isinstance(data, pd.DataFrame):
            if categorical_columns:
                assert all(
                    column_name in data for column_name in categorical_columns
                ), f"Some of the categorical_columns are not included in the dataframe."
            if target_column:
                assert target_column in data, f"The target column is not included in the dataframe."
        else:
            if data.ndim == 1:
                data = np.expand_dims(data, axis=0)
            if data.ndim > 2:
                raise ValueError("The dimension of the data should be <= 2.")
            if categorical_columns:
                assert all(
                    type(column_index) == int and 0 <= column_index < data.shape[1]
                    for column_index in categorical_columns
                ), f"The column index must be an integer in [0, {data.shape[1]})"
            if target_column:
                assert (
                    type(target_column) == int and 0 <= target_column < data.shape[1]
                ), f"The column index must be an integer in [0, {data.shape[1]})"
            data = pd.DataFrame(data)

        self.data = data
        self.categorical_cols = categorical_columns if categorical_columns else []
        self.target_col = target_column

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return repr(self.to_pd())

    def iloc(self, i: Union[int, slice, list]):
        """
        Returns the row(s) given an index or a set of indices.

        :param i: An integer index, slice or list.
        :return: A tabular object with the selected rows.
        :rtype: Tabular
        """
        return self.__getitem__(i)

    def __getitem__(self, i: Union[int, slice, list]):
        """
        Get the row or rows given an index or a set of indices.

        :param i: An integer index, slice or list.
        :return: A tabular object with the selected rows.
        :rtype: Tabular
        """
        if isinstance(i, int):
            indices = [i]
        elif isinstance(i, (slice, list)):
            indices = i
        else:
            raise KeyError(f"Indexing a `Tabular` with key {i} of " f"type {type(i).__name__} is not supported.")
        return Tabular(
            data=self.data.iloc[indices], categorical_columns=self.categorical_columns, target_column=self.target_column
        )

    @property
    def shape(self) -> tuple:
        """
        Returns the data shape, e.g., (num_samples, num_features).

        :return: A tuple for the data shape.
        :rtype: tuple
        """
        return self.data.shape

    def num_samples(self) -> int:
        """
        Returns the number of the examples.

        :return: The number of the examples.
        :rtype: int
        """
        return self.data.shape[0]

    @property
    def values(self) -> np.ndarray:
        """
        Returns the raw values of the data object (without feature column names).

        :return: A numpy array of the data object.
        :rtype: np.ndarray
        """
        return self.data.values

    @property
    def categorical_columns(self) -> List:
        """
        Gets the categorical feature names.

        :return: The list of the categorical feature names.
        :rtype: Union[List[str], List[int]]
        """
        return self.categorical_cols

    @property
    def continuous_columns(self) -> List:
        """
        Gets the continuous-valued feature names.

        :return: The list of the continuous-valued feature names.
        :rtype: Union[List[str], List[int]]
        """
        return [c for c in self.data.columns if c not in self.categorical_cols and c != self.target_column]

    @property
    def feature_columns(self) -> List:
        """
        Gets all feature names.

        :return: The list of all the feature column names except the target column.
        :rtype: Union[List[str], List[int]]
        """
        if self.target_column is None:
            return list(self.data.columns)
        else:
            return [c for c in self.data.columns if c != self.target_column]

    @property
    def target_column(self) -> Union[str, int]:
        """
        Gets the target/label column name.

        :return: The target column name, or None if there is no target column.
        :rtype: Union[str, int]
        """
        return self.target_col

    @property
    def columns(self) -> Sequence:
        """
        Gets all the data columns including both the feature columns and
        target/label column.

        :return: The list of the column names.
        :rtype: Sequence
        """
        return self.data.columns

    def to_pd(self, copy=True) -> pd.DataFrame:
        """
        Converts `Tabular` to `pd.DataFrame`.

        :param copy: `True` if it returns a data copy, or `False` otherwise.
        :return: A pandas DataFrame representing the tabular data.
        :rtype: pd.DataFrame
        """
        return self.data.copy() if copy else self.data

    def to_numpy(self, copy=True) -> np.ndarray:
        """
        Converts `Tabular` to `np.ndarray`.

        :param copy: `True` if it returns a data copy, or `False` otherwise.
        :return: A numpy ndarray representing the tabular data.
        :rtype: np.ndarray
        """
        return self.data.values.copy() if copy else self.data.values

    def copy(self):
        """
        Returns a copy of the tabular data.

        :return: The copied tabular data.
        :rtype: Tabular
        """
        return Tabular(
            data=self.to_pd(), categorical_columns=self.categorical_columns, target_column=self.target_column
        )

    def remove_target_column(self):
        """
        Removes the target/label column and returns a new `Tabular` instance.

        :return: The new tabular data without target/label column.
        :rtype: Tabular
        """
        if self.target_col is None:
            return self.copy()
        else:
            return Tabular(
                data=self.to_pd().drop(columns=[self.target_col]), categorical_columns=self.categorical_columns
            )

    def get_target_column(self):
        """
        Returns the target/label column.

        :return: A list of targets or labels.
        :rtype: List
        """
        assert self.target_column is not None, "The target/label column doesn't exist."
        return self.to_pd(copy=False)[[self.target_column]].values.flatten()

    def get_continuous_medians(self) -> Dict:
        """
        Gets the absolute median values of the continuous-valued features.

        :return: A dict storing the absolute median value for each continuous-valued feature.
        :rtype: Dict
        """
        return {c: np.median(np.abs(self.data[c].values.astype(float))) for c in self.continuous_columns}

    def get_continuous_bounds(self) -> tuple:
        """
        Gets the upper and lower bounds of the continuous-valued features.

        :return: The upper and lower bounds, i.e., a tuple of two numpy arrays.
        :rtype: tuple
        """
        min_vals, max_vals = [], []
        for c in self.continuous_columns:
            d = self.data[c].values.astype(float)
            min_vals.append(np.min(d))
            max_vals.append(np.max(d))
        return np.array(min_vals), np.array(max_vals)

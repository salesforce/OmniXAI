#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
The pre-processing functions for filling NaNs and missing values.
"""
import numpy as np
import pandas as pd
from typing import Union
from .base import TransformBase
from ..data.tabular import Tabular


class FillNaN(TransformBase):
    """
    Fill NaNs in a pandas dataframe or a numpy array.
    """

    def __init__(self, value: Union[str, int, float]):
        """
        :param value: The value to fill NaNs, chosen from ['mean', 'median'] or float values
        """
        super().__init__()
        assert type(value) in [
            str,
            int,
            float,
        ], f"Value {value} is invalid. Please choose from ['mean', 'median'] or float values"
        if type(value) == str:
            assert value in [
                "mean",
                "median",
            ], f"Value {value} is invalid. Please choose from ['mean', 'median'] or float values"
        self.value_type = value
        self.value = value

    def fit(self, x: Union[np.ndarray, pd.DataFrame]) -> TransformBase:
        if isinstance(x, np.ndarray) and x.ndim == 2:
            x = pd.DataFrame(x)
        if isinstance(x, pd.DataFrame):
            if self.value_type == "mean":
                self.value = x.mean()
            elif self.value_type == "median":
                self.value = x.median()
        else:
            assert type(self.value_type) != str, "When ndim != 2, only constant values are allowed to fill NaNs"
        return self

    def transform(self, x: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
        if isinstance(x, np.ndarray):
            if x.ndim == 2:
                x = pd.DataFrame(x)
                return x.fillna(self.value).values
            else:
                return np.nan_to_num(x, nan=self.value)
        else:
            return x.fillna(self.value)

    def invert(self, x: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
        """
        This is a pseudo inverse transform because the positions of
        the NANs in the original data are not stored.

        :param x: The data on which to apply the inverse transform.
        :return: The inverse transformed data.
        :rtype: Union[np.ndarray, pd.DataFrame]
        """
        return x


class FillNaNTabular(TransformBase):
    """
    Fill NaNs in a Tabular object.
    """

    def __init__(self, value: Union[str, int, float]):
        """
        :param value: The value to fill NaNs, chosen from ['mean', 'median'] or float values
        """
        super().__init__()
        self.trans = FillNaN(value)

    @staticmethod
    def _split(x: Tabular) -> (pd.DataFrame, pd.DataFrame):
        """
        Separates categorical features and continuous-valued features.

        :param x: A `Tabular` object.
        :return: The pandas DataFrames of the categorical features and
            the continuous-valued features.
        :rtype: tuple(pd.DataFrame, pd.DataFrame)
        """
        df = x.to_pd()
        cate_df = df[x.categorical_columns] if x.categorical_columns else None
        cont_df = df[x.continuous_columns] if x.continuous_columns else None
        return cate_df, cont_df

    def fit(self, x: Tabular) -> TransformBase:
        """
        Fits a `FillNaN` transformer.

        :param x: A `Tabular` object.
        :return: Itself.
        :rtype: FillNaNTabular
        """
        _, cont_df = self._split(x)
        if cont_df is not None:
            self.trans.fit(cont_df)
        return self

    def transform(self, x: Tabular) -> Tabular:
        """
        Fills NaNs in the continuous-valued features.

        :param x: A `Tabular` object.
        :return: The transformed data.
        :rtype: Tabular
        """
        cate_df, cont_df = self._split(x)
        if cont_df is not None:
            cont_df = self.trans.transform(cont_df)
        if cate_df is not None:
            if cont_df is None:
                df = cate_df
            else:
                df = pd.concat([cate_df, cont_df], axis=1)
                df = df[list(x.columns)]
        else:
            df = cont_df
        return Tabular(df, categorical_columns=x.categorical_columns)

    def invert(self, x: Tabular) -> Tabular:
        """
        This is a pseudo inverse transform because the positions of
        the NANs in the original data are not stored.

        :param x: The data on which to apply the inverse transform.
        :return: The inverse transformed data.
        :rtype: Tabular
        """
        return x

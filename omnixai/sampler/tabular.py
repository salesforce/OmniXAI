#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
The class for re-sampling training data.
"""
import pandas as pd

from ..data.tabular import Tabular


class Sampler:
    """
    The class for re-sampling training data. It includes sub-sampling, under-sampling
    and over-sampling.
    """

    @staticmethod
    def _get_categorical_values(df, categorical_columns):
        """
        Gets the categorical feature values.

        :param df: The input dataframe.
        :param categorical_columns: A list of categorical feature names.
        :return: A dict whose keys are feature names and values are feature values.
        :rtype: Dict
        """
        if categorical_columns is None or len(categorical_columns) == 0:
            return {}
        categorical_values = {}
        for col in categorical_columns:
            categorical_values[col] = set(df[col].values)
        return categorical_values

    @staticmethod
    def _find_extra_samples(df, feature_name, feature_value, n=1):
        """
        Returns a sub-dataframe whose column ``feature_name`` contains value ``feature_value``.

        :param df: The input dataframe.
        :param feature_name: The feature name.
        :param feature_value: The feature value.
        :param n: The number of rows to select.
        :return: The selected rows with ``feature_name = feature_value``.
        :rtype: pd.DataFrame
        """
        x = df[df[feature_name] == feature_value]
        return x.head(n)

    @staticmethod
    def _add_extra_samples(original_df, sampled_df, categorical_columns):
        """
        Checks if all the categorical feature values in ``original_df`` are included in ``sampled_df``.
        If there are some values that are not included, some additional examples extracted from ``original_df``
        will be added into ``sampled_df``. These examples contains the missing feature values.

        :param original_df: The original dataframe.
        :param sampled_df: The sampled dataframe (via subsampling, undersampling, etc.).
        :param categorical_columns: A list of categorical feature names.
        :return: A new dataframe containing all the categorical feature values in ``original_df``.
        :rtype: pd.DataFrame
        """
        dfs = [sampled_df]
        cate_a = Sampler._get_categorical_values(original_df, categorical_columns)
        cate_b = Sampler._get_categorical_values(sampled_df, categorical_columns)
        for col in cate_a.keys():
            a, b = cate_a[col], cate_b[col]
            for value in a.difference(b):
                dfs.append(Sampler._find_extra_samples(original_df, col, value))
        return pd.concat(dfs)

    @staticmethod
    def subsample(tabular_data: Tabular, fraction: float, random_state=None) -> Tabular:
        """
        Samples a subset of the input dataset. It guarantees that all the categorical values
        are included in the sampled dataframe, i.e., there will be no missing categorical values.

        :param tabular_data: The input tabular data.
        :param fraction: The fraction of the sampled instance.
        :param random_state: The random seed.
        :return: A subset extracted from ``tabular_data``.
        :rtype: Tabular
        """
        df = tabular_data.to_pd(copy=False)
        if tabular_data.target_column is None:
            samples = df.sample(frac=fraction, random_state=random_state)
        else:
            dfs = []
            for label in df[tabular_data.target_column].unique():
                split = df[df[tabular_data.target_column] == label]
                dfs.append(split.sample(frac=fraction, random_state=random_state))
            samples = pd.concat(dfs)

        # Add additional samples to make sure no categorical values are missing
        new_df = Sampler._add_extra_samples(
            original_df=df, sampled_df=samples, categorical_columns=tabular_data.categorical_columns
        )
        return Tabular(
            data=new_df.sample(frac=1, random_state=random_state),
            categorical_columns=tabular_data.categorical_columns,
            target_column=tabular_data.target_column,
        )

    @staticmethod
    def undersample(tabular_data: Tabular, random_state=None) -> Tabular:
        """
        Undersamples a class-imbalance dataset to make it more balance, i.e.,
        keeping all of the data in the minority class and decreasing the size of the majority class.
        It guarantees that all the categorical values are included in the sampled dataframe, i.e.,
        there will be no missing categorical values.

        :param tabular_data: The input tabular data.
        :param random_state: The random seed.
        :return: A subset extracted from ``tabular_data``.
        :rtype: Tabular
        """
        assert tabular_data.target_column is not None, "`tabular_data` doesn't have a target column."
        df = tabular_data.to_pd(copy=False)
        splits = {
            label: df[df[tabular_data.target_column] == label] for label in df[tabular_data.target_column].unique()
        }
        min_count = min([len(split) for split in splits.values()])
        samples = pd.concat(
            [split.sample(n=min(min_count, len(split)), random_state=random_state) for label, split in splits.items()]
        )
        # Add additional samples to make sure no categorical values are missing
        new_df = Sampler._add_extra_samples(
            original_df=df, sampled_df=samples, categorical_columns=tabular_data.categorical_columns
        )
        return Tabular(
            data=new_df.sample(frac=1, random_state=random_state),
            categorical_columns=tabular_data.categorical_columns,
            target_column=tabular_data.target_column,
        )

    @staticmethod
    def oversample(tabular_data: Tabular, random_state=None) -> Tabular:
        """
        Oversamples a class-imbalance dataset to make it more balance, i.e.,
        keeping all of the data in the majority class and increasing the size of the minority class.
        It guarantees that all the categorical values are included in the sampled dataframe, i.e.,
        there will be no missing categorical values.

        :param tabular_data: The input tabular data.
        :param random_state: The random seed.
        :return: An oversampled dataset.
        :rtype: Tabular
        """
        assert tabular_data.target_column is not None, "`tabular_data` doesn't have a target column."
        df = tabular_data.to_pd(copy=False)
        splits = {
            label: df[df[tabular_data.target_column] == label] for label in df[tabular_data.target_column].unique()
        }
        max_count = max([len(split) for split in splits.values()])
        samples = pd.concat(
            [split.sample(n=max_count, random_state=random_state, replace=True) for label, split in splits.items()]
        )
        return Tabular(
            data=samples.sample(frac=1, random_state=random_state),
            categorical_columns=tabular_data.categorical_columns,
            target_column=tabular_data.target_column,
        )

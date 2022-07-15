#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
The accumulated local effects plots for tabular data.
"""
import warnings
import numpy as np
import pandas as pd
from typing import List
from collections import OrderedDict

from ..base import TabularExplainer
from ....data.tabular import Tabular
from ....explanations.tabular.ale import ALEExplanation


class ALE(TabularExplainer):
    """
    The accumulated local effects (ALE) plots for tabular data. For more information, please refer to
    https://christophm.github.io/interpretable-ml-book/ale.html.
    """

    explanation_type = "global"
    alias = ["ale", "accumulated_local_effects"]

    def __init__(self, training_data: Tabular, predict_function, mode="classification", **kwargs):
        """
        :param training_data: The data used to initialize the explainer. ``training_data``
            can be the training dataset for training the machine learning model. If the training
            dataset is large, ``training_data`` can be its subset by applying
            `omnixai.sampler.tabular.Sampler.subsample`.
        :param predict_function: The prediction function corresponding to the model to explain.
            When the model is for classification, the outputs of the ``predict_function``
            are the class probabilities. When the model is for regression, the outputs of
            the ``predict_function`` are the estimated values.
        :param mode: The task type, e.g., `classification` or `regression`.
        :param kwargs: Additional parameters, e.g., ``grid_resolution`` -- the number of
            candidates for each feature.
        """
        super().__init__(training_data=training_data, predict_function=predict_function, mode=mode, **kwargs)
        self.grid_resolution = kwargs.get("grid_resolution", 10)

    def _ale_continuous(self, data, column, bins):
        feat_bins = pd.cut(data[:, column], bins, include_lowest=True)

        z = data.copy()
        z[:, column] = [feat_bins.categories[i].left for i in feat_bins.codes]
        ya = self.predict_fn(z)
        z[:, column] = [feat_bins.categories[i].right for i in feat_bins.codes]
        yb = self.predict_fn(z)
        if ya.ndim == 1:
            ya = np.expand_dims(ya, axis=-1)
            yb = np.expand_dims(yb, axis=-1)

        cols = OrderedDict({column: [bins[b + 1] for b in feat_bins.codes]})
        delta_cols = OrderedDict({f"delta_{i}": yb[:, i] - ya[:, i] for i in range(ya.shape[1])})
        cols.update(delta_cols)
        delta_df = pd.DataFrame(cols)

        df = delta_df.groupby([column])[list(delta_cols.keys())].agg(["mean", "size"])
        for col in delta_cols.keys():
            df[(col, "mean")] = df[(col, "mean")].cumsum()
        df.loc[min(bins), :] = 0
        df = df.sort_index()

        for col in delta_cols.keys():
            z = (df[(col, "mean")] + df[(col, "mean")].shift(1, fill_value=0)) * 0.5
            avg = (z * df[(col, "size")]).sum() / (df[(col, "size")].sum() + 1e-6)
            df[(col, "mean")] = df[(col, "mean")] - avg
        df = df[[(col, "mean") for col in delta_cols.keys()]]
        df.columns = list(delta_cols.keys())
        return df

    @staticmethod
    def cmds(mat, k=1):
        """Classical multidimensional scaling. Please refer to:
        https://en.wikipedia.org/wiki/Multidimensional_scaling#Classical_multidimensional_scaling
        """
        n = mat.shape[0]
        mat_square = np.square(mat)
        mat_center = np.eye(n) - np.ones((n, n)) / n
        m = -0.5 * mat_center.dot(mat_square).dot(mat_center)
        eigen_values, eigen_vectors = np.linalg.eigh(m)
        idx = np.argsort(eigen_values)[::-1]
        eigen_values, eigen_vectors = eigen_values[idx], eigen_vectors[:, idx]
        eigen_sqrt_diag = np.diag(np.sqrt(eigen_values[0:k]))
        return eigen_vectors[:, 0:k].dot(eigen_sqrt_diag)

    def _categorical_order(self, column, num_bins=100):
        """
        This function is implemented based on
        https://github.com/DanaJomar/PyALE/blob/master/PyALE/_src/lib.py#L46
        """
        from statsmodels.distributions.empirical_distribution import ECDF

        df = pd.DataFrame(self.data, columns=range(len(self.feature_columns)))
        cate_features = set(self.categorical_features)
        features = sorted(df[column].unique())

        scores = pd.DataFrame(0, index=features, columns=features)
        for i in range(len(self.feature_columns)):
            if i == column:
                continue
            s = pd.DataFrame(0, index=features, columns=features)
            if i in cate_features:
                counts = pd.crosstab(self.data[:, column], self.data[:, i])
                fractions = counts.div(np.sum(counts, axis=1), axis=0)
                for j in features:
                    diff = abs(fractions - fractions.loc[j]).sum(axis=1) / 2
                    s.loc[j, :] = diff
                    s.loc[:, j] = diff
            else:
                seq = np.arange(0, 1, 1 / num_bins)
                q = df[i].quantile(seq).to_list()
                cdf = df.groupby(column)[i].agg(ECDF)
                q_cdf = cdf.apply(lambda x: x(q))
                for j in features:
                    diff = q_cdf.apply(lambda x: max(abs(x - q_cdf[j])))
                    s.loc[j, :] = diff
                    s.loc[:, j] = diff
            scores += s

        z = self.cmds(scores, 1).flatten()
        sorted_indices = z.argsort()
        return [features[i] for i in sorted_indices]

    def _ale_categorical(self, data, features, column):
        predictions = self.predict_fn(data)
        if predictions.ndim == 1:
            predictions = np.expand_dims(predictions, axis=-1)

        x = data[:, column]
        feature_indices = {f: i for i, f in enumerate(features)}
        unique, counts = np.unique(x, return_counts=True)
        count_df = pd.DataFrame(counts, columns=["size"], index=unique).loc[features]
        fractions = count_df / (count_df.sum() + 1e-6)

        z = data.copy()
        z[:, column] = [features[min(feature_indices[f] + 1, len(features) - 1)]
                        for f in data[:, column]]
        ya_indices = (x != features[-1])
        ya = self.predict_fn(z)[ya_indices]
        if ya.ndim == 1:
            ya = np.expand_dims(ya, axis=-1)

        y = predictions[ya_indices]
        cols = OrderedDict({column: z[:, column][ya_indices]})
        delta_cols = OrderedDict({f"delta_{i}": ya[:, i] - y[:, i] for i in range(ya.shape[1])})
        cols.update(delta_cols)
        df_a = pd.DataFrame(cols)

        z[:, column] = [features[max(feature_indices[f] - 1, 0)]
                        for f in data[:, column]]
        yb_indices = (x != features[0])
        yb = self.predict_fn(z)[yb_indices]
        if yb.ndim == 1:
            yb = np.expand_dims(yb, axis=-1)

        y = predictions[yb_indices]
        cols = OrderedDict({column: x[yb_indices]})
        delta_cols = OrderedDict({f"delta_{i}": y[:, i] - yb[:, i] for i in range(yb.shape[1])})
        cols.update(delta_cols)
        df_b = pd.DataFrame(cols)

        delta_df = pd.concat([df_a, df_b])
        df = delta_df.groupby([column]).mean()
        df.loc[features[0]] = 0
        df = df.loc[features].cumsum()

        return pd.DataFrame(
            df.values - np.sum(df.values * fractions.values, axis=0),
            columns=df.columns,
            index=[self.categorical_names[column][int(i)] for i in df.index.values]
        )

    def explain(
            self,
            features: List = None,
            monte_carlo: bool = True,
            monte_carlo_steps: int = 10,
            monte_carlo_frac: float = 0.1,
            **kwargs
    ) -> ALEExplanation:
        """
        Generates accumulated local effects (ALE) plots.

        :param features: The names of the features to be studied.
        :param monte_carlo: Whether computing ALE plots for Monte Carlo samples.
        :param monte_carlo_steps: The number of Monte Carlo sampling steps.
        :param monte_carlo_frac: The number of randomly selected samples in each Monte Carlo step.
        :return: The generated ALE explanations.
        """
        if features is None:
            feature_columns = self.feature_columns
        else:
            if isinstance(features, str):
                features = [features]
            for f in features:
                assert f in self.feature_columns, \
                    f"The dataset doesn't have feature `{f}`."
            feature_columns = features
        if len(feature_columns) > 20:
            warnings.warn(f"Too many features ({len(feature_columns)} > 20) for ALE to process, "
                          f"it will take a while to finish. It is better to choose a subset"
                          f"of features to analyze by setting the parameter `features`.")

        explanations = ALEExplanation(self.mode)
        column_index = {f: i for i, f in enumerate(self.feature_columns)}

        for feature_name in feature_columns:
            sampled_scores = []
            i = column_index[feature_name]
            if i in self.categorical_features:
                features = self._categorical_order(i)
                scores = self._ale_categorical(data=self.data, features=features, column=i)
                if monte_carlo:
                    n = int(self.data.shape[0] * monte_carlo_frac)
                    if n < 10:
                        warnings.warn(f"The number of samples in each Monte Carlo step is "
                                      f"too small, i.e., {n} < 10. The Monte Carlo sampling is ignored.")
                    else:
                        x = pd.DataFrame(self.data[:, i], columns=["value"]).reset_index()
                        group_indices = x.groupby("value")["index"].apply(list).values
                        for _ in range(monte_carlo_steps):
                            indices = []
                            for group in group_indices:
                                n = max(1, int(monte_carlo_frac * len(group)))
                                indices += np.random.choice(group, n, replace=False).tolist()
                            s = self._ale_categorical(data=self.data[indices], features=features, column=i)
                            if len(s) == len(scores):
                                sampled_scores.append(s.sort_index().values)
            else:
                percentiles = np.linspace(0, 100, num=self.grid_resolution)
                bins = sorted(set(np.percentile(self.data[:, i], percentiles)))
                scores = self._ale_continuous(data=self.data, column=i, bins=bins)
                if monte_carlo:
                    n = int(self.data.shape[0] * monte_carlo_frac)
                    if n < 10:
                        warnings.warn(f"The number of samples in each Monte Carlo step is "
                                      f"too small, i.e., {n} < 10. The Monte Carlo sampling is ignored.")
                    else:
                        for _ in range(monte_carlo_steps):
                            indices = np.random.choice(range(self.data.shape[0]), n, replace=False)
                            s = self._ale_continuous(data=self.data[indices], column=i, bins=bins)
                            if len(s) == len(scores):
                                sampled_scores.append(s.sort_index().values)

            scores = scores.sort_index()
            explanations.add(
                feature_name=feature_name,
                values=list(scores.index.values),
                scores=scores.values,
                sampled_scores=sampled_scores if sampled_scores else None
            )
        return explanations

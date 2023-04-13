#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from typing import List, Dict, Callable, Union

from .....data.tabular import Tabular
from .....preprocessing.base import Identity
from .....preprocessing.encode import OneHot, KBins
from .....preprocessing.pipeline import Pipeline
from .....preprocessing.tabular import TabularTransform


class CFRetrieval:
    """
    A KNN-based method for finding the features that may change
    the predicted label for a query instance.
    """

    def __init__(
        self,
        training_data: Tabular,
        predict_function: Callable,
        ignored_features: List = None,
        feature_column_top_k: int = -1,
        feature_value_top_k: int = 3,
        num_cont_bins: int = 10,
        num_neighbors: int = 30,
        hnsw_ef_construction: int = 200,
        hnsw_m: int = 30,
        hnsw_ef: int = 50,
        **kwargs
    ):
        """
        :param training_data: The training data.
        :param predict_function: The predict function.
        :param ignored_features: The features ignored in generating counterfactual examples.
        :param feature_column_top_k: The maximum number of the selected feature columns.
        :param feature_value_top_k: The maximum number of the selected values for each feature.
        :param num_cont_bins: The number of bins for discretizing continuous-valued features.
        :param num_neighbors: The number of neighbors in KNN search.
        :param hnsw_ef_construction: The parameter `ef_construction` in HNSW.
        :param hnsw_m: The parameter `m` in HNSW.
        :param hnsw_ef: The parameter `ef` in HNSW.
        :param kwargs: Other parameters.
        """
        assert isinstance(training_data, Tabular), "`training_data` should be an instance of Tabular."

        self.predict_function = predict_function
        self.subset = training_data.remove_target_column()
        self.ignored_features = ignored_features

        self.column_top_k = feature_column_top_k
        self.value_top_k = feature_value_top_k
        self.num_neighbors = num_neighbors
        self.hnsw_ef_construction = hnsw_ef_construction
        self.hnsw_m = hnsw_m
        self.hnsw_ef = hnsw_ef

        predict_scores = predict_function(self.subset)
        predict_labels = np.argmax(predict_scores, axis=1)
        class_labels = sorted(set(predict_labels))
        classes = defaultdict(list)
        for i, label in enumerate(predict_labels):
            classes[label].append(i)

        self.transformer = TabularTransform(
            cate_transform=Pipeline().step(OneHot()), cont_transform=KBins(n_bins=num_cont_bins)
        ).fit(self.subset)
        data = self.transformer.transform(self.subset)

        self.knn_models = {}
        self.knn_num_elements = {}
        self.knn_dim = data.shape[1]

        for label in class_labels:
            y = classes[label]
            x = data[y]
            self.knn_models[label], self.knn_num_elements[label] = self._build_knn_index(x, y)

    def _build_knn_index(
        self, instances: Union[List, np.ndarray], labels: Union[List, np.ndarray]
    ):
        """
        Builds the KNN search index for each class.

        :param instances: The input instances with different classes.
        :param labels: The class labels of the instances.
        :return: The KNN search index and the number of the elements in the KNN model.
        """
        import hnswlib
        # Remove duplicated examples
        hash_keys, xs, ys = {}, [], []
        for x, y in zip(instances, labels):
            h = hash(x.tobytes())
            if h not in hash_keys:
                hash_keys[h] = True
                xs.append(x)
                ys.append(y)

        # Build HNSW index
        hnsw_index = hnswlib.Index(space="l2", dim=instances.shape[1])
        hnsw_index.init_index(max_elements=len(xs), ef_construction=self.hnsw_ef_construction, M=self.hnsw_m)
        hnsw_index.add_items(xs, ys)
        hnsw_index.set_ef(self.hnsw_ef)
        return hnsw_index, len(xs)

    def _knn_query(self, x: np.ndarray, label: int, k: int) -> List:
        """
        Finds the nearest neighbors given a query instance.

        :param x: The query instance.
        :param label: The desired label.
        :param k: The number of neighbors.
        :return: A list of the indices of the nearest neighbors.
        """
        indices, distances = self.knn_models[label].knn_query(x, k=k)
        neighbors = [[idx[i] for i in range(len(idx)) if dists[i] > 0] for idx, dists in zip(indices, distances)]
        return neighbors

    def _pick_top_columns(self, x: Tabular, candidate_features: Dict, desired_label: int, top_k: int) -> Dict:
        """
        Picks the top k columns from the generated candidate features.

        :param x: The query instance.
        :param candidate_features: The generated candidate features via KNN.
        :param desired_label: The desired label.
        :param top_k: The maximum number of picked columns.
        :return: The picked candidate features.
        """
        assert x.target_column is None, "Input `x` cannot have a target column."

        x_dict = x.to_pd(copy=False).to_dict("records")[0]
        feature_and_values, instances = [], []
        for f, vals in candidate_features.items():
            for v in vals:
                feature_and_values.append((f, v))
                instances.append([x_dict[c] if c != f else v for c in x.columns])

        scores = self.predict_function(
            Tabular(data=pd.DataFrame(instances, columns=x.columns), categorical_columns=x.categorical_columns)
        )
        column_scores = {}
        for (f, v), s in zip(feature_and_values, scores):
            column_scores[f] = max(column_scores.get(f, 0), s[desired_label])

        column_scores = sorted(column_scores.items(), key=lambda z: z[1], reverse=True)
        columns = [c for c, _ in column_scores][:top_k]
        return {f: v for f, v in candidate_features.items() if f in columns}

    def get_nn_samples(self, instance: Tabular, desired_label: int) -> (pd.DataFrame, np.ndarray):
        """
        Finds nearest neighbor samples in a desired class.

        :param instance: The query instance.
        :param desired_label: The desired label.
        :return: The nearest neighbor samples and the corresponding indices.
        """
        assert isinstance(instance, Tabular), "Input ``instance`` should be an instance of Tabular."
        assert instance.shape[0] == 1, "Input ``instance`` can only contain one instance."
        assert instance.target_column is None, "Input ``instance`` cannot have a target column."

        query = self.transformer.transform(
            Tabular(
                data=instance.to_pd(copy=False)[self.subset.columns],
                categorical_columns=self.subset.categorical_columns,
            )
        )
        indices = self._knn_query(query, desired_label, self.num_neighbors)[0]
        y = self.subset.iloc(indices).to_pd(copy=False)
        return y, indices

    def get_cf_features(self, instance: Tabular, desired_label: int) -> (Dict, np.ndarray):
        """
        Finds candidate features for generating counterfactual examples.

        :param instance: The query instance.
        :param desired_label: The desired label.
        :return: The candidate features and the indices of the nearest neighbors.
        """
        x = instance.to_pd(copy=False)
        y, indices = self.get_nn_samples(instance, desired_label)
        cate_candidates, cont_candidates = {}, {}

        # Categorical feature difference
        for f in self.subset.categorical_columns:
            u = x[f].values[0]
            values = [v for v in y[f] if v != u]
            cate_candidates[f] = dict(Counter(values))

        # Continuous-valued feature difference
        for f in self.subset.continuous_columns:
            u = x[f].values[0]
            cont_candidates[f] = [float(v) for v in y[f] if v != u]

        res = defaultdict(list)
        for f, counts in cate_candidates.items():
            candidates = sorted(counts.items(), key=lambda s: s[1], reverse=True)
            values = [v for v, c in candidates[: self.value_top_k] if c > 1]
            if len(values) > 0:
                res[f] = values

        for f, values in cont_candidates.items():
            if len(values) > 0:
                percentiles = np.linspace(0, 100, num=self.value_top_k)
                values = list(np.percentile(values, percentiles))
                res[f] = list(sorted(set(values)))

        # Delete the features in ``ignored_features``
        if self.ignored_features is not None:
            res = {f: v for f, v in res.items() if f not in self.ignored_features}
        # Pick the top feature columns
        if self.column_top_k > 0:
            res = self._pick_top_columns(instance, res, desired_label, self.column_top_k)
        return res, indices


class SimpleCFRetrieval:
    """
    The class for extracting all the feature values in a dataset.
    """

    def __init__(
        self,
        training_data: Tabular,
        ignored_features: List = None,
        num_cont_bins: int = 10,
        **kwargs
    ):
        """
        :param training_data: The training data.
        :param ignored_features: The features ignored in generating counterfactual examples.
        :param num_cont_bins: The number of bins for discretizing continuous-valued features.
        :param kwargs: Other parameters.
        """
        assert isinstance(training_data, Tabular), "`training_data` should be an instance of Tabular."
        self.ignored_features = ignored_features if ignored_features is not None else []
        subset = training_data.remove_target_column()

        transformer = TabularTransform(
            cate_transform=Identity(), cont_transform=KBins(n_bins=num_cont_bins)
        ).fit(subset)
        df = transformer.invert(transformer.transform(subset)).to_pd(copy=False)

        self.features = {}
        for col in df.columns:
            if col not in self.ignored_features:
                self.features[col] = sorted(set(df[col].unique()))

    def get_cf_features(self, instance: Tabular, desired_label: int) -> (Dict, None):
        """
        Finds candidate features for generating counterfactual examples.

        :param instance: The query instance.
        :param desired_label: The desired label.
        :return: The candidate features
        """
        return self.features, None

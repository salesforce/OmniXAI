#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import numpy as np
import pandas as pd
from typing import Dict, Callable, Union

from .....data.tabular import Tabular


class BinarySearchRefinement:
    """
    The module for refining the continuous-valued features
    in the generated counterfactual examples via binary search.
    """

    def __init__(self, training_data: Tabular, predict_function: Callable):
        """
        :param training_data: The training data.
        :param predict_function: The predict function.
        """
        self.predict_function = predict_function
        self.cont_columns = training_data.continuous_columns

    @staticmethod
    def _refine(instance: Tabular, predict_function: Callable, cont_features: Dict, desired_label: int) -> pd.DataFrame:
        """
        Refines the continuous-valued features for the given instance.

        :param instance: The instance to be refined.
        :param predict_function: The predict function.
        :param cont_features: The continuous-valued features to be refined.
        :param desired_label: The desired label.
        :return: The refined instance.
        """
        x = instance.to_pd(copy=False)
        y = x.copy()
        column2loc = {c: x.columns.get_loc(c) for c in cont_features}

        for col, (a, b) in cont_features.items():
            gap, r = b - a, None
            while (b - a) / (gap + 1e-3) > 0.1:
                z = (a + b) * 0.5
                y.iloc[0, column2loc[col]] = z
                scores = predict_function(Tabular(data=y, categorical_columns=instance.categorical_columns))
                if np.argmax(scores[0, :]) == desired_label:
                    b, r = z, z
                else:
                    a = z
            y.iloc[0, column2loc[col]] = r if r is not None else x.iloc[0, column2loc[col]]
        return y

    def refine(self, instance: Tabular, cfs: Tabular, desired_label: int) -> Union[Tabular, None]:
        """
        Refines the continuous-valued features in the counterfactual examples.

        :param instance: The query instance.
        :param cfs: The counterfactual examples.
        :param desired_label: The desired label.
        :return: The refined counterfactual examples.
        """
        assert instance.target_column is None, "Input `instance` cannot have a target column."
        if cfs is None:
            return None

        results = []
        x = instance.to_pd(copy=False)
        for i in range(cfs.shape[0]):
            y = cfs.iloc([i]).to_pd(copy=False)
            cont_features = {}
            for col in self.cont_columns:
                a, b = float(x[col].values[0]), float(y[col].values[0])
                if a != b:
                    cont_features[col] = (a, b) if a <= b else (b, a)
            if len(cont_features) == 0:
                results.append(y)
            else:
                results.append(
                    BinarySearchRefinement._refine(cfs.iloc([i]), self.predict_function, cont_features, desired_label)
                )
        return Tabular(data=pd.concat(results), categorical_columns=cfs.categorical_columns)

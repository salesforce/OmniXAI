#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import os
import unittest
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from omnixai.data.tabular import Tabular
from omnixai.explainers.data.mutual_info import MutualInformation


class TestMutual(unittest.TestCase):

    def test_classification(self):
        feature_names = [
            "Age",
            "Workclass",
            "fnlwgt",
            "Education",
            "Education-Num",
            "Marital Status",
            "Occupation",
            "Relationship",
            "Race",
            "Sex",
            "Capital Gain",
            "Capital Loss",
            "Hours per week",
            "Country",
            "label",
        ]
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../datasets")
        data = np.genfromtxt(os.path.join(data_dir, "adult.data"), delimiter=", ", dtype=str)
        tabular_data = Tabular(
            data,
            feature_columns=feature_names,
            categorical_columns=[feature_names[i] for i in [1, 3, 5, 6, 7, 8, 9, 13]],
            target_column="label",
        )
        explainer = MutualInformation(tabular_data)
        explanations = explainer.explain()
        fig = explanations.plotly_plot()

    def test_regression(self):
        housing = fetch_california_housing()
        df = pd.DataFrame(
            np.concatenate([housing.data, housing.target.reshape((-1, 1))], axis=1),
            columns=list(housing.feature_names) + ["target"],
        )
        tabular_data = Tabular(df, target_column="target")
        explainer = MutualInformation(tabular_data, mode="regression")
        explanations = explainer.explain()
        fig = explanations.plotly_plot()


if __name__ == "__main__":
    unittest.main()

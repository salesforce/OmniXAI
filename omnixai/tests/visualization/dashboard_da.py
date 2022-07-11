#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import os
import unittest
import numpy as np
from omnixai.data.tabular import Tabular
from omnixai.explainers.data import DataAnalyzer
from omnixai.visualization.dashboard import Dashboard


class TestImbalance(unittest.TestCase):
    def test(self):
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
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../datasets")
        data = np.genfromtxt(os.path.join(data_dir, "adult.data"), delimiter=", ", dtype=str)
        tabular_data = Tabular(
            data,
            feature_columns=feature_names,
            categorical_columns=[feature_names[i] for i in [1, 3, 5, 6, 7, 8, 9, 13]],
            target_column="label",
        )

        explainer = DataAnalyzer(
            explainers=["correlation", "imbalance#0", "imbalance#1", "imbalance#2", "imbalance#3", "mutual", "chi2"],
            mode="classification",
            data=tabular_data
        )
        explanations = explainer.explain_global(
            params={
                "imbalance#0": {"features": ["Sex"]},
                "imbalance#1": {"features": ["Race"]},
                "imbalance#2": {"features": ["Sex", "Race"]},
                "imbalance#3": {"features": ["Marital Status", "Age"]},
            }
        )
        dashboard = Dashboard(data_explanations=explanations)
        dashboard.show()


if __name__ == "__main__":
    unittest.main()

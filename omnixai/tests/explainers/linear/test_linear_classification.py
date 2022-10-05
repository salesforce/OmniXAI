#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import os
import unittest
import pprint
import sklearn
import sklearn.datasets
import sklearn.ensemble

import numpy as np
from omnixai.utils.misc import set_random_seed
from omnixai.data.tabular import Tabular
from omnixai.explainers.tabular import LogisticRegression
from omnixai.explanations.base import ExplanationBase


class TestLinearTabular(unittest.TestCase):
    def test_1(self):
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

        set_random_seed()
        model = LogisticRegression()
        model.fit(tabular_data)

        i = 1653
        test_x = tabular_data.iloc(list(range(i, i + 4)))
        explanations = model.explain(test_x)
        e = explanations.get_explanations()
        self.assertAlmostEqual(e["coefficients"]["Age"], 0.3428, delta=1e-3)
        self.assertAlmostEqual(e["coefficients"]["Capital Gain"], 2.3382, delta=1e-3)

        s = explanations.to_json()
        e = ExplanationBase.from_json(s)
        self.assertEqual(s, e.to_json())
        e.plotly_plot()

    def test_2(self):
        iris = sklearn.datasets.load_iris()
        tabular_data = Tabular(
            np.concatenate([iris.data, iris.target.reshape((-1, 1))], axis=1),
            feature_columns=iris.feature_names + ["label"],
            target_column="label",
        )

        set_random_seed()
        model = LogisticRegression()
        model.fit(tabular_data)

        i = np.random.randint(0, tabular_data.shape[0])
        test_x = tabular_data.iloc(i)
        explanations = model.explain(test_x)
        e = explanations.get_explanations()
        self.assertAlmostEqual(e["coefficients"]["petal length (cm)"], -1.8005, delta=1e-3)
        self.assertAlmostEqual(e["coefficients"]["petal width (cm)"], -1.7055, delta=1e-3)

    def test_3(self):
        iris = sklearn.datasets.load_iris()
        tabular_data = Tabular(
            np.concatenate([iris.data, iris.target.reshape((-1, 1))], axis=1),
            feature_columns=iris.feature_names + ["label"],
            target_column="label",
        )

        set_random_seed()
        model = LogisticRegression()
        model.fit(tabular_data)

        base_folder = os.path.dirname(os.path.abspath(__file__))
        directory = f"{base_folder}/../../datasets/tmp"
        model.save(directory=directory)
        model = LogisticRegression.load(directory=directory)

        i = np.random.randint(0, tabular_data.shape[0])
        test_x = tabular_data.iloc(i)
        explanations = model.explain(test_x)
        e = explanations.get_explanations()
        self.assertAlmostEqual(e["coefficients"]["petal length (cm)"], -1.8005, delta=1e-3)
        self.assertAlmostEqual(e["coefficients"]["petal width (cm)"], -1.7055, delta=1e-3)


if __name__ == "__main__":
    unittest.main()

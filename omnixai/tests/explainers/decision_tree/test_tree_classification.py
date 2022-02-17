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
import pandas as pd
from omnixai.data.tabular import Tabular
from omnixai.explainers.tabular.specific.decision_tree import TreeClassifier

pd.set_option("display.max_columns", None)


class TestTreeTabular(unittest.TestCase):
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

        np.random.seed(1)
        model = TreeClassifier()
        model.fit(tabular_data, max_depth=4)

        i = 1653
        test_x = tabular_data.iloc([i, i + 20])
        print(model.class_names())
        print(test_x)
        print(model.predict(test_x))
        explanations = model.explain(test_x)
        for e in explanations.get_explanations()["path"]:
            pprint.pprint([p["text"] for p in e])

    def test_2(self):
        iris = sklearn.datasets.load_iris()
        tabular_data = Tabular(
            np.concatenate([iris.data, iris.target.reshape((-1, 1))], axis=1),
            feature_columns=iris.feature_names + ["label"],
            target_column="label",
        )

        np.random.seed(1)
        model = TreeClassifier()
        model.fit(tabular_data)

        i = np.random.randint(0, tabular_data.shape[0])
        test_x = tabular_data.iloc(i)
        print(model.class_names())
        print(test_x)
        print(model.predict(test_x))
        explanations = model.explain(test_x)
        for e in explanations.get_explanations()["path"]:
            pprint.pprint([p["text"] for p in e])

    def test_3(self):
        feature_names = (
            "label,cap-shape,cap-surface,cap-color,bruises?,odor,gill-attachment,"
            "gill-spacing,gill-size,gill-color,stalk-shape,stalk-root,"
            "stalk-surface-above-ring, stalk-surface-below-ring, "
            "stalk-color-above-ring,stalk-color-below-ring,veil-type,"
            "veil-color,ring-number,ring-type,spore-print-color,"
            "population,habitat".split(",")
        )
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../datasets")
        data = np.genfromtxt(os.path.join(data_dir, "agaricus-lepiota.data"), delimiter=",", dtype="<U20")
        tabular_data = Tabular(
            data, feature_columns=feature_names, categorical_columns=feature_names[1:], target_column="label"
        )

        np.random.seed(1)
        model = TreeClassifier()
        model.fit(tabular_data, max_depth=4)

        i = 137
        test_x = tabular_data.iloc(i)
        print(model.class_names())
        print(test_x)
        print(model.predict(test_x))
        explanations = model.explain(test_x)
        for e in explanations.get_explanations()["path"]:
            pprint.pprint([p["text"] for p in e])


if __name__ == "__main__":
    unittest.main()

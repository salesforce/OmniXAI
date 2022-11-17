#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import os
import unittest
import pandas as pd
from omnixai.explainers.tabular import KNNCounterfactualExplainer
from omnixai.tests.explainers.tasks import TabularClassification

pd.set_option("display.max_columns", None)


class TestKNNCE(unittest.TestCase):
    def setUp(self):
        base_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
        task = TabularClassification(base_folder).train_adult(num_training_samples=2000)
        self.data = task.data
        self.predict_function = lambda z: task.model.predict_proba(task.transform.transform(z))
        self.test_instances = task.test_data.iloc(list(range(5))).remove_target_column()

    def test_explain(self):
        explainer = KNNCounterfactualExplainer(
            training_data=self.data,
            predict_function=self.predict_function,
        )
        explanations = explainer.explain(self.test_instances)
        for explanation in explanations.get_explanations():
            print("Query instance:")
            print(explanation["query"])
            print("Counterfactual examples:")
            print(explanation["counterfactual"])
            print("-----------------")


if __name__ == "__main__":
    unittest.main()

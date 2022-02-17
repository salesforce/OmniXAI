#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import unittest
import pandas as pd
from omnixai.explainers.tabular import MACEExplainer
from omnixai.tests.explainers.tasks import TabularClassification

pd.set_option("display.max_columns", None)


class TestMACE(unittest.TestCase):
    def setUp(self):
        task = TabularClassification.train_adult(num_training_samples=2000)
        self.data = task.data
        self.predict_function = lambda z: task.model.predict_proba(task.transform.transform(z))
        self.test_instances = task.test_data.iloc(list(range(5))).remove_target_column()

    def test_explain(self):
        explainer = MACEExplainer(
            training_data=self.data,
            predict_function=self.predict_function,
            ignored_features=["Sex", "Race", "Relationship", "Capital Loss"],
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

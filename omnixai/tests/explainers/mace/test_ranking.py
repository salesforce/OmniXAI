#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import os
import unittest
import pandas as pd
from omnixai.explainers.ranking.counterfactual.mace import MACEExplainer
from omnixai.tests.explainers.tasks import TabularClassification

pd.set_option("display.max_columns", None)


class TestMACE(unittest.TestCase):
    def setUp(self):
        base_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
        task = TabularClassification(base_folder).train_adult(num_training_samples=2000)
        self.data = task.data
        self.predict_function = lambda z: task.model.predict_proba(task.transform.transform(z))[:, 0]
        self.test_instances = task.test_data.iloc(list(range(5))).remove_target_column()

    def test_candidate_features(self):
        explainer = MACEExplainer(
            training_data=None,
            predict_function=self.predict_function,
            ignored_features=["Sex", "Race", "fnlwgt"],
            method="rl"
        )
        explanations = explainer.explain(
            X=self.test_instances,
            item_a_index=0,
            item_b_index=[1, 2]
        )
        print(explanations)


if __name__ == "__main__":
    unittest.main()

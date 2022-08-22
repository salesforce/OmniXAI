#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import os
import unittest
import pandas as pd
from omnixai.explainers.ranking.agnostic.validity import ValidityRankingExplainer
from omnixai.tests.explainers.tasks import TabularClassification

pd.set_option("display.max_columns", None)


class TestRanking(unittest.TestCase):
    def setUp(self):
        base_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
        task = TabularClassification(base_folder).train_adult(num_training_samples=2000)
        self.data = task.data
        self.predict_function = lambda z: task.model.predict_proba(task.transform.transform(z))[:, 0]
        self.test_instances = task.test_data.iloc(list(range(5))).remove_target_column()

    def test_candidate_features(self):
        explainer = ValidityRankingExplainer(
            training_data=self.data,
            predict_function=self.predict_function
        )
        explanations = explainer.explain(
            X=self.test_instances,
            epsilon=-1.0,
            weighted=True,
            k=8
        )
        print(explanations)
        e = explanations.get_explanations(index=0)
        self.assertAlmostEqual(e["top_features"]["Marital Status"], 5.621, delta=1e-3)
        self.assertAlmostEqual(e["top_features"]["Capital Loss"], 9.580, delta=1e-3)


if __name__ == "__main__":
    unittest.main()

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
from omnixai.explainers.tabular.counterfactual.mace.retrieval import CFRetrieval
from omnixai.explainers.tabular.counterfactual.mace.rl import RLOptimizer
from omnixai.tests.explainers.tasks import TabularClassification

pd.set_option("display.max_columns", None)


class TestMACE(unittest.TestCase):
    def setUp(self):
        base_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
        task = TabularClassification(base_folder).train_adult(num_training_samples=2000)
        self.data = task.data
        self.predict_function = lambda z: task.model.predict_proba(task.transform.transform(z))
        self.test_instance = task.test_data.iloc(1).remove_target_column()
        self.desired_label = 1

    def test_explain(self):
        knn = CFRetrieval(
            training_data=self.data,
            predict_function=self.predict_function,
            ignored_features=["Sex", "Race", "Relationship"],
            feature_column_top_k=5,
        )
        cf_features = knn.get_cf_features(self.test_instance, desired_label=self.desired_label)[0]
        optimizer = RLOptimizer(
            x=self.test_instance,
            predict_function=self.predict_function,
            candidate_features=cf_features,
            oracle_function=lambda _s: int(self.desired_label == np.argmax(_s)),
            desired_label=1
        )
        cfs = optimizer.optimize()
        print(cfs)


if __name__ == "__main__":
    unittest.main()

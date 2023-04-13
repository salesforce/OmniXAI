#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import os
import unittest
from omnixai.utils.misc import set_random_seed
from omnixai.explainers.tabular.agnostic.bias import BiasAnalyzer
from omnixai.tests.explainers.tasks import TabularClassification


class TestClassificationBias(unittest.TestCase):

    def test_classification_metric(self):
        set_random_seed()
        base_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
        task = TabularClassification(base_folder).train_adult(num_training_samples=2000)
        predict_function = lambda z: task.model.predict_proba(task.transform.transform(z))

        explainer = BiasAnalyzer(
            mode="classification",
            predict_function=predict_function,
            training_data=task.test_data,
            training_targets=task.test_targets
        )
        explanations = explainer.explain(
            feature_column="Sex",
            feature_value_or_threshold=["Female", ["Male"]],
            label_value_or_threshold=1
        )
        print(explanations.get_explanations())
        explanations.plotly_plot()


if __name__ == "__main__":
    unittest.main()

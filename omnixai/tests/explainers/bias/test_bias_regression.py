#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import unittest
import numpy as np
from omnixai.utils.misc import set_random_seed
from omnixai.explainers.tabular.agnostic.bias import BiasAnalyzer
from omnixai.tests.explainers.tasks import TabularRegression


class TestRegressionBias(unittest.TestCase):

    def test_classification_metric(self):
        set_random_seed()
        task = TabularRegression().train_boston()
        predict_function = lambda z: task.model.predict(task.transform.transform(z))

        explainer = BiasAnalyzer(
            mode="regression",
            predict_function=predict_function,
            training_data=task.test_data,
            training_targets=task.test_targets
        )
        explanations = explainer.explain(
            feature_column="LSTAT",
            feature_value_or_threshold=10,
            label_value_or_threshold=22
        )
        print(explanations.get_explanations())
        explanations.plotly_plot()


if __name__ == "__main__":
    unittest.main()

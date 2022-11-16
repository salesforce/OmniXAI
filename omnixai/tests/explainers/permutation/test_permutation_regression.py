#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import unittest
from omnixai.utils.misc import set_random_seed
from omnixai.explainers.tabular import PermutationImportance
from omnixai.tests.explainers.tasks import TabularRegression


class TestPDPTabular(unittest.TestCase):
    def test(self):
        set_random_seed()
        task = TabularRegression().train_boston()
        predict_function = lambda z: task.model.predict(task.transform.transform(z))
        explainer = PermutationImportance(
            training_data=task.train_data, predict_function=predict_function, mode="regression"
        )
        explanations = explainer.explain(X=task.test_data, y=task.test_targets)
        explanations.plotly_plot()


if __name__ == "__main__":
    unittest.main()

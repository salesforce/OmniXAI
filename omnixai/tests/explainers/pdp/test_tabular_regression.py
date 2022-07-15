#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import unittest
import pprint
import numpy as np
from omnixai.utils.misc import set_random_seed
from omnixai.explainers.tabular import PartialDependenceTabular
from omnixai.tests.explainers.tasks import TabularRegression


class TestPDPTabular(unittest.TestCase):
    def test_explain(self):
        set_random_seed()
        task = TabularRegression().train_boston()
        predict_function = lambda z: task.model.predict(task.transform.transform(z))
        explainer = PartialDependenceTabular(
            training_data=task.train_data, predict_function=predict_function, mode="regression"
        )
        explanations = explainer.explain(monte_carlo=False)
        pprint.pprint(explanations.get_explanations())
        self.assertAlmostEqual(np.max(explanations.get_explanations()["LSTAT"]["scores"]), 32.1, delta=0.1)


if __name__ == "__main__":
    unittest.main()

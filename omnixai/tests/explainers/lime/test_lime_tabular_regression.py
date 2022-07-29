#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import unittest
import pprint
from omnixai.utils.misc import set_random_seed
from omnixai.explainers.tabular import LimeTabular
from omnixai.tests.explainers.tasks import TabularRegression


class TestLimeTabular(unittest.TestCase):
    def test_explain(self):
        task = TabularRegression().train_boston()
        predict_function = lambda z: task.model.predict(task.transform.transform(z))
        explainer = LimeTabular(
            training_data=task.train_data,
            predict_function=predict_function,
            mode="regression",
            ignored_features=None
        )

        set_random_seed()
        i = 25
        test_x = task.test_data.iloc(i)
        explanations = explainer.explain(test_x, num_features=5)
        for e in explanations.get_explanations():
            print(e["instance"])
            pprint.pprint(list(zip(e["features"], e["values"], e["scores"])))
            self.assertEqual(e["features"][0], "RM")
            self.assertEqual(e["features"][1], "LSTAT")
            self.assertEqual(e["features"][2], "PTRATIO")
            self.assertEqual(e["features"][3], "TAX")


if __name__ == "__main__":
    unittest.main()

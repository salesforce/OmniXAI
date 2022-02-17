#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import unittest
import numpy as np
import pandas as pd
from omnixai.data.tabular import Tabular


class TestTabular(unittest.TestCase):
    def test_values(self):
        x = np.array([1, 2, 3, 4, 5])
        d = pd.DataFrame(x.reshape((1, x.size)), columns=["a", "b", "c", "d", "e"])
        t1 = Tabular(x, categorical_columns=[0, 1])
        t2 = Tabular(d, categorical_columns=["a", "b"])

        self.assertEqual(np.sum(np.abs(t1.values - np.array([[1, 2, 3, 4, 5]]))), 0, "incorrect values")
        self.assertEqual(np.sum(np.abs(t2.values - np.array([[1, 2, 3, 4, 5]]))), 0, "incorrect values")

        self.assertCountEqual(t1.categorical_columns, [0, 1])
        self.assertCountEqual(t2.categorical_columns, ["a", "b"])
        self.assertCountEqual(t1.continuous_columns, [2, 3, 4])
        self.assertCountEqual(t2.continuous_columns, ["c", "d", "e"])
        self.assertCountEqual(t2.feature_columns, ["a", "b", "c", "d", "e"])

        self.assertCountEqual(t1.shape, d.shape)
        self.assertEqual(np.sum(np.abs(t1[0].values - np.array([[1, 2, 3, 4, 5]]))), 0, "incorrect values")


if __name__ == "__main__":
    unittest.main()

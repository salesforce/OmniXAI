#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import unittest
from omnixai.preprocessing.fill import *


class TestFillNaN(unittest.TestCase):
    def test_fill_pd(self):
        x = pd.DataFrame({"A": [1, np.nan, 2, 6], "B": [5, np.nan, np.nan, 2]})

        p = FillNaN(value=0).fit(x)
        y = p.transform(x)
        z = np.array([[1, 5], [0, 0], [2, 0], [6, 2]])
        self.assertEqual(np.sum(np.abs(y.values - z)), 0)

        p = FillNaN(value="mean").fit(x)
        y = p.transform(x)
        z = np.array([[1, 5], [3, 3.5], [2, 3.5], [6, 2]])
        self.assertEqual(np.sum(np.abs(y.values - z)), 0)

        p = FillNaN(value="median").fit(x)
        y = p.transform(x)
        z = np.array([[1, 5], [2, 3.5], [2, 3.5], [6, 2]])
        self.assertEqual(np.sum(np.abs(y.values - z)), 0)

    def test_fill_np_1(self):
        x = np.array([[1, 5], [np.nan, np.nan], [2, np.nan], [6, 2]])

        p = FillNaN(value=0).fit(x)
        y = p.transform(x)
        z = np.array([[1, 5], [0, 0], [2, 0], [6, 2]])
        self.assertEqual(np.sum(np.abs(y - z)), 0)

        p = FillNaN(value="mean").fit(x)
        y = p.transform(x)
        z = np.array([[1, 5], [3, 3.5], [2, 3.5], [6, 2]])
        self.assertEqual(np.sum(np.abs(y - z)), 0)

        p = FillNaN(value="median").fit(x)
        y = p.transform(x)
        z = np.array([[1, 5], [2, 3.5], [2, 3.5], [6, 2]])
        self.assertEqual(np.sum(np.abs(y - z)), 0)

    def test_fill_np_2(self):
        x = np.array([1, 5, np.nan, np.nan, 2, np.nan, 6, 2])
        p = FillNaN(value=0).fit(x)
        y = p.transform(x)
        z = np.array([1, 5, 0, 0, 2, 0, 6, 2])
        self.assertEqual(np.sum(np.abs(y - z)), 0)

    def test_fill_tabular(self):
        x = Tabular(
            pd.DataFrame({"A": [1, np.nan, 2, 6], "B": [5, np.nan, np.nan, 2], "C": ["a", "b", "c", "d"]}),
            categorical_columns=["C"],
        )
        p = FillNaNTabular(value="mean").fit(x)
        y = p.transform(x)
        z = np.array([[1, 5], [3, 3.5], [2, 3.5], [6, 2]])
        self.assertEqual(np.sum(np.abs(y.to_pd()[["A", "B"]].values - z)), 0)

        x = Tabular(pd.DataFrame({"A": [1, np.nan, 2, 6], "B": [5, np.nan, np.nan, 2]}))
        p = FillNaNTabular(value="median").fit(x)
        y = p.transform(x)
        z = np.array([[1, 5], [2, 3.5], [2, 3.5], [6, 2]])
        self.assertEqual(np.sum(np.abs(y.to_pd()[["A", "B"]].values - z)), 0)

        x = Tabular(pd.DataFrame({"C": ["a", "b", "c", "d"]}), categorical_columns=["C"])
        p = FillNaNTabular(value="mean").fit(x)
        y = p.transform(x)
        z = ["a", "b", "c", "d"]
        self.assertCountEqual(y.to_pd()[["C"]].values, z)


if __name__ == "__main__":
    unittest.main()

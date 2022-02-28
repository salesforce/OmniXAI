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
from omnixai.preprocessing.normalize import *
from omnixai.preprocessing.encode import *
from omnixai.preprocessing.tabular import TabularTransform


class TestTabularTransform(unittest.TestCase):
    def test_1(self):
        x = Tabular(
            pd.DataFrame({"A": [1, 2, 2, 6], "B": [5, 4, 3, 2], "C": ["a", "b", "c", "d"]}), categorical_columns=["C"]
        )
        p = TabularTransform(cate_transform=OneHot(), cont_transform=MinMax()).fit(x)
        y = p.transform(x)
        z = p.invert(y)

        self.assertLess(np.sum(np.abs(x.to_pd()[["A", "B"]].values - z.to_pd()[["A", "B"]].values)), 1e-8)
        self.assertCountEqual(x.to_pd()[["C"]].values, z.to_pd()[["C"]].values)

    def test_2(self):
        x = Tabular(
            pd.DataFrame(
                {"A": [1, 2, 2, 6], "B": [5, 4, 3, 2], "C": ["a", "b", "c", "d"], "target": ["1", "0", "0", "1"]}
            ),
            categorical_columns=["C"],
            target_column="target",
        )
        p = TabularTransform(cate_transform=Ordinal(), cont_transform=MinMax()).fit(x)
        y = p.transform(x)
        z = p.invert(y)

        self.assertLess(np.sum(np.abs(x.to_pd()[["A", "B"]].values - z.to_pd()[["A", "B"]].values)), 1e-8)
        self.assertCountEqual(x.to_pd()[["C"]].values, z.to_pd()[["C"]].values)

        x = Tabular(
            pd.DataFrame({"A": [1, 2, 2, 6], "B": [5, 4, 3, 2], "C": ["a", "b", "c", "d"]}), categorical_columns=["C"]
        )
        y = p.transform(x)
        z = p.invert(y)

        self.assertLess(np.sum(np.abs(x.to_pd()[["A", "B"]].values - z.to_pd()[["A", "B"]].values)), 1e-8)
        self.assertCountEqual(x.to_pd()[["C"]].values, z.to_pd()[["C"]].values)


if __name__ == "__main__":
    unittest.main()

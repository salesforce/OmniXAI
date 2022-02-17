#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import unittest
import numpy as np
from omnixai.preprocessing.encode import *


class TestEncode(unittest.TestCase):
    def test_KBins(self):
        n_bins = 5
        x = np.random.rand(10, 5)
        s = KBins(n_bins=n_bins).fit(x)
        y = s.transform(x)
        z = s.invert(y)

        self.assertEqual(np.sum(np.abs(np.min(y, axis=0))), 0)
        self.assertEqual(np.mean(np.abs(np.max(y, axis=0))), n_bins - 1)

    def test_OneHot(self):
        x = [["Male", 1], ["Female", 3], ["Female", 2]]
        s = OneHot().fit(x)
        y = s.transform(x)
        z = s.invert(y)

        self.assertCountEqual(y.shape, (3, 5))
        self.assertCountEqual(z.shape, (3, 2))
        for i in range(3):
            self.assertCountEqual(z[i], x[i])

    def test_Ordinal(self):
        x = [["Male", 1], ["Female", 3], ["Female", 2]]
        s = Ordinal().fit(x)
        y = s.transform(x)
        z = s.invert(y)

        self.assertCountEqual(y.shape, (3, 2))
        self.assertCountEqual(z.shape, (3, 2))
        for i in range(3):
            self.assertCountEqual(z[i], x[i])


if __name__ == "__main__":
    unittest.main()

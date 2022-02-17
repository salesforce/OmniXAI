#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import unittest
import numpy as np
from omnixai.preprocessing.normalize import *


class TestNormalize(unittest.TestCase):
    def test_standard(self):
        x = np.random.rand(10, 5)
        s = Standard().fit(x)
        y = s.transform(x)
        z = s.invert(y)

        self.assertLess(np.sum(np.abs(np.sum(y, axis=0))), 1e-6)
        self.assertLess(np.sum(np.abs(x - z)), 1e-6)

    def test_minmax(self):
        x = np.random.rand(10, 5)
        s = MinMax().fit(x)
        y = s.transform(x)
        z = s.invert(y)

        self.assertEqual(np.sum(np.abs(np.min(y, axis=0))), 0)
        self.assertEqual(np.mean(np.abs(np.max(y, axis=0))), 1)
        self.assertLess(np.sum(np.abs(x - z)), 1e-6)


if __name__ == "__main__":
    unittest.main()

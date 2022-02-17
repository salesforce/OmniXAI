#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import os
import unittest
import numpy as np
from omnixai.preprocessing.pipeline import Pipeline
from omnixai.preprocessing.normalize import *


class TestPipeline(unittest.TestCase):
    def test(self):
        x = np.random.rand(10, 5)
        pipeline = Pipeline().step(Standard()).step(MinMax()).fit(x)
        y = pipeline.transform(x)
        z = pipeline.invert(y)

        self.assertLess(np.sum(np.abs(np.min(y, axis=0))), 1e-8)
        self.assertLess(np.mean(np.abs(np.max(y, axis=0) - 1)), 1e-8)
        self.assertLess(np.sum(np.abs(x - z)), 1e-8)

        directory = os.path.dirname(os.path.abspath(__file__))
        pipeline.dump(directory)
        pipeline = Pipeline()
        pipeline.load(directory)
        os.remove(os.path.join(directory, pipeline.name))

        y = pipeline.transform(x)
        z = pipeline.invert(y)

        self.assertLess(np.sum(np.abs(np.min(y, axis=0))), 1e-8)
        self.assertLess(np.mean(np.abs(np.max(y, axis=0) - 1)), 1e-8)
        self.assertLess(np.sum(np.abs(x - z)), 1e-8)


if __name__ == "__main__":
    unittest.main()

#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
The pipeline for multiple pre-processing transforms.
"""
import os
import dill
from ..utils.misc import AutodocABCMeta
from .base import TransformBase


class Pipeline(metaclass=AutodocABCMeta):
    """
    The pipeline for multiple pre-processing transforms.
    """

    name = "pipeline"

    def __init__(self):
        self.steps = []

    def step(self, transformer: TransformBase):
        """
        Adds a new transform into the pipeline.

        :param transformer: A transformer derived from TransformBase
        :return: The current pipeline instance
        """
        self.steps.append(transformer)
        return self

    def fit(self, x):
        """
        Estimates the parameters of the all transforms.

        :param x: The data for estimating the parameters.
        :return: The current instance.
        """
        for step in self.steps:
            x = step.fit(x).transform(x)
        return self

    def transform(self, x):
        """
        Applies all the transforms to the input data.

        :param x: The data on which to apply the transform.
        :return: The transformed data.
        """
        for step in self.steps:
            x = step.transform(x)
        return x

    def invert(self, x):
        """
        Applies the inverse transforms to the input data.

        :param x: The data on which to apply the inverse transform.
        :return: The inverse transformed data.
        """
        for step in self.steps[::-1]:
            x = step.invert(x)
        return x

    def dump(self, directory):
        """
        Saves the pipeline to the specified file.

        :param directory: The directory to save the pipeline
        """
        path = os.path.join(directory, self.name)
        with open(path, "wb") as f:
            dill.dump(self.steps, f)

    def load(self, directory):
        """
        Loads the pipeline from the specified file.

        :param directory: The directory to load the pipeline from
        """
        path = os.path.join(directory, self.name)
        with open(path, "rb") as f:
            self.steps = dill.load(f)

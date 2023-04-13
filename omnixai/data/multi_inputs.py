#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
The class for multiple inputs.
"""
from typing import Dict, Union
from .base import Data
from .tabular import Tabular
from .image import Image
from .text import Text


class MultiInputs(Data):
    """
    This data class is used for a model with multiple inputs, e.g., a visual-language model with
    images and texts as its inputs, or a ranking model with queries and items as its inputs.
    The data is stored in a dict, e.g., `{"image": Image(), "text": Text()}`.
    """
    data_type = "timeseries"

    def __init__(self, **inputs):
        """
        :param inputs: Multiple input parameters, e.g., ``inputs = {"image": Image(), "text": Text()}``.
        """
        super().__init__()
        num_samples = []
        for key, value in inputs.items():
            assert isinstance(value, (Tabular, Image, Text)), \
                f"The type of input {key} must be `Tabular`, `Image` or `Text` " \
                f"instead of {type(value)}."
            num_samples.append(value.num_samples())
        assert min(num_samples) == max(num_samples), \
            f"The numbers of samples in the inputs are different: {num_samples}."

        for key, value in inputs.items():
            setattr(self, key, value)
        self.inputs = inputs
        self.nsamples = num_samples[0]

    @property
    def values(self) -> Dict:
        """
        Returns the raw values of each input.

        :return: A dict containing the raw values for each input.
        """
        return {key: value.values for key, value in self.inputs.items()}

    def num_samples(self) -> int:
        """
        Returns the number of samples in the inputs.

        :return: The number samples in the inputs.
        """
        return self.nsamples

    def __contains__(self, item):
        return item in self.inputs

    def __getitem__(self, i: Union[int, slice, list]):
        """
        Get a subset of the input samples given an index or a set of indices.

        :param i: An integer index, slice or list.
        :return: A MultiInputs object with the selected samples.
        :rtype: MultiInputs
        """
        inputs = {key: value[i] for key, value in self.inputs.items()}
        return MultiInputs(**inputs)

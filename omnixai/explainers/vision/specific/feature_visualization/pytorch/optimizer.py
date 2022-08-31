#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import torch
import itertools
import torch.nn as nn
import numpy as np
from typing import Union, List
from dataclasses import dataclass


@dataclass
class Objective:
    layer: nn.Module
    weight: float = 1.0
    channel_indices: Union[int, List[int]] = None
    neuron_indices: Union[int, List[int]] = None
    direction_vectors: Union[np.ndarray, List[np.ndarray]] = None


class FeatureOptimizer:
    """
    The optimizer for feature visualization.
    """

    def __init__(
            self,
            model: nn.Module,
            objectives: Union[Objective, List[Objective]],
            **kwargs
    ):
        self.model = model
        self.objectives = objectives if isinstance(objectives, (list, tuple)) \
            else [objectives]



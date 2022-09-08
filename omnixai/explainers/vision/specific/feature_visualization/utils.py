#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import itertools
import numpy as np
from typing import Union, List, Any
from dataclasses import dataclass


@dataclass
class Objective:
    layer: Any
    weight: float = 1.0
    channel_indices: Union[int, List[int]] = None
    neuron_indices: Union[int, List[int]] = None
    direction_vectors: Union[np.ndarray, List[np.ndarray]] = None


class FeatureOptimizerMixin:

    @staticmethod
    def _process_objectives(objectives):
        results = []
        for obj in objectives:
            r = {"weight": obj.weight}
            if obj.direction_vectors is not None:
                r["type"] = "direction"
                vectors = obj.direction_vectors \
                    if isinstance(obj.direction_vectors, list) \
                    else [obj.direction_vectors]
                r["indices"] = list(range(len(vectors)))
                r["vector"] = np.array(vectors, dtype=np.float32)
            elif obj.channel_indices is not None:
                r["type"] = "channel"
                r["indices"] = [obj.channel_indices] \
                    if isinstance(obj.channel_indices, int) \
                    else obj.channel_indices
            elif obj.neuron_indices is not None:
                r["type"] = "neuron"
                r["indices"] = [obj.neuron_indices] \
                    if isinstance(obj.neuron_indices, int) \
                    else obj.neuron_indices
            else:
                r["type"] = "layer"
                r["indices"] = (0,)
            results.append(r)

        # Combinations of different objectives
        indices = np.array(
            [m for m in itertools.product(*[r["indices"] for r in results])], dtype=int)
        assert indices.shape[1] == len(objectives)
        # Set new indices for each combination
        for i, r in enumerate(results):
            r["batch_indices"] = indices[:, i]
            if r["type"] == "direction":
                r["vector"] = r["vector"][r["batch_indices"], ...]
        # Set names
        names = []
        for i in range(indices.shape[0]):
            labels = []
            for j, r in enumerate(results):
                try:
                    layer_name = objectives[j].layer.name
                except:
                    layer_name = type(objectives[j].layer).__name__
                labels.append({"type": r["type"], "layer_name": layer_name, "index": indices[i, j]})
            names.append(labels)
        return results, indices.shape[0], names

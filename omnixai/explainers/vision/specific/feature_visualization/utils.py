#
# Copyright (c) 2023 salesforce.com, inc.
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


def fft_freq(width, height, mode):
    freq_x = np.fft.fftfreq(width)[:, None]
    if mode == "tf":
        cut_off = int(height % 2 == 1)
        freq_y = np.fft.fftfreq(height)[:height // 2 + 1 + cut_off]
        return np.sqrt(freq_y ** 2 + freq_x ** 2)
    else:
        freq_y = np.fft.fftfreq(height)
        return np.sqrt(freq_y ** 2 + freq_x ** 2)


def fft_scale(width, height, mode, decay_power=1.0):
    frequencies = fft_freq(width, height, mode)
    scale = 1.0 / np.maximum(frequencies, 1.0 / max(width, height)) ** decay_power
    scale = scale * np.sqrt(width * height)
    return scale


def fft_inputs(batch_size, channel, width, height, mode, std=0.01):
    freq = fft_freq(width, height, mode)
    inputs = np.random.randn(*((2, batch_size, channel) + freq.shape)) * std
    return inputs.astype(np.float32)

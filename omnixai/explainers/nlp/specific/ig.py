#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
The integrated-gradient explainer for NLP tasks.
"""
import numpy as np
from typing import Callable, Dict

from ...base import ExplainerBase
from ....data.text import Text
from ....explanations.text.word_importance import WordImportance
from ....utils.misc import is_torch_available, is_tf_available


def _calculate_integral(inp, baseline, gradients):
    gradients = (gradients[:-1] + gradients[1:]) / 2.0
    avg_grads = np.average(gradients, axis=0)
    integrated_grads = (inp - baseline) * avg_grads
    integrated_grads = np.sum(integrated_grads, axis=-1)
    return integrated_grads


class _IntegratedGradientTorch:
    def __init__(self, model, embedding_layer):
        self.model = model
        self.embedding_layer = embedding_layer
        self.embeddings = None
        self.embedding_layer_inputs = None

    def compute_integrated_gradients(
        self, inputs, output_index, additional_inputs=None, steps=50, batch_size=8
    ):
        import torch

        assert inputs.shape[0] == 1, "The batch size of `inputs` should be 1."
        device = next(self.model.parameters()).device
        hooks = []

        self.model.eval()
        all_inputs = (inputs,)
        if additional_inputs is not None:
            all_inputs += (additional_inputs,) if not isinstance(additional_inputs, tuple) else additional_inputs

        try:
            # Forward pass for extracting embeddings
            hooks.append(self.embedding_layer.register_forward_hook(self._embedding_hook))
            self.model(*all_inputs)
            baselines = np.zeros(self.embeddings.shape)
            hooks.append(self.embedding_layer.register_forward_hook(self._embedding_layer_hook))

            # Build the inputs for computing integrated gradient
            alphas = np.linspace(start=0.0, stop=1.0, num=steps, endpoint=True)
            gradients = []
            for k in range(0, len(alphas), batch_size):
                self.embedding_layer_inputs = torch.tensor(
                    np.stack([baselines[0] + a * (self.embeddings[0] - baselines[0])
                              for a in alphas[k:k + batch_size]]),
                    dtype=torch.get_default_dtype(),
                    device=device,
                    requires_grad=True,
                )
                repeated_inputs = self._repeat(all_inputs, num_reps=self.embedding_layer_inputs.shape[0])

                # Compute gradients
                predictions = self.model(*repeated_inputs)
                if len(predictions.shape) > 1:
                    assert output_index is not None, "The model has multiple outputs, the output index cannot be None"
                    predictions = predictions[:, output_index]
                grad = (
                    torch.autograd.grad(
                        torch.unbind(predictions), self.embedding_layer_inputs)[0].detach().cpu().numpy()
                )
                gradients.append(grad)
            gradients = np.concatenate(gradients, axis=0)
        finally:
            for hook in hooks:
                hook.remove()
        return _calculate_integral(self.embeddings[0], baselines[0], gradients)

    def _embedding_hook(self, module, inputs, outputs):
        self.embeddings = outputs.detach().cpu().numpy()

    def _embedding_layer_hook(self, module, inputs, outputs):
        return self.embedding_layer_inputs

    @staticmethod
    def _repeat(all_inputs, num_reps):
        return [x.repeat(*((num_reps,) + (1,) * (len(x.shape) - 1))) for x in all_inputs]


class _IntegratedGradientTf:
    def __init__(self, model, embedding_layer):
        self.model = model
        self.embedding_layer = embedding_layer
        self.embeddings = None
        self.embedding_layer_inputs = None

    def compute_integrated_gradients(
            self, inputs, output_index, additional_inputs=None, steps=50, batch_size=8
    ):
        import tensorflow as tf

        original_call = self.embedding_layer.call
        all_inputs = (inputs,)
        if additional_inputs is not None:
            all_inputs += (additional_inputs,) if not isinstance(additional_inputs, tuple) else additional_inputs

        try:
            self._embedding_hook(self.embedding_layer)
            self.model(*all_inputs)
            self.embeddings = self.embedding_layer.res.numpy()
            baselines = np.zeros(self.embeddings.shape)

            # Build the inputs for computing integrated gradient
            alphas = np.linspace(start=0.0, stop=1.0, num=steps, endpoint=True)
            # Compute gradients
            gradients = []
            for k in range(0, len(alphas), batch_size):
                with tf.GradientTape() as tape:
                    self._embedding_layer_hook(self.embedding_layer, tape)
                    self.embedding_layer_inputs = tf.convert_to_tensor(
                        np.stack([baselines[0] + a * (self.embeddings[0] - baselines[0])
                                  for a in alphas[k:k + batch_size]]),
                        dtype=tf.keras.backend.floatx(),
                    )
                    repeated_inputs = [
                        tf.tile(x, (self.embedding_layer_inputs.shape[0],) + (1,) * (len(x.shape) - 1))
                        for x in all_inputs
                    ]
                    predictions = self.model(*repeated_inputs)
                    if len(predictions.shape) > 1:
                        assert output_index is not None, \
                            "The model has multiple outputs, the output index cannot be None"
                        predictions = predictions[:, output_index]
                    grad = tape.gradient(predictions, self.embedding_layer.res).numpy()
                    gradients.append(grad)
            gradients = np.concatenate(gradients, axis=0)
        finally:
            self._remove_hook(self.embedding_layer, original_call)
        return _calculate_integral(self.embeddings[0], baselines[0], gradients)

    def _embedding_hook(self, layer):
        def _hook(func):
            def wrapper(*args, **kwargs):
                layer.res = func(*args, **kwargs)
                return layer.res

            return wrapper

        layer.call = _hook(layer.call)

    def _embedding_layer_hook(self, layer, tape):
        def _hook(func):
            def wrapper(*args, **kwargs):
                layer.res = self.embedding_layer_inputs
                tape.watch(layer.res)
                return layer.res

            return wrapper

        layer.call = _hook(layer.call)

    @staticmethod
    def _remove_hook(layer, original_call):
        layer.call = original_call
        delattr(layer, "res")


class IntegratedGradientText(ExplainerBase):
    """
    The integrated-gradient explainer for NLP tasks.
    If using this explainer, please cite the original work: https://github.com/ankurtaly/Integrated-Gradients.
    """

    explanation_type = "local"
    alias = ["ig", "integrated_gradient"]

    def __init__(
            self,
            model,
            embedding_layer,
            preprocess_function: Callable,
            mode: str = "classification",
            id2token: Dict = None,
            tokenizer: Callable = None,
            **kwargs,
    ):
        """
        :param model: The model to explain, whose type can be `tf.keras.Model` or `torch.nn.Module`.
        :param embedding_layer: The embedding layer in the model, which can be
            `tf.keras.layers.Layer` or `torch.nn.Module`.
        :param preprocess_function: The pre-processing function that converts the raw inputs
            into the inputs of ``model``. The first output of ``preprocess_function`` must
            be the token ids.
        :param mode: The task type, e.g., `classification` or `regression`.
        :param id2token: The mapping from token ids to tokens. If `tokenizer` is set, `id2token` will be ignored.
        :param tokenizer: The tokenizer for processing text inputs, i.e., tokenizers in HuggingFace.
        """
        super().__init__()
        assert preprocess_function is not None, (
            "`preprocess_function` cannot be None, which converts a `Text` " "instance into the inputs of `model`."
        )
        self.mode = mode
        self.model = model
        self.embedding_layer = embedding_layer
        self.preprocess_function = preprocess_function
        self.id2token = id2token
        self.tokenizer = tokenizer

        ig_class = None
        if is_torch_available():
            import torch.nn as nn

            if isinstance(model, nn.Module):
                ig_class = _IntegratedGradientTorch
                self.model_type = "torch"
        if ig_class is None and is_tf_available():
            import tensorflow as tf

            if isinstance(model, tf.keras.Model):
                ig_class = _IntegratedGradientTf
                self.model_type = "tf"
        if ig_class is None:
            raise ValueError(f"`model` should be a tf.keras.Model " f"or a torch.nn.Module instead of {type(model)}")
        self.ig_model = ig_class(self.model, self.embedding_layer)

    def _preprocess(self, X: Text):
        inputs = self.preprocess_function(X)
        if not isinstance(inputs, (tuple, list)):
            inputs = (inputs,)
        if self.model_type == "torch":
            import torch

            device = next(self.model.parameters()).device
            torch_inputs = []
            for x in inputs:
                if isinstance(x, (np.ndarray, list)):
                    x = torch.tensor(x)
                torch_inputs.append(x.to(device))
            return tuple(torch_inputs)
        else:
            import tensorflow as tf

            tf_inputs = []
            for x in inputs:
                if isinstance(x, (np.ndarray, list)):
                    x = tf.convert_to_tensor(x)
                tf_inputs.append(x)
            return tuple(tf_inputs)

    def explain(self, X: Text, y=None, **kwargs) -> WordImportance:
        """
        Generates the word/token-importance explanations for the input instances.

        :param X: A batch of input instances.
        :param y: A batch of labels to explain. For regression, ``y`` is ignored.
            For classification, the top predicted label of each input instance will be explained
            when ``y = None``.
        :param kwargs: Additional parameters, e.g., ``steps`` for
            `IntegratedGradient.compute_integrated_gradients`.
        :return: The explanations for all the instances, e.g., word/token importance scores.
        """
        steps = kwargs.get("steps", 50)
        batch_size = kwargs.get("batch_size", 16)
        explanations = WordImportance(mode=self.mode)

        inputs = self._preprocess(X)
        if self.mode == "classification":
            if y is not None:
                if type(y) == int:
                    y = [y for _ in range(len(X))]
                else:
                    assert len(X) == len(y), (
                        f"Parameter ``y`` is a {type(y)}, the length of y "
                        f"should be the same as the number of images in X."
                    )
            else:
                scores = (
                    self.model(*inputs).detach().cpu().numpy()
                    if self.model_type == "torch"
                    else self.model(*inputs).numpy()
                )
                y = np.argmax(scores, axis=1).astype(int)

        for i, instance in enumerate(X):
            output_index = y[i] if y is not None else None
            inputs = self._preprocess(instance)
            scores = self.ig_model.compute_integrated_gradients(
                inputs=inputs[0],
                output_index=output_index,
                additional_inputs=None if len(inputs) == 1 else inputs[1:],
                steps=steps,
                batch_size=batch_size
            )
            tokens = inputs[0].detach().cpu().numpy() if self.model_type == "torch" \
                else inputs[0].numpy()

            if self.tokenizer is not None:
                input_tokens = [self.tokenizer.decode([t]) for t in tokens[0]]
            elif self.id2token is not None:
                input_tokens = [self.id2token[t] for t in tokens[0]]
            else:
                input_tokens = tokens[0]
            explanations.add(
                instance=instance.to_str(),
                target_label=y[i] if y is not None else None,
                tokens=input_tokens,
                importance_scores=scores,
            )
        return explanations

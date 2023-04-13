#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
The integrated-gradient explainer for vision-language models.
"""
import inspect
import numpy as np
from typing import Callable

from ...base import ExplainerBase
from ....data.multi_inputs import MultiInputs
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
            self, inputs, loss_function, steps, batch_size=8, **kwargs
    ):
        import torch

        device = next(self.model.parameters()).device
        hooks = []

        self.model.eval()
        all_inputs = (inputs,) if not isinstance(inputs, tuple) else inputs
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
                repeated_inputs = []
                num_reps = self.embedding_layer_inputs.shape[0]
                for x in all_inputs:
                    if isinstance(x, torch.Tensor):
                        repeated_inputs.append(x.repeat(*((num_reps,) + (1,) * (len(x.shape) - 1))))
                    elif isinstance(x, (list, tuple)):
                        repeated_inputs.append(x * num_reps)
                    else:
                        raise ValueError(f"Wrong type {type(x)}")

                # Compute gradients
                outputs = self.model(*repeated_inputs)
                loss = torch.stack(
                    [loss_function(outputs[i:i + 1], **kwargs) for i in range(outputs.shape[0])])
                grad = (
                    torch.autograd.grad(torch.unbind(loss), self.embedding_layer_inputs)[0].detach().cpu().numpy()
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


class IntegratedGradient(ExplainerBase):
    """
    The integrated-gradient explainer for visual-language models.
    If using this explainer, please cite the original work: https://github.com/ankurtaly/Integrated-Gradients.
    """

    explanation_type = "local"
    alias = ["ig", "integrated_gradient"]

    def __init__(
            self,
            model,
            embedding_layer,
            preprocess_function: Callable,
            tokenizer: Callable,
            loss_function: Callable = None,
            **kwargs,
    ):
        """
        :param model: The model to explain, whose type can be `tf.keras.Model` or `torch.nn.Module`.
        :param embedding_layer: The embedding layer in the model, which can be
            `tf.keras.layers.Layer` or `torch.nn.Module`.
        :param preprocess_function: The pre-processing function that converts the raw inputs
            into the inputs of ``model``.
        :param tokenizer: The tokenizer for processing text inputs.
        :param loss_function: The loss function used to compute gradients w.r.t the target layer.
        """
        super().__init__()
        assert embedding_layer is not None, "The embedding layer cannot be None."
        assert preprocess_function is not None, "`preprocess_function` cannot be None."
        assert tokenizer is not None, "The tokenizer cannot be None."
        assert loss_function is not None, "The loss_function cannot be None."

        self.model = model
        self.embedding_layer = embedding_layer
        self.preprocess_function = preprocess_function
        self.tokenizer = tokenizer
        self.loss_function = loss_function

        ig_class = None
        if is_torch_available():
            import torch.nn as nn

            if isinstance(model, nn.Module):
                ig_class = _IntegratedGradientTorch
                self.model_type = "torch"
        if ig_class is None and is_tf_available():
            import tensorflow as tf

            if isinstance(model, tf.keras.Model):
                raise ValueError("The tensorflow model is not supported yet.")
        if ig_class is None:
            raise ValueError(f"`model` should be a tf.keras.Model " f"or a torch.nn.Module instead of {type(model)}")
        self.ig_model = ig_class(self.model, self.embedding_layer)

    def explain(self, X: MultiInputs, **kwargs) -> WordImportance:
        """
        Generates the word/token-importance explanations for the input instances.

        :param X: A batch of input instances, e.g., `X.image` contains the input images
            and `X.text` contains the input texts.
        :param kwargs: Additional parameters.
        :return: The explanations for all the instances, e.g., word/token importance scores.
        """
        assert "image" in X, "The input doesn't have attribute `image`."
        assert "text" in X, "The input doesn't have attribute `text`."
        steps = kwargs.get("steps", 32)
        batch_size = kwargs.get("batch_size", 8)
        explanations = WordImportance(mode="vlm")

        tokenizer_params = {}
        signature = inspect.signature(self.tokenizer).parameters
        if "padding" in signature:
            tokenizer_params["padding"] = True
        tokenized_texts = self.tokenizer(X.text.values, **tokenizer_params)

        for i, instance in enumerate(X):
            inputs = self.preprocess_function(instance)
            if not isinstance(inputs, (tuple, list)):
                inputs = (inputs,)
            scores = self.ig_model.compute_integrated_gradients(
                inputs=inputs,
                loss_function=self.loss_function,
                steps=steps,
                batch_size=batch_size,
                **kwargs
            )
            tokens = [t for t, m in zip(tokenized_texts["input_ids"][i],
                                        tokenized_texts["attention_mask"][i])
                      if m > 0]
            explanations.add(
                instance=instance.text.to_str(),
                tokens=[self.tokenizer.decode([t]) for t in tokens],
                importance_scores=scores,
                target_label=None
            )
        return explanations

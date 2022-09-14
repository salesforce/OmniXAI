#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import numpy as np
from typing import Callable
from omnixai.utils.misc import is_tf_available, is_torch_available
from omnixai.explainers.base import ExplainerBase
from omnixai.data.image import Image
from omnixai.explanations.image.pixel_importance import PixelImportance


def _smooth_grad_torch(
        X,
        y,
        model,
        preprocess_function,
        mode: str,
        num_samples: int,
        sigma: float
):
    import torch

    model.eval()
    device = next(model.parameters()).device
    inputs = preprocess_function(X) if preprocess_function is not None else X.to_numpy()
    inputs = inputs if isinstance(inputs, torch.Tensor) else \
        torch.tensor(inputs, dtype=torch.get_default_dtype())

    outputs = model(inputs.to(device))
    if mode == "classification":
        if y is not None:
            if type(y) == int:
                y = [y for _ in range(len(X))]
            else:
                assert len(X) == len(y), (
                    f"Parameter ``y`` is a {type(y)}, the length of y "
                    f"should be the same as the number of images in X."
                )
        else:
            scores = outputs.detach().cpu().numpy()
            y = np.argmax(scores, axis=1).astype(int)
    else:
        y = None

    gradients = 0
    idx = torch.arange(outputs.shape[0])
    input_images = inputs.detach().cpu().numpy()
    sigma = sigma * (np.max(input_images) - np.min(input_images))
    for i in range(num_samples):
        noise = np.random.randn(*inputs.shape) * sigma
        x = torch.tensor(
            input_images + noise,
            dtype=torch.get_default_dtype(),
            device=device,
            requires_grad=True
        )
        outputs = model(x.to(device))
        if y is not None:
            outputs = outputs[idx, y]
        grad = torch.autograd.grad(torch.unbind(outputs), x)[0]
        gradients += grad.detach().cpu().numpy()

    gradients = gradients / num_samples
    gradients = np.transpose(gradients, (0, 2, 3, 1))
    return gradients, y


def _smooth_grad_tf(
        X,
        y,
        model,
        preprocess_function,
        mode: str,
        num_samples,
        sigma: float
):
    import tensorflow as tf

    inputs = preprocess_function(X) if preprocess_function is not None else X.to_numpy()
    inputs = tf.convert_to_tensor(inputs)
    if mode == "classification":
        if y is not None:
            if type(y) == int:
                y = [y for _ in range(len(X))]
            else:
                assert len(X) == len(y), (
                    f"Parameter ``y`` is a {type(y)}, the length of y "
                    f"should be the same as the number of images in X."
                )
        else:
            predictions = model(inputs)
            y = tf.argmax(predictions, axis=-1).numpy().astype(int)
    else:
        y = None

    gradients = 0
    input_images = inputs.numpy()
    sigma = sigma * (np.max(input_images) - np.min(input_images))
    for i in range(num_samples):
        noise = np.random.randn(*inputs.shape) * sigma
        x = tf.Variable(
            noise + input_images,
            dtype=tf.float32,
            trainable=True
        )
        with tf.GradientTape() as tape:
            tape.watch(x)
            outputs = model(x)
            if y is not None:
                outputs = tf.reshape(tf.gather(outputs, y, axis=1), shape=(-1,))
            grad = tape.gradient(outputs, x)
            gradients += grad.numpy()

    gradients = gradients / num_samples
    return gradients, y


def _smooth_grad(
        X,
        y,
        model,
        preprocess_function,
        mode: str,
        num_samples: int,
        sigma: float
):
    if is_torch_available():
        import torch.nn as nn

        if isinstance(model, nn.Module):
            return _smooth_grad_torch(
                X=X,
                y=y,
                model=model,
                preprocess_function=preprocess_function,
                mode=mode,
                num_samples=num_samples,
                sigma=sigma
            )

    if is_tf_available():
        import tensorflow as tf

        if isinstance(model, tf.keras.Model):
            return _smooth_grad_tf(
                X=X,
                y=y,
                model=model,
                preprocess_function=preprocess_function,
                mode=mode,
                num_samples=num_samples,
                sigma=sigma
            )

    raise ValueError(f"`model` should be a tf.keras.Model "
                     f"or a torch.nn.Module instead of {type(model)}")


class SmoothGrad(ExplainerBase):
    """
    The Smooth-Grad method for generating visual explanations.
    If using this explainer, please cite `SmoothGrad: removing noise by adding noise,
    Smilkov et al., https://arxiv.org/abs/1706.03825`.
    """

    explanation_type = "local"
    alias = ["smoothgrad", "smooth-grad"]

    def __init__(
            self,
            model,
            preprocess_function: Callable,
            mode: str = "classification",
            **kwargs
    ):
        """
        :param model: The model to explain, whose type can be `tf.keras.Model` or `torch.nn.Module`.
        :param preprocess_function: The preprocessing function that converts the raw data
            into the inputs of ``model``.
        :param mode: The task type, e.g., `classification` or `regression`.
        """
        super().__init__()
        if not is_tf_available() and not is_torch_available():
            raise EnvironmentError("Both Torch and Tensorflow cannot be found.")

        self.model = model
        self.preprocess_function = preprocess_function
        self.mode = mode

    def _resize(self, image):
        """
        Rescales the raw input image to the input size of the model.

        :param image: The raw input image.
        :return: The resized image.
        """
        assert image.shape[0] == 1, "`image` can contain one instance only."
        if self.preprocess_function is None:
            return image

        y = image.to_numpy()
        x = self.preprocess_function(image)
        if not isinstance(x, np.ndarray):
            try:
                x = x.detach().cpu().numpy()
            except:
                x = x.numpy()
        x = x.squeeze()
        if x.shape[0] == 3:
            x = np.transpose(x, (1, 2, 0))

        min_a, max_a = np.min(y), np.max(y)
        min_b, max_b = np.min(x), np.max(x)
        r = (max_a - min_a) / (max_b - min_b + 1e-8)
        return Image(data=(r * x + min_a - r * min_b).astype(int), batched=False, channel_last=True)

    def explain(self, X: Image, y=None, num_samples=50, sigma=0.1, **kwargs):
        """
        Generates the explanations for the input instances.

        :param X: A batch of input instances.
        :param y: A batch of labels to explain. For regression, ``y`` is ignored.
            For classification, the top predicted label of each input instance will be explained
            when `y = None`.
        :param num_samples: The number of images used to compute smooth gradients.
        :param sigma: The sigma for calculating standard deviation of noise.
        :param kwargs: Additional parameters.
        :return: The explanations for all the instances, e.g., pixel importance scores.
        :rtype: PixelImportance
        """
        explanations = PixelImportance(self.mode)

        gradients, y = _smooth_grad(
            X=X,
            y=y,
            model=self.model,
            preprocess_function=self.preprocess_function,
            mode=self.mode,
            num_samples=num_samples,
            sigma=sigma
        )
        for i in range(len(X)):
            label = y[i] if y is not None else None
            explanations.add(
                image=self._resize(X[i]).to_numpy()[0],
                target_label=label,
                importance_scores=gradients[i]
            )
        return explanations

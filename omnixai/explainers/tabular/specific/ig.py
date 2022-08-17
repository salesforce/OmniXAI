#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
The integrated-gradient explainer for tabular data.
"""
import numpy as np
from typing import Callable

from ....utils.misc import is_tf_available, is_torch_available, tensor_to_numpy
from ..base import TabularExplainer
from ....data.tabular import Tabular
from ....explanations.tabular.feature_importance import FeatureImportance


class IntegratedGradient:
    """
    The class for computing integrated gradients. It can handle both
    tabular and image data.
    """

    @staticmethod
    def compute_integrated_gradients(model, inp, baseline, output_index, steps=50):
        """
        Computes integrated gradients given the model and the inputs. The model should be either
        `tf.keras.Model` or `torch.nn.Module`.

        :param model: The model, e.g., a `tf.keras.Model` or a `torch.nn.Module`.
        :param inp: The input instances to explain.
        :param baseline: The baselines to compare with.
        :param output_index: `None` for regression or the label index for classification.
        :param steps: The number of steps when computing integrated gradients.
        :return: The integrated gradients.
        :rtype: np.ndarray
        """
        if baseline is None:
            baseline = 0 * inp
        assert baseline.shape == inp.shape
        alphas = np.linspace(start=0.0, stop=1.0, num=steps, endpoint=True)
        inputs = np.stack([baseline + a * (inp - baseline) for a in alphas])
        gradients = None

        if not is_tf_available() and not is_torch_available():
            raise EnvironmentError("Both Torch and Tensorflow cannot be found.")

        if is_tf_available():
            import tensorflow as tf

            if isinstance(model, tf.keras.Model):
                inputs = tf.convert_to_tensor(inputs)
                with tf.GradientTape() as tape:
                    tape.watch(inputs)
                    predictions = model(inputs)
                    if len(predictions.shape) > 1:
                        assert (
                            output_index is not None
                        ), "The model has multiple outputs, the output index cannot be None"
                        predictions = predictions[:, output_index]
                    gradients = tape.gradient(predictions, inputs).numpy()

        if gradients is None and is_torch_available():
            import torch
            import torch.nn as nn
            from torch.autograd import grad

            if isinstance(model, nn.Module):
                model.eval()
                param = next(model.parameters())
                inputs = torch.tensor(inputs, dtype=torch.get_default_dtype(), requires_grad=True).to(param.device)
                predictions = model(inputs)
                if len(predictions.shape) > 1:
                    assert output_index is not None, "The model has multiple outputs, the output index cannot be None"
                    predictions = predictions[:, output_index]
                gradients = (
                    grad(
                        outputs=predictions, inputs=inputs, grad_outputs=torch.ones_like(predictions).to(param.device)
                    )[0]
                    .detach()
                    .cpu()
                    .numpy()
                )

        if gradients is None:
            raise ValueError(f"`model` should be a tf.keras.Model " f"or a torch.nn.Module instead of {type(model)}")

        # Use trapezoidal rule to approximate the integral.
        # See Section 4 of the following paper for an accuracy comparison between
        # left, right, and trapezoidal IG approximations:
        # "Computing Linear Restrictions of Neural Networks", Matthew Sotoudeh, Aditya V. Thakur
        # https://arxiv.org/abs/1908.06214
        gradients = (gradients[:-1] + gradients[1:]) / 2.0
        avg_grads = np.average(gradients, axis=0)
        integrated_grads = (inp - baseline) * avg_grads
        return integrated_grads


class IntegratedGradientTabular(TabularExplainer, IntegratedGradient):
    """
    The integrated-gradient explainer for tabular data. It only supports continuous-valued features.
    If using this explainer, please cite the original work: https://github.com/ankurtaly/Integrated-Gradients.
    """

    explanation_type = "local"
    alias = ["ig", "integrated_gradient"]

    def __init__(
        self,
        training_data: Tabular,
        model,
        preprocess_function: Callable = None,
        mode: str = "classification",
        **kwargs,
    ):
        """
        :param training_data: The data used to construct baselines. ``training_data``
            can be the training dataset for training the machine learning model. If the training
            dataset is large, ``training_data`` can be its subset by applying
            `omnixai.sampler.tabular.Sampler.subsample`.
        :param model: The ML model to explain, whose type can be `tf.keras.Model` or `torch.nn.Module`.
            When the model is for classification, the outputs of the ``model``
            are the class probabilities or logits. When the model is for regression, the outputs of
            the ``model`` are the estimated values.
        :param preprocess_function: The pre-processing function that converts the raw input data
            into the inputs of ``model``.
        :param mode: The task type, e.g., `classification` or `regression`.
        :param kwargs: Additional parameters to initialize the IG explainer,
            e.g., ``num_random_trials`` -- the number of trials in generating baselines
            (when ``num_random_trials`` is negative, the baseline will be the mean of ``training_data``).
        """
        super().__init__(training_data=training_data, predict_function=model, mode=mode, **kwargs)
        assert (
            len(self.categorical_columns) == 0
        ), "The integrated-gradient explainer only supports continuous-valued features"
        if not is_tf_available() and not is_torch_available():
            raise EnvironmentError("Both Torch and Tensorflow cannot be found.")

        model_type = None
        if is_tf_available():
            import tensorflow as tf

            if isinstance(model, tf.keras.Model):
                model_type = "tf"
        if model_type is None and is_torch_available():
            import torch.nn as nn

            if isinstance(model, nn.Module):
                model_type = "torch"
        if model_type is None:
            raise ValueError(
                f"`predict_function` should be a tf.keras.Model " f"or a torch.nn.Module instead of {type(model)}"
            )

        self.model = model
        self.preprocess_function = preprocess_function
        self.predict_fn = None

        num_random_trials = kwargs.get("num_random_trials", -1)
        if num_random_trials < 0:
            self.baselines = self._sample_baseline(use_mean=True)
        else:
            self.baselines = self._sample_baseline(use_mean=False, num_random_trials=num_random_trials)

    def _sample_baseline(self, use_mean=True, num_random_trials=10) -> np.ndarray:
        """
        Constructs the baselines for computing the integrated gradients.

        :param use_mean: When ``use_mean`` is True, it returns the mean of the training data.
        :param num_random_trials: When ``use_mean`` is False, it returns ``num_random_trials``
            randomly sampled instances as the baselines.
        :return: The constructed baselines.
        :rtype: np.ndarray
        """
        samples = self.transformer.invert(self.data).to_numpy(copy=False)
        if use_mean:
            baseline = np.mean(samples, axis=0)
            return np.expand_dims(baseline, axis=0)
        else:
            replace = samples.shape[0] < num_random_trials
            indices = np.random.choice(samples.shape[0], size=num_random_trials, replace=replace)
            return samples[indices]

    def _predict(self, inputs: np.ndarray) -> np.ndarray:
        """
        Predicts class labels in classification.

        :param inputs: The input instances.
        :return: The predicted labels.
        :rtype: np.ndarray
        """
        try:
            import torch

            self.model.eval()
            param = next(self.model.parameters())
            X = inputs if isinstance(inputs, torch.Tensor) else torch.tensor(inputs, dtype=torch.get_default_dtype())
            scores = self.model(X.to(param.device)).detach().cpu().numpy()
        except:
            scores = self.model(inputs).numpy()
        y = np.argmax(scores, axis=1).astype(int)
        return y

    def explain(self, X, y=None, baseline=None, **kwargs) -> FeatureImportance:
        """
        Generates the explanations for the input instances.

        :param X: A batch of input instances. When ``X`` is `pd.DataFrame`
            or `np.ndarray`, ``X`` will be converted into `Tabular` automatically.
        :param y: A batch of labels to explain. For regression, ``y`` is ignored.
            For classification, the top predicted label of each input instance will be explained
            when ``y = None``.
        :param baseline: The baselines for computing integrated gradients. When it is `None`,
            the baselines will be automatically generated by `IntegratedGradientTabular._sample_baseline`.
        :param kwargs: Additional parameters, e.g., ``steps`` for
            `IntegratedGradient.compute_integrated_gradients`.
        :return: The explanations for all the input instances.
        """
        steps = kwargs.get("steps", 50)
        explanations = FeatureImportance(self.mode)
        X = self._to_tabular(X).remove_target_column()

        baselines = self.baselines
        if baseline is not None:
            baselines = baseline if isinstance(baseline, list) else [baseline]
            baselines = np.array(baselines)
        baselines = Tabular(data=baselines, feature_columns=X.feature_columns)

        if self.preprocess_function is not None:
            instances = self.preprocess_function(X)
            instances = tensor_to_numpy(instances)
            baselines = self.preprocess_function(baselines)
            baselines = tensor_to_numpy(baselines)
        else:
            instances = self._to_numpy(X)
            baselines = self._to_numpy(baselines)

        if self.mode == "classification":
            if y is not None:
                if type(y) == int:
                    y = [y for _ in range(len(instances))]
                else:
                    assert len(instances) == len(y), (
                        f"Parameter ``y`` is a {type(y)}, the length of y "
                        f"should be the same as the number of instances in X."
                    )
            else:
                y = self._predict(instances)

        for i, instance in enumerate(instances):
            output_index = y[i] if y is not None else None
            all_gradients = []
            for baseline in baselines:
                integrated_grads = self.compute_integrated_gradients(
                    model=self.model, inp=instance, baseline=baseline, output_index=output_index, steps=steps
                )
                all_gradients.append(integrated_grads)
            grads = np.average(np.array(all_gradients), axis=0)
            explanations.add(
                instance=X.iloc(i).to_pd(),
                target_label=output_index,
                feature_names=self.feature_columns,
                feature_values=list(instance),
                importance_scores=grads,
                sort=True,
            )
        return explanations

    def save(
            self,
            directory: str,
            filename: str = None,
            **kwargs
    ):
        """
        Saves the initialized explainer.

        :param directory: The folder for the dumped explainer.
        :param filename: The filename (the explainer class name if it is None).
        """
        super().save(
            directory=directory,
            filename=filename,
            ignored_attributes=["data"],
            **kwargs
        )

#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
The L2X explainer for image data.
"""
import warnings
import sklearn
import numpy as np
from typing import Callable

from ...base import ExplainerBase
from ....data.image import Image
from ....explanations.image.pixel_importance import PixelImportance
from ....utils.misc import is_torch_available

if is_torch_available():
    import torch
    import torch.nn as nn
    from ...tabular.agnostic.L2X.utils import L2XModel, Trainer


    class _DefaultModelBase(nn.Module):
        def __init__(self, explainer, **kwargs):
            super().__init__()
            self.image_shape = explainer.data.shape[1:]
            self.output_dim = np.max(explainer.predictions) + 1 if explainer.mode == "classification" else 1
            self.scale = 255.0 if np.max(explainer.data) > 1 else 1.0


    class DefaultSelectionModel(_DefaultModelBase):
        """
        The default selection model in L2X, which is designed for MNIST.
        """

        def __init__(self, explainer, **kwargs):
            """
            :param explainer: A `L2XImage` explainer.
            :param kwargs: Additional parameters.
            """
            super().__init__(explainer, **kwargs)
            self.conv_layers = nn.Sequential(
                nn.Conv2d(self.image_shape[0], 10, kernel_size=3),
                nn.MaxPool2d(2),
                nn.ReLU(),
                nn.Conv2d(10, 20, kernel_size=3),
                nn.ReLU(),
                nn.Conv2d(20, 1, kernel_size=1),
            )
            self.out_size = self.conv_layers(torch.tensor(explainer.data[0:1], dtype=torch.get_default_dtype())).shape[1:]
            self.upsampling_layer = nn.Upsample(size=self.image_shape[1:], mode="bilinear")

        def forward(self, inputs):
            """
            :param inputs: The model inputs.
            """
            inputs = inputs / self.scale
            outputs = self.conv_layers(inputs)
            return outputs.view((inputs.shape[0], -1))

        def postprocess(self, inputs):
            """
            Upsamples to the original image size.

            :param inputs: The outputs of ``forward``.
            """
            inputs = inputs.view((-1, self.out_size[0], self.out_size[1], self.out_size[2]))
            return self.upsampling_layer(inputs)


    class DefaultPredictionModel(_DefaultModelBase):
        """
        The default prediction model in L2X, which is designed for MNIST.
        """

        def __init__(self, explainer, **kwargs):
            """
            :param explainer: A `L2XImage` explainer.
            :param kwargs: Additional parameters.
            """
            super().__init__(explainer, **kwargs)
            self.conv_layers = nn.Sequential(
                nn.Conv2d(self.image_shape[0], 10, kernel_size=5),
                nn.MaxPool2d(2),
                nn.ReLU(),
                nn.Conv2d(10, 20, kernel_size=5),
                nn.Dropout(),
                nn.MaxPool2d(2),
                nn.ReLU(),
            )
            conv_out_size = self.conv_layers(torch.tensor(explainer.data[0:1], dtype=torch.get_default_dtype())).shape[1:]
            self.fc_layers = nn.Sequential(
                nn.Linear(int(np.prod(conv_out_size)), 50), nn.ReLU(), nn.Dropout(), nn.Linear(50, self.output_dim)
            )

        def forward(self, inputs, weights):
            """
            :param inputs: The model inputs.
            :param weights: The weights generated via Gumbel-Softmax sampling.
            """
            inputs = inputs / self.scale
            outputs = self.conv_layers(inputs * weights)
            outputs = self.fc_layers(torch.flatten(outputs, 1))
            if outputs.shape[1] == 1:
                outputs = outputs.squeeze(dim=1)
            return outputs


class L2XImage(ExplainerBase):
    """
    The LIME explainer for vision tasks.
    If using this explainer, please cite the original work:
    `Learning to Explain: An Information-Theoretic Perspective on Model Interpretation,
    Jianbo Chen, Le Song, Martin J. Wainwright, Michael I. Jordan, https://arxiv.org/abs/1802.07814`.
    """

    explanation_type = "local"
    alias = ["l2x", "L2X"]

    def __init__(
        self,
        training_data: Image,
        predict_function: Callable,
        mode: str = "classification",
        tau: float = 0.5,
        k: int = 10,
        selection_model=None,
        prediction_model=None,
        loss_function: Callable = None,
        optimizer=None,
        learning_rate: float = 1e-3,
        batch_size: int = None,
        num_epochs: int = 20,
        **kwargs,
    ):
        """
        :param training_data: The data used to train the explainer. ``training_data``
            should be the training dataset for training the machine learning model.
        :param predict_function: The prediction function corresponding to the model to explain.
            When the model is for classification, the outputs of the ``predict_function``
            are the class probabilities. When the model is for regression, the outputs of
            the ``predict_function`` are the estimated values.
        :param mode: The task type, e.g., `classification` or `regression`.
        :param tau: Parameter ``tau`` in Gumbel-Softmax.
        :param k: The maximum number of the selected features in L2X.
        :param selection_model: A pytorch model class for estimating P(S|X) in L2X. If
            ``selection_model = None``, a default model `DefaultSelectionModel` will be used.
        :param prediction_model: A pytorch model class for estimating Q(X_S) in L2X. If
            ``prediction_model = None``, a default model `DefaultPredictionModel` will be used.
        :param loss_function: The loss function for the task, e.g., `nn.CrossEntropyLoss()`
            for classification.
        :param optimizer: The optimizer class for training the explainer, e.g., `torch.optim.Adam`.
        :param learning_rate: The learning rate for training the explainer.
        :param batch_size: The batch size for training the explainer. If ``batch_size`` is `None`,
            ``batch_size`` will be picked from `[32, 64, 128, 256]` based on the sample size.
        :param num_epochs: The number of epochs for training the explainer.
        :param kwargs: Additional parameters, e.g., parameters for ``selection_model``
            and ``prediction_model``.
        """
        super().__init__()
        assert is_torch_available(), \
            "PyTorch is not installed. L2XImage requires the installation of PyTorch."
        assert training_data.values is not None, "`training_data` cannot be empty."
        assert mode in [
            "classification",
            "regression",
        ], f"Unknown mode: {mode}, please choose `classification` or `regression`"
        self.device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.verbose = kwargs.get("verbose", True)
        self.dtype = kwargs.get("dtype", "float32")

        self.mode = mode
        self.predict_function = predict_function
        self.data = training_data.to_numpy(hwc=False, keepdim=True).astype(self.dtype)

        if mode == "classification":
            predicted_scores = predict_function(training_data)
            self.predictions = np.argmax(predicted_scores, axis=1).astype(int)
        else:
            self.predictions = predict_function(training_data).astype(self.dtype)

        selection_model = (
            DefaultSelectionModel(self, **kwargs) if selection_model is None else selection_model(self, **kwargs)
        )
        prediction_model = (
            DefaultPredictionModel(self, **kwargs) if prediction_model is None else prediction_model(self, **kwargs)
        )

        self.l2x = L2XModel(selection_model=selection_model, prediction_model=prediction_model, tau=tau, k=k).to(
            self.device
        )

        if loss_function is None:
            if mode == "classification":
                loss_function = nn.CrossEntropyLoss()
            else:
                loss_function = nn.MSELoss()
                warnings.warn("MSELoss is used in L2X, which may not generate reasonable results.")

        if batch_size is None:
            batch_size = 32
            for size in [256, 128, 64, 32]:
                if self.data.shape[0] / size > 100:
                    batch_size = size
                    break
        Trainer(
            optimizer_class=torch.optim.Adam if optimizer is None else optimizer,
            learning_rate=learning_rate,
            batch_size=batch_size,
            num_epochs=num_epochs,
        ).train(
            model=self.l2x, loss_func=loss_function, train_x=self.data, train_y=self.predictions, verbose=self.verbose
        )
        self._check_performance()

    def _check_performance(self):
        """
        Checks the performance of the trained L2X model. If the performance is bad,
        the generated explanations may not be reliable.
        """
        self.l2x.eval()
        outputs = self.l2x.prediction_model(
            torch.tensor(self.data, dtype=torch.get_default_dtype()).to(self.device),
            weights=torch.ones(self.data.shape).to(self.device),
        )
        if self.mode == "classification":
            predictions = np.argmax(outputs.detach().cpu().numpy(), axis=1)
            accuracy = sklearn.metrics.accuracy_score(self.predictions, predictions)
            if self.verbose:
                print(f"L2X prediction model accuracy: {accuracy}")
            if accuracy < 0.7:
                warnings.warn(
                    f"The L2X prediction model accuracy is too low, " "please tuning the training parameters."
                )
        else:
            predictions = outputs.detach().cpu().numpy()
            errors = np.mean(np.abs(self.predictions - predictions) / np.abs(self.predictions))
            if self.verbose:
                print(f"L2X prediction model error: {errors}")
            if errors > 0.3:
                warnings.warn(
                    f"The L2X prediction model accuracy is too low, " "please tuning the training parameters."
                )

    def explain(self, X: Image, **kwargs) -> PixelImportance:
        """
        Generates the explanations for the input instances. For classification,
        it explains the top predicted label for each input instance.

        :param X: A batch of input instances.
        :return: The explanations for all the input instances.
        """
        explanations = PixelImportance(self.mode)
        inputs = X.to_numpy(hwc=False, keepdim=True)
        inputs = torch.tensor(inputs, dtype=torch.get_default_dtype(), device=next(self.l2x.parameters()).device)
        importance_scores, _ = self.l2x.explain(inputs)
        importance_scores = importance_scores.detach().cpu().numpy()

        y = None
        if self.mode == "classification":
            prediction_scores = self.predict_function(X)
            y = np.argmax(prediction_scores, axis=1)

        for i, image in enumerate(X):
            scores = importance_scores[i].squeeze()
            if scores.ndim == 3 and scores.shape[0] == 3:
                scores = np.transpose(scores, (1, 2, 0))
            explanations.add(
                image=image.to_numpy()[0], target_label=y[i] if y is not None else None, importance_scores=scores
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
            ignored_attributes=["data", "predictions"],
            **kwargs
        )

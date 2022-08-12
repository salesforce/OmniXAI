#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
The L2X explainer for NLP tasks.
"""
import warnings
import sklearn
import numpy as np
from typing import Callable
from omnixai.utils.misc import is_torch_available
from omnixai.data.text import Text
from omnixai.explainers.base import ExplainerBase
from omnixai.preprocessing.text import Word2Id
from omnixai.explanations.text.word_importance import WordImportance

if is_torch_available():
    import torch
    import torch.nn as nn
    from ...tabular.agnostic.L2X.utils import L2XModel, Trainer
    from ...tabular.agnostic.L2X.utils import InputData, DataLoader

    class _DefaultModelBase(nn.Module):
        def __init__(self, explainer, **kwargs):
            super().__init__()
            self.output_dim = np.max(explainer.predictions) + 1 if explainer.mode == "classification" else 1
            self.num_embeddings = explainer.transform.vocab_size
            self.embedding_size = kwargs.get("embedding_size", 50)
            self.embedding = nn.Embedding(self.num_embeddings, self.embedding_size)
            self.embedding.weight.data.normal_(mean=0.0, std=0.01)

    class DefaultSelectionModel(_DefaultModelBase):
        """
        The default selection model in L2X for text data. It consists of five 1D convolution layers
        and one linear output layer.
        """

        def __init__(self, explainer, **kwargs):
            """
            :param explainer: A `L2XText` explainer.
            :param kwargs: Additional parameters, e.g., ``hidden_size`` -- the hidden layer size,
                and ``kernel_size`` -- the kernel size in 1D convolution.
            """
            super().__init__(explainer, **kwargs)
            hidden_size = kwargs.get("hidden_size", 100)
            kernel_size = kwargs.get("kernel_size", 3)

            self.activation = nn.ReLU()
            self.conv_1 = nn.Conv1d(self.embedding_size, hidden_size, kernel_size, padding=1)
            self.conv_2 = nn.Conv1d(hidden_size, hidden_size, kernel_size, padding=1)
            self.conv_3 = nn.Conv1d(hidden_size, hidden_size, kernel_size, padding=1)
            self.conv_4 = nn.Conv1d(hidden_size * 2, hidden_size, 1)
            self.conv_5 = nn.Conv1d(hidden_size, 1, 1)
            self.dense = nn.Linear(hidden_size, hidden_size)

        def forward(self, inputs, masks):
            """
            :param inputs: The input IDs.
            :param masks: The input masks.
            """
            embeddings = self.embedding(inputs).permute(0, 2, 1)
            first_layer = self.activation(self.conv_1(embeddings))
            a = self.activation(self.dense(first_layer.max(dim=2)[0]))
            b = self.activation(self.conv_3(self.activation(self.conv_2(first_layer))))
            x = a.unsqueeze(dim=-1).repeat(1, 1, b.shape[-1])
            x = torch.cat([x, b], dim=1)
            x = nn.Dropout(0.2)(x)

            outputs = self.activation(self.conv_4(x))
            outputs = self.conv_5(outputs).squeeze(dim=1)
            return outputs + (1 - masks) * -10000

    class DefaultPredictionModel(_DefaultModelBase):
        """
        The default prediction model in L2X for text data. It consists of three 1D convolution layers
        and one linear output layer by default.
        """

        def __init__(self, explainer, **kwargs):
            """
            :param explainer: A `L2XText` explainer.
            :param kwargs: Additional parameters, e.g., ``hidden_size`` -- the hidden layer size,
                and ``kernel_sizes`` -- the kernel sizes in the 1D convolution layers.
            """
            super().__init__(explainer, **kwargs)
            hidden_size = kwargs.get("hidden_size", 100)
            kernel_sizes = kwargs.get("kernel_sizes", [3, 4, 5])
            if type(kernel_sizes) == int:
                kernel_sizes = [kernel_sizes]

            self.activation = nn.ReLU()
            self.conv_layers = nn.ModuleList([nn.Conv1d(self.embedding_size, hidden_size, k) for k in kernel_sizes])
            self.dropout = nn.Dropout(0.2)
            self.output_layer = nn.Linear(len(kernel_sizes) * hidden_size, self.output_dim)

        def forward(self, inputs, masks, weights):
            """
            :param inputs: The input IDs.
            :param masks: The input masks.
            :param weights: The weights generated via Gumbel-Softmax sampling.
            """
            embeddings = self.embedding(inputs)
            x = embeddings * (masks * weights).unsqueeze(dim=-1)
            x = x.permute(0, 2, 1)
            x = [self.activation(layer(x).max(2)[0]) for layer in self.conv_layers]
            outputs = self.output_layer(self.dropout(torch.cat(x, dim=1)))
            if outputs.shape[1] == 1:
                outputs = outputs.squeeze(dim=1)
            return outputs


class L2XText(ExplainerBase):
    """
    The LIME explainer for text data.
    If using this explainer, please cite the original work:
    `Learning to Explain: An Information-Theoretic Perspective on Model Interpretation,
    Jianbo Chen, Le Song, Martin J. Wainwright, Michael I. Jordan, https://arxiv.org/abs/1802.07814`.
    """

    explanation_type = "local"
    alias = ["l2x", "L2X"]

    def __init__(
        self,
        training_data: Text,
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
            ``selection_model = None``, a default model `DefaultPredictionModel` will be used.
        :param loss_function: The loss function for the task, e.g., `nn.CrossEntropyLoss()`
            for classification.
        :param optimizer: The optimizer class for training the explainer, e.g., `torch.optim.AdamW`.
        :param learning_rate: The learning rate for training the explainer.
        :param batch_size: The batch size for training the explainer. If ``batch_size`` is `None`,
            ``batch_size`` will be picked from `[32, 64, 128, 256]` based on the sample size.
        :param num_epochs: The number of epochs for training the explainer.
        :param kwargs: Additional parameters, e.g., parameters for ``selection_model``
            and ``prediction_model``.
        """
        super().__init__()
        if not is_torch_available():
            raise EnvironmentError("Torch cannot be found.")
        assert training_data.values is not None, "`training_data` cannot be empty."
        assert mode in [
            "classification",
            "regression",
        ], f"Unknown mode: {mode}, please choose `classification` or `regression`"
        self.device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.verbose = kwargs.get("verbose", True)
        self.dtype = kwargs.get("dtype", "float32")
        self.max_length = kwargs.get("max_length", 256)

        self.mode = mode
        self.predict_function = predict_function
        self.transform = Word2Id().fit(training_data)
        self.data = self.transform.transform(training_data)

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
                if len(self.data) / size > 100:
                    batch_size = size
                    break
        Trainer(
            optimizer_class=torch.optim.AdamW if optimizer is None else optimizer,
            learning_rate=learning_rate,
            batch_size=batch_size,
            num_epochs=num_epochs,
        ).train(
            model=self.l2x,
            loss_func=loss_function,
            train_x=self.data,
            train_y=self.predictions,
            padding=True,
            max_length=self.max_length,
            verbose=self.verbose,
        )
        self._check_performance()

    def _check_performance(self):
        """
        Checks the performance of the trained L2X model. If the performance is bad,
        the generated explanations may not be reliable.
        """
        self.l2x.eval()
        data_loader = DataLoader(
            dataset=InputData(self.data, [0] * len(self.data), self.max_length),
            batch_size=32,
            collate_fn=InputData.collate_func,
            shuffle=False,
        )
        outputs = []
        for inputs in data_loader:
            value, mask, target = inputs
            value = value.to(self.device)
            mask = mask.to(self.device)
            y = self.l2x.prediction_model(value, mask, weights=torch.ones(value.shape).to(self.device))
            outputs.append(y.detach().cpu().numpy())
        outputs = np.concatenate(outputs, axis=0)

        if self.mode == "classification":
            predictions = np.argmax(outputs, axis=1)
            accuracy = sklearn.metrics.accuracy_score(self.predictions, predictions)
            if self.verbose:
                print(f"L2X prediction model accuracy: {accuracy}")
            if accuracy < 0.7:
                warnings.warn(
                    f"The L2X prediction model accuracy is too low, " "please tuning the training parameters."
                )
        else:
            predictions = outputs
            errors = np.mean(np.abs(self.predictions - predictions) / np.abs(self.predictions))
            if self.verbose:
                print(f"L2X prediction model error: {errors}")
            if errors > 0.3:
                warnings.warn(
                    f"The L2X prediction model accuracy is too low, " "please tuning the training parameters."
                )

    def explain(self, X: Text, **kwargs) -> WordImportance:
        """
        Generates the explanations for the input instances. For classification,
        it explains the top predicted label for each input instance.

        :param X: A batch of input instances.
        :return: The explanations for all the input instances.
        """
        explanations = WordImportance(self.mode)
        instances = self.transform.transform(X)
        data_loader = DataLoader(
            dataset=InputData(instances, [0] * len(instances), self.max_length),
            batch_size=32,
            collate_fn=InputData.collate_func,
            shuffle=False,
        )
        importance_scores = []
        for inputs in data_loader:
            value, mask, target = inputs
            value = value.to(self.device)
            mask = mask.to(self.device)
            weights, _ = self.l2x.explain(value, mask)
            weights = weights.detach().cpu().numpy()
            importance_scores += weights.tolist()

        y = None
        if self.mode == "classification":
            predicted_scores = self.predict_function(X)
            y = np.argmax(predicted_scores, axis=1)

        for i in range(len(X)):
            tokens = self.transform.invert([instances[i]])[0]
            scores = importance_scores[i][: len(tokens)]
            explanations.add(
                instance=X[i].to_str(),
                target_label=y[i] if y is not None else None,
                tokens=tokens,
                importance_scores=scores,
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

#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
The L2X explainer for tabular data.
"""
import warnings
import sklearn
import numpy as np
from typing import Callable
from omnixai.utils.misc import is_torch_available
from omnixai.data.tabular import Tabular
from omnixai.explainers.tabular.base import TabularExplainer
from omnixai.explanations.tabular.feature_importance import FeatureImportance

if is_torch_available():
    import torch
    import torch.nn as nn
    from .utils import L2XModel, Trainer

    class _DefaultModelBase(nn.Module):
        def __init__(self, explainer, **kwargs):
            super().__init__()
            assert explainer.data.shape[1] == len(explainer.categorical_columns) + len(explainer.continuous_columns)
            self.num_cates = len(explainer.categorical_columns)
            self.num_conts = len(explainer.continuous_columns)
            self.output_dim = np.max(explainer.predictions) + 1 if explainer.mode == "classification" else 1

            self.embedding_size = kwargs.get("embedding_size", 8)
            self.embedding, self.offsets = self._build_embedding(
                cate_data=explainer.data[:, : self.num_cates], embedding_size=self.embedding_size
            )
            if self.offsets is not None:
                self.offsets = nn.Parameter(torch.LongTensor(self.offsets), requires_grad=False)

        @staticmethod
        def _build_embedding(cate_data, embedding_size):
            if cate_data is None or cate_data.shape[1] == 0:
                return None, None
            num_embeddings = 1
            offsets = np.zeros((cate_data.shape[1],), dtype=int)
            for i in range(cate_data.shape[1]):
                offsets[i] = num_embeddings
                max_value = int(np.max(cate_data[:, i])) + 1
                num_embeddings += max_value
            embedding = nn.Embedding(num_embeddings, embedding_size)
            embedding.weight.data.uniform_(-0.001, 0.001)
            return embedding, np.expand_dims(offsets, axis=0)

        @staticmethod
        def _init_weights(module):
            if isinstance(module, nn.Embedding):
                module.weight.data.uniform_(-0.001, 0.001)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

    class DefaultSelectionModel(_DefaultModelBase):
        """
        The default selection model in L2X for tabular data. It is a simple feedforward
        neural network with three linear layers. The categorical features are mapped to
        embeddings.
        """

        def __init__(self, explainer, **kwargs):
            """
            :param explainer: A `L2XTabular` explainer.
            :param kwargs: Additional parameters, e.g., ``hidden_size`` --
                the hidden layer size.
            """
            super().__init__(explainer, **kwargs)
            hidden_size = kwargs.get("hidden_size", 100)
            input_size = self.num_cates * self.embedding_size + self.num_conts
            self.layers = nn.Sequential(
                nn.BatchNorm1d(input_size),
                nn.Linear(input_size, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, self.num_cates + self.num_conts),
            )

        def forward(self, inputs):
            """
            :param inputs: The model inputs.
            """
            if self.num_cates == 0:
                outputs = self.layers(inputs)
            else:
                x = inputs[:, : self.num_cates].long() + self.offsets
                x = self.embedding(x).view((inputs.shape[0], -1))
                if self.num_conts > 0:
                    y = inputs[:, self.num_cates :]
                    x = torch.cat([x, y], dim=1)
                outputs = self.layers(x)
            return outputs

    class DefaultPredictionModel(_DefaultModelBase):
        """
        The default prediction model in L2X for tabular data. It is a simple feedforward
        neural network with three linear layers. The categorical features are mapped to
        embeddings.
        """

        def __init__(self, explainer, **kwargs):
            """
            :param explainer: A `L2XTabular` explainer.
            :param kwargs: Additional parameters, e.g., ``hidden_size`` --
                the hidden layer size.
            """
            super().__init__(explainer, **kwargs)
            hidden_size = kwargs.get("hidden_size", 200)
            input_size = self.num_cates * self.embedding_size + self.num_conts
            self.layers = nn.Sequential(
                nn.BatchNorm1d(input_size),
                nn.Linear(input_size, hidden_size),
                nn.GELU(),
                nn.BatchNorm1d(hidden_size),
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.BatchNorm1d(hidden_size),
                nn.Linear(hidden_size, self.output_dim),
            )
            self.layers.apply(self._init_weights)

        def forward(self, inputs, weights):
            """
            :param inputs: The model inputs.
            :param weights: The weights generated via Gumbel-Softmax sampling.
            """
            if self.num_cates == 0:
                outputs = self.layers(inputs * weights)
            else:
                cate_weights = weights[:, : self.num_cates].unsqueeze(dim=-1)
                x = inputs[:, : self.num_cates].long() + self.offsets
                x = self.embedding(x) * cate_weights
                x = x.view((inputs.shape[0], -1))
                if self.num_conts > 0:
                    y = inputs[:, self.num_cates :] * weights[:, self.num_cates :]
                    x = torch.cat([x, y], dim=1)
                outputs = self.layers(x)

            if outputs.shape[1] == 1:
                outputs = outputs.squeeze(dim=1)
            return outputs


class L2XTabular(TabularExplainer):
    """
    The L2X explainer for tabular data.
    If using this explainer, please cite the original work:
    `Learning to Explain: An Information-Theoretic Perspective on Model Interpretation,
    Jianbo Chen, Le Song, Martin J. Wainwright, Michael I. Jordan, https://arxiv.org/abs/1802.07814`.
    """

    explanation_type = "local"
    alias = ["l2x", "L2X"]

    def __init__(
        self,
        training_data: Tabular,
        predict_function: Callable,
        mode: str = "classification",
        tau: float = 0.5,
        k: int = 8,
        selection_model=None,
        prediction_model=None,
        loss_function: Callable = None,
        optimizer=None,
        learning_rate: float = 1e-3,
        batch_size: int = None,
        num_epochs: int = 10,
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
        :param optimizer: The optimizer class for training the L2X explainer, e.g., `torch.optim.Adam`.
        :param learning_rate: The learning rate for training the L2X explainer.
        :param batch_size: The batch size for training the L2X explainer. If ``batch_size`` is `None`,
            ``batch_size`` will be picked from `[32, 64, 128, 256]` based on the sample size.
        :param num_epochs: The number of epochs for training the L2X explainer.
        :param kwargs: Additional parameters, e.g., parameters for ``selection_model``
            and ``prediction_model``.
        """
        super().__init__(training_data=training_data, predict_function=predict_function, mode=mode, **kwargs)
        assert training_data.shape[0] > 0, "`training_data` cannot be empty."
        if not is_torch_available():
            raise EnvironmentError("Torch cannot be found.")
        self.device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.verbose = kwargs.get("verbose", True)
        self.dtype = kwargs.get("dtype", "float32")

        self.data = self.data.astype(self.dtype)
        if mode == "classification":
            predicted_scores = self.predict_fn(self.data)
            self.predictions = np.argmax(predicted_scores, axis=1).astype(int)
        else:
            self.predictions = self.predict_fn(self.data).astype(self.dtype)

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
                if self.data.shape[0] / size > 50:
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

    def explain(self, X, **kwargs) -> FeatureImportance:
        """
        Generates the explanations corresponding to the input instances. For classification,
        it explains the top predicted label for each input instance.

        :param X: A batch of input instances. When ``X`` is `pd.DataFrame`
            or `np.ndarray`, ``X`` will be converted into `Tabular` automatically.
        :param kwargs: Not used here.
        :return: The feature-importance explanations for all the input instances.
        """
        X = self._to_tabular(X).remove_target_column()
        explanations = FeatureImportance(self.mode)
        instances = self.transformer.transform(X).astype(self.dtype)
        inputs = torch.tensor(instances, dtype=torch.get_default_dtype(), device=next(self.l2x.parameters()).device)
        scores, _ = self.l2x.explain(inputs)
        scores = scores.detach().cpu().numpy()

        y = None
        if self.mode == "classification":
            prediction_scores = self.predict_fn(instances)
            y = np.argmax(prediction_scores, axis=1)

        for i, instance in enumerate(instances):
            df = X.iloc(i).to_pd()
            feature_values = [df[self.feature_columns[feat]].values[0] for feat in range(len(self.feature_columns))]
            explanations.add(
                instance=df,
                target_label=y[i] if y is not None else None,
                feature_names=self.feature_columns,
                feature_values=feature_values,
                importance_scores=scores[i],
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
            ignored_attributes=["data", "predictions"],
            **kwargs
        )

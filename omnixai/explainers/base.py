#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
The base classes for the supported explainers.
"""
import inspect
import numpy as np
from abc import abstractmethod
from collections import OrderedDict, defaultdict
from typing import Collection, Callable, Any, Dict

from ..utils.misc import AutodocABCMeta, build_predict_function
from ..data.base import Data
from ..explanations.base import PredictedResults

_EXPLAINERS = defaultdict(list)


class ExplainerABCMeta(AutodocABCMeta):
    """
    The meta class for an explainer. It will automatically register an explainer class,
    i.e., storing it in ``_EXPLAINERS``.
    """

    def __new__(mcls, classname, bases, cls_dict):
        cls = super().__new__(mcls, classname, bases, cls_dict)
        if not inspect.isabstract(cls):
            _module = cls.__module__.split(".")[2]
            _name = cls.__name__
            if _name in _EXPLAINERS[_module]:
                raise RuntimeError(
                    f"Explainer class name `{_name}` exists in `{_module}`. " f"Please use a different class name."
                )
            _EXPLAINERS[_module].append(cls)
        return cls


class ExplainerBase(metaclass=ExplainerABCMeta):
    """
    The abstract base class for an explainer. If an explainer inherits from this class,
    it will be registered automatically.
    """

    def __init__(self):
        pass

    @abstractmethod
    def explain(self, **kwargs):
        raise NotImplementedError

    @property
    def explanation_type(self):
        """
        :return: A string indicates the explanation type, e.g., local, global or both
        """
        return "local"


class AutoExplainerBase(metaclass=AutodocABCMeta):
    """
    The base class for task-specific explainers. The class derived from `AutoExplainerBase`
    acts as a explainer factory and provides a unified interface to create explainers, e.g.,
    allowing users to choose multiple explainers and generate different explanations at the same time.
    """

    _EXPLAINERS = _EXPLAINERS

    def __init__(
        self,
        explainers: Collection[str],
        mode: str,
        data: Data,
        model: Any,
        preprocess: Callable = None,
        postprocess: Callable = None,
        params: Dict = None,
    ):
        """
        :param explainers: The names or alias of the explainers to use.
        :param mode: The task type, e.g. classification or regression.
        :param data: The training data used to initialize explainers.
            For image or text explainers, it can be empty, e.g., `data = Image()` or `data = Text()`.
        :param model: The machine learning model to explain, which can be a scikit-learn model,
            a tensorflow model, a torch model, or a black-box prediction function.
        :param preprocess: The preprocessing function that converts the raw input features
            into the inputs of ``model``.
        :param postprocess: The postprocessing function that transforms the outputs of ``model``
            to a user-specific form, e.g., the predicted probability for each class.
        :param params: A dict containing the additional parameters for initializing each explainer,
            e.g., `params["lime"] = {"param_1": param_1, ...}`.
        """
        super().__init__()
        self._NAME_TO_CLASS = {_class.__name__: _class for _class in self._MODELS}
        for _class in self._MODELS:
            if hasattr(_class, "alias"):
                for name in _class.alias:
                    assert name not in self._NAME_TO_CLASS, f"Alias {name} exists, please use a different one."
                    self._NAME_TO_CLASS[name] = _class

        for name in explainers:
            name = name.split("#")[0]
            assert (
                name in self._NAME_TO_CLASS
            ), f"Explainer {name} is not found. Please choose explainers from {self._NAME_TO_CLASS.keys()}"

        self.names = explainers
        self.mode = mode
        self.data = data
        self.model = model
        self.preprocess = preprocess
        self.postprocess = postprocess
        self.predict_function = self._build_predict_function()
        self.explainers = self._build_explainers(params)

    def _build_predict_function(self):
        """
        Constructs the prediction function based on the preprocessing function,
        the model and the postprocessing function.

        :return: The prediction function.
        :rtype: Callable
        """
        return build_predict_function(
            model=self.model,
            preprocess=self.preprocess,
            postprocess=self.postprocess,
            mode=self.mode
        )

    def _build_explainers(self, params):
        """
        Creates instances for the specified explainers. It checks the signatures of
        the ``__init__`` function of each explainer and determines how to initialize them.

        :return: A dict of initialized explainers.
        :rtype: Dict
        """
        if params is None:
            params = {}
        explainers = {}
        for name in self.names:
            explainer_name = name.split("#")[0]
            _class = self._NAME_TO_CLASS[explainer_name]
            _param = params.get(name, {})
            _signature = inspect.signature(_class.__init__).parameters
            try:
                if "predict_function" in _signature:
                    explainer = _class(
                        predict_function=self.predict_function,
                        mode=self.mode,
                        training_data=self.data,
                        **_param
                    )
                elif "model" in _signature:
                    explainer = _class(
                        model=self.model,
                        preprocess_function=self.preprocess,
                        postprocess_function=self.postprocess,
                        mode=self.mode,
                        training_data=self.data,
                        **_param,
                    )
                else:
                    explainer = _class(
                        mode=self.mode,
                        training_data=self.data,
                        **_param
                    )
                explainers[name] = explainer
            except Exception as e:
                raise type(e)(f"Explainer {name} -- {str(e)}")
        return explainers

    def _predict(self, X):
        """
        Gets the predictions given input instances.

        :return: The predictions results.
        :rtype: PredictedResults
        """
        predictions = self.predict_function(X)
        if not isinstance(predictions, np.ndarray):
            try:
                predictions = predictions.detach().cpu().numpy()
            except AttributeError:
                predictions = predictions.numpy()
        return PredictedResults(predictions)

    def explain(self, X, params=None):
        """
        Generates local explanations for the specific instances.

        :param X: The instances to explain.
        :param params: A dict containing the additional parameters for generating explanations,
            e.g., `params["lime"] = {"param_1": param_1, ...}`.
        :return: A dict of explanation results generated by the explainers that support local explanation,
            e.g. `{"lime": lime_explanations, "shap": shap_explanations, ...}`.
        :rtype: OrderedDict
        """
        if params is None:
            params = {}
        if self.mode != "data_analysis":
            explanations = OrderedDict({"predict": self._predict(X)})
        else:
            explanations = OrderedDict()

        for name in self.names:
            if self.explainers[name].explanation_type in ["local", "both"]:
                try:
                    param = params.get(name, {})
                    explanations[name] = self.explainers[name].explain(X=X, **param)
                except Exception as e:
                    raise type(e)(f"Explainer {name} -- {str(e)}")
        return explanations

    def explain_global(self, params=None):
        """
        Generates global explanations.

        :param params: A dict containing the additional parameters for generating explanations,
            e.g., `params["pdp"] = {"param_1": param_1, ...}`.
        :return: A dict of explanation results generated by the explainers that support global explanation,
            e.g. `{"pdp": pdp_explanations, ...}`.
        :rtype: OrderedDict
        """
        if params is None:
            params = {}
        explanations = OrderedDict()
        for name in self.names:
            if self.explainers[name].explanation_type in ["global", "both"]:
                try:
                    param = params.get(name, {})
                    explanations[name] = self.explainers[name].explain(**param)
                except Exception as e:
                    raise type(e)(f"Explainer {name} -- {str(e)}")
        return explanations

    @property
    def explainer_names(self):
        """
        Gets the names of the specified explainers.

        :return: Explainer names.
        :rtype: Collection
        """
        return self.names

    @staticmethod
    def list_explainers():
        """
        List the supported explainers.
        """
        pass

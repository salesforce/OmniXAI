#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
from typing import Collection, Callable, Any, Dict

from ...data.image import Image
from ..base import AutoExplainerBase


class VisionExplainer(AutoExplainerBase):
    """
    The class derived from `AutoExplainerBase` for vision tasks,
    allowing users to choose multiple explainers and generate
    different explanations at the same time.

    .. code-block:: python

        explainer = VisionExplainer(
            explainers=["gradcam", "lime", "ig"],
            mode="classification",
            model=model,
            preprocess=preprocess_function,
            postprocess=postprocess_function,
            params={"gradcam": {"target_layer": model.layer4[-1]}}
        )
        local_explanations = explainer.explain(img)
    """

    _MODELS = AutoExplainerBase._EXPLAINERS[__name__.split(".")[2]]

    def __init__(
        self,
        explainers: Collection,
        mode: str,
        model: Any,
        data: Image = Image(),
        preprocess: Callable = None,
        postprocess: Callable = None,
        params: Dict = None,
    ):
        """
        :param explainers: The names or alias of the explainers to use.
        :param mode: The task type, e.g. `classification` or `regression`.
        :param model: The machine learning model to explain, which can be a scikit-learn model,
            a tensorflow model, a torch model, or a black-box prediction function.
        :param data: The training data used to initialize explainers.
            It can be empty, e.g., `data = Image()`, for those explainers such as
            `IntegratedGradient` and `Grad-CAM` that don't require training data.
        :param preprocess: The preprocessing function that converts the raw input features
            into the inputs of ``model``.
        :param postprocess: The postprocessing function that transforms the outputs of ``model``
            to a user-specific form, e.g., the predicted probability for each class.
        :param params: A dict containing the additional parameters for initializing each explainer,
            e.g., `params["gradcam"] = {"param_1": param_1, ...}`.
        """
        super().__init__(
            explainers=explainers,
            mode=mode,
            data=data,
            model=model,
            preprocess=preprocess,
            postprocess=postprocess,
            params=params,
        )

    @staticmethod
    def list_explainers():
        """
        List the supported explainers.
        """
        from tabulate import tabulate
        lists = []
        for _class in VisionExplainer._MODELS:
            alias = _class.alias if hasattr(_class, "alias") else _class.__name__
            explanation_type = _class.explanation_type \
                if _class.explanation_type != "both" else "global & local"
            lists.append([_class.__module__, _class.__name__, alias, explanation_type])
        table = tabulate(
            lists,
            headers=["Package", "Explainer Class", "Alias", "Explanation Type"],
            tablefmt='orgtbl'
        )
        print(table)

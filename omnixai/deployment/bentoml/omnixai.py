#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
from __future__ import annotations

import json
import os.path
import typing as t
from types import ModuleType

import bentoml
from bentoml import Tag
from bentoml.models import Model
from bentoml.models import ModelContext
from bentoml.exceptions import NotFound
from bentoml.exceptions import BentoMLException
from omnixai.explainers.base import AutoExplainerBase
from omnixai.utils.misc import get_pkg_version
from omnixai.explanations.utils import DefaultJsonEncoder

MODULE_NAME = "omnixai.deployment.bentoml.omnixai"
MODEL_PATH = "saved_model"
API_VERSION = "v1"


def get(tag_like: str | Tag) -> Model:
    model = bentoml.models.get(tag_like)
    if model.info.module not in (MODULE_NAME, __name__):
        raise NotFound(
            f"Model {model.tag} was saved with module {model.info.module}, not loading with {MODULE_NAME}."
        )
    return model


def load_model(
        bento_model: str | Tag | Model,
) -> AutoExplainerBase:
    """
    Load the OmniXAI explainer with the given tag from the local BentoML model store.

    :param bento_model: Either the tag of the model to get from the store, or a BentoML `~bentoml.Model`
        instance to load the model from.
    :return: The dumped OmniXAI explainer, e.g., ``TabularExplainer``, ``VisionExplainer``.
    """
    if not isinstance(bento_model, Model):
        bento_model = get(bento_model)

    if bento_model.info.module not in (MODULE_NAME, __name__):
        raise BentoMLException(
            f"Model {bento_model.tag} was saved with module {bento_model.info.module}, not loading with {MODULE_NAME}."
        )
    model_path = bento_model.path_of(MODEL_PATH)
    return AutoExplainerBase.load(model_path)


def save_model(
        name: str,
        model: AutoExplainerBase,
        *,
        mode: str = "model_and_data",
        signatures: t.Dict = None,
        labels: t.Dict[str, str] | None = None,
        custom_objects: t.Dict[str, t.Any] | None = None,
        external_modules: t.List[ModuleType] | None = None,
        metadata: t.Dict[str, t.Any] | None = None,
) -> bentoml.Model:
    """
    Save an OmniXAI explainer to BentoML modelstore.

    :param name: The name for given model instance. This should pass Python identifier check.
    :param model: The explainer to be saved.
    :param mode: How to save the explainers, i.e., "model_and_data" and "individual".
        "model_and_data" for saving the model, preprocessing function, postprocessing function
        and dataset for initialization, or "individual" for saving each initialized explainer
        in the AutoExplainer. When there is no explainer that needs to train a post-hoc
        explanation model (e.g., L2X) and the dataset for initialization is not too large,
        "model_and_data" is a better option. Otherwise, "individual" is a proper option.
    :param signatures: The methods to expose for running inference on the target model.
    :param labels: User-defined labels for managing models, e.g. team=nlp, stage=dev.
    :param custom_objects: User-defined additional python objects to be saved alongside the model.
    :param external_modules: User-defined additional python modules to be saved alongside the model or custom objects.
    :param metadata: Custom metadata for the given explainer.
    :return: A bentoml model.
    """
    if not isinstance(model, AutoExplainerBase):
        raise TypeError(
            f"Given model ({model}) is not an instance of ``AutoExplainerBase``."
        )

    context = ModelContext(
        framework_name="omnixai",
        framework_versions={"omnixai": get_pkg_version("omnixai")},
    )

    if signatures is None:
        signatures = {
            "predict": {"batchable": True},
            "explain": {"batchable": False}
        }

    with bentoml.models.create(
            name,
            module=MODULE_NAME,
            api_version=API_VERSION,
            labels=labels,
            custom_objects=custom_objects,
            external_modules=external_modules,
            metadata=metadata,
            context=context,
            signatures=signatures,
    ) as bento_model:
        model.save(bento_model.path_of(MODEL_PATH), mode=mode)
        return bento_model


def get_runnable(bento_model: Model):

    class OmniXAIRunnable(bentoml.Runnable):
        SUPPORTED_RESOURCES = ("cpu", "nvidia.com/gpu")
        SUPPORTS_CPU_MULTI_THREADING = True

        def __init__(self):
            super().__init__()
            self.model = load_model(bento_model)

    def add_runnable_method(method_name, options):
        def _run(self, *args, **kwargs):
            results = getattr(self.model, method_name)(*args, **kwargs)
            return results

        OmniXAIRunnable.add_method(
            _run,
            name=method_name,
            batchable=options.batchable,
            batch_dim=options.batch_dim,
            input_spec=options.input_spec,
            output_spec=options.output_spec,
        )

    for method_name, options in bento_model.info.signatures.items():
        add_runnable_method(method_name, options)

    return OmniXAIRunnable


def init_service(
        model_tag,
        task_type,
        service_name,
        api_name=None,
        api_doc=None,
        api_route=None
):
    """
    :param model_tag: The tag of the model.
    :param task_type: The task type, e.g., tabular, vision or nlp.
    :param service_name: The name of the service.
    :param api_name: The base api name. For example, if `api_name = 'abc'`,
        the name of the "predict" and "explain" apis will be "abc_predict" and "abc_explain", respectively.
    :param api_doc: The api docs. It can either be a string or a tuple (list) of length 2. If it is a tuple (list),
        the first element will be the doc of the "predict" api, and the second one will be the doc of the "explain" api.
    :param api_route: The base api route. For example, if `api_route = '/abc'`,
        the name of the "predict" and "explain" apis will be "/abc/predict" and "/abc/explain", respectively.
    """
    from bentoml.io import NumpyNdarray, JSON, Text, Multipart
    assert task_type in ["tabular", "vision", "nlp"], \
        f"`task_type` should be 'tabular', 'vision' or 'nlp' other than {task_type}."

    model = get(model_tag)
    runner = model.to_runner()
    svc = bentoml.Service(service_name, runners=[runner])

    if api_doc is None:
        predict_doc, explain_doc = None, None
    elif isinstance(api_doc, str):
        predict_doc, explain_doc = api_doc, api_doc
    elif isinstance(api_doc, (list, tuple)):
        assert len(api_doc) == 2, \
            f"`api_doc` is a {type(api_doc)} but has {len(api_doc)} element(s), " \
            f"which can only have 2 elements."
        predict_doc, explain_doc = api_doc[0], api_doc[1]
    else:
        raise TypeError(f"`api_doc` should be str, list or tuple instead of {type(api_doc)}.")

    if task_type == "tabular":
        predict_input_spec = NumpyNdarray()
        explain_input_spec = Multipart(data=NumpyNdarray(), params=JSON())
    elif task_type == "vision":
        predict_input_spec = NumpyNdarray()
        explain_input_spec = Multipart(data=NumpyNdarray(), params=JSON())
    elif task_type == "nlp":
        predict_input_spec = Text()
        explain_input_spec = Multipart(data=Text(), params=JSON())
    else:
        raise ValueError(f"Unknown `task_type`: {task_type}")

    @svc.api(
        input=predict_input_spec,
        output=Text(),
        name=f"{api_name}_predict" if api_name is not None else None,
        doc=predict_doc,
        route=os.path.join(api_route, "predict") if api_route is not None else None
    )
    def predict(data):
        result = runner.predict.run(data)
        return result.to_json()

    @svc.api(
        input=explain_input_spec,
        output=Text(),
        name=f"{api_name}_explain" if api_name is not None else None,
        doc=explain_doc,
        route=os.path.join(api_route, "explain") if api_route is not None else None
    )
    def explain(data, params):
        result = runner.explain.run(data, params, run_predict=False)
        return json.dumps(result, cls=DefaultJsonEncoder)

    return svc

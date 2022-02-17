#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
from typing import Collection, Dict

from ...data.tabular import Tabular
from ..base import AutoExplainerBase


class DataAnalyzer(AutoExplainerBase):
    """
    The class derived from `AutoExplainerBase` for data analysis,
    allowing users to choose multiple explainers and generate
    different explanations at the same time.

    .. code-block:: python

        explainers = TabularExplainer(
            explainers=["imbalance"],
            data=data,
            params={"imbalance": {"n_bins": 10}}
        )
        explanations = explainers.explain()
    """

    _MODELS = AutoExplainerBase._EXPLAINERS[__name__.split(".")[2]]

    def __init__(self, explainers: Collection, data: Tabular, params: Dict = None):
        """
        :param explainers: The names or alias of the analyzers to use.
        :param data: The training data used to initialize explainers.
        :param params: A dict containing the additional parameters for initializing each analyzer,
            e.g., `params["imbalance"] = {"param_1": param_1, ...}`.
        """
        super().__init__(
            explainers=explainers,
            mode="data_analysis",
            data=data,
            model=None,
            preprocess=None,
            postprocess=None,
            params=params,
        )

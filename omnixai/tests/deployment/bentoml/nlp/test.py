#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
from omnixai.data.text import Text
from omnixai.deployment.bentoml.omnixai import init_service


def test():
    svc = init_service(
        model_tag="nlp_explainer:latest",
        task_type="nlp",
        service_name="nlp_explainer"
    )
    for runner in svc.runners:
        runner.init_local()

    x = Text([
        "it was a fantastic performance!",
        "best film ever",
    ])
    predictions = svc.apis["predict"].func(x)
    print(predictions)
    local_explanations = svc.apis["explain"].func(x, {})

    from omnixai.explainers.base import AutoExplainerBase
    from omnixai.visualization.dashboard import Dashboard
    exp = AutoExplainerBase.parse_explanations_from_json(local_explanations)
    dashboard = Dashboard(instances=x, local_explanations=exp)
    dashboard.show()


if __name__ == "__main__":
    test()

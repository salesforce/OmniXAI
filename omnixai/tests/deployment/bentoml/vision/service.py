#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
from omnixai.deployment.bentoml.omnixai import init_service

print("Loading vision service...")

svc = init_service(
    model_tag="vision_explainer:latest",
    task_type="vision",
    service_name="vision_explainer"
)

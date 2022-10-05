from omnixai.deployment.bentoml.omnixai import get, init_service


model = get("tabular_explainer:latest")
svc = init_service(model, "tabular_explainer")

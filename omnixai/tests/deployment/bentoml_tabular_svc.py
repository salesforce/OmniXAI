from omnixai.deployment.bentoml.omnixai import init_service


svc = init_service(
    model_tag="tabular_explainer:latest",
    task_type="tabular",
    service_name="tabular_explainer"
)

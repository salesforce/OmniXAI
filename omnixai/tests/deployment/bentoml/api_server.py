import uvicorn
from bentoml._internal.log import configure_server_logging
from bentoml._internal.context import component_context
from omnixai.deployment.bentoml.omnixai import init_service


def main():
    component_context.component_type = "dev_api_server"
    configure_server_logging()

    svc = init_service(
        model_tag="tabular_explainer:latest",
        task_type="tabular",
        service_name="tabular_explainer"
    )

    component_context.component_name = svc.name
    if svc.tag is None:
        component_context.bento_name = f"*{svc.__class__.__name__}"
        component_context.bento_version = "not available"
    else:
        component_context.bento_name = svc.tag.name
        component_context.bento_version = svc.tag.version

    uvicorn_options = {
        "backlog": 2048,
        "log_config": None,
        "workers": 1,
        "lifespan": "on",
        "port": 5000
    }
    config = uvicorn.Config(svc.asgi_app, **uvicorn_options)
    uvicorn.Server(config).run()


if __name__ == "__main__":
    main()

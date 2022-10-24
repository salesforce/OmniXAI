import uvicorn
import socket
import click
from bentoml._internal.log import configure_server_logging
from bentoml._internal.context import component_context
from omnixai.deployment.bentoml.omnixai import init_service


async def app(scope, receive, send):
    assert scope['type'] == 'http'

    await send({
        'type': 'http.response.start',
        'status': 200,
        'headers': [
            [b'content-type', b'text/plain'],
        ],
    })
    await send({
        'type': 'http.response.body',
        'body': b'Hello, world!',
    })


@click.command()
@click.option("--fd", type=click.INT, required=False, default=-1)
def main(fd):
    print("Starting server...")
    component_context.component_type = "dev_api_server"
    configure_server_logging()

    svc = init_service(
        model_tag="nlp_explainer:latest",
        task_type="nlp",
        service_name="nlp_explainer"
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
        # "lifespan": "on",
        "port": 5000
    }
    config = uvicorn.Config(svc.asgi_app, **uvicorn_options)

    if fd != -1:
        sock = socket.socket(fileno=fd)
        uvicorn.Server(config).run(sockets=[sock])
    else:
        uvicorn.Server(config).run()
    print("finished.")


if __name__ == "__main__":
    main()

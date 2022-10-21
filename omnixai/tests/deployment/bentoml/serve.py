import sys
import logging
from circus.sockets import CircusSocket
from circus.watcher import Watcher


SCRIPT_DEV_API_SERVER = "omnixai.tests.deployment.bentoml.api_server"
logger = logging.getLogger(__name__)


def create_standalone_arbiter(watchers, **kwargs):
    from circus.arbiter import Arbiter
    return Arbiter(
        watchers,
        endpoint=f"tcp://127.0.0.1:5123",
        pubsub_endpoint=f"tcp://127.0.0.1:5124",
        **kwargs,
    )


def serve():
    host = "0.0.0.0"
    port = 3000
    backlog = 2048

    watchers = []
    circus_sockets = []
    circus_sockets.append(
        CircusSocket(
            name="_bento_api_server",
            host=host,
            port=port,
            backlog=backlog,
        )
    )
    args = [
        "-m",
        SCRIPT_DEV_API_SERVER,
        "--fd",
        "$(circus.sockets._bento_api_server)",
    ]

    watchers.append(
        Watcher(
            name="dev_api_server",
            cmd=sys.executable,
            args=args,
            copy_env=True,
            stop_children=True,
            use_sockets=True,
            close_child_stdin=False,
        )
    )

    arbiter = create_standalone_arbiter(
        watchers,
        sockets=circus_sockets,
        debug=True,
        loglevel="WARNING",
    )
    arbiter.start(
        cb=lambda _: logger.info(
            f'Starting development BentoServer from '
            f"running on http://{host}:{port} (Press CTRL+C to quit)"
        ),
    )


if __name__ == "__main__":
    serve()

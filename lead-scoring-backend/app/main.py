import sentry_sdk
from fastapi import FastAPI
from sentry_sdk.integrations.asgi import SentryAsgiMiddleware

from app.routers import load_routers
from app.config import SENTRY_DSN, ENVIRONMENT, SENTRY_TRACE_SAMPLE_RATE

# TODO: exclude healthcheck when using traces_sample https://elsacorp.atlassian.net/browse/DATA-214
sentry_sdk.init(dsn=SENTRY_DSN,
                environment=ENVIRONMENT,
                traces_sample_rate=SENTRY_TRACE_SAMPLE_RATE)


def get_app():
    create_app = FastAPI()
    load_routers(create_app)
    return create_app


fast_api_app = get_app()

app = SentryAsgiMiddleware(fast_api_app)

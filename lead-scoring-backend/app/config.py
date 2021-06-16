import os
from os.path import join, dirname

from dotenv import load_dotenv
from pydantic import BaseSettings

dotenv_path = join(dirname(__file__), '../.env')
load_dotenv(dotenv_path)

API_KEY = os.environ.get("API_KEY")
API_KEY_NAME = os.environ.get("API_KEY_NAME")
SENTRY_DSN = os.environ.get("SENTRY_DSN", '')
ENVIRONMENT = os.environ.get('ENVIRONMENT', 'dev')
SENTRY_TRACE_SAMPLE_RATE = float(os.environ.get('SENTRY_TRACE_SAMPLE_RATE', 0))


class Settings(BaseSettings):
    app_name: str = "Lead Scoring API"


settings = Settings()

import os
from pathlib import Path

DATA_DIR = Path(os.environ.get("DATA_DIR", "data"))

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Env(BaseSettings):
    config_dir: str = Field(default="config")
    config_name: str = Field(default="config.yaml")

    wandb_api_key: str = Field(default="")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

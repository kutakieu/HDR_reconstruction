from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Env(BaseSettings):
    config_dir: str = Field(default="config")
    config_name: str = Field(default="config.yaml")

    data_dir: Path = Field(default="data")
    hdr_dir: str = Field(default="hdr")
    ldr_dir: str = Field(default="ldr")

    wandb_api_key: str = Field(default="")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Env(BaseSettings):
    config_dir: str = Field(default="config")
    config_name: str = Field(default="config.yaml")

    data_dir: Path = Field(default="data")
    hdr_dir: str = Field(default="calibrated_hdrs")
    ldr_dir: str = Field(default="calibrated_ldrs")

    wandb_api_key: str = Field(default="")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

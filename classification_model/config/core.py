from pathlib import Path
from typing import Sequence

from pydantic import BaseModel, ConfigDict
from strictyaml import YAML, load

import classification_model

# Project Directories
PACKAGE_ROOT = Path(classification_model.__file__).resolve().parent
ROOT = PACKAGE_ROOT.parent
CONFIG_FILE_PATH = PACKAGE_ROOT / "config.yml"
DATASET_DIR = PACKAGE_ROOT / "datasets"
TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_models"


class AppConfig(BaseModel):
    """
    Application-level config.
    """

    package_name: str
    training_data_file: str
    pipeline_save_file: str


class ModelConfig(BaseModel):
    """
    All configuration relevant to model
    training and feature engineering.
    """

    target: str
    # unused_fields: Sequence[str]
    features: Sequence[str]
    test_size: float
    random_state: int
    # numerical_vars: Sequence[str]
    # categorical_vars: Sequence[str]
    # cabin_vars: Sequence[str]


class Config(BaseModel):
    """Master config object."""
    model_config = ConfigDict(
        protected_namespaces=()
    )
    app_conf: AppConfig
    model_conf: ModelConfig


def find_config_file() -> Path:
    """Locate the configuration file."""
    if CONFIG_FILE_PATH.is_file():
        return CONFIG_FILE_PATH
    raise Exception(f"Config not found at {CONFIG_FILE_PATH!r}")


def fetch_config_from_yaml(cfg_path: Path = None) -> YAML:
    """Parse YAML containing the package configuration."""

    if not cfg_path:
        cfg_path = find_config_file()

    if cfg_path:
        with open(cfg_path, "r") as conf_file:
            parsed_config = load(conf_file.read())
            return parsed_config
    raise OSError(f"Did not find config file at path: {cfg_path}")


def create_and_validate_config(parsed_config: YAML = None) -> Config:
    """Run validation on config values."""
    if parsed_config is None:
        parsed_config = fetch_config_from_yaml()
    # print("parsed_config.data", parsed_config.data)
    # {'package_name': 'classification_model', 'training_data_file': 'train.csv', 'test_data_file': 'test.csv', 'target': 'Survived', 'pipeline_name': 'classification_model', 'pipeline_save_file': 'classification_model_output_v', 'features': ['Pclass', 'Age'], 'test_size': '0.1', 'random_state': '42', 'C': '0.0005'}

    # specify the data attribute from the strictyaml YAML type.
    _config = Config(
        app_conf=AppConfig(**parsed_config.data),
        model_conf=ModelConfig(**parsed_config.data),
    )
    return _config


config = create_and_validate_config()

from typing import Any

import yaml


def load_yaml_data(path: str) -> Any:
    with open(path, 'r') as yaml_file:
        data = yaml.safe_load(yaml_file)
    return data

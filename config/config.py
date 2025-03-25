import os
import yaml

def get_config():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_config_path = os.path.join(base_dir, "config.yaml")
    return yaml.safe_load(open(yaml_config_path, encoding="utf-8"))
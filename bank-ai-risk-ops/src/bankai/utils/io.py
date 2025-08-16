import os, yaml
def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

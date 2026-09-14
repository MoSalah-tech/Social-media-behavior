from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def load_config(path: str | Path = "configs/config.yaml") -> dict:
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with open(path, "r") as f:
        return yaml.safe_load(f)
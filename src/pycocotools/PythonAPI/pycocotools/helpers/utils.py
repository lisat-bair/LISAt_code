try:
    import simdjson as json
except:  # noqa: E722
    import json
from pathlib import Path
from typing import Any, Dict


def load_json(json_path: Path) -> Dict[str, Any]:
    """
    Args:
        json_path: Path to json file

    Returns: json dictionary
    """
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    return data


def save_json(json_path: Path, data: Dict, indent: int = 4, sort_keys: bool = True) -> None:
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=indent,
            # cls=CustomJSONEncoder,
            sort_keys=sort_keys,
        )

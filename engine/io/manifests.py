import json
import logging
from typing import Dict, Any, TypeVar, Optional
from pathlib import Path
import jsonschema

from engine.io import paths

logger = logging.getLogger("VisionForge.IO")

SCHEMAS_DIR = Path(__file__).parent.parent.parent / "shared" / "schemas"

def load_schema(schema_name: str) -> Dict[str, Any]:
    schema_path = SCHEMAS_DIR / f"{schema_name}.json"
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema {schema_name} not found at {schema_path}")
    with open(schema_path, "r") as f:
        return json.load(f)

def validate(data: Dict[str, Any], schema_name: str):
    """Validates data against a named schema located in shared/schemas."""
    schema = load_schema(schema_name)
    try:
        jsonschema.validate(instance=data, schema=schema)
    except jsonschema.ValidationError as e:
        logger.error(f"Validation failed for {schema_name}: {e.message}")
        raise ValueError(f"Invalid {schema_name}: {e.message}")

def save_manifest(path: Path, data: Dict[str, Any], schema_name: Optional[str] = None):
    """Saves dictionary to JSON file with optional validation."""
    if schema_name:
        validate(data, schema_name)
    
    paths.ensure_dirs(path.parent)
    
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def load_manifest(path: Path) -> Dict[str, Any]:
    """Loads JSON file."""
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    with open(path, "r") as f:
        return json.load(f)

def update_manifest(path: Path, updates: Dict[str, Any], schema_name: Optional[str] = None) -> Dict[str, Any]:
    """Updates existing manifest with new values."""
    data = load_manifest(path)
    data.update(updates)
    save_manifest(path, data, schema_name)
    return data

"""Small helpers shared by the RF-DETR export entry points."""

from pathlib import Path
from typing import Optional, Union


def resolve_exported_path(export_result: Optional[Union[str, Path]], format_name: str) -> Path:
    """Return and validate the artifact path reported by ``model.export``."""
    if export_result is None:
        raise RuntimeError(
            f"RF-DETR {format_name} export completed without returning an artifact path"
        )
    exported_path = Path(export_result)
    if not exported_path.is_file():
        raise RuntimeError(
            f"RF-DETR {format_name} export returned {exported_path}, but that file does not exist"
        )
    return exported_path

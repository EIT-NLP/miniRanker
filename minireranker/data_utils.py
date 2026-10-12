import os
from pathlib import Path
from typing import Optional

def infer_video_root(data_root: str, explicit_video_root: Optional[str]) -> Optional[str]:
    """Resolve a frame root without embedding a machine-specific fallback."""
    root = explicit_video_root or os.environ.get("SHAREGPTVIDEO_ROOT") or os.environ.get("VIDEO_ROOT")
    if root:
        return str(Path(root).expanduser())
    for candidate in [Path(data_root).parent / "ShareGPTVideo", Path("data/ShareGPTVideo")]:
        if candidate.is_dir():
            return str(candidate)
    return None

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional
import json
import re

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp"}

def list_images_sorted(folder: Path) -> List[Path]:
    files = [p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS]
    # sort by numeric chunks when possible (0001.png etc.)
    def key_fn(p: Path):
        m = re.findall(r"\d+", p.stem)
        return (int(m[-1]) if m else 10**12, p.name)
    return sorted(files, key=key_fn)

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def load_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))

def save_json(p: Path, data: dict) -> None:
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

@dataclass
class BatchConfig:
    rough_dir: Path
    out_dir: Path
    preset_line: Path
    preset_paint: Path
    mode: str  # "line" | "paint" | "both"
    base_seed: int = 12345
    seed_stride: int = 17
    skip_existing: bool = True

def seed_for_frame(base_seed: int, idx: int, stride: int) -> int:
    # deterministic but frame-varying
    return base_seed + idx * stride


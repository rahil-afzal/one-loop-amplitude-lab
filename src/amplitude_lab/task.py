from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path


@dataclass(frozen=True)
class Task:
    name: str
    derivation_prompt: str

    @classmethod
    def load(cls, name_or_path: str | Path = "phi4_dim4") -> Task:
        folder = Path(name_or_path)
        if not folder.is_dir():
            folder = Path(str(resources.files("amplitude_lab") / "tasks" / name_or_path))
        if not folder.is_dir():
            raise FileNotFoundError(f"no task folder {name_or_path!r}")
        derivation_file = folder / "derivation.md"
        if not derivation_file.is_file():
            raise FileNotFoundError(f"task folder {folder} has no derivation.md")
        return cls(
            name=folder.name,
            derivation_prompt=derivation_file.read_text(encoding="utf-8"),
        )

import hashlib
import json
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path


@dataclass
class Task:
    name: str
    suite: str
    statement: str
    max_attempts: int = 10
    max_turns: int = 40
    run_timeout: float = 60.0
    eval_timeout: float = 60.0
    derivation_prompt: str = ""
    stages: list[dict[str, str]] = field(default_factory=list)

    @classmethod
    def load(cls, name_or_path="phi4_dim4"):
        folder = Path(name_or_path)
        if not folder.is_dir():
            folder = Path(str(resources.files("amplitude_lab") / "tasks" / name_or_path))
        if not folder.is_dir():
            raise FileNotFoundError(f"no task folder {name_or_path!r}")
        meta = json.loads((folder / "task.json").read_text(encoding="utf-8"))
        statement = (folder / "statement.md").read_text(encoding="utf-8")
        statement = statement.replace("{max_attempts}", str(meta["max_attempts"]))
        derivation_file = folder / "derivation.md"
        derivation_prompt = (
            derivation_file.read_text(encoding="utf-8") if derivation_file.is_file() else ""
        )
        return cls(
            name=meta["name"],
            suite=meta["suite"],
            statement=statement,
            derivation_prompt=derivation_prompt,
            max_attempts=meta["max_attempts"],
            max_turns=meta["max_turns"],
            run_timeout=meta["run_timeout"],
            eval_timeout=meta["eval_timeout"],
            stages=meta.get("stages", []),
        )

    @property
    def statement_sha256(self):
        return hashlib.sha256(self.statement.encode("utf-8")).hexdigest()

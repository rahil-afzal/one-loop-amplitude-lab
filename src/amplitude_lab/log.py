import hashlib
import json
import platform
import time
from pathlib import Path


def environment():
    import mpmath
    import sympy

    from . import __version__

    return {
        "amplitude_lab": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "sympy": sympy.__version__,
        "mpmath": mpmath.__version__,
    }


class RunLog:
    """One directory per run. Everything is appended; nothing is rewritten except outcome.json.

    meta.json      what was run: task, statement hash, model settings, library versions
    events.jsonl   one row per turn: what the agent did and how it went
    attempts.jsonl one row per submission with the verifier's full detail (seed included)
    attempt_NN.py  the submitted files
    transcript.md  the conversation
    """

    def __init__(self, base, task, llm):
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.dir = Path(base) / f"{stamp}-{task.name}"
        n = 1
        while self.dir.exists():
            n += 1
            self.dir = Path(base) / f"{stamp}-{task.name}-{n}"
        self.dir.mkdir(parents=True)
        meta = {
            "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "task": task.name,
            "suite": task.suite,
            "statement_sha256": task.statement_sha256,
            "limits": {
                "max_attempts": task.max_attempts,
                "max_turns": task.max_turns,
                "run_timeout_s": task.run_timeout,
                "eval_timeout_s": task.eval_timeout,
            },
            "stages": getattr(task, "stages", []),
            "llm": llm.describe() if hasattr(llm, "describe") else {"name": str(llm)},
            "environment": environment(),
        }
        (self.dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        (self.dir / "statement.md").write_text(task.statement, encoding="utf-8")

    def say(self, who, text):
        with open(self.dir / "transcript.md", "a", encoding="utf-8") as fh:
            fh.write(f"## {who}\n\n{text.rstrip()}\n\n")

    def event(self, **row):
        row["time"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        with open(self.dir / "events.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")

    def save_code(self, number, code):
        path = self.dir / f"attempt_{number:02d}.py"
        path.write_text(code, encoding="utf-8")
        return path

    def record(self, number, code, report):
        row = {
            "attempt": number,
            "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "file": f"attempt_{number:02d}.py",
            "sha256": hashlib.sha256(code.encode("utf-8")).hexdigest(),
            **report.to_dict(),
        }
        with open(self.dir / "attempts.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")

    def finish(self, outcome):
        with open(self.dir / "outcome.json", "w", encoding="utf-8") as fh:
            json.dump(outcome, fh, indent=2)

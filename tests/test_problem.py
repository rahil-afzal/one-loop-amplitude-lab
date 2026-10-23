import json
from pathlib import Path

import pytest

from amplitude_lab import benchmarks
from amplitude_lab.problem import ProblemSpec

ROOT = Path(__file__).resolve().parent.parent


def test_roundtrip():
    p = benchmarks.yhg_dim4()
    assert ProblemSpec.from_dict(p.to_dict()) == p


def test_coupling_sq():
    p = ProblemSpec(name="x", c4=3.0, c42=4.0)
    assert p.coupling_sq == 25.0


@pytest.mark.parametrize(
    "changes",
    [
        {"mu2": 0.0},
        {"mu2": -1.0},
        {"s0": 0.0},
        {"c4": float("inf")},
        {"loop_order": 2},
        {"scheme": "MS"},
        {"process": "phi2 phi2 -> phi2 phi2"},
        {"name": ""},
    ],
)
def test_invalid_specs_cannot_be_built(changes):
    with pytest.raises(ValueError):
        benchmarks.yhg_dim4().with_(**changes)


def test_unknown_coupling_rejected():
    data = benchmarks.yhg_dim4().to_dict()
    data["couplings"]["c6"] = 1.0
    with pytest.raises(ValueError, match="c6"):
        ProblemSpec.from_dict(data)


def test_missing_c4_rejected():
    with pytest.raises(ValueError):
        ProblemSpec.from_dict({"name": "x", "couplings": {"c42": 1.0}})


@pytest.mark.parametrize("name", list(benchmarks.BUILTIN))
def test_json_files_match_builtins(name):
    from_file = ProblemSpec.from_json(ROOT / "benchmarks" / f"{name}.json")
    assert from_file == benchmarks.BUILTIN[name]()


def test_resolve_errors():
    with pytest.raises(FileNotFoundError):
        benchmarks.resolve("no_such_benchmark")


def test_resolve_path(tmp_path):
    path = tmp_path / "p.json"
    path.write_text(json.dumps(benchmarks.pure_phi4().to_dict()), encoding="utf-8")
    assert benchmarks.resolve(str(path)) == benchmarks.pure_phi4()

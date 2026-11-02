import json

import pytest

from amplitude_lab import __version__
from amplitude_lab.cli import main
from amplitude_lab.pipeline import run_pipeline
from amplitude_lab.reporting import to_markdown


def test_run_writes_reports_and_exits_zero(tmp_path, capsys):
    code = main(["run", "--out-dir", str(tmp_path)])
    assert code == 0
    report = json.loads((tmp_path / "run.json").read_text(encoding="utf-8"))
    assert report["overall_status"] == "pass" and report["failed_checks"] == []
    assert (tmp_path / "run.md").read_text(encoding="utf-8").startswith("# amplitude-lab")
    assert "Overall status: PASS" in capsys.readouterr().out


def test_report_structure(problem):
    report = run_pipeline(problem)
    for key in (
        "schema_version",
        "environment",
        "problem",
        "conventions",
        "integral",
        "amplitudes",
        "checks",
        "overall_status",
        "limitations",
        "unverified_claims",
    ):
        assert key in report
    assert report["environment"]["amplitude_lab"] == __version__
    assert len(report["amplitudes"]) == 4
    json.dumps(report)
    assert "Limitations" in to_markdown(report)


def test_assistant_never_changes_overall_status(problem):
    from amplitude_lab.assistant.mock import mock_client

    for kind in ("correct", "wrong_sign", "malicious"):
        report = run_pipeline(problem, mock_client(kind))
        assert report["overall_status"] == "pass"
        assert len(report["assistant"]) == 1


def test_mock_wrong_reported_but_does_not_fail_run(tmp_path):
    code = main(["run", "--assistant", "mock-wrong_sign", "--out-dir", str(tmp_path)])
    assert code == 0
    report = json.loads((tmp_path / "run.json").read_text(encoding="utf-8"))
    assert report["assistant"][0]["status"] == "rejected"


def test_missing_key_exit_code_and_no_secret_leak(tmp_path, capsys):
    code = main(["run", "--assistant", "gemini", "--out-dir", str(tmp_path)])
    captured = capsys.readouterr()
    assert code == 2
    assert "GEMINI_API_KEY" in captured.err
    assert not (tmp_path / "run.json").exists()


def test_smoke_without_key_exits_two(capsys):
    assert main(["smoke"]) == 2
    assert "GEMINI_API_KEY" in capsys.readouterr().err


def test_validate_command(capsys):
    assert main(["validate", "--problem", "pure_phi4", "--only", "amplitude_crossing"]) == 0
    assert "amplitude_crossing" in capsys.readouterr().out


def test_amplitude_command(capsys):
    assert main(["amplitude", "--problem", "pure_phi4", "--s", "5", "--t", "-2"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["tree"]["re"] == 1.0 and data["one_loop"]["im"] > 0


def test_invalid_kinematics_is_a_clean_error(capsys):
    assert main(["amplitude", "--s", "5", "--t", "0"]) == 2
    assert "singular" in capsys.readouterr().err


def test_describe_and_list_checks(capsys):
    assert main(["describe"]) == 0
    assert json.loads(capsys.readouterr().out)["timelike_bubble_imaginary_part"] == "+pi"
    assert main(["list-checks"]) == 0
    assert "physics_optical_theorem" in capsys.readouterr().out


def test_unknown_problem_is_a_clean_error(capsys):
    assert main(["validate", "--problem", "nope"]) == 2
    assert "nope" in capsys.readouterr().err


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert __version__ in capsys.readouterr().out

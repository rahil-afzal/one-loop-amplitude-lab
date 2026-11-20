import json

import pytest

from amplitude_lab import __version__
from amplitude_lab.cli import main
from amplitude_lab.workflows.llm import ScriptedLLM


def test_help_lists_only_retained_commands(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    output = capsys.readouterr().out
    assert "derive" in output
    assert "task" in output
    assert "verify" in output
    assert "{task,derive,verify}" in output
    removed_commands = (
        "run ",
        "validate ",
        "amplitude ",
        "describe ",
        "list-checks ",
        "refine ",
        "serve ",
    )
    for removed in removed_commands:
        assert removed not in output


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_task_prints_packaged_prompt(capsys):
    assert main(["task", "--task", "phi4_dim4"]) == 0
    prompt = capsys.readouterr().out
    assert "analytical amplitude derivation" in prompt.lower()
    assert "A_loop" in prompt


def test_derive_help_is_offline(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["derive", "--task", "phi4_dim4", "--help"])
    assert exc.value.code == 0
    assert "--jsonl-events" in capsys.readouterr().out


def test_verify_valid_and_invalid_saved_responses(tmp_path):
    valid = tmp_path / "valid.md"
    valid.write_text(
        "```final\n"
        "A_tree = c4\n"
        "A_loop = c4**2/(32*pi**2) * "
        "(6 - log(-s/mu2) - log(-t/mu2) - log(-u/mu2))\n"
        "beta_c4 = -3*c4**2/(16*pi**2)\n"
        "```",
        encoding="utf-8",
    )
    assert main(["verify", str(valid)]) == 0

    invalid = tmp_path / "invalid.md"
    invalid.write_text(
        "```final\nA_tree = c4\nA_loop = 0\nbeta_c4 = 0\n```",
        encoding="utf-8",
    )
    assert main(["verify", str(invalid)]) == 1


def test_derive_cli_runs_scripted_offline_workflow(monkeypatch, tmp_path, capsys):
    import amplitude_lab.cli as cli

    plan = json.dumps(
        {
            "interpretation": "Derive the phi4 amplitude.",
            "lagrangian": "$\\mathcal{L}=\\frac12(\\partial\\phi)^2+\\frac{c_4}{4!}\\phi^4$",
            "strategy": "Derive the tree and loop amplitudes.",
            "assumptions": [],
            "steps": [
                {
                    "id": "final",
                    "goal": "Establish the Lagrangian and derive the final amplitude",
                    "depends_on": [],
                }
            ],
        }
    )
    good = (
        "Derivation.\n"
        "```final\n"
        "A_tree = c4\n"
        "A_loop = c4**2/(32*pi**2) * "
        "(6 - log(-s/mu2) - log(-t/mu2) - log(-u/mu2))\n"
        "beta_c4 = -3*c4**2/(16*pi**2)\n"
        "```"
    )
    audit = json.dumps(
        {
            "verdict": "ACCEPT",
            "target_step_id": "final",
            "diagnostic": "The derivation is consistent.",
        }
    )
    derivation = ScriptedLLM([plan, good])
    auditor = ScriptedLLM([audit])
    monkeypatch.setattr(
        cli,
        "role_llm_from_env",
        lambda role, *_args, **_kwargs: derivation if role == "derivation" else auditor,
    )

    code = main(
        [
            "derive",
            "--task",
            "phi4_dim4",
            "--runs-dir",
            str(tmp_path),
            "--jsonl-events",
        ]
    )
    assert code == 0
    assert derivation.calls
    assert auditor.calls
    assert '"type":' in capsys.readouterr().out

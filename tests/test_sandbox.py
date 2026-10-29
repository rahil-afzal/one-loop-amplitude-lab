import os
import sys

import pytest

import amplitude_lab
from amplitude_lab.sandbox import evaluate_file, run_snippet

REFERENCE_PY = os.path.join(os.path.dirname(amplitude_lab.__file__), "reference.py")


def out(code, **kw):
    return run_snippet(code, **kw).output


def test_prints():
    assert out("print(6 * 7)").strip() == "42"


def test_sympy_works():
    assert (
        out("import sympy as sp\nx = sp.symbols('x')\nprint(sp.diff(x**3, x))").strip() == "3*x**2"
    )


def test_scratch_directory_is_writable_and_readable():
    assert out("open('a.txt', 'w').write('hi')\nprint(open('a.txt').read())").strip() == "hi"


def test_import_statements_outside_python_and_sympy_are_refused():
    for mod in ("os", "subprocess", "socket", "numpy", "amplitude_lab", "ctypes"):
        assert "not allowed" in out(f"import {mod}"), mod
        assert "not allowed" in out(f"__import__({mod!r})"), mod


def test_traceback_comes_back():
    assert "ZeroDivisionError" in out("1/0")


def test_timeout():
    r = run_snippet("while True:\n    pass", timeout=1)
    assert r.timed_out and "timed out" in r.output


def test_long_output_is_clipped():
    r = run_snippet("print('x' * 100000)", limit=500)
    assert len(r.output) < 600 and "truncated" in r.output


# Attacks that get around the `import` wrapper by taking a module from sympy. The audit hook
# has to stop the operation itself. A snippet prints LEAK only if it got what it wanted.
VIA_SYMPY = "import sympy\nm = sympy.external.import_module\n"

ATTACKS = {
    "import reference by __import__": (
        "m = __import__('amplitude_lab.reference', fromlist=['x'])\nprint('LEAK', m.__file__)"
    ),
    "import reference via sympy": (
        VIA_SYMPY + "ref = m('amplitude_lab.reference')\nprint('LEAK' if ref else 'none')"
    ),
    "read reference source": f"print('LEAK', open({REFERENCE_PY!r}).read()[:30])",
    "read reference with os.open": (
        VIA_SYMPY + f"os = m('os')\nfd = os.open({REFERENCE_PY!r}, os.O_RDONLY)\nprint('LEAK', fd)"
    ),
    "list the filesystem": VIA_SYMPY + "os = m('os')\nprint('LEAK', os.listdir('/'))",
    "run a command": VIA_SYMPY + "os = m('os')\nos.system('echo LEAK')",
    "subprocess": VIA_SYMPY + "sp = m('subprocess')\nsp.run(['echo', 'LEAK'])",
    "network": VIA_SYMPY
    + "s = m('socket')\ns.create_connection(('127.0.0.1', 9), 1)\nprint('LEAK')",
    "ctypes": VIA_SYMPY + "c = m('ctypes')\nc.CDLL('libc.so.6')\nprint('LEAK')",
    "kill the parent": VIA_SYMPY + "os = m('os')\nos.kill(os.getppid(), 0)\nprint('LEAK')",
    "write outside scratch": "open('/tmp/amplitude_lab_leak_test', 'w').write('x')\nprint('LEAK')",
    "exec an import": "exec('import os')\nprint('LEAK')",
}


@pytest.mark.skipif(
    sys.platform == "win32", reason="paths and libc differ; hooks untested on Windows"
)
@pytest.mark.parametrize("name", sorted(ATTACKS))
def test_attack_is_stopped(name):
    result = out(ATTACKS[name])
    assert "LEAK" not in result, result
    assert not os.path.exists("/tmp/amplitude_lab_leak_test")


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX paths")
def test_walking_the_filesystem_finds_nothing():
    result = out(VIA_SYMPY + "os = m('os')\nprint(list(os.walk('/home')))")
    assert result.strip() == "[]"


@pytest.mark.skipif(sys.platform == "win32", reason="no resource limits on Windows")
def test_memory_bomb_is_stopped():
    assert "MemoryError" in out("x = bytearray(8 * 2**30)")


@pytest.mark.skipif(sys.platform == "win32", reason="no resource limits on Windows")
def test_huge_file_is_stopped():
    result = run_snippet("open('big', 'wb').write(b'0' * 60_000_000)\nprint('WROTE')")
    assert "WROTE" not in result.output


def test_other_interpreter_can_be_chosen(monkeypatch):
    monkeypatch.setenv("AMPLITUDE_LAB_SANDBOX_PYTHON", sys.executable)
    assert out("print('ok')").strip() == "ok"


def test_missing_interpreter_is_a_clear_message(monkeypatch, tmp_path):
    monkeypatch.setenv("AMPLITUDE_LAB_SANDBOX_PYTHON", str(tmp_path / "nope"))
    assert "could not start the sandbox interpreter" in out("print(1)")
    src = tmp_path / "c.py"
    src.write_text("def amplitude(s, t, u, c4, mu2):\n    return c4\n")
    values, error = evaluate_file(str(src), [(1 + 1j, 1, -2 - 1j, 1.0, 1.0)])
    assert values is None and "could not start" in error


def test_evaluate_file_roundtrip(tmp_path):
    src = tmp_path / "c.py"
    src.write_text("def amplitude(s, t, u, c4, mu2):\n    return c4 * s + 1j * mu2\n")
    values, error = evaluate_file(str(src), [(2 + 1j, 0, -2 - 1j, 3.0, 5.0)])
    assert error is None and values == [(3 * (2 + 1j) + 5j)]


def test_answer_key_rule_covers_any_install_location(monkeypatch):
    # a second installed copy (site-packages) is not the directory the sandbox was loaded from,
    # so the rule has to work on the path alone, for the source and for cached bytecode
    from amplitude_lab import sandbox

    monkeypatch.setattr(sys, "dont_write_bytecode", sys.dont_write_bytecode)
    ns = {}
    exec(sandbox._prelude(), ns)
    key = ns["_is_answer_key"]
    assert key("/venv/lib/site-packages/amplitude_lab/reference.py", "/")
    assert key("/venv/lib/site-packages/amplitude_lab/__pycache__/reference.cpython-312.pyc", "/")
    assert not key("/venv/lib/site-packages/sympy/reference.py", "/")
    assert not key("/venv/lib/site-packages/amplitude_lab/physics.py", "/")
    assert not key("/home/me/amplitude_lab/.venv/lib/site-packages/sympy/core/basic.py", "/")

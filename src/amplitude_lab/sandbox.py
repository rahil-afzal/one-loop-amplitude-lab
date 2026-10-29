"""Running untrusted python (agent snippets and candidate files) in a subprocess.

Two layers, neither a perfect boundary but together enough to keep a model away from the
answer key by accident or curiosity:

* an `__import__` wrapper that limits `import` statements written in the snippet itself to
  python + sympy (this is the "tools" restriction, and it is only about tools);
* a PEP 578 audit hook that refuses the operations that would matter: reading anything
  outside the interpreter's own library and the scratch directory (and never the amplitude_lab
  package), writing outside scratch, spawning processes, network access, killing processes,
  most of ctypes. Unlike the import wrapper this does not depend on how the code got hold of
  a module (`__import__`, `sympy.external.import_module`, `os` pulled out of another module).

On POSIX the child also gets memory and file-size limits. Not covered: native code that
tampers with the interpreter, kernel-level escapes, anything on Windows beyond the hooks.
For runs that matter, use a separate interpreter that has no amplitude_lab and no numpy
(AMPLITUDE_LAB_SANDBOX_PYTHON) and, better, a container.
"""

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass

ALLOWED = (
    "math", "cmath", "fractions", "decimal", "itertools", "functools", "collections",
    "operator", "numbers", "json", "re", "random", "typing", "dataclasses", "__future__",
    "sympy", "mpmath",
)  # fmt: skip

MEM_MB = 2048
PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))


def sandbox_python():
    return os.environ.get("AMPLITUDE_LAB_SANDBOX_PYTHON") or sys.executable


_PRELUDE = r"""
import builtins, os, sys
sys.dont_write_bytecode = True


def _limits():
    try:
        import resource
        mem = __MEM__ * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
        resource.setrlimit(resource.RLIMIT_FSIZE, (20_000_000, 20_000_000))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except Exception:
        pass


def _is_answer_key(path, sep):
    # .../amplitude_lab/reference.py and its cached bytecode, wherever it is installed
    parts = path.split(sep)
    if len(parts) > 1 and parts[-1] == "reference.py" and parts[-2] == "amplitude_lab":
        return True
    return (
        len(parts) > 2
        and parts[-2] == "__pycache__"
        and parts[-3] == "amplitude_lab"
        and parts[-1].startswith("reference.")
    )


def _install(protected_files):
    import sysconfig

    norm = lambda p: os.path.normcase(os.path.realpath(p))
    sep = os.sep
    under = lambda p, base: p == base or p.startswith(base + sep)

    scratch = norm(os.getcwd())
    roots = set()
    for key in ("stdlib", "platstdlib", "purelib", "platlib"):
        path = sysconfig.get_paths().get(key)
        if path:
            roots.add(norm(path))
    for path in (sys.prefix, sys.base_prefix, sys.exec_prefix, sys.base_exec_prefix, *sys.path):
        if path:
            roots.add(norm(path))
    package = norm(__PACKAGE__)
    protected = {norm(p) for p in protected_files}
    devices = {"/dev/null", "/dev/urandom", "/dev/zero"}
    cache = {}

    def readable(path):
        if path in cache:
            return cache[path]
        ok = (
            path not in protected
            and not under(path, package)
            and not _is_answer_key(path, sep)
            and (path in devices or under(path, scratch) or any(under(path, r) for r in roots))
        )
        cache[path] = ok
        return ok

    def as_path(value):
        if isinstance(value, bytes):
            value = os.fsdecode(value)
        return norm(value)

    write_flags = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC
    blocked = (
        "os.system", "os.exec", "os.posix_spawn", "os.spawn", "os.fork", "os.forkpty",
        "subprocess.", "os.kill", "os.killpg", "os.remove", "os.rename", "os.rmdir",
        "os.mkdir", "os.chmod", "os.chown", "os.link", "os.symlink", "os.truncate",
        "os.utime", "os.chdir", "os.chroot", "shutil.", "socket.", "winreg.", "pty.",
        "webbrowser.", "urllib.", "mmap.",
    )  # fmt: skip

    def hook(event, args):
        if event == "open":
            path, mode, flags = args[0], args[1], args[2]
            if isinstance(path, int):
                return
            path = as_path(path)
            writing = bool(mode and any(c in str(mode) for c in "wax+")) or bool(
                isinstance(flags, int) and flags & write_flags
            )
            if writing:
                if not under(path, scratch) or path in protected:
                    raise PermissionError("writing outside the scratch directory is not allowed")
            elif not readable(path):
                raise PermissionError("reading this path is not allowed in the sandbox")
        elif event == "import":
            if str(args[0]).split(".")[0] == "amplitude_lab":
                raise ImportError("amplitude_lab is not available in the sandbox")
        elif event in ("os.listdir", "os.scandir"):
            target = args[0]
            if target is None:
                target = "."
            if not isinstance(target, int) and not readable(as_path(target)):
                raise PermissionError("listing this directory is not allowed in the sandbox")
        elif event.startswith("ctypes."):
            # sympy's import pulls in ctypes and calls dlopen(None); nothing else is needed
            if not (event == "ctypes.dlopen" and args[0] is None):
                raise PermissionError("ctypes is not available in the sandbox")
        elif event.startswith(blocked):
            raise PermissionError(event + " is not allowed in the sandbox")

    sys.addaudithook(hook)

    allowed = set(__ALLOWED__)
    user_modules = ("__main__", "candidate")
    real_import = builtins.__import__

    def guarded(name, globals=None, locals=None, fromlist=(), level=0):
        if level == 0:
            caller = sys._getframe(1).f_globals.get("__name__", "")
            top = name.split(".")[0]
            if caller in user_modules and top not in allowed:
                raise ImportError("import of " + repr(top) + " is not allowed (python + sympy)")
        return real_import(name, globals, locals, fromlist, level)

    builtins.__import__ = guarded
"""


def _prelude():
    return (
        _PRELUDE.replace("__MEM__", str(MEM_MB))
        .replace("__PACKAGE__", repr(PACKAGE_DIR))
        .replace("__ALLOWED__", repr(ALLOWED))
    )


_LAUNCH = (
    "import runpy, sys\n"
    + "__PRELUDE__\n"
    + "_limits()\n_install([])\n"
    + "runpy.run_path(sys.argv[1], run_name='__main__')\n"
)

_WORKER = r"""
import sys, json, importlib.util, linecache
__PRELUDE__
_limits()

cand_path, points_file, out_file = sys.argv[1:4]
out = open(out_file, "w")
with open(points_file) as fh:
    points = json.load(fh)
_install([points_file, out_file])


def where(exc):
    tb, found = exc.__traceback__, ""
    while tb is not None:
        if tb.tb_frame.f_code.co_filename == cand_path:
            line = linecache.getline(cand_path, tb.tb_lineno).strip()
            found = " at line %d: %s" % (tb.tb_lineno, line[:120])
        tb = tb.tb_next
    return found


def finish(obj):
    json.dump(obj, out)
    out.close()
    sys.exit(0)


try:
    spec = importlib.util.spec_from_file_location("candidate", cand_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    func = mod.amplitude
except BaseException as exc:
    finish({"error": "could not load amplitude(): %s: %s%s"
            % (type(exc).__name__, str(exc)[:300], where(exc))})

values = []
for p in points:
    s, t, u = complex(p[0], p[1]), complex(p[2], p[3]), complex(p[4], p[5])
    try:
        v = complex(func(s, t, u, p[6], p[7]))
    except BaseException as exc:
        # the exception text can quote the arguments, which are the verifier's test points;
        # report the type and where it happened, never the message (except for import errors)
        text = "amplitude() raised %s" % type(exc).__name__
        if isinstance(exc, ImportError):
            text += ": " + str(exc)[:200]
        finish({"error": text + where(exc)})
    values.append([v.real, v.imag])
finish({"values": values})
"""


@dataclass
class RunResult:
    output: str
    timed_out: bool = False


def _clip(text, limit):
    if len(text) > limit:
        return text[:limit] + "\n[output truncated]"
    return text


def _text(data):
    if data is None:
        return ""
    return data.decode("utf-8", "replace") if isinstance(data, bytes) else data


def run_snippet(code, timeout=60.0, limit=4000):
    """Run a snippet in a scratch directory and return what it printed."""
    with tempfile.TemporaryDirectory() as tmp:
        snippet = os.path.join(tmp, "snippet.py")
        launch = os.path.join(tmp, "launch.py")
        with open(snippet, "w", encoding="utf-8") as fh:
            fh.write(code)
        with open(launch, "w", encoding="utf-8") as fh:
            fh.write(_LAUNCH.replace("__PRELUDE__", _prelude()))
        try:
            done = subprocess.run(
                [sandbox_python(), "-I", launch, snippet],
                cwd=tmp, capture_output=True, timeout=timeout,
            )  # fmt: skip
        except subprocess.TimeoutExpired as exc:
            partial = _text(exc.stdout) + _text(exc.stderr)
            return RunResult(_clip(partial + f"\n[timed out after {timeout:g} s]", limit), True)
        except OSError as exc:
            return RunResult(f"could not start the sandbox interpreter: {exc}")
        return RunResult(_clip(_text(done.stdout) + _text(done.stderr), limit))


def evaluate_file(path, points, timeout=60.0):
    """Call amplitude() from a candidate file in a subprocess.

    points: list of (s, t, u, c4, mu2). Returns (values, error); exactly one is None. The error
    text never contains the argument values.
    """
    try:
        with open(path, encoding="utf-8") as fh:
            source = fh.read()
    except OSError as exc:
        return None, f"could not read the candidate file: {exc.__class__.__name__}"
    rows = []
    for s, t, u, c4, mu2 in points:
        s, t, u = complex(s), complex(t), complex(u)
        rows.append([s.real, s.imag, t.real, t.imag, u.real, u.imag, float(c4), float(mu2)])
    with tempfile.TemporaryDirectory() as tmp:
        names = {k: os.path.join(tmp, k) for k in ("worker.py", "candidate_src.py", "points.json")}
        out_file = os.path.join(tmp, "out.json")
        with open(names["worker.py"], "w", encoding="utf-8") as fh:
            fh.write(_WORKER.replace("__PRELUDE__", _prelude()))
        with open(names["candidate_src.py"], "w", encoding="utf-8") as fh:
            fh.write(source)
        with open(names["points.json"], "w") as fh:
            json.dump(rows, fh)
        try:
            subprocess.run(
                [sandbox_python(), "-I", names["worker.py"], names["candidate_src.py"],
                 names["points.json"], out_file],
                cwd=tmp, capture_output=True, timeout=timeout,
            )  # fmt: skip
        except subprocess.TimeoutExpired:
            return None, f"timed out after {timeout:g} s"
        except OSError as exc:
            return None, f"could not start the sandbox interpreter: {exc}"
        try:
            with open(out_file) as fh:
                data = json.load(fh)
        except (OSError, ValueError):
            return None, "the process died before returning values (memory or size limit?)"
    if "error" in data:
        return None, data["error"]
    return [complex(re, im) for re, im in data["values"]], None

import cmath
import importlib.util

from .sandbox import evaluate_file


class CandidateError(Exception):
    """The candidate could not be evaluated (crash, timeout, non-finite value)."""


class Candidate:
    """A submitted amplitude(s, t, u, c4, mu2).

    isolated=True runs the file in a subprocess with the import guard; use it for anything an
    agent wrote. isolated=False imports the file directly and is meant for our own examples.
    """

    def __init__(self, path=None, func=None, isolated=True, timeout=60.0):
        if (path is None) == (func is None):
            raise ValueError("give either path or func")
        self.path = path
        self.isolated = isolated if path is not None else False
        self.timeout = timeout
        self._func = func

    @classmethod
    def from_function(cls, func):
        return cls(func=func)

    def _load(self):
        if self._func is None:
            spec = importlib.util.spec_from_file_location("candidate", self.path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            self._func = mod.amplitude
        return self._func

    def evaluate(self, points):
        """points: iterable of (s, t, u, c4, mu2). Returns a list of complex numbers."""
        points = list(points)
        if self.isolated:
            values, error = evaluate_file(self.path, points, self.timeout)
            if error:
                raise CandidateError(error)
        else:
            try:
                func = self._load()
                values = [complex(func(complex(s), complex(t), complex(u), c4, mu2))
                          for s, t, u, c4, mu2 in points]  # fmt: skip
            except Exception as exc:
                raise CandidateError(f"amplitude() raised {type(exc).__name__}: {exc}") from None
        if not all(cmath.isfinite(v) for v in values):
            raise CandidateError("amplitude() returned a non-finite value")
        return values

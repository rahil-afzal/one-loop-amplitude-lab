"""Safe parsing and evaluation of LLM-proposed formulas.

LLM text is NEVER executed as Python. It is parsed with ``ast`` and converted node by
node into a SymPy expression through a strict whitelist. Rejected: attribute access,
subscripts, imports, calls to anything but whitelisted math functions, lambdas,
comprehensions, strings, complex literals, symbolic or large exponents, oversized input.
"""

from __future__ import annotations

import ast
import math
import operator
from contextvars import ContextVar

import mpmath as mp
import sympy as sp

MAX_LENGTH = 500
MAX_NODES = 200
MAX_EXPONENT = 8
MAX_CONSTANT = 1e6
MAX_NUMERIC_DECADES = 50  # any purely numeric power must stay below 10**50

S = sp.Symbol("s", complex=True)
MU2 = sp.Symbol("mu2", positive=True)

_NAMES = {"s": S, "mu2": MU2, "pi": sp.pi, "I": sp.I}
_EXTRA_NAMES: ContextVar[dict] = ContextVar("extra_names", default={})
_FUNCTIONS = {"log": sp.log, "ln": sp.log, "sqrt": sp.sqrt, "exp": sp.exp}
_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}


class CandidateParseError(ValueError):
    """The proposal is not a safe, supported arithmetic expression."""


def _convert(node: ast.AST) -> sp.Expr:
    if isinstance(node, ast.Expression):
        return _convert(node.body)

    if isinstance(node, ast.Constant):
        value = node.value
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise CandidateParseError("Only real numeric constants are allowed.")
        if abs(value) > MAX_CONSTANT:
            raise CandidateParseError("Numeric constant is too large.")
        return sp.sympify(value)

    if isinstance(node, ast.Name):
        names = {**_NAMES, **_EXTRA_NAMES.get()}
        if node.id not in names:
            raise CandidateParseError(f"Disallowed name: {node.id!r}.")
        return names[node.id]

    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_convert(node.operand))

    if isinstance(node, ast.BinOp):
        if type(node.op) in _BINOPS:
            return _BINOPS[type(node.op)](_convert(node.left), _convert(node.right))
        if isinstance(node.op, ast.Pow):
            exponent = _convert(node.right)
            if exponent.free_symbols or not exponent.is_number:
                raise CandidateParseError("Exponents must be numeric constants.")
            if not exponent.is_real or abs(float(exponent)) > MAX_EXPONENT:
                raise CandidateParseError(f"Exponent magnitude must be <= {MAX_EXPONENT}.")
            base = _convert(node.left)
            if not base.free_symbols:
                # Nested numeric powers like ((10**6)**8)**8... must not build huge integers.
                magnitude = abs(complex(base))
                if magnitude > 0.0 and (
                    abs(float(exponent)) * abs(math.log10(magnitude)) > MAX_NUMERIC_DECADES
                ):
                    raise CandidateParseError("Numeric power is too large.")
            return base**exponent
        raise CandidateParseError(f"Disallowed operator: {type(node.op).__name__}.")

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCTIONS:
            raise CandidateParseError("Only whitelisted math functions may be called.")
        if node.keywords or len(node.args) != 1:
            raise CandidateParseError("Functions take exactly one positional argument.")
        return _FUNCTIONS[node.func.id](_convert(node.args[0]))

    raise CandidateParseError(f"Disallowed syntax: {type(node).__name__}.")


def extract_expression(text: str) -> str:
    """Strip Markdown fences, a leading 'B(s;mu2) =', and normalize '^' and unicode minus."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = "\n".join(
            line for line in cleaned.splitlines() if not line.strip().startswith("```")
        ).strip()
    if "=" in cleaned and "==" not in cleaned:
        cleaned = cleaned.split("=", 1)[1].strip()
    return cleaned.replace("^", "**").replace("\u2212", "-")


def parse_expression(text: str, extra_names: tuple[str, ...] = ()) -> sp.Expr:
    """Parse a proposal; ``extra_names`` adds plain complex symbols (e.g. ``t``, ``c4``)."""
    token = _EXTRA_NAMES.set({name: sp.Symbol(name, complex=True) for name in extra_names})
    try:
        return _parse_expression(text)
    finally:
        _EXTRA_NAMES.reset(token)


def _parse_expression(text: str) -> sp.Expr:
    expression = extract_expression(text)
    if not expression:
        raise CandidateParseError("Empty proposal.")
    if len(expression) > MAX_LENGTH:
        raise CandidateParseError(f"Proposal exceeds {MAX_LENGTH} characters.")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise CandidateParseError("Proposal is not a valid arithmetic expression.") from exc
    if sum(1 for _ in ast.walk(tree)) > MAX_NODES:
        raise CandidateParseError("Proposal is too complex.")
    expr = _convert(tree)
    if expr.has(sp.zoo, sp.nan, sp.oo, -sp.oo):
        raise CandidateParseError("Proposal contains a non-finite constant (e.g. division by 0).")
    return expr


def evaluate(expr: sp.Expr, s: complex, mu2: float, precision: int = 30) -> complex:
    """Evaluate with mpmath (principal branches). Raises on any numerical failure."""
    try:
        function = sp.lambdify((S, MU2), expr, modules="mpmath")
        with mp.workdps(precision):
            value = function(mp.mpc(complex(s).real, complex(s).imag), mp.mpf(mu2))
            result = complex(value)
    except Exception as exc:
        raise ArithmeticError(f"Evaluation failed: {type(exc).__name__}") from None
    if result != result or abs(result) == float("inf"):
        raise ArithmeticError("Expression evaluated to a non-finite value.")
    return result

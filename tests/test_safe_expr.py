"""Security and correctness of LLM-proposal parsing. Nothing here may execute code."""

import pytest

from amplitude_lab.assistant.safe_expr import CandidateParseError, evaluate, parse_expression


@pytest.mark.parametrize(
    "payload",
    [
        "__import__('os').system('echo unsafe')",
        "s.real",
        "s.__class__",
        "[x for x in range(3)]",
        "(lambda: 1)()",
        "open('/etc/passwd')",
        "eval('1+1')",
        "exec('x=1')",
        "'a string'",
        "1j",
        "True",
        "s[0]",
        "{1: 2}",
        "s if s else mu2",
        "not s",
        "s @ s",
        "s // 2",
        "s % 2",
        "log(s, mu2)",
        "log(x=s)",
        "log()",
        "unknown_fn(s)",
        "x + 1",
        "2 - log(-s/mu2); import os",
        "2 - log(-s/mu2)\nimport os",
        "1/0",
        "s/(1-1)",
        "",
        "   ",
    ],
)
def test_unsafe_or_unsupported_rejected(payload):
    with pytest.raises(CandidateParseError):
        parse_expression(payload)


@pytest.mark.parametrize(
    "payload",
    [
        "9**9**9",
        "s**s",
        "s**(1/0)",
        "2**100",
        "1e12 + s",
        "s**2.5e3",
        "((((((((((1000000**8)**8)**8)**8)**8)**8)**8)**8)**8)**8)**8)**8",
        "((((((((((9**8)**8)**8)**8)**8)**8)**8)**8)**8)**8)**8)**8 * s",
    ],
)
def test_resource_exhaustion_rejected(payload):
    with pytest.raises(CandidateParseError):
        parse_expression(payload)


def test_length_and_complexity_limits():
    with pytest.raises(CandidateParseError):
        parse_expression("s+" * 400 + "s")
    with pytest.raises(CandidateParseError):
        parse_expression("(" * 300 + "s" + ")" * 300)


@pytest.mark.parametrize(
    "text",
    [
        "2 - log(-s/mu2)",
        "2 - ln(-s/mu2)",
        "B(s; mu2) = 2 - log(-s/mu2)",
        "```\n2 - log(-s/mu2)\n```",
        "```python\n2 - log(-s/mu2)\n```",
        "2 \u2212 log(\u2212s/mu2)",
        "2 - 2*log(sqrt(-s/mu2))",
        "2 - log(-s/mu2)**1",
        "2 - log((-s/mu2)**1)",
    ],
)
def test_valid_forms_parse_and_agree(text):
    expr = parse_expression(text)
    value = evaluate(expr, -3 + 1j, 1.0)
    reference = evaluate(parse_expression("2 - log(-s/mu2)"), -3 + 1j, 1.0)
    assert value == pytest.approx(reference, abs=1e-12)


def test_caret_is_power():
    assert evaluate(parse_expression("s^2"), 3.0, 1.0) == pytest.approx(9.0)


def test_nonfinite_evaluation_raises_arithmetic_error():
    with pytest.raises(ArithmeticError):
        evaluate(parse_expression("1/(s - 3)"), 3.0, 1.0)


def test_nested_numeric_power_is_fast():
    import time

    start = time.perf_counter()
    with pytest.raises(CandidateParseError):
        parse_expression("(((((((1000000**8)**8)**8)**8)**8)**8)**8)**8")
    assert time.perf_counter() - start < 1.0

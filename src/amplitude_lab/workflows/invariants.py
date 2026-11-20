"""Prompt-anchored convention lock and oracle-free invariant checks.

No reference amplitudes are used here. The only external input is the convention
stated in the problem text; everything else is a physics identity.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import sympy as sp

c4, s, t, u, mu2 = sp.symbols("c4 s t u mu2")


@dataclass(frozen=True)
class Conventions:
    lagrangian_sign: int          # L_int = sign * c4 * phi^4/4!
    amplitude: str = "iA"         # iA = sum of connected amputated diagrams
    scheme: str = "MSbar"

    @property
    def tree(self) -> sp.Expr:
        # vertex = i*sign*c4  =>  iA_tree = i*sign*c4  =>  A_tree = sign*c4
        return self.lagrangian_sign * c4

    def as_dict(self) -> dict:
        return {
            "lagrangian_interaction": f"{self.lagrangian_sign:+d} * c4 * phi^4/4!",
            "vertex_factor": f"{self.lagrangian_sign:+d} * I * c4",
            "A_tree": str(self.tree),
            "amplitude_definition": self.amplitude,
            "scheme": self.scheme,
            "binding": True,
            "note": (
                "These come from the problem statement and outrank the planner, "
                "the notation contract, and every accepted checkpoint."
            ),
        }


_LITERAL = re.compile(r"coefficient\s+of\s+phi\^4/4!\s+in\s+the\s+Lagrangian", re.I)
_STABLE = re.compile(r"stable\s+(quartic\s+)?potential", re.I)


def conventions_from_prompt(text: str) -> Conventions | None:
    if _LITERAL.search(text or ""):
        return Conventions(+1)
    if _STABLE.search(text or ""):
        return Conventions(-1)
    return None


def lagrangian_sign_from_text(text: str) -> int | None:
    """Best-effort read of the interaction sign from a planner Lagrangian string.

    Returns None when unsure, so callers never act on a guess.
    """
    if not text:
        return None
    flat = re.sub(r"\\[a-zA-Z]+", " ", text)
    flat = flat.replace("{", " ").replace("}", " ").replace("$", " ")
    flat = re.sub(r"\s+", "", flat)
    for match in re.finditer(r"c_?4", flat):
        tail = flat[match.end():match.end() + 12]
        if "4!" not in tail and "^4" not in tail:
            continue
        signs = re.findall(r"[+-]", flat[:match.start()])
        if signs:
            return -1 if signs[-1] == "-" else +1
    return None


def _chk(name: str, ok: bool, message: str) -> dict:
    return {"name": name, "passed": bool(ok), "message": message, "details": {}}


def verify_invariants(exprs: dict, conv: Conventions | None) -> list[dict]:
    out: list[dict] = []
    tree, loop, beta = exprs.get("A_tree"), exprs.get("A_loop"), exprs.get("beta_c4")

    if tree is not None and conv is not None:
        delta = sp.simplify(tree - conv.tree)
        out.append(_chk(
            "tree_convention", delta == 0,
            f"A_tree must equal {conv.tree} for L_int={conv.lagrangian_sign:+d}*c4*phi^4/4! "
            f"(binding prompt convention); got {tree}",
        ))

    if loop is not None:
        symmetric = all(
            sp.simplify(loop - loop.subs([(a, b), (b, a)], simultaneous=True)) == 0
            for a, b in ((s, t), (t, u))
        )
        out.append(_chk(
            "crossing", symmetric,
            "A_loop must be the analytic channel sum, invariant under s<->t and t<->u. "
            "A fixed-region boundary form (log(s/mu2) with +I*pi on one channel) fails this; "
            "use 6 - log(-s/mu2) - log(-t/mu2) - log(-u/mu2).",
        ))
        lam = sp.symbols("lam", positive=True)
        scaled = loop.subs({s: lam * s, t: lam * t, u: lam * u, mu2: lam * mu2})
        out.append(_chk(
            "mass_dimension", sp.simplify(scaled - loop) == 0,
            "A_loop must be invariant under (s,t,u,mu2) -> lam*(s,t,u,mu2) in 4D",
        ))

    if tree is not None and loop is not None and beta is not None:
        total = tree + loop
        residual = sp.series(
            2 * mu2 * sp.diff(total, mu2) + beta * sp.diff(total, c4), c4, 0, 3
        ).removeO()
        out.append(_chk(
            "rg_invariance", sp.simplify(residual) == 0,
            f"mu dA/dmu must vanish through O(c4^2); residual = {sp.simplify(residual)}",
        ))

    if tree is not None and loop is not None:
        # 2 Im A = (1/2!) * Phi2 * |A_tree|^2, Phi2(distinguishable, massless) = 1/(8 pi)
        point = {
            s: sp.Float(5) + sp.I * sp.Float("1e-12"),
            t: sp.Float(-2), u: sp.Float(-3),
            mu2: sp.Float(1), c4: sp.Float("0.7"),
        }
        try:
            got = complex(sp.N(loop.subs(point), 20)).imag
            tree_value = complex(sp.N(tree.subs(point), 20))
            want = float(abs(tree_value) ** 2 / (32 * sp.pi))
            ok = abs(got - want) < 1e-9
            message = (
                f"Im A_loop(s+i0)={got:.6g}; unitarity with |A_tree|^2 and the "
                f"identical-particle 1/2! phase space predicts {want:.6g}"
            )
        except Exception as exc:
            ok, message = False, f"evaluation failed: {type(exc).__name__}: {exc}"
        out.append(_chk("optical_theorem", ok, message))

    return out
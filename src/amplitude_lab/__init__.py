"""amplitude-lab: deterministically validated one-loop scalar scattering amplitudes.

The physics core (``conventions``, ``kinematics``, ``problem``, ``integrals``,
``amplitude``, ``crosssection``, ``dispersion``, ``validation``) never imports an LLM
SDK. The optional ``assistant`` package may propose formulas, but every proposal is
parsed safely and verified by deterministic code before it is reported.
"""

__version__ = "0.1.0"

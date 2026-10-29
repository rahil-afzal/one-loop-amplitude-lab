"""Tree-level inputs used by the verifier. Nothing in here knows the one-loop answer."""

import math

# Massless two-body phase space integrated over the solid angle:
# dPhi2 = |p| dOmega / (16 pi^2 sqrt(s)), |p| = sqrt(s)/2  ->  1/(8 pi).
PHASE_SPACE_2 = 1.0 / (8.0 * math.pi)

# Two identical particles in the intermediate state.
IDENTICAL = 0.5


def cut_im_part(c4):
    """Im A in the physical region from unitarity, with tree amplitude A = c4.

    2 Im A = (1/2) * Phi2 * |c4|^2, i.e. Im A = c4^2 / (32 pi).
    """
    return 0.5 * IDENTICAL * c4**2 * PHASE_SPACE_2


# mu dc4/dmu = BETA_COEFF * c4^2 at one loop (c4 = -lambda in the usual lambda phi^4 / 4! sign).
BETA_COEFF = -3.0 / (16.0 * math.pi**2)

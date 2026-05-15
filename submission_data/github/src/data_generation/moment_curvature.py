"""Monotonic moment-curvature analysis for a single composite section.

Implements the standard OpenSeesPy zero-length-section integrator pattern:
a single ``zeroLengthSection`` element with a fixed node and a free node
that carries axial + rotational DOFs. Axial load is zero (pure bending),
and curvature is incremented via displacement control on the rotational
DOF of the free node.

At each curvature step we record:

- ``curvature``  (the controlled rotation, 1/in)
- ``moment``     (kip-in, from ``sectionForce``)
- ``axial_strain`` (at the reference fibre, from ``sectionDeformation``)
- ``neutral_axis_in`` (computed from the strain field: -eps0 / phi)
- ``slip_in``    (analytical, see :func:`_interface_slip`)

If the Newton solver fails to converge mid-sweep, the analysis is stopped
and the recorded arrays are truncated. The caller decides whether to keep
the partial curve or discard the sample.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import openseespy.opensees as ops

from .composite_section import SectionInfo, build_section
from .lhs_sampler import SectionParams


_SECTION_TAG = 1
_NODE_FIXED = 1
_NODE_FREE = 2
_ELEMENT_TAG = 1


@dataclass
class MomentCurvatureResult:
    """Output of one moment-curvature sweep.

    All arrays have the same length (the number of *converged* steps). If
    ``converged_steps < n_requested`` the analysis terminated early.
    """

    curvature: np.ndarray       # 1/in
    moment: np.ndarray          # kip-in
    axial_strain: np.ndarray    # in/in
    neutral_axis_in: np.ndarray # in (from top of deck)
    slip_in: np.ndarray         # in (analytical)
    converged_steps: int
    n_requested: int
    section_info: SectionInfo

    @property
    def fully_converged(self) -> bool:
        return self.converged_steps == self.n_requested

    @property
    def physically_valid(self) -> bool:
        """True iff the sweep produced enough converged steps with a
        plausibly located neutral axis.

        Two checks:

        1. At least 20 of the requested steps converged. (Sweeps cut short
           by Newton failure produce too-truncated M-phi curves for the
           PINN to learn from.)
        2. The *initial* neutral axis (small-curvature, elastic regime)
           and the *final* neutral axis both lie inside ``[0, d_total]``.
           Sections that converge to the wrong equilibrium branch (a
           hogging response under our sagging-direction loading) end up
           with y_na outside the section.

        We deliberately do **not** require monotonic moment — real
        composite sections show a moment dip when the deck concrete
        crushes, followed by a slow climb as the steel hardens. That's
        physical and must be preserved in the training set.
        """
        if self.converged_steps < 20:
            return False
        d_total = self.section_info.total_depth_in
        y_na_init = self.neutral_axis_in[3]   # skip the first few near-zero-phi points
        y_na_final = self.neutral_axis_in[-1]
        in_range = lambda y: 0.0 <= y <= d_total
        return in_range(y_na_init) and in_range(y_na_final)


def analyze(
    params: SectionParams,
    n_steps: int = 80,
    curvature_max: Optional[float] = None,
    axial_load_kip: float = 0.0,
    newton_tol: float = 1e-8,
    newton_max_iter: int = 30,
) -> MomentCurvatureResult:
    """Run a single moment-curvature sweep on ``params``.

    Parameters
    ----------
    params
        One sampled section.
    n_steps
        Number of curvature increments. Curvatures recorded at each step.
    curvature_max
        Upper bound on curvature (1/in). If ``None``, set to a depth-aware
        default that is conservatively large for the section depth.
    axial_load_kip
        Applied axial load at the free node (positive = tension). Default 0.
    newton_tol, newton_max_iter
        Convergence controls for the Newton solver.
    """
    info = build_section(params, section_tag=_SECTION_TAG)

    if curvature_max is None:
        # Yield curvature scale: phi_y ~ 2 * fy / (E_s * d_total)
        # Use 12 * phi_y as the upper bound (covers fully plastic + some).
        phi_y_scale = 2.0 * params.fy_ksi / (29_000.0 * info.total_depth_in)
        curvature_max = 12.0 * phi_y_scale

    # We integrate in the sagging direction. OpenSees fibre sections use
    # eps(y) = eps0 - phi * y; with the deck at small y and the girder
    # bottom at large y, *negative* curvature puts the deck in compression
    # (sagging) — the physically intended loading for bridge girders.
    d_phi = -curvature_max / n_steps

    # ---- model setup --------------------------------------------------
    ops.node(_NODE_FIXED, 0.0, 0.0)
    ops.node(_NODE_FREE, 0.0, 0.0)
    # Fix everything at node 1; at node 2, restrain vertical translation
    # only — leave axial and rotation free so the section settles into
    # equilibrium with the applied axial load.
    ops.fix(_NODE_FIXED, 1, 1, 1)
    ops.fix(_NODE_FREE, 0, 1, 0)

    ops.element(
        "zeroLengthSection",
        _ELEMENT_TAG,
        _NODE_FIXED, _NODE_FREE,
        _SECTION_TAG,
    )

    # Axial-load pattern (applied first, then held constant during the
    # curvature sweep via loadConst).
    ops.timeSeries("Constant", 1)
    ops.pattern("Plain", 1, 1)
    ops.load(_NODE_FREE, axial_load_kip, 0.0, 0.0)

    ops.system("UmfPack")
    ops.numberer("Plain")
    ops.constraints("Plain")
    ops.test("NormUnbalance", newton_tol, newton_max_iter)
    ops.algorithm("Newton")

    ops.integrator("LoadControl", 1.0)
    ops.analysis("Static")
    if ops.analyze(1) != 0:
        return _empty_result(info, n_steps)
    ops.loadConst("-time", 0.0)

    # Second pattern: provides a NON-ZERO reference moment so that
    # DisplacementControl has a valid load direction to scale. Sign chosen
    # to match the sagging direction we want to integrate toward (d_phi is
    # negative in OpenSees convention for sagging in this section layout).
    ops.timeSeries("Linear", 2)
    ops.pattern("Plain", 2, 2)
    ops.load(_NODE_FREE, 0.0, 0.0, -1.0)
    ops.integrator("DisplacementControl", _NODE_FREE, 3, d_phi)

    curvature_arr = np.zeros(n_steps)
    moment_arr = np.zeros(n_steps)
    axial_strain_arr = np.zeros(n_steps)
    neutral_axis_arr = np.zeros(n_steps)

    converged = 0
    for k in range(n_steps):
        status = ops.analyze(1)
        if status != 0:
            break
        # For a zeroLengthSection element the section deformation equals
        # the relative displacement between the two nodes: eps0 is the
        # axial DOF (1), curvature phi is the rotation DOF (3). Both come
        # out negative under sagging in OpenSees' eps(y) = eps0 - phi*y
        # convention; we store magnitudes so the dataset uses the
        # conventional positive-sagging sign.
        eps0_raw = ops.nodeDisp(_NODE_FREE, 1)
        phi_raw = ops.nodeDisp(_NODE_FREE, 3)
        sec_force = ops.eleResponse(_ELEMENT_TAG, "section", "force")
        if not sec_force or len(sec_force) < 2:
            break
        moment_raw = sec_force[1]
        curvature_arr[k] = abs(phi_raw)
        moment_arr[k] = abs(moment_raw)
        axial_strain_arr[k] = eps0_raw            # signed: < 0 = compression at deck top
        neutral_axis_arr[k] = _neutral_axis_depth(eps0_raw, phi_raw)
        converged += 1

    # Trim to converged length
    curvature_arr = curvature_arr[:converged]
    moment_arr = moment_arr[:converged]
    axial_strain_arr = axial_strain_arr[:converged]
    neutral_axis_arr = neutral_axis_arr[:converged]

    slip_arr = _interface_slip(
        moment_arr, params, info.plastic_moment_kip_in
    )

    return MomentCurvatureResult(
        curvature=curvature_arr,
        moment=moment_arr,
        axial_strain=axial_strain_arr,
        neutral_axis_in=neutral_axis_arr,
        slip_in=slip_arr,
        converged_steps=converged,
        n_requested=n_steps,
        section_info=info,
    )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _neutral_axis_depth(eps0: float, phi: float) -> float:
    """Neutral axis depth from the section reference fibre (y=0, deck top).

    OpenSees fibre sections use ``eps(y) = eps0 - phi * y``. Setting to
    zero gives ``y_na = eps0 / phi``. Under sagging both ``eps0`` and
    ``phi`` come out negative, so the ratio is positive — a depth below
    the deck top. We guard against ``phi`` very near zero (start of
    analysis) by returning NaN, which downstream code masks before
    aggregation.
    """
    if abs(phi) < 1e-12:
        return float("nan")
    return eps0 / phi


def _interface_slip(
    moment: np.ndarray, params: SectionParams, mp_est: float
) -> np.ndarray:
    """Analytical interface slip estimate.

    Closed-form slip at the supports of a simply supported composite beam
    under partial composite action is non-trivial, so we use a physically
    motivated *scaling* form rather than a derivation:

    .. math::

        \\delta(M) = \\delta_0 \\cdot (1 - \\eta_c)
                    \\cdot (M / M_p)
                    \\cdot \\sqrt{L / L_{ref}}
                    \\cdot K_s^{-1/2}

    where :math:`\\delta_0 = 0.10` in is calibrated to give slip values in
    the 0.02-0.15 in range observed in Oehlers/Bradford push-out and
    beam tests. This is acknowledged as a placeholder; refining the slip
    surrogate against a beam-level OpenSees model with discrete shear
    connectors is on the v2 backlog.
    """
    eta_c = params.composite_action
    if eta_c >= 0.999 or mp_est <= 0.0:
        return np.zeros_like(moment)

    delta_0 = 0.10  # in
    L_ref = 240.0   # in (20 ft)
    span_factor = np.sqrt(max(params.span_in, 1.0) / L_ref)
    ks_factor = 1.0 / np.sqrt(max(params.shear_stud_stiffness_ratio, 0.05))
    moment_ratio = np.clip(np.abs(moment) / mp_est, 0.0, 1.2)

    return delta_0 * (1.0 - eta_c) * moment_ratio * span_factor * ks_factor


def _empty_result(info: SectionInfo, n_requested: int) -> MomentCurvatureResult:
    z = np.zeros(0)
    return MomentCurvatureResult(
        curvature=z, moment=z, axial_strain=z,
        neutral_axis_in=z, slip_in=z,
        converged_steps=0, n_requested=n_requested,
        section_info=info,
    )

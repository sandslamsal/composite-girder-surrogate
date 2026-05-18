"""Physics-informed loss terms for the composite-girder surrogate.

The surrogate predicts (y_na, curvature) in *normalised* space.
The losses operate partly in physical units (compatibility / equilibrium
on strains and integrated moments) and partly in normalised space
(data-fit). The normaliser is plugged in via :class:`PhysicsLossContext`
so we can convert back and forth without re-introducing the per-target
scale/offset bookkeeping at every call site.

The losses fall into two classes:

1. **Compatibility** -- Euler--Bernoulli strain at K sampled fibre
   depths must equal ``phi (y - y_na)``. Penalises both out-of-section
   neutral axes and inconsistent strain signs.

2. **Equilibrium** -- the moment implied by a simplified bilinear-elastic
   integration of fibre stresses across the section must match the input
   moment level ``r * Mp_est``. This is the loss term that genuinely
   enforces section equilibrium given the model's (phi, y_na) predictions.
   It depends only on the two predicted quantities (curvature and
   neutral-axis depth); no moment output is required.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch import Tensor


# Material constants (consistent with composite_section.py)
_E_STEEL_KSI = 29_000.0
_EPS_STEEL_YIELD_REF_KSI = _E_STEEL_KSI    # used as f_y / E_s scale
_EPS_C0 = -0.002                            # concrete strain at peak compression
# Concrete tension capacity: Concrete01 (used in dataset gen) is zero
# tension, but for the equilibrium loss we use a small linear tension
# branch up to ft and then zero. Negligible numerically but stabilises
# gradient near zero strain.
_FT_OVER_FC = 0.07
# Bilinear strain-hardening ratio for steel (matches Steel02 b in the
# dataset generator).
_STEEL_B = 0.01


@dataclass
class PhysicsLossContext:
    """Per-batch physical context built from the feature dataframe + normaliser.

    All tensors have shape ``(B,)`` and live on the same device as
    predictions. Section geometry is required so the fibre-integration
    equilibrium loss can compute proper integrals; callers that only
    need the compatibility loss can leave the geometry fields unset
    (None), in which case the equilibrium loss returns zero.
    """

    total_depth_in: Tensor          # d_total
    mp_estimate_kip_in: Tensor      # estimated plastic moment for normalisation
    moment_ratio: Tensor            # applied load level M / M_p (input feature)
    target_scale: Tensor            # shape (2,) -- y_phys = y_norm * scale + offset
    target_offset: Tensor           # shape (2,)
    # Optional fields used by the fibre-integration equilibrium loss.
    deck_thickness_in: Tensor | None = None
    deck_width_in: Tensor | None = None        # effective deck width AFTER eta_c scaling
    steel_depth_in: Tensor | None = None
    flange_width_in: Tensor | None = None
    flange_thickness_in: Tensor | None = None
    web_thickness_in: Tensor | None = None
    fc_deck_ksi: Tensor | None = None
    fy_ksi: Tensor | None = None
    composite_action: Tensor | None = None


def _denorm(pred: Tensor, ctx: PhysicsLossContext) -> Tensor:
    """Map normalised predictions (y_na, curvature) to physical units.

    Must stay consistent with
    :meth:`FeatureNormalizer.inverse_transform_targets`.
    """
    return pred * ctx.target_scale + ctx.target_offset


# ---------------------------------------------------------------------------
# Compatibility loss
# ---------------------------------------------------------------------------

def compatibility_loss(
    pred: Tensor, ctx: PhysicsLossContext, n_fibres: int = 10,
) -> Tensor:
    """Euler--Bernoulli compatibility residual evaluated at sampled fibres.

    At each fibre depth ``y_k`` (uniformly spaced in ``[0, d_total]``)
    the implied strain is ``eps_k = phi (y_k - y_na)``. The constraint
    is automatic from the formula, so this term turns into a structural
    well-posedness check:

    * ``y_na`` must lie inside the section: ``relu(-y_na) + relu(y_na - d)``.
    * Strain magnitude at every sampled fibre must stay below a physical
      bound (0.05, two orders of magnitude beyond yield), to prevent the
      optimiser from finding pathological ``phi`` values during the
      early-training transient.
    * The signs of the extreme-fibre strains must be opposite (top in
      compression, bottom in tension), otherwise the section is in pure
      axial loading rather than bending.
    """
    y_na, phi = _denorm(pred, ctx).unbind(dim=-1)
    d = ctx.total_depth_in

    # (1) NA inside section
    out_top = torch.relu(-y_na)
    out_bot = torch.relu(y_na - d)

    # (2) Sampled-fibre strain magnitudes
    # Build a (B, n_fibres) tensor of y values per section.
    alpha = torch.linspace(0.0, 1.0, n_fibres, device=y_na.device)        # (n_fibres,)
    y_fibres = d.unsqueeze(-1) * alpha                                    # (B, n_fibres)
    eps = phi.unsqueeze(-1) * (y_fibres - y_na.unsqueeze(-1))             # (B, n_fibres)
    cap = 0.05
    mag = torch.relu(eps.abs() - cap).pow(2).mean(dim=-1)                 # (B,)

    # (3) Sign consistency at the deck top (y=0) and section bottom (y=d)
    eps_top = phi * (0.0 - y_na)
    eps_bot = phi * (d - y_na)
    sign = torch.relu(eps_top).pow(2) + torch.relu(-eps_bot).pow(2)

    residual = out_top.pow(2) + out_bot.pow(2) + sign + mag
    return residual.mean()


# ---------------------------------------------------------------------------
# Equilibrium loss -- fibre-stress integration
# ---------------------------------------------------------------------------

def _bilinear_steel_stress(eps: Tensor, fy: Tensor) -> Tensor:
    """Simplified elastic-strain-hardening steel: linear up to yield,
    then ``b * E_s`` slope. Matches Steel02 with ``b=0.01`` used in the
    dataset generator."""
    eps_y = fy / _E_STEEL_KSI
    abs_eps = eps.abs()
    elastic = _E_STEEL_KSI * eps
    hardening = torch.sign(eps) * (fy + _STEEL_B * _E_STEEL_KSI * (abs_eps - eps_y))
    return torch.where(abs_eps <= eps_y, elastic, hardening)


def _concrete_stress(eps: Tensor, fc: Tensor) -> Tensor:
    """Simplified concrete: parabolic compression up to ``-fc`` at
    ``eps = -0.002``, then constant; small linear tension up to
    ``ft = 0.07 fc`` then zero. Compression-negative convention."""
    eps_c0 = torch.full_like(fc, _EPS_C0)
    ft = _FT_OVER_FC * fc
    eps_t = ft / (_E_STEEL_KSI / 8.7)   # rough E_c ~ E_s / 8.7
    # Compression branch (eps <= 0)
    ratio = (eps / eps_c0).clamp(min=0.0)           # 0 at eps=0, 1 at eps=eps_c0
    sigma_c = -fc * (2.0 * ratio - ratio.pow(2))    # parabolic; -fc at eps_c0
    sigma_c = torch.where(
        eps < eps_c0,
        -fc.clone(),                                # flat post-peak (no softening for stability)
        sigma_c,
    )
    # Tension branch (eps > 0): linear up to ft, then 0
    sigma_t = torch.where(eps <= eps_t, (ft / eps_t.clamp(min=1e-6)) * eps,
                          torch.zeros_like(eps))
    return torch.where(eps <= 0.0, sigma_c, sigma_t)


def equilibrium_loss_fibre(
    pred: Tensor, ctx: PhysicsLossContext, n_fibres: int = 10,
) -> Tensor:
    """Compute fibre stresses from the model's (phi, y_na) prediction and
    integrate ``int sigma(y) (y - y_na) dA`` over the section. The
    result is the moment implied by the predicted strain field; it
    should match the input ``r * Mp_est`` (in the elastic-ish regime
    where the simplified bilinear materials are accurate).

    The loss depends only on the model's two predicted quantities
    (curvature and neutral-axis depth) -- no moment output is used.

    Requires the optional section-geometry fields in
    :class:`PhysicsLossContext`. If any are missing, returns zero.
    """
    geom_required = [
        ctx.deck_thickness_in, ctx.deck_width_in, ctx.steel_depth_in,
        ctx.flange_width_in, ctx.flange_thickness_in, ctx.web_thickness_in,
        ctx.fc_deck_ksi, ctx.fy_ksi,
    ]
    if any(g is None for g in geom_required):
        return torch.zeros((), device=pred.device)

    y_na, phi = _denorm(pred, ctx).unbind(dim=-1)

    t_s = ctx.deck_thickness_in
    b_eff = ctx.deck_width_in
    d_s = ctx.steel_depth_in
    b_f = ctx.flange_width_in
    t_f = ctx.flange_thickness_in
    t_w = ctx.web_thickness_in
    fc = ctx.fc_deck_ksi
    fy = ctx.fy_ksi
    d_total = ctx.total_depth_in

    # Build (B, n_fibres) of y values uniformly across the section depth.
    alpha = torch.linspace(0.0, 1.0, n_fibres, device=y_na.device)
    dy = d_total.unsqueeze(-1) * (alpha[1] - alpha[0])              # (B, 1)
    y = d_total.unsqueeze(-1) * alpha                                # (B, n_fibres)

    # Width at each fibre depth:
    #   y < t_s             -> deck:           width = b_eff
    #   t_s <= y < t_s+t_f  -> top flange:     width = b_f
    #   inner web           -> width = t_w
    #   bottom flange       -> width = b_f
    is_deck = y < t_s.unsqueeze(-1)
    is_top_flange = (y >= t_s.unsqueeze(-1)) & (y < (t_s + t_f).unsqueeze(-1))
    is_bottom_flange = y >= (t_s + d_s - t_f).unsqueeze(-1)
    is_web = (~is_deck) & (~is_top_flange) & (~is_bottom_flange)

    width = torch.zeros_like(y)
    width = torch.where(is_deck, b_eff.unsqueeze(-1).expand_as(y), width)
    width = torch.where(is_top_flange, b_f.unsqueeze(-1).expand_as(y), width)
    width = torch.where(is_web, t_w.unsqueeze(-1).expand_as(y), width)
    width = torch.where(is_bottom_flange, b_f.unsqueeze(-1).expand_as(y), width)

    # Strain field: eps(y) = phi * (y - y_na)   (positive => tension)
    eps = phi.unsqueeze(-1) * (y - y_na.unsqueeze(-1))

    # Fibre stresses (broadcast scalar material properties).
    sigma_concrete = _concrete_stress(eps, fc.unsqueeze(-1).expand_as(eps))
    sigma_steel = _bilinear_steel_stress(eps, fy.unsqueeze(-1).expand_as(eps))
    sigma = torch.where(is_deck, sigma_concrete, sigma_steel)

    # Moment integral: M = int sigma(y) * (y - y_na) * width(y) dy
    # We use compression-negative sign so the integrand corresponds to
    # *sagging-positive* M when the section is bent sagging.
    moment_integrand = -sigma * (y - y_na.unsqueeze(-1)) * width
    M_implied = (moment_integrand * dy).sum(dim=-1)                # (B,)

    target_m = ctx.moment_ratio * ctx.mp_estimate_kip_in

    # Smooth mask: trust the bilinear-elastic integration for small r
    # (purely elastic). For large r, the simplified material model
    # diverges from OpenSees' richer concrete behaviour, so we soften.
    r = ctx.moment_ratio
    weight = torch.clamp((0.7 - r) / 0.2, min=0.0, max=1.0)

    rel = (M_implied - target_m) / (ctx.mp_estimate_kip_in + 1e-6)
    rel = torch.clamp(rel, min=-20.0, max=20.0)
    return ((weight * rel) ** 2).mean()


def equilibrium_loss(
    pred: Tensor, ctx: PhysicsLossContext, mode: str = "fibre",
) -> Tensor:
    """Equilibrium term dispatch. ``"fibre"`` is the fibre-stress
    integration; ``"none"`` returns zero."""
    if mode == "fibre":
        return equilibrium_loss_fibre(pred, ctx)
    if mode == "none":
        return torch.zeros((), device=pred.device)
    raise ValueError(f"unknown equilibrium-loss mode: {mode!r}")


# ---------------------------------------------------------------------------
# Data loss + total
# ---------------------------------------------------------------------------

def data_loss(pred: Tensor, target: Tensor, weights: Tensor | None = None) -> Tensor:
    """MSE in normalised target space. Optional per-target weight vector."""
    diff_sq = (pred - target) ** 2
    if weights is not None:
        diff_sq = diff_sq * weights
    return diff_sq.mean()


def total_loss(
    pred: Tensor,
    target: Tensor,
    ctx: PhysicsLossContext,
    lambda_compat: float,
    lambda_equil: float = 0.0,
    data_weights: Tensor | None = None,
    equil_mode: str = "fibre",
    lambda_capacity: float = 0.0,
) -> Dict[str, Tensor]:
    """Combined data + compatibility + equilibrium loss for the
    two-output (y_na, curvature) surrogate.

    ``equil_mode`` selects the equilibrium term: ``"fibre"`` is the
    fibre-stress integration; ``"none"`` disables it. ``lambda_capacity``
    is retained for call-signature compatibility but is inert: the
    capacity term was moment-based and does not apply to the two-output
    model.
    """
    l_data = data_loss(pred, target, data_weights)
    l_compat = compatibility_loss(pred, ctx)
    if equil_mode == "none":
        l_equil = torch.zeros((), device=pred.device)
    else:
        l_equil = equilibrium_loss(pred, ctx, mode=equil_mode)
    zero = torch.zeros((), device=pred.device)
    l_total = l_data + lambda_compat * l_compat + lambda_equil * l_equil
    return {
        "total": l_total,
        "data": l_data,
        "compat": l_compat,
        "equil": l_equil,
        "capacity": zero,
    }

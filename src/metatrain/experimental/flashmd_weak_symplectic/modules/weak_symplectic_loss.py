"""
The weak symplectic loss: the stochastic estimate of |J^T Omega J - Omega|_F^2 / dim, with
J = d(q_next, p_next) / d(q, p) the Jacobian of the phase-space map of the model and
dim = 6 * n_atoms.

For each random probe v the loss needs a JVP, J v, and a VJP, J^T (Omega J v):
l(v) = |J^T Omega J v - Omega v|^2 / dim, whose expectation over probes with identity
covariance is the exact loss. Both products go through the graph of the training forward
(reverse mode only):

- J v = d/dw [J^T w] (double backward): the graph of g(w) = J^T w is built once per
  forward, and differentiated with respect to w for every probe;
- J^T (Omega J v): one more reverse pass through the forward graph.

The probes are processed ``chunk_size`` at a time (batched with ``is_grads_batched``), and
the chunks are yielded lazily so that the trainer can backpropagate each before the next
is built.

The weight of the loss in the training is fixed, or follows a cosine schedule
(increasing or decreasing), see :class:`SymplecticWeightScheduleHypers` and
:func:`symplectic_weight_factor`.

Normalization ("data_scales", the default): in physical units the two residual
components (``residual_q``, ``residual_p``) have different units, so their balance
depends on the unit system. The loss is therefore computed in the coordinates of the
data loss, x~ = D^-1 x with D = diag(Sigma_q, Sigma_p) the per-atom (per atomic type)
scales that the data loss divides the position and momentum targets by. In these
coordinates the symplectic form is Omega~ = D Omega D = [[0, Lambda], [-Lambda, 0]],
Lambda = Sigma_q Sigma_p, the condition J~^T Omega~ J~ = Omega~ is exactly equivalent to
J^T Omega J = Omega, and the residual is D M D (M = J^T Omega J - Omega), whose blocks all
have units of action. Divided by one reference action c (the mean of sigma_q * sigma_p
over the atomic types), the loss is

    |D M D|_F^2 / (c^2 dim) = E_v~ |D M D v~|^2 / (c^2 dim),

estimated without forming Omega~: the probes become v = D v~, the raw residual M v is
computed as before, and it is multiplied by D / c (see
``weak_symplectic_normalization_proof.md``, Theorem 3.1 and Proposition 5.1). With
``"none"`` the loss is the raw |M|_F^2 / dim.

Everything that only *inspects* this loss (dense Jacobian, neighbor statistics,
variance and gradient diagnostics of the estimator, normalization check, cost
benchmark, plots) lives in :mod:`.symplectic_diagnostics`.
"""

import math
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Tuple,
)

import torch
from metatensor.torch import TensorMap
from metatomic.torch import ModelOutput, System
from torch import Tensor
from typing_extensions import TypedDict


if TYPE_CHECKING:
    from metatrain.utils.wrapper import MetatrainModel

ProbeDistribution = Literal["rademacher", "gaussian"]
Normalization = Literal["data_scales", "none"]
NORMALIZATIONS = ("data_scales", "none")

# (Vq, Vp) with a leading probe dimension -> (J v)_q, (J v)_p with the same leading dim
JvpFn = Callable[[Tensor, Tensor], Tuple[Tensor, Tensor]]
# (start, n_chunk) -> (Vq, Vp): the probes of the chunk of probes [start, start + n_chunk)
ProbeFn = Callable[[int, int], Tuple[Tensor, Tensor]]


class SymplecticWeightScheduleHypers(TypedDict):
    """Schedule of the weight of the weak symplectic loss over the training.

    The weight used at optimizer step ``t`` is ``weak_symplectic_weight * f(t)``. With
    ``x = t / (total_steps - 1)`` the progress of the training (0 at the first step, 1
    at the last one), ``f`` follows half a period of a cosine between
    ``start_fraction`` and ``end_fraction`` and is constant outside of it:

    - ``increase``: ``f`` rises from ``min_weight_ratio`` (before ``start_fraction``)
      to 1 (after ``end_fraction``), ``f = r + (1 - r) * (1 - cos(pi * s)) / 2``;
    - ``decrease``: ``f`` falls from 1 to ``min_weight_ratio``,
      ``f = r + (1 - r) * (1 + cos(pi * s)) / 2``;

    with ``r = min_weight_ratio`` and ``s = (x - start) / (end - start)`` clipped to
    [0, 1]. The cosine is flat at both ends, so the weight changes slowly when the
    ramp starts and when it ends.

    When the scheduled weight is exactly 0 (``min_weight_ratio: 0``, before an
    ``increase`` ramp or after a ``decrease`` ramp), the symplectic loss is not computed
    at all for that step (no jvp/vjp, and the training forward uses the fast attention).
    It then counts as 0 in the logged ``symplectic_loss``.

    The step is counted from the start of the run (including the epochs of a restarted
    run) over the same total number of steps as the learning-rate schedule, so a
    restart resumes the schedule where it stopped.
    """

    enabled: bool = False
    """Whether to schedule the weight. If ``False``, the weight is
    ``weak_symplectic_weight`` during the whole training."""
    mode: Literal["increase", "decrease"] = "increase"
    """``increase``: from ``min_weight_ratio * weak_symplectic_weight`` up to
    ``weak_symplectic_weight``. ``decrease``: from ``weak_symplectic_weight`` down to
    ``min_weight_ratio * weak_symplectic_weight``."""
    start_fraction: float = 0.0
    """Fraction of the training steps at which the cosine ramp starts."""
    end_fraction: float = 1.0
    """Fraction of the training steps at which the cosine ramp ends (equal to
    ``start_fraction``: a step from one weight to the other)."""
    min_weight_ratio: float = 0.0
    """The low end of the schedule, as a fraction of ``weak_symplectic_weight``, in
    [0, 1]."""


WEIGHT_SCHEDULE_MODES = ("increase", "decrease")


def resolve_weight_schedule_hypers(
    hypers: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    The ``weak_symplectic_schedule`` hypers with the defaults of
    :class:`SymplecticWeightScheduleHypers` filled in (a config or checkpoint written
    before the schedule may not have them), checked.
    """
    from metatrain.utils.hypers import init_with_defaults

    defaults = init_with_defaults(SymplecticWeightScheduleHypers)
    hypers = dict(hypers or {})
    unknown = set(hypers) - set(defaults)
    if unknown:
        raise ValueError(
            f"Unknown keys in weak_symplectic_schedule: {sorted(unknown)} "
            f"(expected some of {sorted(defaults)})"
        )
    schedule = {**defaults, **hypers}

    if schedule["mode"] not in WEIGHT_SCHEDULE_MODES:
        raise ValueError(
            f"weak_symplectic_schedule.mode must be one of {WEIGHT_SCHEDULE_MODES}, "
            f"got {schedule['mode']!r}"
        )
    start = float(schedule["start_fraction"])
    end = float(schedule["end_fraction"])
    if not 0.0 <= start <= end <= 1.0:
        raise ValueError(
            "weak_symplectic_schedule needs 0 <= start_fraction <= end_fraction <= 1, "
            f"got start_fraction={start}, end_fraction={end}"
        )
    ratio = float(schedule["min_weight_ratio"])
    if not 0.0 <= ratio <= 1.0:
        raise ValueError(
            "weak_symplectic_schedule.min_weight_ratio must be in [0, 1], "
            f"got {ratio}"
        )
    schedule["enabled"] = bool(schedule["enabled"])
    schedule["start_fraction"] = start
    schedule["end_fraction"] = end
    schedule["min_weight_ratio"] = ratio
    return schedule


def symplectic_weight_factor(
    step: int, total_steps: int, schedule: Dict[str, Any]
) -> float:
    """
    The factor ``f`` in [0, 1] that multiplies ``weak_symplectic_weight`` at optimizer
    step ``step`` (0-based) out of ``total_steps`` (see
    :class:`SymplecticWeightScheduleHypers`; ``schedule`` comes from
    :func:`resolve_weight_schedule_hypers`). It is exactly 0 or 1 outside the ramp.
    """
    if not schedule["enabled"]:
        return 1.0

    progress = step / max(1, total_steps - 1)
    start = schedule["start_fraction"]
    end = schedule["end_fraction"]
    if end > start:
        s = min(max((progress - start) / (end - start), 0.0), 1.0)
    else:
        # zero-length ramp: a step at ``start``
        s = 1.0 if progress >= start else 0.0

    # 0 -> 1 along half a cosine period (exactly 0 at s = 0 and 1 at s = 1)
    ramp = 0.5 * (1.0 - math.cos(math.pi * s))
    ratio = schedule["min_weight_ratio"]
    if schedule["mode"] == "increase":
        return ratio + (1.0 - ratio) * ramp
    return 1.0 - (1.0 - ratio) * ramp


def sample_probes(like: Tensor, n_probes: int, distribution: ProbeDistribution) -> Tensor:
    """``n_probes`` random probes shaped like ``like`` (first dimension: probes)."""
    shape = (n_probes, *like.shape)
    if distribution == "rademacher":
        probes = torch.randint(0, 2, shape, device=like.device).to(dtype=like.dtype)
        return 2.0 * probes - 1.0

    if distribution == "gaussian":
        return torch.randn(shape, dtype=like.dtype, device=like.device)

    raise ValueError(f"Unknown probe distribution: {distribution}")


def num_loss_chunks(n_probes: int, chunk_size: int) -> int:
    """Number of chunks yielded by :func:`calc_symplectic_loss_chunks`."""
    return math.ceil(n_probes / chunk_size)


def phase_graph_from_predictions(
    model: "MetatrainModel",
    systems: list[System],
    raw_predictions: Dict[str, TensorMap],
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    ``(q, p, q_next, p_next)``, all ``[n_atoms, 3]`` and in physical units, of the last
    training forward of ``model`` (the ``MetatrainModel`` wrapper, not wrapped in DDP),
    with the graph attached: ``q`` and ``p`` are the leaves stored by the forward of the
    core model (see ``FlashMDWeakSymplectic.weak_symplectic_leaves``).

    In training mode the wrapper returns the raw (unscaled) outputs of the core model:
    the full scales of the scaler (per target and per property) are applied here, and
    the additive part is the current state, ``q_next = q + delta_q`` and ``p_next = p +
    delta_p`` with ``predict_momenta_as_difference`` (``p_next = p_prediction``
    otherwise).

    :param model: The wrapper, whose core was called in training mode with the weak
        symplectic graph enabled.
    :param systems: The systems of that forward.
    :param raw_predictions: The ``position`` and ``momentum`` predictions of that
        forward.
    :return: The leaves and the predicted next state.
    """
    q, p = model.core.weak_symplectic_leaves()
    if model.scaler is None:
        physical = {name: raw_predictions[name] for name in ("position", "momentum")}
    else:
        physical = model.scaler.apply_scales(
            systems,
            {name: raw_predictions[name] for name in ("position", "momentum")},
            remove=False,
            use_per_target_scales=True,
            use_per_property_scales=True,
        )
    delta_q = physical["position"].block().values.reshape(-1, 3).to(q.dtype)
    p_prediction = physical["momentum"].block().values.reshape(-1, 3).to(p.dtype)

    q_next = q + delta_q
    if model.core.hypers["predict_momenta_as_difference"]:
        p_next = p + p_prediction
    else:
        p_next = p_prediction

    return q, p, q_next, p_next


@dataclass
class PhaseSpaceScales:
    """
    The diagonal D = diag(Sigma_q, Sigma_p) of the normalization, per degree of freedom
    of a batch, and the reference action c (see the module docstring).

    :param q: ``[n_atoms, 3]`` scales of the positions (sigma_q of each atom's type).
    :param p: ``[n_atoms, 3]`` scales of the momenta.
    :param action: the reference action c, mean of sigma_q * sigma_p over the atomic
        types of the model (a float, the same for every batch).
    """

    q: Tensor
    p: Tensor
    action: float

    def probes(self, Vq_tilde: Tensor, Vp_tilde: Tensor) -> Tuple[Tensor, Tensor]:
        """The physical probes v = D v~ of normalized probes (first dim: probes)."""
        return Vq_tilde * self.q, Vp_tilde * self.p

    def residual(
        self, residual_q: Tensor, residual_p: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """The normalized residual D (M v) / c of a physical residual M v."""
        return (
            residual_q * (self.q / self.action),
            residual_p * (self.p / self.action),
        )


_PHASE_TARGETS = ("position", "momentum")


def _per_atom_target_scales(
    model: "MetatrainModel", systems: List[System], like: Tensor
) -> Tuple[Tensor, Tensor]:
    """``[n_atoms, 3]`` per-target scales of the position and momentum targets."""
    n_atoms = sum(len(system) for system in systems)
    if model.scaler is None:
        ones = torch.ones((n_atoms, 3), dtype=like.dtype, device=like.device)
        return ones, ones
    outputs = {name: ModelOutput(sample_kind="atom") for name in _PHASE_TARGETS}
    with torch.no_grad():
        scales = model.scaler.get_scales(
            systems,
            outputs,
            # the scales the data loss divides the targets by (the per-property scales
            # are applied to the predictions, inside the data loss' coordinates)
            use_per_target_scales=True,
            use_per_property_scales=False,
        )
    return tuple(  # type: ignore[return-value]
        scales[name]
        .block()
        .values.reshape(n_atoms, 3)
        .to(dtype=like.dtype, device=like.device)
        .detach()
        for name in _PHASE_TARGETS
    )


def scales_by_atomic_type(
    model: "MetatrainModel", like: Tensor
) -> Dict[int, Tuple[float, float]]:
    """
    ``{atomic type: (sigma_q, sigma_p)}``, the per-target scales of the position and
    momentum targets for every atomic type of the model (averaged over x, y, z).

    :param model: The ``MetatrainModel`` wrapper.
    :param like: A tensor with the dtype and device to compute them with.
    """
    types = [int(t) for t in model.core.atomic_types]
    system = System(
        types=torch.tensor(types, dtype=torch.int32, device=like.device),
        positions=torch.zeros((len(types), 3), dtype=like.dtype, device=like.device),
        cell=torch.zeros((3, 3), dtype=like.dtype, device=like.device),
        pbc=torch.zeros(3, dtype=torch.bool, device=like.device),
    )
    sq, sp = _per_atom_target_scales(model, [system], like)
    return {
        t: (float(sq[i].mean()), float(sp[i].mean())) for i, t in enumerate(types)
    }


def reference_action(model: "MetatrainModel", like: Tensor) -> float:
    """c = mean over the atomic types of the model of sigma_q * sigma_p."""
    by_type = scales_by_atomic_type(model, like)
    return sum(sq * sp for sq, sp in by_type.values()) / max(len(by_type), 1)


def phase_space_scales(
    model: "MetatrainModel", systems: List[System], like: Tensor
) -> PhaseSpaceScales:
    """
    The normalization of the loss on a batch: D from the per-target scales of the
    ``position`` and ``momentum`` targets of every atom (the coordinates in which the
    data loss is measured), and the reference action c (:func:`reference_action`).

    :param model: The ``MetatrainModel`` wrapper.
    :param systems: The systems of the batch.
    :param like: ``q`` of the batch (dtype and device).
    """
    sq, sp = _per_atom_target_scales(model, systems, like)
    return PhaseSpaceScales(q=sq, p=sp, action=reference_action(model, like))


def resolve_normalization(normalization: str) -> str:
    if normalization not in NORMALIZATIONS:
        raise ValueError(
            f"Unknown weak symplectic normalization: {normalization!r} (expected one "
            f"of {NORMALIZATIONS})"
        )
    return normalization


def make_double_backward_jvp(
    q: Tensor, p: Tensor, q_next: Tensor, p_next: Tensor
) -> JvpFn:
    """
    J v for a batch of probes (first dimension: probes), as d/dw [J^T w], with two
    reverse-mode passes through the graph of the training forward.

    The graph of g(w) = J^T w, which does not depend on the probes, is built once here
    and shared by all the calls of the returned function (so by all the chunks).
    """
    wq = torch.zeros_like(q_next, requires_grad=True)
    wp = torch.zeros_like(p_next, requires_grad=True)

    # g(w) = J^T w, linear in w; differentiating it with respect to w gives J v.
    gq, gp = torch.autograd.grad(
        outputs=(q_next, p_next),
        inputs=(q, p),
        grad_outputs=(wq, wp),
        create_graph=True,
        retain_graph=True,
    )

    def jvp(Vq: Tensor, Vp: Tensor) -> Tuple[Tensor, Tensor]:
        Jv_q, Jv_p = torch.autograd.grad(
            outputs=(gq, gp),
            inputs=(wq, wp),
            grad_outputs=(Vq.to(gq.dtype), Vp.to(gp.dtype)),
            is_grads_batched=True,
            create_graph=True,
            retain_graph=True,
        )
        return Jv_q, Jv_p

    return jvp


def symplectic_vjp(
    q: Tensor,
    p: Tensor,
    q_next: Tensor,
    p_next: Tensor,
    Jv_q: Tensor,
    Jv_p: Tensor,
) -> Tuple[Tensor, Tensor]:
    """
    J^T (Omega J v) for a batch of J v (first dimension: probes), with one reverse-mode
    pass through the graph of the training forward (batched over the probes with
    ``is_grads_batched``), with Omega (x_q, x_p) = (x_p, -x_q).
    """
    JT_omega_Jv_q, JT_omega_Jv_p = torch.autograd.grad(
        outputs=(q_next, p_next),
        inputs=(q, p),
        grad_outputs=(Jv_p.to(q_next.dtype), -Jv_q.to(p_next.dtype)),
        is_grads_batched=True,
        create_graph=True,
        retain_graph=True,
    )
    return JT_omega_Jv_q, JT_omega_Jv_p


def probe_residual_parts(
    Vq: Tensor,
    Vp: Tensor,
    JT_omega_Jvq: Tensor,
    JT_omega_Jvp: Tensor,
    dim: int,
    scales: Optional[PhaseSpaceScales] = None,
) -> Tuple[Tensor, Tensor]:
    """
    The two terms of the probe losses, each ``[n_probes]``: the q-rows
    ``|(J^T Omega J v - Omega v)_q|^2 / dim`` and the p-rows
    ``|(J^T Omega J v - Omega v)_p|^2 / dim``, with ``Omega v = (v_p, -v_q)``, for the
    physical probes ``v = (Vq, Vp)``.

    With ``scales``, the residual is normalized (multiplied by D / c) before squaring,
    so that the terms are the q-rows and the p-rows of ``|D M D v~|^2 / (c^2 dim)`` for
    ``v = D v~``. Without, in physical units they have different units, which is why
    the diagnostics log them separately.
    """
    residual_q = JT_omega_Jvq - Vp
    residual_p = JT_omega_Jvp + Vq
    if scales is not None:
        residual_q, residual_p = scales.residual(residual_q, residual_p)
    q_rows = residual_q.square().flatten(start_dim=1).sum(dim=1) / dim
    p_rows = residual_p.square().flatten(start_dim=1).sum(dim=1) / dim
    return q_rows, p_rows


def probe_residual_losses(
    Vq: Tensor,
    Vp: Tensor,
    JT_omega_Jvq: Tensor,
    JT_omega_Jvp: Tensor,
    dim: int,
    scales: Optional[PhaseSpaceScales] = None,
) -> Tensor:
    """
    ``[n_probes]``: l_k = |J^T Omega J v_k - Omega v_k|^2 / dim (normalized with
    ``scales``), the sum of the two terms of :func:`probe_residual_parts`.
    """
    q_rows, p_rows = probe_residual_parts(
        Vq, Vp, JT_omega_Jvq, JT_omega_Jvp, dim, scales
    )
    return q_rows + p_rows


def calc_symplectic_probe_loss_chunks(
    q: Tensor,
    p: Tensor,
    q_next: Tensor,
    p_next: Tensor,
    *,
    n_probes: int,
    probe_dist: ProbeDistribution,
    chunk_size: int,
    probe_fn: Optional[ProbeFn] = None,
    with_parts: bool = False,
    scales: Optional[PhaseSpaceScales] = None,
) -> Iterator[Any]:
    """
    Lazily yields, chunk by chunk (``chunk_size`` probes per chunk), the *individual*
    symplectic loss of every probe: a tensor ``[n_chunk]`` with the graph attached,
    whose k-th element is

        l_k = (|J^T Omega J v_k - Omega v_k|^2) / dim,    dim = q.numel() + p.numel(),

    or, with ``scales`` (the normalization D, c; see the module docstring),

        l_k = |D M D v~_k|^2 / (c^2 dim),   M = J^T Omega J - Omega,

    for the normalized probes v~_k, computed with the physical probes v_k = D v~_k.
    Either way the probes v~ (or v) have identity covariance, so the mean over probes
    is an unbiased estimate of |M|_F^2 / dim, resp. |D M D|_F^2 / (c^2 dim).

    The training loss is the mean over all the probes of all the chunks (see
    :func:`calc_symplectic_loss_chunks`).

    J v is a double backward (:func:`make_double_backward_jvp`, whose J^T w graph is
    built once, before the first chunk) and J^T (Omega J v) one more reverse pass
    (:func:`symplectic_vjp`), both batched over the probes of the chunk. The probes are
    sampled by :func:`sample_probes` unless ``probe_fn`` is given (the dense reference
    of the diagnostics uses the canonical basis vectors; with ``scales`` they are basis
    vectors of the normalized coordinates).

    Since this is a generator, the graph of a chunk (the expensive part of the memory,
    kept alive by ``create_graph=True``) only exists while its chunk is being consumed:
    call ``.backward(retain_graph=True)`` on each chunk before asking for the next one,
    and its higher-order graph is freed in between. Only the graph of the model forward
    and the graph of J^T w, which are shared by all chunks, stay alive.

    :param with_parts: yield ``(losses, parts)`` instead, with ``parts`` the detached
        ``[n_chunk, 2]`` q-row and p-row terms of the losses
        (:func:`probe_residual_parts`).
    :param scales: the normalization, ``None`` for the raw loss.
    """
    jvp_fn = make_double_backward_jvp(q, p, q_next, p_next)

    dim = q.numel() + p.numel()

    for start in range(0, n_probes, chunk_size):
        n_chunk = min(chunk_size, n_probes - start)

        if probe_fn is None:
            Vq = sample_probes(q, n_chunk, probe_dist)
            Vp = sample_probes(p, n_chunk, probe_dist)
        else:
            Vq, Vp = probe_fn(start, n_chunk)
        if scales is not None:
            # normalized probes v~ -> physical probes v = D v~
            Vq, Vp = scales.probes(Vq, Vp)

        Jv_q, Jv_p = jvp_fn(Vq, Vp)
        JT_omega_Jvq, JT_omega_Jvp = symplectic_vjp(q, p, q_next, p_next, Jv_q, Jv_p)

        q_rows, p_rows = probe_residual_parts(
            Vq, Vp, JT_omega_Jvq, JT_omega_Jvp, dim, scales
        )
        losses = q_rows + p_rows
        if with_parts:
            yield losses, torch.stack([q_rows.detach(), p_rows.detach()], dim=1)
        else:
            yield losses


def calc_symplectic_loss_chunks(
    q: Tensor,
    p: Tensor,
    q_next: Tensor,
    p_next: Tensor,
    *,
    n_probes: int,
    probe_dist: ProbeDistribution,
    chunk_size: int,
    scales: Optional[PhaseSpaceScales] = None,
) -> Iterator[Tensor]:
    """
    Lazily yields the symplectic loss in chunks of ``chunk_size`` probes; the yielded
    values sum to the mean over all ``n_probes`` probes (see
    :func:`calc_symplectic_probe_loss_chunks`, which yields the individual probe losses
    these are built from).
    """
    for probe_losses in calc_symplectic_probe_loss_chunks(
        q,
        p,
        q_next,
        p_next,
        n_probes=n_probes,
        probe_dist=probe_dist,
        chunk_size=chunk_size,
        scales=scales,
    ):
        # divided by the total number of probes, so that the chunks sum to the mean
        yield probe_losses.sum() / n_probes


def symplectic_loss_chunks(
    model: "MetatrainModel",
    systems: list[System],
    raw_predictions: Dict[str, TensorMap],
    *,
    n_probes: int,
    probe_distribution: ProbeDistribution,
    chunk_size: int,
    normalization: Normalization = "data_scales",
) -> Iterator[Tensor]:
    """
    The symplectic loss of the last training forward, chunk by chunk (see
    :func:`calc_symplectic_loss_chunks`). The caller is expected to backpropagate each
    chunk before requesting the next one (``retain_graph=True`` for all but the last,
    since the model forward graph is shared), and to sum the chunks for logging. The
    number of chunks is given by :func:`num_loss_chunks`.

    :param model: The ``MetatrainModel`` wrapper (not wrapped in DDP).
    :param normalization: ``"data_scales"`` (the loss in the coordinates of the data
        loss, see the module docstring) or ``"none"`` (physical units).
    """
    q, p, q_next, p_next = phase_graph_from_predictions(model, systems, raw_predictions)
    scales = (
        phase_space_scales(model, systems, q)
        if resolve_normalization(normalization) == "data_scales"
        else None
    )
    return calc_symplectic_loss_chunks(
        q,
        p,
        q_next,
        p_next,
        n_probes=n_probes,
        probe_dist=probe_distribution,
        chunk_size=chunk_size,
        scales=scales,
    )


def symplectic_loss(
    model: "MetatrainModel",
    systems: list[System],
    raw_predictions: Dict[str, TensorMap],
    *,
    n_probes: int,
    probe_distribution: ProbeDistribution,
    chunk_size: Optional[int] = None,
    normalization: Normalization = "data_scales",
) -> Tensor:
    """
    The full symplectic loss as a single tensor (all chunks kept in one graph, so
    memory grows with ``n_probes``). ``chunk_size=None`` batches all the probes at once.
    For the memory savings of chunking use :func:`symplectic_loss_chunks` instead.
    """
    chunks = symplectic_loss_chunks(
        model,
        systems,
        raw_predictions,
        n_probes=n_probes,
        probe_distribution=probe_distribution,
        chunk_size=n_probes if chunk_size is None else chunk_size,
        normalization=normalization,
    )
    return sum(chunks)  # type: ignore[return-value]

"""
Diagnostics of the weak symplectic loss, independent of the loss itself.

Nothing in here is needed to train with the weak symplectic loss: the loss lives in
:mod:`.weak_symplectic_loss`, and this module only *looks* at it. The diagnostics work
whether the loss is active (``weak_symplectic_weight > 0``) or not, which makes it
possible to measure how symplectic a model is without ever regularizing it: every
diagnostic runs its own forward pass in training mode with the weak symplectic graph
enabled, on a fixed batch, and restores the model afterwards. Like the loss, they only
use reverse-mode differentiation through the graph of the training forward (J v by
double backward, rows of J by batched vector-Jacobian products), so they also work with
the long-range featurizer.

Everything is configured by the ``diagnostics`` dictionary of the trainer hypers
(:class:`SymplecticDiagnosticsHypers`) and driven by :class:`SymplecticDiagnostics`:

- periodic diagnostics of the stochastic loss and of its gradient
  (:func:`run_symplectic_diagnostics_for_ks`), every ``every`` optimizer steps: the
  estimator's mean and variance against an exact (dense) or high-K reference, the
  split of the residual into its q-rows and p-rows (which have different units, see
  :func:`.weak_symplectic_loss.probe_residual_parts`), the agreement of the gradient
  with the reference gradient and between probes, and the agreement and relative size
  of the gradient of the data loss and of the symplectic loss;
- one-off checks: neighbor statistics of the whole data (:func:`print_neighbor_info`),
  the sparsity of the dense Jacobian on the first diagnostic batch
  (:func:`print_jacobian_sparsity`, optionally repeated during the training), and the
  time and memory of the stages of the loss (forward, J v, J^T Omega J v, backward)
  against the number of probes per chunk (:func:`measure_probe_costs`), on the
  structures of the data with the most neighbors, with the batch size it allows.

They never touch the optimizer, the scheduler, the parameters or their ``.grad``
(gradients are obtained with ``torch.autograd.grad``), and they leave the global random
number generators as they found them.
"""

import contextlib
import csv
import gc
import json
import logging
import math
import time
import warnings
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import torch
from metatensor.torch import TensorMap
from metatomic.torch import NeighborListOptions, System
from torch import Tensor
from typing_extensions import TypedDict

from metatrain.utils.data import CollateFn, unpack_batch
from metatrain.utils.per_atom import average_by_num_atoms
from metatrain.utils.transfer import batch_to

from .weak_symplectic_loss import (
    Normalization,
    PhaseSpaceScales,
    ProbeDistribution,
    ProbeFn,
    calc_symplectic_probe_loss_chunks,
    make_double_backward_jvp,
    phase_graph_from_predictions,
    phase_space_scales,
    probe_residual_losses,
    resolve_normalization,
    sample_probes,
    scales_by_atomic_type,
    symplectic_vjp,
)


if TYPE_CHECKING:
    from metatrain.utils.wrapper import MetatrainModel


ReferenceMode = Literal["auto", "dense", "high_k"]


# ------------------------------------------------------------------------------------
# Hyperparameters
# ------------------------------------------------------------------------------------


class SymplecticDiagnosticsHypers(TypedDict):
    """Diagnostics of the weak symplectic loss, to be used with or without the loss.

    They are evaluated on rank 0 only, on a fixed (unaugmented, unshuffled) batch,
    without touching the optimizer. With distributed training the other ranks wait at
    their next gradient all-reduce while rank 0 works, so keep one diagnostic well
    below a few minutes.
    """

    every: int = 0
    """Number of optimizer steps between two evaluations of the stochastic symplectic
    loss and of its gradient (``0`` disables them). They are also evaluated after the
    last step. Results are written to ``symplectic_diagnostics.{csv,jsonl}`` and
    plotted in ``symplectic_diagnostics_<split>.png`` in ``output_dir``."""
    ks: list[int] = []
    """Probe counts K to compare on the same batch, sharing the forward pass and the
    reference (e.g. ``[2, 8]``). If empty, the number of probes of the loss
    (``weak_symplectic_num_probes``) is used."""
    reference: Literal["auto", "dense", "high_k"] = "auto"
    """Reference the stochastic estimators are compared with. ``dense`` is exact
    (``6 * n_atoms`` probes: the canonical basis), ``high_k`` uses
    ``reference_probes`` random probes, and ``auto`` is ``dense`` if
    ``6 * n_atoms <= dense_max_dim`` and ``high_k`` otherwise."""
    reference_probes: int = 256
    """Number of random probes of the ``high_k`` reference."""
    dense_max_dim: int = 600
    """Largest ``6 * n_atoms`` for which ``auto`` picks the ``dense`` reference."""
    chunk_size: Optional[int] = None
    """Number of probes whose graph is alive at once during the diagnostics. If
    ``None``, the chunk size of the loss (``weak_symplectic_chunk_size``) is used."""
    batch_size: Optional[int] = None
    """Number of structures of the fixed diagnostic batches (the first ones of the
    first training and validation datasets). If ``None``, the ``batch_size`` of the
    training is used. Keep it small: the dense reference scales with the number of
    atoms."""
    validation: bool = True
    """Also evaluate the periodic diagnostics on a fixed validation batch."""
    seed: int = 0
    """Seed of the probes of the diagnostics (combined with the step and the split)."""
    probe_gradients: bool = True
    """Also backpropagate every probe on its own, to log how much the gradients of the
    individual probes agree with each other and with the reference (one backward per
    probe, and a ``K x n_params`` float32 matrix in CPU memory)."""
    data_gradient: bool = True
    """Also compute the gradient of the data loss on the diagnostic batch (one more
    backward), to log its cosine with the gradients of the symplectic loss and the
    ratio ``weight * |grad symplectic| / |grad data|``: whether the two losses pull the
    parameters in compatible directions, and which one dominates the update."""
    output_dir: Optional[str] = None
    """Directory of the CSV, JSON lines and PNG files. If ``None``, the checkpoint
    directory is used."""
    neighbor_info: bool = False
    """Once, at the start of the training: print the neighbor statistics of *all* the
    training (and validation) structures, and what they mean for the cost of the
    symplectic loss. It collates the whole dataset once, so it can take a while on
    large datasets."""
    jacobian_sparsity: bool = False
    """Once, on the first diagnostic batch: compute the dense Jacobian of the phase-space
    map and print which atom blocks of J and of ``J^T Omega J - Omega`` are exactly
    zero, grouped by hop distance in the neighbor graph, and the three symplectic
    identities (the qq, qp and pp blocks of ``J^T Omega J - Omega``)."""
    normalization_check: bool = False
    """Once, on the first diagnostic batch: check the normalized (``data_scales``) loss
    |D M D|_F^2 / (c^2 dim) four ways -- from the dense Jacobian, in the normalized
    coordinates with the transformed symplectic form, with the loss code and basis
    probes, and with ``normalization_check_probes`` random probes -- and print them with
    the raw loss, the balance of the residual blocks before and after the normalization
    and the scales used (see :func:`check_normalization`). The result is also written to
    ``normalization_check.json``. It needs the dense Jacobian, so it is skipped if
    ``6 * n_atoms > jacobian_max_dim``. Runs whatever ``weak_symplectic_normalization``
    is."""
    normalization_check_probes: int = 1024
    """Number of random probes of the stochastic part of ``normalization_check``."""
    scaling_check: bool = False
    """Once, on the first diagnostic batch: follow q and p through the whole chain --
    dataset, rotation of the augmentation, removal of the additive part and of the
    scales, inputs and outputs of the training forward, physical map of the weak loss,
    model in evaluation mode -- and check every step numerically (see
    :func:`check_scaling_pipeline`). Written to ``scaling_check.json``."""
    jacobian_max_dim: int = 1500
    """Largest ``6 * n_atoms`` for which ``jacobian_sparsity`` is computed (it costs
    that many vector-Jacobian products)."""
    jacobian_chunk_size: int = 64
    """Number of rows of the dense Jacobian computed together (one batched
    vector-Jacobian product per chunk; lower it if it does not fit in memory)."""
    jacobian_plot: bool = True
    """With ``jacobian_sparsity``: also save heat maps of the normalized ``|J|`` and
    ``|M|`` (M = J^T Omega J - Omega), of their atom blocks (log scale) and of their decay
    with the hop distance, in ``output_dir``: ``jacobian_sparsity.png``, or
    ``jacobian_sparsity_step<step>.png`` if ``jacobian_every`` is set."""
    jacobian_plot_pairs: bool = True
    """With ``jacobian_sparsity``: also save atom-pair maps in the style of the Hessian
    figures of the truncated automatic sparse differentiation paper (Langer, Hill,
    Ceriotti): atoms ordered so that graph neighbors are adjacent, the hop distance
    above the diagonal and the block norm below it, and the block norm against the hop
    distance (median and 10-90% band) for J and for M. Saved as ``jacobian_pairs.png``,
    or ``jacobian_pairs_step<step>.png`` if ``jacobian_every`` is set."""
    jacobian_every: int = 0
    """With ``jacobian_sparsity``: also recompute the dense Jacobian every this many
    optimizer steps (and after the last one) on the same fixed batch, to follow how its
    sparsity and the symplectic residual change during the training (``0``: only at the
    start). Each time costs ``6 * n_atoms`` vector-Jacobian products, so use a large
    interval. A line per computation is appended to ``jacobian_sparsity.csv`` and logged,
    and the heat maps are saved if ``jacobian_plot``; the full tables are only printed
    at the start."""
    cost_benchmark: bool = False
    """Once, before the first optimizer step: time and peak memory (CUDA only) of each
    stage of the symplectic loss -- the training forward, the J^T w graph built once
    per forward (``setup``), J v (``jvp``), J^T Omega J v (``vjp``) and the backward to
    the parameters -- for a chunk of ``cost_benchmark_probes`` probes and of the
    fractions ``cost_benchmark_fractions`` of it (and of the chunk size of the
    training). Also gives the time of the whole loss in chunks of each size, the peak
    memory above the forward graph, and the largest batch size that fits on the device,
    to choose ``batch_size`` and ``weak_symplectic_chunk_size``."""
    cost_benchmark_probes: Optional[int] = None
    """Number of probes of ``cost_benchmark``. If ``None``, the number of probes of the
    loss (``weak_symplectic_num_probes``) is used."""
    cost_benchmark_fractions: list[float] = [1.0, 0.5, 0.25]
    """Chunk sizes of ``cost_benchmark``, as fractions of ``cost_benchmark_probes``."""
    cost_benchmark_repeats: int = 3
    """Timed repeats of every chunk of ``cost_benchmark`` (the best time is kept)."""
    cost_benchmark_headroom: float = 0.15
    """With ``cost_benchmark`` on CUDA: the fraction of the device memory that the batch
    size recommendation leaves free, for the CUDA context, the fragmentation of the
    allocator and what the linear estimate misses."""
    cost_benchmark_max_neighbors: bool = True
    """With ``cost_benchmark``: measure on the training structures with the largest
    number of neighbors of any atom (the padded width of a batch is set by its most
    connected atom, and the attention is quadratic in it), i.e. on the most expensive
    structures of the data. If ``False``, the first structures of the training set are
    used. The neighbor statistics of the whole data are computed once for this, if
    ``neighbor_info`` has not already done it."""


def _resolve_hypers(hypers: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """The hypers with the defaults of :class:`SymplecticDiagnosticsHypers` filled in
    (a checkpoint or a config written before the diagnostics may not have them)."""
    from metatrain.utils.hypers import init_with_defaults

    return {**init_with_defaults(SymplecticDiagnosticsHypers), **(hypers or {})}


# ------------------------------------------------------------------------------------
# Running the model for the diagnostics
# ------------------------------------------------------------------------------------


@contextlib.contextmanager
def isolated_rng(device=None):
    """
    Restore the global (CPU and CUDA) random number generator states on exit, so that
    what runs inside does not change the random numbers seen by the rest of training.
    """
    devices: List[int] = []
    if device is not None and torch.device(device).type == "cuda":
        index = torch.device(device).index
        devices = [index if index is not None else torch.cuda.current_device()]
    with torch.random.fork_rng(devices=devices):
        yield


@contextlib.contextmanager
def symplectic_forward(
    model: "MetatrainModel",
    batch: Union[Callable[[], List[System]], List[System]],
):
    """
    Run the model once, in training mode with the weak symplectic graph enabled (whether
    or not the loss is active in the training), and yield ``(systems, raw_predictions)``
    with the graph attached. The state of the model (training mode, symplectic flag,
    stored phase-space leaves) and the random number generators are restored on exit.

    :param model: the ``MetatrainModel`` wrapper (not wrapped in DDP).
    :param batch: the systems of the fixed diagnostic batch, or a function returning
        fresh ones (preferred: ``forward`` replaces ``system.positions`` in place).
    """
    from metatrain.utils.evaluate_model import evaluate_model

    core = model.core
    params = [p for p in model.parameters() if p.requires_grad]
    device = params[0].device
    previous_state = (
        model.training,
        core._weak_symplectic_enabled,
        core._weak_symplectic_positions,
        core._weak_symplectic_momenta,
    )
    try:
        # the phase-space leaves are only created in training mode
        model.train()
        core.set_weak_symplectic_enabled(True)

        with isolated_rng(device), torch.enable_grad():
            systems = batch() if callable(batch) else batch
            targets = {
                name: model.dataset_info.targets[name]
                for name in ("position", "momentum")
            }
            raw_predictions = evaluate_model(model, systems, targets, is_training=True)
            yield systems, raw_predictions
    finally:
        (
            was_training,
            core._weak_symplectic_enabled,
            core._weak_symplectic_positions,
            core._weak_symplectic_momenta,
        ) = previous_state
        model.train(was_training)


def neighbor_list_options(model: "MetatrainModel") -> NeighborListOptions:
    """The neighbor list of the core model (the one its graph is built from)."""
    return model.core.requested_neighbor_lists()[0]


def neighbor_counts(system: System, options: NeighborListOptions) -> Tensor:
    """``[n_atoms]`` number of neighbors of every atom of ``system`` (CPU)."""
    first_atom = system.get_neighbor_list(options).samples.column("first_atom")
    return torch.bincount(first_atom.cpu(), minlength=len(system))


def batch_neighbor_width(systems: List[System], options: NeighborListOptions) -> int:
    """The padded neighbor width of the graph of a batch: its largest neighbor count."""
    return max(
        (int(neighbor_counts(system, options).max()) for system in systems if len(system)),
        default=0,
    )


# ------------------------------------------------------------------------------------
# Neighbor statistics of the data
# ------------------------------------------------------------------------------------

# how many of the most connected structures of each split the statistics remember
_N_WORST_STRUCTURES = 256


def neighbor_statistics(
    model: "MetatrainModel",
    datasets: Mapping[str, Sequence[Any]],
    collate_fn: Callable,
    *,
    n_probes: int,
    chunk_size: int,
    batch_size: int = 64,
) -> Dict[str, Dict[str, Any]]:
    """
    Neighbor statistics of *all* the structures of ``datasets`` (one entry per name, each
    a list of datasets), for the neighbor list the model requests. Independent of the
    forward pass and of the batching: the structures are collated (which attaches the
    neighbor lists) ``batch_size`` at a time, only to be counted.

    Every atom has ``n_i`` neighbors; the graph of a batch is padded (NEF format) to the
    largest ``n_i`` of the batch, and the padded slots have a cutoff factor of exactly
    0, so they do not enter J = d(q_next, p_next) / d(q, p), only the real neighbors do.
    Consequently:

    - ``dim = 6 * n_atoms`` (the normalization of every probe loss) does *not* depend on
      the number of neighbors;
    - the sparsity of J (hence of J^T Omega J - Omega) does: an atom only feels the atoms
      inside ``num_gnn_layers * cutoff`` (without the long-range featurizer);
    - the cost of one J v / one J^T w pass scales with ``n_atoms * max_num_neighbors``
      (the padded size), and is paid ``n_probes`` times (``chunk_size`` at a time).

    Each entry also has ``worst_structures``: ``[dataset index, structure index, largest
    neighbor count, n_atoms]`` of the most connected structures (largest neighbor count
    first), which the cost benchmark measures.
    """
    options = neighbor_list_options(model)
    result: Dict[str, Dict[str, Any]] = {}
    for name, split_datasets in datasets.items():
        counts: List[Tensor] = []
        atoms_per_structure: List[int] = []
        structures: List[List[int]] = []
        for dataset_index, dataset in enumerate(split_datasets):
            for start in range(0, len(dataset), batch_size):
                stop = min(start + batch_size, len(dataset))
                with isolated_rng():
                    batch = collate_fn([dataset[i] for i in range(start, stop)])
                systems, _, _ = unpack_batch(batch)
                for offset, system in enumerate(systems):
                    system_counts = neighbor_counts(system, options)
                    counts.append(system_counts)
                    atoms_per_structure.append(len(system))
                    structures.append(
                        [
                            dataset_index,
                            start + offset,
                            int(system_counts.max()) if len(system) else 0,
                            len(system),
                        ]
                    )
        if len(counts) == 0:
            continue

        per_atom = torch.cat(counts).double()
        sorted_counts, _ = per_atom.sort()

        def percentile(fraction: float) -> int:
            return int(sorted_counts[int(fraction * (len(sorted_counts) - 1))])

        n_atoms = int(per_atom.numel())
        max_neighbors = int(per_atom.max())
        mean_atoms = n_atoms / len(atoms_per_structure)
        structures.sort(key=lambda s: (-s[2], -s[3]))
        result[name] = {
            "n_structures": len(atoms_per_structure),
            "n_atoms": n_atoms,
            "atoms_min": min(atoms_per_structure),
            "atoms_mean": mean_atoms,
            "atoms_max": max(atoms_per_structure),
            "neighbors_min": int(per_atom.min()),
            "neighbors_mean": float(per_atom.mean()),
            "neighbors_median": percentile(0.5),
            "neighbors_p95": percentile(0.95),
            "neighbors_p99": percentile(0.99),
            "neighbors_max": max_neighbors,
            "real_edges": int(per_atom.sum()),
            # padded slots if a batch were padded to the largest neighbor count
            "padding_fraction_at_max": 1.0
            - float(per_atom.mean()) / max(max_neighbors, 1),
            "dim_mean": 6.0 * mean_atoms,
            "n_probes": int(n_probes),
            "n_chunks": math.ceil(n_probes / chunk_size),
            "edge_slots_x_probes_per_structure": mean_atoms * max_neighbors * n_probes,
            "worst_structures": structures[:_N_WORST_STRUCTURES],
        }
    return result


def print_neighbor_info(
    model: "MetatrainModel",
    datasets: Mapping[str, Sequence[Any]],
    collate_fn: Callable,
    *,
    n_probes: int,
    chunk_size: int,
    batch_size: int = 64,
) -> Dict[str, Dict[str, Any]]:
    """Print :func:`neighbor_statistics` with a short explanation; returns its dict."""
    info = neighbor_statistics(
        model,
        datasets,
        collate_fn,
        n_probes=n_probes,
        chunk_size=chunk_size,
        batch_size=batch_size,
    )
    lines = ["", "[weak symplectic] neighbors in the whole dataset"]
    for name, s in info.items():
        lines += [
            f"  {name}: {s['n_structures']} structures, {s['n_atoms']} atoms "
            f"(per structure: min {s['atoms_min']}, mean {s['atoms_mean']:.1f}, "
            f"max {s['atoms_max']})",
            f"    neighbors per atom (real)   : min {s['neighbors_min']}, "
            f"mean {s['neighbors_mean']:.1f}, median {s['neighbors_median']}, "
            f"p95 {s['neighbors_p95']}, p99 {s['neighbors_p99']}, "
            f"max {s['neighbors_max']}",
            f"    real edges                  : {s['real_edges']}",
            f"    padding if padded to the max: {s['padding_fraction_at_max']:.1%} "
            "(cutoff factor 0 -> not in J)",
            f"    dim = 6 * n_atoms           : {s['dim_mean']:.0f} per structure "
            "on average",
            f"    {s['n_probes']} probes in {s['n_chunks']} chunk(s) -> "
            f"~{s['edge_slots_x_probes_per_structure']:.0f} edge-slot evaluations per "
            "J v / J^T w, per structure",
        ]
    lines += [
        "  effect on the symplectic loss:",
        "    dim is the normalization of every probe loss, independent of the neighbor "
        "count",
        "    neighbors define which atoms enter J (sparsity of J^T Omega J - Omega)",
        "    cost per probe ~ n_atoms * padded neighbor width",
        "",
    ]
    print("\n".join(lines), flush=True)
    return info


# ------------------------------------------------------------------------------------
# Cost of the stages of the loss (forward, setup, jvp, vjp, backward)
# ------------------------------------------------------------------------------------

LOSS_STAGES = ("jvp", "vjp", "backward")

_OUT_OF_MEMORY = getattr(torch, "OutOfMemoryError", torch.cuda.OutOfMemoryError)


def _measured(device: torch.device, function: Callable[[], Any]) -> Tuple[Any, Dict]:
    """
    ``(function(), cost)``, with ``cost`` the wall time ``time_s`` and, on CUDA (NaN on
    CPU, where torch does not track memory): ``peak_memory``, the peak allocated during
    the call above what was allocated before it, ``kept_memory``, what the call left
    allocated (the graph it built), and ``peak_total``, the absolute peak.
    """
    is_cuda = device.type == "cuda"
    if is_cuda:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        before = torch.cuda.memory_allocated(device)
    start = time.perf_counter()
    result = function()
    if is_cuda:
        torch.cuda.synchronize(device)
    cost = {"time_s": time.perf_counter() - start}
    if is_cuda:
        peak = torch.cuda.max_memory_allocated(device)
        cost["peak_memory"] = float(peak - before)
        cost["kept_memory"] = float(torch.cuda.memory_allocated(device) - before)
        cost["peak_total"] = float(peak)
    else:
        cost["peak_memory"] = cost["kept_memory"] = cost["peak_total"] = math.nan
    return result, cost


def measure_probe_costs(
    model: "MetatrainModel",
    batch: Callable[[], List[System]],
    *,
    n_probes: int,
    probe_distribution: ProbeDistribution,
    fractions: Sequence[float] = (1.0, 0.5, 0.25),
    repeats: int = 3,
    chunk_sizes: Sequence[int] = (),
    normalization: Normalization = "data_scales",
) -> Dict[str, Any]:
    """
    Time and memory of the stages of the symplectic loss on one batch, for chunks of
    ``n_probes``, half and a quarter of them (``fractions``):

    - ``forward``: the training forward on the batch (data collation excluded), whose
      graph is shared by everything below (``kept_memory`` is what that graph costs);
    - ``setup``: what is built once per forward and shared by all the chunks: the graph
      of g(w) = J^T w that the double backward differentiates to get J v;
    - ``jvp``: J v of the probes of a chunk;
    - ``vjp``: J^T Omega J v and the residuals of the chunk;
    - ``backward``: the gradient of the chunk's loss to the parameters (with
      ``torch.autograd.grad``: no ``.grad`` is written).

    One chunk is measured (after one untimed warm-up of the same shape; ``time_s`` is the
    best of ``repeats``, memory is of the last repeat). The graph of a chunk is freed
    before the next one, so the whole loss of ``n_probes`` probes in chunks of ``n``
    takes ``ceil(n_probes / n)`` times the time of a chunk, and a peak memory of the
    forward graph and setup plus one chunk: both are in the rows (``total_time_s``,
    ``peak_over_forward``). Memory is CUDA only.

    The model is left as it was (see :func:`symplectic_forward`).

    :param batch: function returning fresh systems of the batch (``forward`` replaces
        ``system.positions`` in place).
    :param chunk_sizes: chunk sizes measured on top of the ``fractions`` of
        ``n_probes`` (e.g. the one of the training), each at most ``n_probes``.
    :param normalization: the normalization of the loss, as in the training.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    device = params[0].device
    on_cuda = device.type == "cuda"
    normalize = resolve_normalization(normalization) == "data_scales"
    sizes = sorted(
        {max(1, math.ceil(f * n_probes)) for f in fractions}
        | {min(max(1, int(c)), n_probes) for c in chunk_sizes},
        reverse=True,
    )

    def collect() -> None:
        gc.collect()
        if on_cuda:
            torch.cuda.empty_cache()

    def oom_row(n: int) -> Dict[str, Any]:
        return {"n_chunk": int(n), "n_chunks": math.ceil(n_probes / n), "oom": True}

    # warm-up of the forward (allocator, kernels), not timed
    with symplectic_forward(model, batch()):
        pass
    collect()
    # what is allocated without any batch: the weights and whatever else lives on the GPU
    memory_before = float(torch.cuda.memory_allocated(device)) if on_cuda else math.nan

    forward_cm = symplectic_forward(model, batch())
    (systems, raw_predictions), forward_cost = _measured(device, forward_cm.__enter__)
    try:
        q, p, q_next, p_next = phase_graph_from_predictions(
            model, systems, raw_predictions
        )
        dim = q.numel() + p.numel()
        scales = phase_space_scales(model, systems, q) if normalize else None
        options = neighbor_list_options(model)
        counts = torch.cat([neighbor_counts(system, options) for system in systems])
        info: Dict[str, Any] = {
            "n_atoms": int(q.shape[0]),
            "n_systems": len(systems),
            "dim": int(dim),
            "n_probes": int(n_probes),
            "memory_before": memory_before,
            "param_bytes": float(sum(p.numel() * p.element_size() for p in params)),
            "device_total_memory": (
                float(torch.cuda.get_device_properties(device).total_memory)
                if on_cuda
                else math.nan
            ),
            "neighbor_width": int(counts.max()) if counts.numel() else 0,
            "neighbors_mean": float(counts.double().mean()) if counts.numel() else 0.0,
            "device": str(device),
            "dtype": str(q.dtype),
            "forward": forward_cost,
            "setup": None,
            "rows": [],
        }
        memory_after_forward = (
            float(torch.cuda.memory_allocated(device)) if on_cuda else 0.0
        )

        try:
            jvp_fn, setup_cost = _measured(
                device, lambda: make_double_backward_jvp(q, p, q_next, p_next)
            )
        except _OUT_OF_MEMORY:
            jvp_fn = None
        if jvp_fn is None:
            # even the setup does not fit: nothing can be measured
            collect()
            info["setup"] = {"time_s": math.nan, "kept_memory": math.nan}
            info["rows"] = [oom_row(n) for n in sizes]
            return info
        info["setup"] = setup_cost
        memory_with_setup = (
            float(torch.cuda.memory_allocated(device)) if on_cuda else math.nan
        )

        def measure_row(n: int) -> Dict[str, Any]:
            def one_chunk() -> Dict[str, Dict[str, float]]:
                Vq = sample_probes(q, n, probe_distribution)
                Vp = sample_probes(p, n, probe_distribution)
                if scales is not None:
                    Vq, Vp = scales.probes(Vq, Vp)
                (Jv_q, Jv_p), jvp_cost = _measured(device, lambda: jvp_fn(Vq, Vp))

                def vjp_and_loss() -> Tensor:
                    JTq, JTp = symplectic_vjp(q, p, q_next, p_next, Jv_q, Jv_p)
                    losses = probe_residual_losses(Vq, Vp, JTq, JTp, dim, scales)
                    return losses.sum() / n

                loss, vjp_cost = _measured(device, vjp_and_loss)
                _, backward_cost = _measured(
                    device,
                    lambda: torch.autograd.grad(
                        loss, params, retain_graph=True, allow_unused=True
                    ),
                )
                return {"jvp": jvp_cost, "vjp": vjp_cost, "backward": backward_cost}

            one_chunk()  # not timed: warm-up for this shape
            collect()
            runs = []
            for _ in range(repeats):
                runs.append(one_chunk())
                collect()
            stages = {
                stage: {
                    **runs[-1][stage],
                    "time_s": min(run[stage]["time_s"] for run in runs),
                }
                for stage in LOSS_STAGES
            }
            chunk_time = sum(stages[stage]["time_s"] for stage in LOSS_STAGES)
            n_chunks = math.ceil(n_probes / n)
            peak_total = max(stages[stage]["peak_total"] for stage in LOSS_STAGES)
            return {
                "n_chunk": int(n),
                "n_chunks": int(n_chunks),
                "stages": stages,
                "chunk_time_s": chunk_time,
                "time_per_probe_s": chunk_time / n,
                # the whole loss of n_probes probes, in chunks of n
                "total_time_s": n_chunks * chunk_time,
                "peak_total": peak_total,
                "peak_over_forward": peak_total - memory_after_forward,
                "memory_after_setup": memory_with_setup,
            }

        for n in sizes:
            # a chunk size that does not fit is reported, the smaller ones still run
            try:
                row = measure_row(n)
            except _OUT_OF_MEMORY:
                row = None
            if row is None:
                row = oom_row(n)
            info["rows"].append(row)
            collect()
        jvp_fn = None  # free the J^T w graph before the forward graph
        collect()
    finally:
        forward_cm.__exit__(None, None, None)
    return info


def print_probe_costs(info: Dict[str, Any]) -> None:
    """Print :func:`measure_probe_costs`."""

    def mib(value: float) -> str:
        return "n/a" if value != value else f"{value / 2**20:.0f}"

    forward = info["forward"]
    setup = info["setup"]
    lines = [
        "",
        f"[weak symplectic] cost of the stages of the loss: {info['n_systems']} "
        f"structures, {info['n_atoms']} atoms (dim {info['dim']}), neighbors per atom "
        f"{info['neighbors_mean']:.1f} on average, padded to {info['neighbor_width']}, "
        f"{info['dtype']} on {info['device']}, {info['n_probes']} probes",
        f"  forward (training forward, graph kept): {forward['time_s']:.4f} s, "
        f"peak {mib(forward['peak_memory'])} MiB, graph kept "
        f"{mib(forward['kept_memory'])} MiB",
        f"  setup (J^T w graph, once per forward, shared by the chunks): "
        f"{setup['time_s']:.4f} s, kept {mib(setup['kept_memory'])} MiB",
        f"    {'probes/':>7} {'':>6} | {'time of one chunk [s]':^38} | "
        f"{'K probes':>9} {'per probe':>10} | "
        f"{'peak mem above stage start [MiB]':^26} | {'peak above':>10}",
        f"    {'chunk':>7} {'chunks':>6} | {'jvp':>8} {'vjp':>8} {'bwd':>8} "
        f"{'chunk':>9} | {'[s]':>9} {'[ms]':>10} | {'jvp':>8} {'vjp':>8} "
        f"{'bwd':>8}   | {'forward':>10}",
    ]
    for row in info["rows"]:
        if row.get("oom"):
            lines.append(
                f"    {row['n_chunk']:>7} {row['n_chunks']:>6} | "
                "out of memory: this chunk size does not fit on the device"
            )
            continue
        s = row["stages"]
        lines.append(
            f"    {row['n_chunk']:>7} {row['n_chunks']:>6} | "
            f"{s['jvp']['time_s']:>8.4f} {s['vjp']['time_s']:>8.4f} "
            f"{s['backward']['time_s']:>8.4f} {row['chunk_time_s']:>9.4f} | "
            f"{row['total_time_s']:>9.4f} {1000 * row['time_per_probe_s']:>10.2f} | "
            f"{mib(s['jvp']['peak_memory']):>8} {mib(s['vjp']['peak_memory']):>8} "
            f"{mib(s['backward']['peak_memory']):>8}   | "
            f"{mib(row['peak_over_forward']):>10}"
        )
    lines += [
        "  'K probes': the whole loss in chunks of that size (chunks x chunk time); the "
        "peak memory is the one of the forward graph plus one chunk (graph freed between "
        "chunks): 'peak above forward' = chunk peak - memory kept by the forward"
        + ("" if info["device"].startswith("cuda") else " (memory: CUDA only)"),
        "",
    ]
    print("\n".join(lines), flush=True)


# copies of the weights that training holds on top of them: the gradients, the two
# moments of Adam and the buffer of the gradient all-reduce
_TRAINING_STATE_COPIES = 4

# options whose time per probe is within this factor of the fastest one count as equally
# fast
_TIME_TOLERANCE = 1.03


def plan_batch_size(
    info: Dict[str, Any],
    *,
    batch_size: int,
    chunk_size: int,
    headroom: float = 0.15,
) -> Optional[Dict[str, Any]]:
    """
    What the memory measured by :func:`measure_probe_costs` on its batch means for the
    training batch: the memory of the forward graph and of a chunk grows linearly with the
    number of atoms, so for every chunk size that fitted on the benchmark batch, the peak
    at ``batch_size`` structures per device is

        memory_before + 4 x weights + batch_size x (forward graph + chunk) per structure,

    and the largest ``batch_size`` that stays within the device memory minus ``headroom``
    follows. ``None`` without CUDA memory numbers.

    The estimate assumes structures like the ones of the benchmark batch (same atoms per
    structure and the same padded neighbor width: by default the most connected
    structures of the data, so it is the worst case), and leaves out what is not
    measured (the data loader's batches, the allocator's fragmentation: ``headroom``).

    :param batch_size: the structures per device of the training.
    :param chunk_size: the chunk size of the training.
    :param headroom: the fraction of the device memory left free.
    :return: ``options`` (one per measured chunk size that fitted on the benchmark
        batch), ``current`` (the option of the training chunk size, ``None`` if it did
        not fit on the benchmark batch: ``current_oom``), ``best`` (see below) and the
        memory budget. ``best`` is the fastest option that fits ``batch_size`` (among
        equally fast ones, the one with the largest ``max_batch_size``), or, if none
        fits, the fastest one with the largest ``max_batch_size``.
    """
    total = info["device_total_memory"]
    if total != total or not info["rows"]:
        return None
    budget = total * (1.0 - headroom)
    fixed = info["memory_before"] + _TRAINING_STATE_COPIES * info["param_bytes"]
    available = budget - fixed
    forward_kept = info["forward"]["kept_memory"] + info["setup"]["kept_memory"]
    current_chunk = min(chunk_size, info["n_probes"])

    options = []
    for row in info["rows"]:
        if row.get("oom"):
            continue
        per_system = (row["peak_over_forward"] + forward_kept) / info["n_systems"]
        peak = fixed + batch_size * per_system
        options.append(
            {
                "n_chunk": row["n_chunk"],
                "time_per_probe_s": row["time_per_probe_s"],
                "per_system": per_system,
                "peak": peak,
                "fits": peak <= budget,
                "max_batch_size": max(0, int(available // max(per_system, 1.0))),
            }
        )

    def fastest(candidates: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not candidates:
            return None
        limit = _TIME_TOLERANCE * min(c["time_per_probe_s"] for c in candidates)
        equal = [c for c in candidates if c["time_per_probe_s"] <= limit]
        return max(equal, key=lambda c: c["max_batch_size"])

    fitting = [o for o in options if o["fits"]]
    return {
        "batch_size": int(batch_size),
        "chunk_size": int(current_chunk),
        "headroom": float(headroom),
        "budget": budget,
        "fixed": fixed,
        "available": available,
        "options": options,
        "current": next((o for o in options if o["n_chunk"] == current_chunk), None),
        "current_oom": any(
            r.get("oom") and r["n_chunk"] == current_chunk for r in info["rows"]
        ),
        "best": fastest(fitting) or fastest(options),
    }


def print_batch_size_plan(plan: Dict[str, Any], info: Dict[str, Any]) -> None:
    """Print :func:`plan_batch_size` and what to set."""

    def gib(value: float) -> str:
        return f"{value / 2**30:.1f}"

    B = plan["batch_size"]
    atoms_per_system = info["n_atoms"] / info["n_systems"]
    lines = [
        "",
        f"[weak symplectic] memory and batch size: {info['device']} has "
        f"{gib(info['device_total_memory'])} GiB, {gib(plan['budget'])} GiB usable with "
        f"{plan['headroom']:.0%} headroom; the weights, the gradients, the Adam moments "
        f"and the all-reduce buffer take about {gib(plan['fixed'])} GiB",
        f"  measured on {info['n_systems']} structures ({info['n_atoms']} atoms, "
        f"{atoms_per_system:.0f} per structure, padded to {info['neighbor_width']} "
        f"neighbors); the training has batch_size {B} per device, i.e. about "
        f"{B * atoms_per_system:.0f} atoms, and memory grows linearly with the atoms",
        f"  {'chunk':>5} | {'MiB per':>9} {'peak at':>11} {'fits':>5} {'largest':>8} "
        f"| {'per probe':>9}",
        f"  {'size':>5} | {'structure':>9} {f'batch {B} [GiB]':>11} {'':>5} "
        f"{'batch':>8} | {'[ms]':>9}",
    ]
    for option in plan["options"]:
        marker = "  <- training" if option is plan["current"] else ""
        lines.append(
            f"  {option['n_chunk']:>5} | "
            f"{option['per_system'] / 2**20:>9.0f} {gib(option['peak']):>11} "
            f"{'yes' if option['fits'] else 'no':>5} {option['max_batch_size']:>8} | "
            f"{1000 * option['time_per_probe_s']:>9.1f}{marker}"
        )

    current, best = plan["current"], plan["best"]
    mine = f"chunk_size={plan['chunk_size']}"
    if current is None:
        lines.append(
            f"  the training settings ({mine}) "
            + (
                "do not fit even the benchmark batch."
                if plan["current_oom"]
                else "were not measured."
            )
        )
    elif current["fits"]:
        lines.append(
            f"  the training settings ({mine}, batch_size {B}) are predicted to fit: "
            f"{gib(current['peak'])} of {gib(plan['budget'])} GiB; at most batch_size "
            f"{current['max_batch_size']} with them."
        )
    else:
        lines.append(
            f"  the training settings ({mine}, batch_size {B}) are predicted NOT to fit: "
            f"{gib(current['peak'])} GiB needed, {gib(plan['budget'])} GiB usable; at "
            f"most batch_size {current['max_batch_size']} with them."
        )
    if best is None:
        lines.append("  no chunk size fitted on the benchmark batch.")
    elif best["fits"]:
        if best is current:
            lines.append(
                "  recommendation: keep the training settings, they are the fastest "
                f"that fit (batch_size up to {best['max_batch_size']} would fit)."
            )
        else:
            lines.append(
                f"  recommendation: batch_size {B} fits; the fastest chunk size that "
                f"fits is weak_symplectic_chunk_size={best['n_chunk']} "
                f"({1000 * best['time_per_probe_s']:.0f} ms per probe, batch up to "
                f"{best['max_batch_size']} fits), the training uses {mine}"
                + (
                    f" ({1000 * current['time_per_probe_s']:.0f} ms per probe)."
                    if current is not None
                    else "."
                )
            )
    elif best["max_batch_size"] > 0:
        lines.append(
            f"  recommendation: batch_size {B} does not fit with any measured chunk size; "
            f"use batch_size <= {best['max_batch_size']} per device with "
            f"weak_symplectic_chunk_size={best['n_chunk']} (the fastest, "
            f"{1000 * best['time_per_probe_s']:.0f} ms per probe)."
        )
        widest = max(plan["options"], key=lambda o: o["max_batch_size"])
        if widest["max_batch_size"] > best["max_batch_size"]:
            lines.append(
                f"  for a larger batch: batch_size <= {widest['max_batch_size']} with "
                f"weak_symplectic_chunk_size={widest['n_chunk']} "
                f"({1000 * widest['time_per_probe_s']:.0f} ms per probe)."
            )
    else:
        lines.append(
            "  recommendation: not even one structure fits with the measured chunk "
            "sizes; use a smaller chunk_size, a smaller model or a larger device."
        )
    lines += [
        "  (the effective batch is the number of devices times batch_size; the estimate "
        "is for structures like the benchmark ones)",
        "",
    ]
    print("\n".join(lines), flush=True)


# ------------------------------------------------------------------------------------
# Dense Jacobian and its sparsity
# ------------------------------------------------------------------------------------


def _batched_vjp(
    outputs: Tuple[Tensor, ...],
    inputs: Tuple[Tensor, ...],
    grad_outputs: Tuple[Tensor, ...],
) -> Tuple[Tensor, ...]:
    """
    ``J^T w`` for a batch of ``w`` (first dimension of ``grad_outputs``), detached,
    with zeros for inputs that do not reach the outputs. Uses ``is_grads_batched`` and
    falls back to one backward per ``w`` if the model does not support it.
    """
    try:
        grads = torch.autograd.grad(
            outputs,
            inputs,
            grad_outputs=grad_outputs,
            is_grads_batched=True,
            retain_graph=True,
            allow_unused=True,
        )
    except (RuntimeError, NotImplementedError):
        n = grad_outputs[0].shape[0]
        per_w = [
            torch.autograd.grad(
                outputs,
                inputs,
                grad_outputs=tuple(w[i] for w in grad_outputs),
                retain_graph=True,
                allow_unused=True,
            )
            for i in range(n)
        ]
        grads = tuple(
            None
            if all(g[j] is None for g in per_w)
            else torch.stack(
                [
                    torch.zeros_like(inputs[j]) if g[j] is None else g[j]
                    for g in per_w
                ]
            )
            for j in range(len(inputs))
        )
    n = grad_outputs[0].shape[0]
    return tuple(
        (
            torch.zeros((n, *x.shape), dtype=x.dtype, device=x.device)
            if g is None
            else g.detach()
        )
        for g, x in zip(grads, inputs)
    )


def dense_phase_jacobian(
    q: Tensor,
    p: Tensor,
    q_next: Tensor,
    p_next: Tensor,
    chunk_size: int = 64,
) -> Tensor:
    """
    The dense Jacobian J = d(q_next, p_next) / d(q, p) of the graph of a training forward
    (see :func:`symplectic_forward` and
    :func:`.weak_symplectic_loss.phase_graph_from_predictions`), as a ``[dim, dim]``
    float64 CPU matrix (dim = 6 * n_atoms), rows and columns in the order of
    ``cat([q.flatten(), p.flatten()])``.

    Row i is ``J^T e_i``: one reverse-mode pass per row, ``chunk_size`` rows batched
    together. Nothing is backpropagated into the parameters.
    """
    n_q = q_next.numel()
    dim = n_q + p_next.numel()
    rows: List[Tensor] = []
    for start in range(0, dim, chunk_size):
        n = min(chunk_size, dim - start)
        W = torch.zeros(n, dim, dtype=q_next.dtype, device=q_next.device)
        W[torch.arange(n), start + torch.arange(n)] = 1.0
        Wq = W[:, :n_q].reshape(n, *q_next.shape)
        Wp = W[:, n_q:].reshape(n, *p_next.shape).to(p_next.dtype)
        gq, gp = _batched_vjp((q_next, p_next), (q, p), (Wq, Wp))
        rows.append(
            torch.cat([gq.reshape(n, -1), gp.reshape(n, -1)], dim=1).double().cpu()
        )
    return torch.cat(rows, dim=0)


def symplectic_residual_matrix(J: Tensor) -> Tensor:
    """M = J^T Omega J - Omega, with Omega (q, p) = (p, -q) as in the loss."""
    n = J.shape[0] // 2
    eye = torch.eye(n, dtype=J.dtype)
    zero = torch.zeros(n, n, dtype=J.dtype)
    omega = torch.cat([torch.cat([zero, eye], 1), torch.cat([-eye, zero], 1)], 0)
    return J.T @ omega @ J - omega


def atom_hop_distances(
    systems: List[System], options: NeighborListOptions
) -> Tensor:
    """
    ``[n_atoms, n_atoms]`` shortest-path length between the atoms of the batch in the
    neighbor graph of ``options`` (periodic images map to the same atom). -1: same
    system but not connected; -2: different systems.
    """
    atoms_per_system = [len(system) for system in systems]
    n_atoms = sum(atoms_per_system)
    adjacency = torch.zeros(n_atoms, n_atoms, dtype=torch.bool)
    offset = 0
    for system in systems:
        samples = system.get_neighbor_list(options).samples
        first = samples.column("first_atom").cpu().long() + offset
        second = samples.column("second_atom").cpu().long() + offset
        adjacency[first, second] = True
        offset += len(system)
    adjacency = (adjacency | adjacency.T).double()

    hops = torch.full((n_atoms, n_atoms), -1, dtype=torch.long)
    reached = torch.eye(n_atoms, dtype=torch.bool)
    hops[reached] = 0
    for h in range(1, n_atoms):
        new = reached | ((reached.double() @ adjacency) > 0)
        if torch.equal(new, reached):
            break
        hops[new & ~reached] = h
        reached = new

    system_index = torch.repeat_interleave(
        torch.arange(len(atoms_per_system)), torch.tensor(atoms_per_system)
    )
    hops[system_index[:, None] != system_index[None, :]] = -2
    return hops


def atom_block_norms(matrix: Tensor, n_atoms: int) -> Tensor:
    """Frobenius norm of every 6x6 (q and p, 3 components each) atom-pair block."""
    blocks = matrix.reshape(2, n_atoms, 3, 2, n_atoms, 3).permute(1, 4, 0, 2, 3, 5)
    return blocks.reshape(n_atoms, n_atoms, 36).norm(dim=-1)


# the four 3x3 sub-blocks of an atom-pair block: "qp" is d q_next_i / d p_j for J, and
# the (q_i, p_j) block for M
SUBBLOCKS = ("qq", "qp", "pq", "pp")


def atom_subblock_norms(matrix: Tensor, n_atoms: int) -> Dict[str, Tensor]:
    """Frobenius norm of every 3x3 sub-block (see :data:`SUBBLOCKS`), ``[n, n]`` each."""
    blocks = matrix.reshape(2, n_atoms, 3, 2, n_atoms, 3).permute(0, 3, 1, 4, 2, 5)
    norms = blocks.reshape(2, 2, n_atoms, n_atoms, 9).norm(dim=-1)
    return {name: norms["qp".index(name[0]), "qp".index(name[1])] for name in SUBBLOCKS}


def _blocks_by_hops(block_norms: Tensor, hops: Tensor, rel_tol: float) -> List[Dict]:
    tol = rel_tol * float(block_norms.max())
    rows = []
    for h in sorted(hops.unique().tolist(), key=lambda h: (h < 0, abs(h))):
        values = block_norms[hops == h]
        rows.append(
            {
                "hops": int(h),
                "pairs": int(values.numel()),
                "exact_zero": int((values == 0).sum()),
                "below_tol": int((values <= tol).sum()),
                "max": float(values.max()),
                "mean": float(values.mean()),
            }
        )
    return rows


def receptive_field_hops(model: "MetatrainModel") -> int:
    """
    How many hops of the neighbor graph J can couple (without the long-range
    featurizer, which couples all the atoms of a system). Each GNN layer adds one hop;
    the feedforward featurizer adds one more, as it mixes the reversed edge messages
    (built around the neighbor) into the edge features once more after the last layer.
    """
    core = model.core
    L = int(core.num_gnn_layers)
    return L + 1 if core.featurizer_type == "feedforward" else L


def jacobian_sparsity(
    model: "MetatrainModel",
    systems: List[System],
    raw_predictions: Dict[str, TensorMap],
    *,
    chunk_size: int = 64,
    rel_tol: float = 1e-6,
) -> Dict[str, Any]:
    """
    Sparsity of the dense J and of M = J^T Omega J - Omega on the batch of a training
    forward (call it inside :func:`symplectic_forward`, with its systems and
    predictions). The 6x6 atom-pair blocks are grouped by their hop distance in the
    neighbor graph: without the long-range featurizer, J can only couple atoms up to
    :func:`receptive_field_hops` apart (beyond, its blocks are structurally zero, up to
    the ~1e-15 weight that the attention mask ``log(clamp(cutoff, 1e-15))`` leaves on
    padded slots), and M up to twice that.

    Returns the matrices (CPU, float64), the block norms, the hops and, for J and M,
    one row per hop distance (``exact_zero``: blocks that are exactly 0; ``below_tol``:
    blocks below ``rel_tol`` times the largest block), for the 6x6 blocks (``J_table``)
    and for each 3x3 sub-block of :data:`SUBBLOCKS` (``J_subtables``), and the three
    symplectic identities. ``exact_symplectic_loss`` is |M|_F^2 / dim, the value the
    stochastic loss estimates; since |Omega|_F^2 = dim it is also |M|_F^2 / |Omega|_F^2.
    """
    q, p, q_next, p_next = phase_graph_from_predictions(model, systems, raw_predictions)
    J = dense_phase_jacobian(q, p, q_next, p_next, chunk_size=chunk_size)
    M = symplectic_residual_matrix(J)
    # the three symplectic identities, with J = [[A, B], [C, D]] (A = d q_next / d q,
    # B = d q_next / d p, C = d p_next / d q, D = d p_next / d p) they are the blocks of
    # M: E1 = |A^T C - C^T A|^2 (qq), E2 = |A^T D - C^T B - I|^2 (qp), E3 =
    # |B^T D - D^T B|^2 (pp). M is antisymmetric (its pq block is -qp^T), so
    # |M|_F^2 = E1 + 2 E2 + E3.
    n = J.shape[0] // 2
    identities = {
        "E1": float(M[:n, :n].square().sum()),
        "E2": float(M[:n, n:].square().sum()),
        "E3": float(M[n:, n:].square().sum()),
    }
    core = model.core
    atoms_per_system = [len(system) for system in systems]
    n_atoms = sum(atoms_per_system)
    hops = atom_hop_distances(systems, neighbor_list_options(model))
    J_blocks = atom_block_norms(J, n_atoms)
    M_blocks = atom_block_norms(M, n_atoms)
    J_subblocks = atom_subblock_norms(J, n_atoms)
    M_subblocks = atom_subblock_norms(M, n_atoms)
    return {
        "J": J,
        "M": M,
        "hops": hops,
        "J_blocks": J_blocks,
        "M_blocks": M_blocks,
        "J_table": _blocks_by_hops(J_blocks, hops, rel_tol),
        "M_table": _blocks_by_hops(M_blocks, hops, rel_tol),
        "J_subtables": {
            name: _blocks_by_hops(norms, hops, rel_tol)
            for name, norms in J_subblocks.items()
        },
        "M_subtables": {
            name: _blocks_by_hops(norms, hops, rel_tol)
            for name, norms in M_subblocks.items()
        },
        "symplectic_identities": identities,
        "receptive_field_hops": receptive_field_hops(model),
        "featurizer_type": core.featurizer_type,
        "long_range": bool(core.long_range),
        "J_zero_fraction": float((J == 0).double().mean()),
        "M_zero_fraction": float((M == 0).double().mean()),
        "exact_symplectic_loss": float(M.square().sum()) / J.shape[0],
        "n_atoms": n_atoms,
        "atoms_per_system": atoms_per_system,
        "dim": J.shape[0],
        "num_gnn_layers": int(core.num_gnn_layers),
        "cutoff": float(core.cutoff),
        "rel_tol": rel_tol,
    }


def print_jacobian_sparsity(
    model: "MetatrainModel",
    systems: List[System],
    raw_predictions: Dict[str, TensorMap],
    *,
    max_dim: int = 1500,
    max_print_atoms: int = 24,
    chunk_size: int = 64,
    verbose: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    Compute and print :func:`jacobian_sparsity` (and the atom-block matrices if there
    are at most ``max_print_atoms`` atoms); returns the same dict. Skipped (None) if
    dim = 6 * n_atoms > ``max_dim`` (it costs dim vector-Jacobian products). With
    ``verbose=False`` nothing is printed (the dict is only computed).
    """
    import numpy as np

    dim = 6 * sum(len(system) for system in systems)
    if dim > max_dim:
        if verbose:
            print(
                f"\n[weak symplectic] dense Jacobian skipped: dim = {dim} > max_dim = "
                f"{max_dim}\n",
                flush=True,
            )
        return None

    info = jacobian_sparsity(model, systems, raw_predictions, chunk_size=chunk_size)
    if not verbose:
        return info
    L = info["num_gnn_layers"]
    reach = info["receptive_field_hops"]
    labels = {-1: "inf", -2: "other sys"}
    lines = [
        "",
        "[weak symplectic] dense Jacobian of (q, p) -> (q_next, p_next)",
        f"  atoms {info['n_atoms']} {info['atoms_per_system']}, dim {info['dim']}, "
        f"num_gnn_layers {L}, featurizer {info['featurizer_type']} "
        f"(receptive field {reach} hops), cutoff {info['cutoff']}",
        f"  exact symplectic loss |J^T Omega J - Omega|_F^2 / dim = "
        f"{info['exact_symplectic_loss']:.6e} (= relative to |Omega|_F^2 / dim = 1)",
    ]
    if info["long_range"]:
        lines.append(
            "  long-range featurizer: J couples all the atoms of a system, the hop "
            "distance only tells how the short-range part decays"
        )
    E = info["symplectic_identities"]
    total = E["E1"] + 2 * E["E2"] + E["E3"]
    lines += [
        "  symplectic identities, J = [[A, B], [C, D]] "
        f"(E1 + 2 E2 + E3) / dim = {total / info['dim']:.6e}",
        f"    E1 = |A^T C - C^T A|_F^2     = {E['E1']:.6e}  "
        f"({E['E1'] / max(total, 1e-300):6.1%} of the loss)",
        f"    E2 = |A^T D - C^T B - I|_F^2 = {E['E2']:.6e}  "
        f"({2 * E['E2'] / max(total, 1e-300):6.1%} of the loss, counted twice)",
        f"    E3 = |B^T D - D^T B|_F^2     = {E['E3']:.6e}  "
        f"({E['E3'] / max(total, 1e-300):6.1%} of the loss)",
        f"  exactly-zero entries: J {info['J_zero_fraction']:.1%}, "
        f"M {info['M_zero_fraction']:.1%}",
    ]
    for name, table, subtables, support in (
        ("J", info["J_table"], info["J_subtables"], reach),
        ("M = J^T Omega J - Omega", info["M_table"], info["M_subtables"], 2 * reach),
    ):
        lines.append(
            f"  {name}: 6x6 atom blocks by hop distance"
            + (
                ""
                if info["long_range"]
                else f" (structurally zero beyond {support} hops, up to a ~1e-15 leak "
                "through padded attention slots)"
            )
        )
        lines.append(
            f"    {'hops':>9} {'pairs':>7} {'exact 0':>8} "
            f"{'<=tol':>7} {'max |B|_F':>11} {'mean |B|_F':>11}"
        )
        for row in table:
            hop = labels.get(row["hops"], str(row["hops"]))
            lines.append(
                f"    {hop:>9} {row['pairs']:>7} {row['exact_zero']:>8} "
                f"{row['below_tol']:>7} {row['max']:>11.3e} {row['mean']:>11.3e}"
            )
        lines.append(
            "    3x3 sub-blocks, exact zeros / max |B|_F "
            "(qp: rows q_next or q, columns p)"
        )
        lines.append(
            f"    {'hops':>9} {'pairs':>7} " + " ".join(f"{sub:>18}" for sub in SUBBLOCKS)
        )
        for i, row in enumerate(table):
            hop = labels.get(row["hops"], str(row["hops"]))
            cells = " ".join(
                f"{subtables[sub][i]['exact_zero']:>8} {subtables[sub][i]['max']:>9.2e}"
                for sub in SUBBLOCKS
            )
            lines.append(f"    {hop:>9} {row['pairs']:>7} {cells}")
    if info["n_atoms"] <= max_print_atoms:

        def matrix(values: Tensor) -> str:
            return np.array2string(
                values.numpy(),
                max_line_width=1000,
                threshold=10**6,
                formatter={"float_kind": lambda x: f"{x:8.1e}"},  # type: ignore
            )

        lines.append("  hop distances (-1: not connected, -2: other system):")
        lines.append(matrix(info["hops"]))
        lines.append("  |J| atom blocks:")
        lines.append(matrix(info["J_blocks"]))
        lines.append("  |M| atom blocks:")
        lines.append(matrix(info["M_blocks"]))
    print("\n".join(lines) + "\n", flush=True)
    return info


# ------------------------------------------------------------------------------------
# Scales of the normalization: species, masses, data space
# ------------------------------------------------------------------------------------


def species_scale_table(model: "MetatrainModel", like: Tensor) -> List[Dict[str, Any]]:
    """
    One row per atomic type of the model: its mass (as registered in the model), the
    scales sigma_q and sigma_p of the normalization, and the mass-reduced combinations

    - ``sigma_q_sqrt_m`` = sigma_q * sqrt(m) and ``sigma_p_over_sqrt_m`` = sigma_p /
      sqrt(m): with momenta predicted directly and ballistic motion over the time step,
      equipartition gives sigma_p ~ sqrt(m k T) and sigma_q ~ dt sqrt(k T / m), so both
      are about the same for every species;
    - ``lambda_over_c`` = sigma_q sigma_p / c: Omega~ / c is the canonical Omega only
      if it is 1 for every species (otherwise the transformed form is used, which is
      exact either way).

    The masses never enter the normalization explicitly: it only depends on species
    (and mass) through the per-type scales fitted by the scaler. With
    ``fixed_scaling_weights`` given as one float per target, the scales are the same
    for every species and carry no mass dependence.

    :param model: The ``MetatrainModel`` wrapper.
    :param like: A tensor with the dtype and device to compute the scales with.
    """
    by_type = scales_by_atomic_type(model, like)
    c = sum(sq * sp for sq, sp in by_type.values()) / max(len(by_type), 1)
    rows = []
    for atomic_type, (sq, sp) in by_type.items():
        mass = float(model.core.masses[atomic_type])
        root = math.sqrt(mass) if mass > 0 else math.nan
        rows.append(
            {
                "atomic_type": atomic_type,
                "mass": mass,
                "sigma_q": sq,
                "sigma_p": sp,
                "sigma_q_sqrt_m": sq * root,
                "sigma_p_over_sqrt_m": sp / root,
                "lambda_over_c": sq * sp / c,
            }
        )
    return rows


def species_scaling_warning(rows: List[Dict[str, Any]]) -> Optional[str]:
    """A warning if the scales do not depend on the species although the masses do
    (e.g. ``fixed_scaling_weights`` as one float), None otherwise."""
    if len(rows) < 2:
        return None
    masses = [r["mass"] for r in rows]
    same_scales = all(
        math.isclose(r["sigma_q"], rows[0]["sigma_q"], rel_tol=1e-9)
        and math.isclose(r["sigma_p"], rows[0]["sigma_p"], rel_tol=1e-9)
        for r in rows
    )
    if same_scales and max(masses) > 1.001 * min(masses):
        return (
            "the position and momentum scales are the same for every species although "
            "the masses differ: the normalization has no species or mass dependence "
            "(probably fixed_scaling_weights given as one float per target; leave them "
            "out to fit per-type scales, or give a dict per atomic type)"
        )
    return None


def data_space_predictions(
    model: "MetatrainModel",
    systems: List[System],
    raw_predictions: Dict[str, TensorMap],
) -> Tuple[Tensor, Tensor]:
    """
    The ``position`` and ``momentum`` predictions ``[n_atoms, 3]`` exactly as the data
    loss of the trainer sees them: the raw outputs of the training forward with the
    per-property scales applied (they are compared there with the targets divided by
    the per-target scales). In the coordinates of the normalized loss they must be
    ``(q_next - q) / sigma_q`` and ``(p_next - p) / sigma_p`` (``p_next / sigma_p``
    without ``predict_momenta_as_difference``), which the diagnostics check.
    """
    phase = {name: raw_predictions[name] for name in ("position", "momentum")}
    if model.scaler is not None:
        phase = model.scaler.apply_scales(
            systems,
            phase,
            remove=False,
            use_per_target_scales=False,
            use_per_property_scales=True,
        )
    return tuple(  # type: ignore[return-value]
        phase[name].block().values.reshape(-1, 3) for name in ("position", "momentum")
    )


# ------------------------------------------------------------------------------------
# Check of the scaling chain of q and p (data -> model -> weak loss)
# ------------------------------------------------------------------------------------
#
# How q and p are scaled and rescaled, from the data to the weak symplectic loss:
#
#  1. dataset: ``position`` / ``momentum`` targets are the future (absolute) positions
#     and momenta q', p' (physical units); the current q is ``system.positions`` and
#     the current p the ``momentum`` data of the system (physical units).
#  2. training collate: random O(3) rotation of the systems (positions, cell and the
#     ``momentum`` data) and of the targets; then the PositionAdditive model is
#     subtracted from the targets (q' - q, and p' - p with
#     ``predict_momenta_as_difference``); then the targets are divided by the
#     per-target scales of the scaler (one per atomic type): the data loss works on
#     (q' - q) / sigma_q and p' / sigma_p (or (p' - p) / sigma_p).
#  3. training forward: the inputs are the *physical* q (through the edge vectors) and
#     p (through the node encoder), not scaled; the outputs are raw, i.e. in the scaled
#     space of step 2 (times the per-property scales, 1 for these targets).
#  4. weak loss: the raw outputs are multiplied by the full scales and the additive
#     part is added back, q_next = q + sigma_q * out_q (p_next likewise), which is the
#     physical map (q, p) -> (q', p') that the exported model applies at evaluation
#     (scaler then additive models); its Jacobian is taken with respect to the
#     physical leaves (q, p), and the normalization D = diag(sigma_q, sigma_p) brings
#     it back to the coordinates of step 2.
#
# :func:`check_scaling_pipeline` verifies every arrow numerically.


def _rows3(tensor_map: TensorMap) -> Tensor:
    return tensor_map.block().values.reshape(-1, 3)


def _momentum_data(system: System) -> Tensor:
    return system.get_data("momentum").block().values.reshape(-1, 3)


def check_scaling_pipeline(
    model: "MetatrainModel",
    make_raw_batch: Callable[[], Tuple[List[System], Dict[str, TensorMap]]],
    make_batch: Callable[[], Tuple[List[System], Dict[str, TensorMap]]],
    *,
    augmenter: Optional[Any] = None,
    seed: int = 0,
) -> Dict[str, Any]:
    """
    Numerical check of the scaling chain of q and p (see the comment above this
    function), on one batch:

    - ``targets_round_trip``: the targets of the training collate, times the scales D
      of the normalization, plus the additive part (q, and p with
      ``predict_momenta_as_difference``), give back the targets of the dataset: D is
      what the data was divided by, and the additive convention is q' = q + delta_q;
    - ``target_magnitudes``: rms of q, of q', of q' - q (and of p, p', p' - p). If
      |q' - q| is comparable to |q - mean(q)|, the ``position`` targets of the dataset
      are probably displacements already rather than future positions (the additive
      model would then subtract q once too often);
    - ``leaves``: the leaves (q, p) of the graph of the training forward are the
      positions and the ``momentum`` data of the systems (physical, not scaled);
    - ``train_vs_eval``: the physical map of the weak loss (raw outputs times the full
      scales, plus the additive part) against the outputs of the model in evaluation
      mode (the wrapper applies the scaler and the additive models itself), i.e. the
      map the exported model applies in MD. They differ only by the attention
      implementation (manual in training, fused at evaluation), and, with the
      long-range featurizer, by Ewald (training) against P3M (evaluation);
    - ``rotation`` (with ``augmenter``): a fixed rotation applied by the augmenter of the
      training must rotate the positions, the ``momentum`` data of the systems and both
      targets the same way. If the input momenta were not rotated, the model would see
      (R q, p) and the symplectic loss a map whose input frame is inconsistent.

    :param make_raw_batch: ``() -> (systems, targets)`` of the dataset, untransformed.
    :param make_batch: ``() -> (systems, targets)`` of the same structures through the
        validation collate (neighbor lists, additive and scale removed, no rotation).
    :param augmenter: the ``O3Augmenter`` of the training, for the rotation check.
    :return: the measured differences and ``passed``.
    """
    from metatrain.utils.evaluate_model import evaluate_model

    core = model.core
    as_difference = bool(core.hypers["predict_momenta_as_difference"])
    raw_systems, raw_targets = make_raw_batch()
    systems, targets = make_batch()
    q = torch.cat([system.positions for system in systems]).detach()
    p = torch.cat([_momentum_data(system) for system in systems]).detach().to(q.dtype)
    scales = phase_space_scales(model, systems, q)
    tolerance = 1e-4 if q.dtype == torch.float32 else 1e-9

    def max_rel_diff(a: Tensor, b: Tensor) -> float:
        a = a.detach().double().cpu()
        b = b.detach().double().cpu()
        return float((a - b).abs().max()) / max(float(b.abs().max()), 1e-300)

    def rms(x: Tensor) -> float:
        return float(x.detach().double().square().mean().sqrt())

    # --- 1-2: the targets, back to physical units
    q_future = _rows3(raw_targets["position"]).to(q)
    p_future = _rows3(raw_targets["momentum"]).to(q)
    # compared as displacements: relative to the absolute positions, an error in
    # q' - q would be invisible
    dq_back = _rows3(targets["position"]).to(q) * scales.q
    p_back = _rows3(targets["momentum"]).to(q) * scales.p
    if as_difference:
        dp_back, dp_future = p_back, p_future - p
    else:
        dp_back, dp_future = p_back, p_future
    round_trip = {
        "position": max_rel_diff(dq_back, q_future - q),
        "momentum": max_rel_diff(dp_back, dp_future),
    }
    magnitudes = {
        "q_centered": rms(q - q.mean(dim=0)),
        "q_target": rms(q_future),
        "q_target_minus_q": rms(q_future - q),
        "p": rms(p),
        "p_target": rms(p_future),
        "p_target_minus_p": rms(p_future - p),
    }
    target_warning = None
    if magnitudes["q_target_minus_q"] > 0.5 * magnitudes["q_centered"]:
        target_warning = (
            f"rms(q' - q) = {magnitudes['q_target_minus_q']:.3g} is comparable to the "
            f"spread of the positions ({magnitudes['q_centered']:.3g}): the "
            "'position' targets look like displacements, not future positions (or "
            "the positions wrap through the cell)"
        )

    # --- 3-4: the training forward, its leaves and its physical map
    targets_info = {name: model.dataset_info.targets[name] for name in targets}
    with symplectic_forward(model, lambda: make_batch()[0]) as (fw_systems, raw):
        q_leaf, p_leaf, q_next, p_next = phase_graph_from_predictions(
            model, fw_systems, raw
        )
        # against an independent collate of the same structures (q, p above)
        leaves = {
            "q_is_positions": max_rel_diff(q_leaf, q),
            "p_is_momentum_data": max_rel_diff(p_leaf, p),
            "q_is_leaf": bool(q_leaf.is_leaf and q_leaf.requires_grad),
            "p_is_leaf": bool(p_leaf.is_leaf and p_leaf.requires_grad),
        }
        dq_weak = (q_next - q_leaf).detach()
        p_weak = (p_next - p_leaf if as_difference else p_next).detach()

    was_training = model.training
    try:
        model.eval()
        with torch.no_grad():
            evaluated = evaluate_model(
                model, make_batch()[0], targets_info, is_training=False
            )
    finally:
        model.train(was_training)
    # as displacements again (and p' - p with predict_momenta_as_difference)
    p_eval = _rows3(evaluated["momentum"]).to(p_weak)
    train_vs_eval = {
        "position": max_rel_diff(dq_weak, _rows3(evaluated["position"]).to(q) - q),
        "momentum": max_rel_diff(p_weak, p_eval - p if as_difference else p_eval),
    }
    eval_tolerance = 1e-3 if q.dtype == torch.float32 else 1e-7

    # --- the rotation of the augmentation
    rotation: Optional[Dict[str, float]] = None
    if augmenter is not None:
        generator = torch.Generator().manual_seed(seed)
        matrix, _ = torch.linalg.qr(
            torch.randn(3, 3, generator=generator, dtype=torch.float64)
        )
        if torch.det(matrix) < 0:
            matrix[:, 0] = -matrix[:, 0]
        r_systems, r_targets = make_raw_batch()
        matrix = matrix.to(dtype=r_systems[0].positions.dtype)
        new_systems, new_targets, _ = augmenter.apply_augmentations(
            r_systems, r_targets, [matrix] * len(r_systems), extra_data={}
        )
        old_q = torch.cat([s.positions for s in r_systems]).detach()
        new_q = torch.cat([s.positions for s in new_systems]).detach()
        # the linear map the augmenter applied to the positions (row vectors), whatever
        # its convention, then the same map must hold for the other vectors
        applied = torch.linalg.lstsq(old_q.double(), new_q.double()).solution
        old_p = torch.cat([_momentum_data(s) for s in r_systems]).double()
        new_p = torch.cat([_momentum_data(s) for s in new_systems]).double()
        rotation = {
            "positions_fit": max_rel_diff(old_q.double() @ applied, new_q),
            "is_rotation": float(
                (applied.T @ applied - torch.eye(3, dtype=applied.dtype)).abs().max()
            ),
            "momentum_data": max_rel_diff(old_p @ applied, new_p),
            "momentum_data_unchanged": max_rel_diff(new_p, old_p),
            "position_target": max_rel_diff(
                _rows3(r_targets["position"]).double() @ applied,
                _rows3(new_targets["position"]),
            ),
            "momentum_target": max_rel_diff(
                _rows3(r_targets["momentum"]).double() @ applied,
                _rows3(new_targets["momentum"]),
            ),
        }

    passed = (
        max(round_trip.values()) < tolerance
        and leaves["q_is_positions"] == 0.0
        and leaves["p_is_momentum_data"] == 0.0
        and leaves["q_is_leaf"]
        and leaves["p_is_leaf"]
        and (bool(core.long_range) or max(train_vs_eval.values()) < eval_tolerance)
        and (
            rotation is None
            or max(
                rotation["momentum_data"],
                rotation["position_target"],
                rotation["momentum_target"],
            )
            < 1e-6
        )
    )
    return {
        "n_atoms": int(q.shape[0]),
        "dtype": str(q.dtype),
        "predict_momenta_as_difference": as_difference,
        "long_range": bool(core.long_range),
        "targets_round_trip": round_trip,
        "target_magnitudes": magnitudes,
        "target_warning": target_warning,
        "leaves": leaves,
        "train_vs_eval": train_vs_eval,
        "rotation": rotation,
        "tolerance": tolerance,
        "eval_tolerance": eval_tolerance,
        "passed": bool(passed),
    }


def print_scaling_pipeline(info: Dict[str, Any]) -> None:
    """Print :func:`check_scaling_pipeline`."""
    rt, mag, lv, te = (
        info["targets_round_trip"],
        info["target_magnitudes"],
        info["leaves"],
        info["train_vs_eval"],
    )
    p_target = "p' - p" if info["predict_momenta_as_difference"] else "p'"
    lines = [
        "",
        f"[weak symplectic] scaling chain of q and p ({info['n_atoms']} atoms, "
        f"{info['dtype']}): "
        + ("PASSED" if info["passed"] else "FAILED")
        + (", with a WARNING (see below)" if info["target_warning"] else ""),
        f"  data: targets (q' - q) / sigma_q and ({p_target}) / sigma_p, back to "
        f"physical units with D, vs the dataset's q' - q and {p_target}: max rel. diff "
        f"position {rt['position']:.1e}, momentum {rt['momentum']:.1e} (tolerance "
        f"{info['tolerance']:.0e})",
        f"  magnitudes (rms): q - mean {mag['q_centered']:.4g}, q' - q "
        f"{mag['q_target_minus_q']:.4g}; p {mag['p']:.4g}, p' {mag['p_target']:.4g}, "
        f"p' - p {mag['p_target_minus_p']:.4g}",
        f"  model inputs: q leaf = positions (rel. diff {lv['q_is_positions']:.1e}), "
        f"p leaf = momentum data ({lv['p_is_momentum_data']:.1e}), both leaves that "
        f"require grad: {lv['q_is_leaf'] and lv['p_is_leaf']} (physical, not scaled)",
        f"  map of the weak loss (raw * full scales + additive) vs the model in "
        f"evaluation mode, as q' - q and {p_target}: max rel. diff position "
        f"{te['position']:.1e}, momentum "
        f"{te['momentum']:.1e} (tolerance {info['eval_tolerance']:.0e}"
        + (
            "; not enforced with the long-range featurizer: Ewald vs P3M)"
            if info["long_range"]
            else "; manual vs fused attention)"
        ),
    ]
    rotation = info["rotation"]
    if rotation is not None:
        lines.append(
            f"  augmentation, a fixed rotation (positions fit {rotation['positions_fit']:.1e}, "
            f"orthogonal to {rotation['is_rotation']:.1e}): momentum data "
            f"{rotation['momentum_data']:.1e}, position target "
            f"{rotation['position_target']:.1e}, momentum target "
            f"{rotation['momentum_target']:.1e}"
        )
        if rotation["momentum_data"] > 1e-6 and rotation["momentum_data_unchanged"] < 1e-12:
            lines.append(
                "  ERROR: the augmentation rotates the positions but not the momentum "
                "data of the systems: the inputs (R q, p) are inconsistent"
            )
    if info["target_warning"]:
        lines.append(f"  WARNING: {info['target_warning']}")
    lines.append("")
    print("\n".join(lines), flush=True)


# ------------------------------------------------------------------------------------
# Sanity check of the normalized loss
# ------------------------------------------------------------------------------------


def _relative_difference(a: float, b: float) -> float:
    return abs(a - b) / max(abs(a), abs(b), 1e-300)


def _omega(n: int, dtype: torch.dtype = torch.float64) -> Tensor:
    eye = torch.eye(n, dtype=dtype)
    zero = torch.zeros(n, n, dtype=dtype)
    return torch.cat([torch.cat([zero, eye], 1), torch.cat([-eye, zero], 1)], 0)


def _block_fractions(matrix: Tensor) -> Dict[str, float]:
    """The shares of the qq, qp (counted twice) and pp blocks in |matrix|_F^2."""
    n = matrix.shape[0] // 2
    E1 = float(matrix[:n, :n].square().sum())
    E2 = float(matrix[:n, n:].square().sum())
    E3 = float(matrix[n:, n:].square().sum())
    total = max(E1 + 2 * E2 + E3, 1e-300)
    return {"qq": E1 / total, "qp": 2 * E2 / total, "pp": E3 / total}


def check_normalization(
    model: "MetatrainModel",
    systems: List[System],
    raw_predictions: Dict[str, TensorMap],
    *,
    n_probes: int = 1024,
    probe_distribution: ProbeDistribution = "rademacher",
    chunk_size: int = 2,
    jacobian_chunk_size: int = 64,
    seed: int = 0,
) -> Dict[str, Any]:
    """
    Sanity check of the normalized ("data_scales") symplectic loss on the batch of a
    training forward (call it inside :func:`symplectic_forward`). With J the dense
    Jacobian of the batch, M = J^T Omega J - Omega, D = diag(Sigma_q, Sigma_p) and c the
    reference action (:func:`.weak_symplectic_loss.phase_space_scales`), it computes the
    exact normalized loss L~ = |D M D|_F^2 / (c^2 dim) in independent ways that must
    agree:

    - ``exact_direct``: |D M D|_F^2 / (c^2 dim), from the dense M (float64);
    - ``exact_transformed``: the residual in the normalized coordinates themselves,
      |J~^T Omega~ J~ - Omega~|_F^2 / dim with J~ = D^-1 J D and Omega~ = D Omega D / c
      (Theorem 3.1 of the proof: it is D M D / c entry by entry, so the difference is
      round-off, ``transformed_max_abs_diff``);
    - ``exact_loss_code``: the loss code of the training (double backward through the
      model, normalized probes v~ = the canonical basis, v = D v~, residual times D / c),
      which checks the implementation, in the precision of the model;
    - ``stochastic_mean`` +- ``stochastic_stderr``: the training estimator with
      ``n_probes`` random normalized probes, which must be within a few standard errors
      of the exact value (its ``z_score``).

    It also checks that the normalized loss lives in the space of the data loss and
    handles the species:

    - ``data_space_max_rel_diff``: the phase-space map in the normalized coordinates,
      ``(q_next - q) / sigma_q`` and ``(p_next - p) / sigma_p`` (``p_next / sigma_p``
      without ``predict_momenta_as_difference``), against the predictions exactly as the
      data loss compares them with its targets
      (:func:`data_space_predictions`): they must be the same
      numbers, i.e. both losses measure the same coordinates;
    - ``species_max_rel_diff``: every atom's entries of D against the scales of its
      atomic type (no atom gets the scale of another species);
    - ``species``: per atomic type, the mass registered in the model, sigma_q, sigma_p,
      sigma_q sqrt(m), sigma_p / sqrt(m) and Lambda / c
      (:func:`species_scale_table`), and ``species_warning`` if
      the scales carry no species (mass) dependence although the masses differ.

    For context it also gives the raw loss (``raw_exact``, and ``raw_loss_code`` from the
    loss code without normalization), the share of the qq, qp and pp blocks of M and of
    D M D (the balance the normalization is about), the q-row / p-row split of both, the
    spread of Lambda / c = sigma_q sigma_p / c over the atoms (Omega~ / c is the
    canonical Omega only if it is 1 everywhere, Proposition 3.2), and
    ``canonical_omega_loss``: |J~^T Omega J~ - Omega|_F^2 / dim, what the (wrong) loss
    with the canonical Omega in the normalized coordinates would give (Proposition
    3.3). ``passed`` tells whether the first three agree within the tolerance of the
    dtype, the stochastic estimate is within 4 standard errors, and the data-space and
    species checks agree within the tolerance.
    """
    q, p, q_next, p_next = phase_graph_from_predictions(model, systems, raw_predictions)
    scales = phase_space_scales(model, systems, q)
    dim = q.numel() + p.numel()
    n = dim // 2
    c = float(scales.action)

    def max_rel_diff(a: Tensor, b: Tensor) -> float:
        a = a.detach().double().cpu()
        b = b.detach().double().cpu()
        return float((a - b).abs().max()) / max(float(b.abs().max()), 1e-300)

    # the same coordinates as the data loss
    with torch.no_grad():
        data_q, data_p = data_space_predictions(model, systems, raw_predictions)
        normalized_dq = (q_next - q) / scales.q
        if model.core.hypers["predict_momenta_as_difference"]:
            normalized_dp = (p_next - p) / scales.p
        else:
            normalized_dp = p_next / scales.p
    data_space_max_rel_diff = max(
        max_rel_diff(normalized_dq, data_q), max_rel_diff(normalized_dp, data_p)
    )

    # every atom gets the scales of its own species
    types = torch.cat([system.types for system in systems]).cpu()
    species_rows = species_scale_table(model, q)
    species_max_rel_diff = 0.0
    for row in species_rows:
        atoms = types == row["atomic_type"]
        if not bool(atoms.any()):
            continue
        for values, expected in (
            (scales.q[atoms.to(scales.q.device)], row["sigma_q"]),
            (scales.p[atoms.to(scales.p.device)], row["sigma_p"]),
        ):
            species_max_rel_diff = max(
                species_max_rel_diff,
                float((values.double() - expected).abs().max()) / max(abs(expected), 1e-300),
            )
    present_types = sorted({int(t) for t in types.tolist()})
    d = torch.cat([scales.q.reshape(-1), scales.p.reshape(-1)]).detach().double().cpu()

    J = dense_phase_jacobian(q, p, q_next, p_next, chunk_size=jacobian_chunk_size)
    M = symplectic_residual_matrix(J)
    DMD = d[:, None] * M * d[None, :] / c
    exact_direct = float(DMD.square().sum()) / dim
    raw_exact = float(M.square().sum()) / dim

    # the same in the normalized coordinates, with the transformed symplectic form
    omega = _omega(n)
    J_tilde = J * d[None, :] / d[:, None]
    omega_tilde = d[:, None] * omega * d[None, :] / c
    R_tilde = J_tilde.T @ omega_tilde @ J_tilde - omega_tilde
    exact_transformed = float(R_tilde.square().sum()) / dim
    canonical_omega_loss = float(
        (J_tilde.T @ omega @ J_tilde - omega).square().sum()
    ) / dim

    # the loss code: dense (basis) probes, normalized and raw
    def basis_loss(with_scales: bool) -> Tuple[float, Tensor]:
        total = 0.0
        parts = torch.zeros(2, dtype=torch.float64)
        for losses, chunk_parts in calc_symplectic_probe_loss_chunks(
            q,
            p,
            q_next,
            p_next,
            n_probes=dim,
            probe_dist=probe_distribution,
            chunk_size=max(chunk_size, 1),
            probe_fn=basis_probe_fn(q, p),
            with_parts=True,
            scales=scales if with_scales else None,
        ):
            total += float(losses.detach().double().sum())
            parts += chunk_parts.double().cpu().sum(dim=0)
        return total, parts

    exact_loss_code, normalized_parts = basis_loss(True)
    raw_loss_code, raw_parts = basis_loss(False)

    # the stochastic estimator of the training, with normalized probes
    torch.manual_seed(_probe_seed(seed, 0, "normalization", n_probes))
    samples: List[Tensor] = []
    for losses in calc_symplectic_probe_loss_chunks(
        q,
        p,
        q_next,
        p_next,
        n_probes=n_probes,
        probe_dist=probe_distribution,
        chunk_size=max(chunk_size, 1),
        scales=scales,
    ):
        samples.append(losses.detach().double().cpu())
    stochastic = torch.cat(samples)
    stochastic_mean = float(stochastic.mean())
    stochastic_stderr = (
        float(stochastic.std(unbiased=True)) / math.sqrt(n_probes)
        if n_probes > 1
        else math.nan
    )
    z_score = (
        (stochastic_mean - exact_direct) / stochastic_stderr
        if stochastic_stderr > 0
        else (0.0 if stochastic_mean == exact_direct else math.inf)
    )

    lam = (d[:n] * d[n:]) / c  # Lambda / c per degree of freedom
    tolerance = 1e-3 if q.dtype == torch.float32 else 1e-8
    checks = {
        "direct_vs_transformed": _relative_difference(exact_direct, exact_transformed),
        "direct_vs_loss_code": _relative_difference(exact_direct, exact_loss_code),
        "raw_vs_loss_code": _relative_difference(raw_exact, raw_loss_code),
        "data_space": data_space_max_rel_diff,
        "species": species_max_rel_diff,
    }
    passed = (
        checks["direct_vs_transformed"] < 1e-10
        and checks["direct_vs_loss_code"] < tolerance
        and checks["raw_vs_loss_code"] < tolerance
        and checks["data_space"] < tolerance
        and checks["species"] < tolerance
        and (abs(z_score) <= 4.0 or exact_direct == 0.0)
    )

    def rows_split(parts: Tensor) -> Dict[str, float]:
        total = max(float(parts.sum()), 1e-300)
        return {"q_rows": float(parts[0]) / total, "p_rows": float(parts[1]) / total}

    by_type = scales_by_atomic_type(model, q)
    return {
        "n_atoms": int(q.shape[0]),
        "atomic_types_in_batch": present_types,
        "data_space_max_rel_diff": data_space_max_rel_diff,
        "species_max_rel_diff": species_max_rel_diff,
        "species": species_rows,
        "species_warning": species_scaling_warning(species_rows),
        "dim": int(dim),
        "dtype": str(q.dtype),
        "reference_action": c,
        "scales_by_type": {str(t): list(v) for t, v in by_type.items()},
        "lambda_over_c_min": float(lam.min()),
        "lambda_over_c_max": float(lam.max()),
        "raw_exact": raw_exact,
        "raw_loss_code": raw_loss_code,
        "exact_direct": exact_direct,
        "exact_transformed": exact_transformed,
        "transformed_max_abs_diff": float((R_tilde - DMD).abs().max()),
        "exact_loss_code": exact_loss_code,
        "stochastic_mean": stochastic_mean,
        "stochastic_stderr": stochastic_stderr,
        "stochastic_n_probes": int(n_probes),
        "z_score": z_score,
        "canonical_omega_loss": canonical_omega_loss,
        "raw_blocks": _block_fractions(M),
        "normalized_blocks": _block_fractions(DMD),
        "raw_rows": rows_split(raw_parts),
        "normalized_rows": rows_split(normalized_parts),
        "relative_differences": checks,
        "tolerance": tolerance,
        "passed": bool(passed),
    }


def print_normalization_check(info: Dict[str, Any]) -> None:
    """Print :func:`check_normalization`."""

    def pct(split: Dict[str, float]) -> str:
        return ", ".join(f"{k} {v:.1%}" for k, v in split.items())

    rel = info["relative_differences"]
    lines = [
        "",
        f"[weak symplectic] normalization check ({info['n_atoms']} atoms, dim "
        f"{info['dim']}, {info['dtype']}): "
        + ("PASSED" if info["passed"] else "FAILED"),
        "  scales (sigma_q, sigma_p) per atomic type: "
        + ", ".join(
            f"{t}: ({v[0]:.4g}, {v[1]:.4g})" for t, v in info["scales_by_type"].items()
        )
        + f"; c = {info['reference_action']:.4g}; Lambda / c in "
        f"[{info['lambda_over_c_min']:.4g}, {info['lambda_over_c_max']:.4g}]"
        + (
            " (= 1: Omega~/c is the canonical Omega)"
            if abs(info["lambda_over_c_min"] - 1) < 1e-9
            and abs(info["lambda_over_c_max"] - 1) < 1e-9
            else " (not 1 everywhere: Omega~/c is not the canonical Omega)"
        ),
        "  normalized loss |D M D|^2 / (c^2 dim):",
        f"    dense M, direct                     {info['exact_direct']:.8e}",
        f"    dense, normalized coordinates (J~)  {info['exact_transformed']:.8e}  "
        f"(rel. diff {rel['direct_vs_transformed']:.1e}, max |R~ - DMD/c| "
        f"{info['transformed_max_abs_diff']:.1e})",
        f"    loss code, basis probes             {info['exact_loss_code']:.8e}  "
        f"(rel. diff {rel['direct_vs_loss_code']:.1e}, tolerance "
        f"{info['tolerance']:.0e})",
        f"    loss code, {info['stochastic_n_probes']} random probes"
        f"{'':>{max(0, 11 - len(str(info['stochastic_n_probes'])))}}"
        f"{info['stochastic_mean']:.8e} +- {info['stochastic_stderr']:.2e}  "
        f"(z = {info['z_score']:+.2f})",
        f"  same space as the data loss: max rel. diff between (q_next - q) / sigma_q, "
        f"(p_next [- p]) / sigma_p and the data-loss predictions "
        f"{rel['data_space']:.1e}",
        f"  species: atomic types in the batch {info['atomic_types_in_batch']}, max rel. "
        f"diff of each atom's D to its type's scales {rel['species']:.1e}",
        f"    {'type':>6} {'mass':>9} {'sigma_q':>10} {'sigma_p':>10} "
        f"{'sigma_q*sqrt(m)':>16} {'sigma_p/sqrt(m)':>16} {'Lambda/c':>9}",
        *(
            f"    {r['atomic_type']:>6} {r['mass']:>9.4g} {r['sigma_q']:>10.4g} "
            f"{r['sigma_p']:>10.4g} {r['sigma_q_sqrt_m']:>16.4g} "
            f"{r['sigma_p_over_sqrt_m']:>16.4g} {r['lambda_over_c']:>9.4f}"
            for r in info["species"]
        ),
        "    (mass dependence only through the fitted per-type scales; with momenta "
        "predicted directly, sigma_q*sqrt(m) and sigma_p/sqrt(m) are ~equal across "
        "species)",
        *(
            [f"  WARNING: {info['species_warning']}"]
            if info["species_warning"]
            else []
        ),
        f"  raw loss |M|^2 / dim: dense {info['raw_exact']:.8e}, loss code "
        f"{info['raw_loss_code']:.8e} (rel. diff {rel['raw_vs_loss_code']:.1e})",
        f"  canonical Omega in normalized coordinates (wrong unless Lambda/c = 1): "
        f"{info['canonical_omega_loss']:.8e}",
        f"  blocks of the residual: raw {pct(info['raw_blocks'])}; normalized "
        f"{pct(info['normalized_blocks'])}",
        f"  residual rows: raw {pct(info['raw_rows'])}; normalized "
        f"{pct(info['normalized_rows'])}",
        "",
    ]
    print("\n".join(lines), flush=True)


def plot_jacobian_sparsity(info: Dict[str, Any], path: str, floor_decades: int = 16) -> str:
    """
    Heat maps of the sparsity of J and of M = J^T Omega J - Omega (``info`` from
    :func:`jacobian_sparsity`), each normalized by its own largest value, so that the
    colors run from 0 to 1 (or from ``-floor_decades`` to 0 in log10):

    - top row, linear scale: ``|J| / max|J|`` and ``|M| / max|M|`` entry by entry, in the
      order of ``cat([q.flatten(), p.flatten()])`` (the dashed lines separate the q
      and the p parts), and the hop distance between atoms in the neighbor graph;
    - bottom row, log scale (exact zeros at the bottom of the scale): the same for the
      Frobenius norm of the 6x6 atom-pair blocks, and the largest and mean block norm as
      a function of the hop distance, which shows how fast J and M decay with it.

    :return: ``path``.
    """
    import matplotlib
    import numpy as np

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def normalized(values: Tensor) -> np.ndarray:
        values = values.abs().numpy()
        return values / max(float(values.max()), 1e-300)

    def log_normalized(values: Tensor) -> np.ndarray:
        with np.errstate(divide="ignore"):
            logs = np.log10(normalized(values))
        return np.clip(logs, -floor_decades, 0.0)  # exact zeros -> the floor

    J, M = info["J"], info["M"]
    n_atoms, dim = info["n_atoms"], info["dim"]
    hops = info["hops"].numpy().copy()
    reach = info["receptive_field_hops"]
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    def image(ax, data, title, cmap, vmin, vmax, label, boundary=None):
        shown = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(title, fontsize=10)
        fig.colorbar(shown, ax=ax, shrink=0.8, label=label)
        if boundary is not None:
            ax.axhline(boundary - 0.5, color="w", lw=0.8, ls="--")
            ax.axvline(boundary - 0.5, color="w", lw=0.8, ls="--")

    half = dim // 2
    image(axes[0, 0], normalized(J), "|J| / max|J|  (entries; q rows/cols, then p)",
          "magma", 0.0, 1.0, "normalized", half)
    image(axes[0, 1], normalized(M), "|M| / max|M|, M = J^T Omega J - Omega",
          "magma", 0.0, 1.0, "normalized", half)

    # hop distance, with "not connected" and "other system" after the finite distances
    finite_max = int(hops.max()) if (hops >= 0).any() else 0
    labels = {}
    if (hops == -1).any():
        hops[hops == -1] = finite_max + 1
        labels[finite_max + 1] = "inf"
    if (hops == -2).any():
        hops[hops == -2] = finite_max + 2
        labels[finite_max + 2] = "other system"
    top = int(hops.max())
    image(axes[0, 2], hops, f"hop distance in the neighbor graph (J reaches {reach})",
          "viridis", 0, max(top, 1), "hops")
    colorbar = fig.axes[-1]
    ticks = list(range(top + 1))
    colorbar.set_yticks(ticks)
    colorbar.set_yticklabels([labels.get(t, str(t)) for t in ticks])

    for ax, blocks, name in (
        (axes[1, 0], info["J_blocks"], "J"),
        (axes[1, 1], info["M_blocks"], "M"),
    ):
        image(ax, log_normalized(blocks),
              f"log10 of |{name}| 6x6 atom blocks / largest block",
              "magma", -floor_decades, 0.0, "log10 (normalized)")
        ax.set_xlabel("atom j")
        ax.set_ylabel("atom i")

    ax = axes[1, 2]
    for table, name, color in ((info["J_table"], "J", "C0"), (info["M_table"], "M", "C1")):
        rows = [r for r in table if r["hops"] >= 0]
        if not rows:
            continue
        scale = max(max(r["max"] for r in rows), 1e-300)
        xs = [r["hops"] for r in rows]
        ax.semilogy(xs, [max(r["max"] / scale, 10.0 ** -floor_decades) for r in rows], "o-",
                    color=color, label=f"{name}: max block")
        ax.semilogy(xs, [max(r["mean"] / scale, 10.0 ** -floor_decades) for r in rows], "s--",
                    color=color, alpha=0.6, label=f"{name}: mean block")
    ax.axvline(reach + 0.5, color="C0", ls=":", lw=1.0)
    ax.axvline(2 * reach + 0.5, color="C1", ls=":", lw=1.0)
    ax.set_xlabel("hop distance")
    ax.set_ylabel("block norm / largest block")
    ax.set_title("decay with the hop distance (dotted: where J / M must vanish)", fontsize=10)
    ax.set_ylim(10.0 ** -floor_decades, 3.0)  # exact zeros sit on the floor
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    fig.suptitle(
        f"sparsity of the Jacobian: {n_atoms} atoms, dim {dim}, "
        f"{info['num_gnn_layers']} GNN layers, {info['featurizer_type']} featurizer, "
        f"cutoff {info['cutoff']}; exact zeros: J {info['J_zero_fraction']:.1%}, "
        f"M {info['M_zero_fraction']:.1%}"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def graph_neighbor_order(hops: Tensor) -> List[int]:
    """
    An ordering of the atoms in which graph neighbors are adjacent (reverse
    Cuthill-McKee of the neighbor graph, per connected component; a breadth-first search
    from the least connected atom if scipy is not available), so that the atom-pair
    maps become banded.
    """
    import numpy as np

    adjacency = (hops == 1).numpy()
    try:
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import reverse_cuthill_mckee

        return [int(i) for i in reverse_cuthill_mckee(csr_matrix(adjacency), True)]
    except Exception:
        degree = adjacency.sum(axis=1)
        order: List[int] = []
        seen = set()
        for start in np.argsort(degree, kind="stable"):
            if int(start) in seen:
                continue
            seen.add(int(start))
            queue = [int(start)]
            while queue:
                atom = queue.pop(0)
                order.append(atom)
                for other in sorted(np.nonzero(adjacency[atom])[0], key=degree.__getitem__):
                    if int(other) not in seen:
                        seen.add(int(other))
                        queue.append(int(other))
        return order


def plot_jacobian_pairs(info: Dict[str, Any], path: str, floor_decades: int = 16) -> str:
    """
    Atom-pair maps of J and of M = J^T Omega J - Omega in the style of the Hessian
    figures of "Truncated automatic sparse differentiation for machine learning
    interatomic potentials" (Langer, Hill, Ceriotti): one row per matrix, with

    - left: one cell per pair of atoms, the atoms ordered so that graph neighbors are
      adjacent (:func:`graph_neighbor_order`). Above the diagonal the cell is the hop
      distance between the two atoms in the neighbor graph, below it (and on it) the
      Frobenius norm of the 6x6 block, as ``log10(norm / largest norm)``. Pairs that are
      not connected, and blocks that are exactly zero (or below ``10**-floor_decades``
      of the largest one), are white;
    - right: the block norm against the hop distance k: median and 10-90% band over all
      the pairs at distance k (norms below ``10**-floor_decades`` of the largest block
      are drawn at that floor), with the hop distance at which the matrix must vanish
      dotted and the fitted decay in orders of magnitude per hop in the title.

    :return: ``path``.
    """
    import matplotlib
    import numpy as np

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hops = info["hops"]
    n_atoms = hops.shape[0]
    order = graph_neighbor_order(hops)
    reach = info["receptive_field_hops"]
    ordered_hops = hops.numpy()[np.ix_(order, order)].astype(float)
    ordered_hops[ordered_hops < 0] = np.nan  # not connected / other system: white
    lower = np.tril(np.ones((n_atoms, n_atoms), dtype=bool))

    fig, axes = plt.subplots(2, 2, figsize=(14, 11), gridspec_kw={"width_ratios": [1.1, 1]})
    for row, (name, blocks, vanish) in enumerate(
        (
            ("J", info["J_blocks"], reach),
            ("M = J^T Omega J - Omega", info["M_blocks"], 2 * reach),
        )
    ):
        norms = blocks.numpy()
        largest = max(float(norms.max()), 1e-300)
        floor = largest * 10.0**-floor_decades

        # --- the pair map
        ax = axes[row, 0]
        with np.errstate(divide="ignore"):
            logs = np.log10(norms[np.ix_(order, order)] / largest)
        shown_norms = np.ma.masked_where(
            ~lower | ~(norms[np.ix_(order, order)] > floor), logs
        )
        shown_hops = np.ma.masked_where(lower | np.isnan(ordered_hops), ordered_hops)
        norm_cmap = plt.get_cmap("magma").copy()
        hop_cmap = plt.get_cmap("viridis_r").copy()
        # masked cells stay transparent (white background), so the two maps can overlap
        image_norms = ax.imshow(
            shown_norms, cmap=norm_cmap, vmin=-floor_decades, vmax=0.0,
            interpolation="nearest",
        )
        max_hop = int(np.nanmax(ordered_hops)) if np.isfinite(ordered_hops).any() else 1
        image_hops = ax.imshow(
            shown_hops, cmap=hop_cmap, vmin=0, vmax=max(max_hop, 1), interpolation="nearest"
        )
        ax.set_facecolor("white")
        ax.set_title(
            f"{name}: below the diagonal log10 |block| / largest, above the hop distance",
            fontsize=9,
        )
        ax.set_xlabel("atom (graph-neighbor order)")
        ax.set_ylabel("atom (graph-neighbor order)")
        fig.colorbar(image_norms, ax=ax, shrink=0.7, location="left", pad=0.12,
                     label="log10 (norm / largest norm)")
        colorbar = fig.colorbar(image_hops, ax=ax, shrink=0.7, label="hop distance")
        colorbar.set_ticks(list(range(max(max_hop, 1) + 1)))

        # --- block norm against the hop distance
        ax = axes[row, 1]
        hop_values = sorted({int(h) for h in hops.unique().tolist() if h >= 0})
        stats = []
        for h in hop_values:
            values = torch.from_numpy(norms[(hops == h).numpy()]).double().clamp_min(floor)
            if values.numel() == 0:
                continue
            q10, q50, q90 = (float(torch.quantile(values, q)) for q in (0.1, 0.5, 0.9))
            stats.append((h, q10, q50, q90, int(values.numel())))
        ks = np.array([s[0] for s in stats])
        q10, q50, q90 = (np.array([s[i] for s in stats]) for i in (1, 2, 3))
        ax.fill_between(ks, q10, q90, alpha=0.25, color="C0", label="10-90%")
        ax.semilogy(ks, q50, "o-", color="C0", label="median")
        ax.axvline(vanish + 0.5, color="k", ls=":", lw=1.0,
                   label=f"structurally zero beyond {vanish} hops")
        ax.axhline(floor, color="gray", lw=0.6, ls="--")
        ax.set_ylim(floor / 2, largest * 3)
        ax.set_xlabel("hop distance k")
        ax.set_ylabel("Frobenius norm of the 6x6 block")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        # decay per hop over the hops where the median is above the floor
        usable = (ks >= 1) & (q50 > floor * 10)
        if usable.sum() >= 2:
            slope = np.polyfit(ks[usable], np.log10(q50[usable]), 1)[0]
            ax.set_title(
                f"{name}: decay of the median ~ {-slope:.1f} orders of magnitude per hop",
                fontsize=9,
            )
        else:
            ax.set_title(f"{name}: block norm against the hop distance", fontsize=9)

    fig.suptitle(
        f"atom-pair maps of the Jacobian: {n_atoms} atoms, "
        f"{info['num_gnn_layers']} GNN layers, {info['featurizer_type']} featurizer"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# ------------------------------------------------------------------------------------
# Diagnostics of the stochastic symplectic loss and of its gradient
# ------------------------------------------------------------------------------------
#
# These evaluate the K-probe estimator used in training on a *fixed* batch, with the
# current parameters, and compare its gradient to a reference gradient and to the
# gradient of the data loss. They return plain values and do no logging.

# scalar fields of a diagnostic record, in the order used for CSV files
DIAGNOSTIC_FIELDS = (
    "step",
    "split",
    "K",
    "symp_mean",
    "symp_probe_variance",
    "symp_estimator_variance",
    "symp_stderr",
    "symp_q_rows",
    "symp_p_rows",
    "grad_cosine",
    "grad_norm_stochastic",
    "grad_norm_reference",
    "symp_reference",
    "symp_reference_q_rows",
    "symp_reference_p_rows",
    "grad_rel_error",
    "reference_mode",
    "reference_n_probes",
    "normalization",
    "data_loss",
    "grad_norm_data",
    "grad_cosine_data_stochastic",
    "grad_cosine_data_reference",
    "weighted_grad_ratio",
    "symplectic_weight",
    "n_params_without_grad_stochastic",
    "n_params_without_grad_reference",
    "n_atoms",
    "neighbor_width",
    "memory_before_mib",
    "peak_memory_mib",
    "diag_time_s",
    "probe_grad_cos_mean",
    "probe_grad_cos_min",
    "probe_grad_cos_to_others_mean",
    "probe_grad_cos_to_ref_mean",
    "probe_grad_norm_mean",
    "n_samples",
)


def basis_probe_fn(q: Tensor, p: Tensor) -> ProbeFn:
    """
    Probes that are the canonical basis vectors of the (q, p) space, in the order of
    ``cat([q.flatten(), p.flatten()])``. Their individual losses sum to
    ``|J^T Omega J - Omega|_F^2 / dim``, which is the expectation of the loss of a
    random probe with identity covariance (rademacher or gaussian), i.e. the exact
    value the stochastic estimator is estimating.
    """
    n_q = q.numel()
    dim = n_q + p.numel()

    def probe_fn(start: int, n_chunk: int) -> Tuple[Tensor, Tensor]:
        rows = torch.zeros(n_chunk, dim, dtype=q.dtype, device=q.device)
        rows[
            torch.arange(n_chunk, device=q.device),
            start + torch.arange(n_chunk, device=q.device),
        ] = 1.0
        return rows[:, :n_q].reshape(n_chunk, *q.shape), rows[:, n_q:].reshape(
            n_chunk, *p.shape
        )

    return probe_fn


def _loss_and_gradient(
    probe_loss_chunks: Iterator[Tuple[Tensor, Tensor]],
    weight: float,
    params: List[Tensor],
) -> Tuple[Tensor, Tensor, List[Optional[Tensor]]]:
    """
    Consume the chunks of individual probe losses (``(losses, parts)``, see
    ``calc_symplectic_probe_loss_chunks(..., with_parts=True)``): collect the losses and
    their q-row / p-row parts, and accumulate the parameter gradient of
    ``weight * sum(probe losses)``.

    The gradient is taken with ``torch.autograd.grad``, so no ``.grad`` is written, and
    the graph is always retained (it is shared with whatever is evaluated next).
    Parameters that do not receive a gradient are returned as ``None``.

    :return: the losses ``[n]``, the parts ``[n, 2]`` (CPU, float64) and the gradient.
    """
    losses: List[Tensor] = []
    parts: List[Tensor] = []
    grad_sums: List[Optional[Tensor]] = [None] * len(params)
    for chunk_losses, chunk_parts in probe_loss_chunks:
        losses.append(chunk_losses.detach().double().cpu())
        parts.append(chunk_parts.double().cpu())
        grads = torch.autograd.grad(
            weight * chunk_losses.sum(),
            params,
            retain_graph=True,
            allow_unused=True,
        )
        for i, grad in enumerate(grads):
            if grad is None:
                continue
            grad = grad.detach().double()
            grad_sums[i] = grad if grad_sums[i] is None else grad_sums[i] + grad
    return torch.cat(losses), torch.cat(parts), grad_sums


def _loss_and_probe_gradients(
    probe_loss_chunks: Iterator[Tuple[Tensor, Tensor]],
    weight: float,
    params: List[Tensor],
) -> Tuple[Tensor, Tensor, List[Optional[Tensor]], Tensor]:
    """
    Like :func:`_loss_and_gradient`, but with one backward per probe, so that the
    gradient of every individual probe loss is also returned: a ``(n_probes, n_params)``
    float32 CPU matrix of the flattened gradients of the probe losses (unweighted, a
    parameter without gradient contributes zeros). The summed gradient is identical to
    that of :func:`_loss_and_gradient` (up to round-off); the cost is one backward per
    probe instead of one per chunk.
    """
    losses: List[Tensor] = []
    parts: List[Tensor] = []
    grad_sums: List[Optional[Tensor]] = [None] * len(params)
    probe_gradients: List[Tensor] = []
    for chunk_losses, chunk_parts in probe_loss_chunks:
        losses.append(chunk_losses.detach().double().cpu())
        parts.append(chunk_parts.double().cpu())
        for i in range(chunk_losses.shape[0]):
            grads = torch.autograd.grad(
                chunk_losses[i], params, retain_graph=True, allow_unused=True
            )
            flat, _ = _flatten_gradient(
                [None if g is None else g.detach().double() for g in grads], params
            )
            probe_gradients.append(flat.float().cpu())
            for j, grad in enumerate(grads):
                if grad is None:
                    continue
                grad = weight * grad.detach().double()
                total = grad_sums[j]
                grad_sums[j] = grad if total is None else total + grad
    return (
        torch.cat(losses),
        torch.cat(parts),
        grad_sums,
        torch.stack(probe_gradients),
    )


def probe_gradient_agreement(
    probe_gradients: Tensor, reference_gradient: Optional[Tensor] = None
) -> Dict[str, Any]:
    """
    How much the gradients of the individual probe losses agree with each other.

    :param probe_gradients: ``(K, n_params)``, the gradient of each probe loss.
    :param reference_gradient: ``(n_params,)``, the reference gradient (optional).
    :return: scalars and lists (all computed in float64):

        - ``probe_grad_cosines``: the ``K x K`` matrix of cosines between the gradients
          of every pair of probes (diagonal 1);
        - ``probe_grad_cos_mean`` / ``probe_grad_cos_min``: mean and minimum of its
          off-diagonal entries (NaN for K = 1);
        - ``probe_grad_cos_to_others``: for every probe, the cosine between its gradient
          and the summed gradient of the *other* K - 1 probes (leave-one-out), and its
          mean ``probe_grad_cos_to_others_mean``;
        - ``probe_grad_cos_to_ref`` / ``probe_grad_cos_to_ref_mean``: the cosine of every
          probe gradient with the reference gradient (if given);
        - ``probe_grad_norms`` / ``probe_grad_norm_mean``: the norms of the probe
          gradients.

        A single probe gradient is a very noisy estimate of the full gradient, so
        cosines well below 1 are expected; what matters is that they are clearly
        positive (the probes share a common direction) and how they change in training.
    """
    eps = 1e-30
    G = probe_gradients.double()
    K = G.shape[0]
    gram = G @ G.T
    norms = gram.diagonal().clamp_min(0).sqrt()
    cos = gram / (norms[:, None] * norms[None, :] + eps)

    result: Dict[str, Any] = {
        "probe_grad_cosines": cos.tolist(),
        "probe_grad_norms": norms.tolist(),
        "probe_grad_norm_mean": float(norms.mean()),
    }
    if K > 1:
        off_diagonal = cos[~torch.eye(K, dtype=torch.bool)]
        result["probe_grad_cos_mean"] = float(off_diagonal.mean())
        result["probe_grad_cos_min"] = float(off_diagonal.min())
        # leave-one-out: v_k = sum_{j != k} g_j
        total = gram.sum()
        row = gram.sum(dim=1)
        dot = row - gram.diagonal()  # g_k . v_k
        v_norm_sq = (total - 2 * row + gram.diagonal()).clamp_min(0)
        to_others = dot / (norms * v_norm_sq.sqrt() + eps)
        result["probe_grad_cos_to_others"] = to_others.tolist()
        result["probe_grad_cos_to_others_mean"] = float(to_others.mean())
    else:
        result["probe_grad_cos_mean"] = math.nan
        result["probe_grad_cos_min"] = math.nan
        result["probe_grad_cos_to_others"] = [math.nan]
        result["probe_grad_cos_to_others_mean"] = math.nan
    if reference_gradient is not None:
        g_ref = reference_gradient.detach().double().cpu()
        to_ref = (G @ g_ref) / (norms * float(g_ref.norm()) + eps)
        result["probe_grad_cos_to_ref"] = to_ref.tolist()
        result["probe_grad_cos_to_ref_mean"] = float(to_ref.mean())
    return result


def _flatten_gradient(
    grad_sums: List[Optional[Tensor]], params: List[Tensor]
) -> Tuple[Tensor, int]:
    """
    One flat float64 vector with the gradients of all the trainable parameters, in
    parameter order. A parameter without gradient contributes zeros (the same rule for
    every gradient we flatten); the number of such parameters is returned too.
    """
    pieces: List[Tensor] = []
    n_missing = 0
    for grad, param in zip(grad_sums, params):
        if grad is None:
            n_missing += 1
            pieces.append(
                torch.zeros(param.numel(), dtype=torch.float64, device=param.device)
            )
        else:
            pieces.append(grad.reshape(-1))
    return torch.cat(pieces), n_missing


def _probe_seed(seed: int, step: int, split: str, k: int) -> int:
    # the hash of a tuple of ints is deterministic (unlike the one of a str, which
    # depends on PYTHONHASHSEED), so the split is turned into ints first
    split_code = tuple(ord(c) for c in split)
    return hash((int(seed), int(step), split_code, int(k))) & 0x7FFFFFFFFFFFFFFF


def run_symplectic_diagnostics_for_ks(
    model: "MetatrainModel",
    batch: Union[Callable[[], List[System]], List[System]],
    Ks: Sequence[int],
    reference_mode: ReferenceMode = "auto",
    step: int = 0,
    split: str = "train",
    *,
    probe_distribution: ProbeDistribution = "rademacher",
    chunk_size: int = 2,
    reference_probes: int = 256,
    reference_chunk_size: Optional[int] = None,
    dense_max_dim: int = 600,
    seed: int = 0,
    return_probe_losses: bool = True,
    probe_gradients: bool = False,
    data_loss_fn: Optional[Callable[[List[System], Dict[str, TensorMap]], Any]] = None,
    data_gradient: bool = True,
    symplectic_weight: float = math.nan,
    normalization: Normalization = "data_scales",
) -> List[Dict[str, Any]]:
    """
    Diagnostics of the K-probe stochastic symplectic loss and of its gradient, for
    several probe counts ``Ks`` at once, on one batch with the current parameters.

    The model is run once (one training forward on ``batch``, see
    :func:`symplectic_forward`); every K estimator and the reference are then evaluated
    on that same graph, with independent probes, so all the records of one call refer
    to the same parameters and the same configurations.

    Everything is computed with the ``normalization`` of the training: with
    ``"data_scales"`` the losses, their q-row / p-row terms, the reference (the dense
    one uses the canonical basis of the normalized coordinates) and all the gradients
    are those of the normalized loss |D M D|_F^2 / (c^2 dim).

    For each K, with l_1..l_K the individual probe losses (see
    :func:`calc_symplectic_probe_loss_chunks`):

    - ``symp_mean``: mean(l), which is exactly the symplectic loss used in training;
    - ``symp_q_rows`` / ``symp_p_rows``: its two terms, from the q-rows and the p-rows
      of the residual J^T Omega J v - Omega v (they sum to ``symp_mean``; in physical
      units they are not normalized against each other, and their ratio shows which
      one the loss is dominated by);
    - ``symp_probe_variance``: unbiased sample variance of the l_k (NaN for K = 1);
    - ``symp_estimator_variance``: that variance divided by K, the variance of the
      K-probe averaged estimator (what the optimizer actually sees);
    - ``symp_stderr``: its square root;
    - ``grad_norm_stochastic``: norm of g_K = grad_theta mean(l), and
    - ``grad_cosine``, ``grad_norm_reference``, ``grad_rel_error`` against the reference
      gradient g_ref (same batch, same parameters).

    The reference (``reference_mode``), with its own ``symp_reference_q_rows`` /
    ``symp_reference_p_rows``:

    - ``"dense"``: the exact symplectic loss |J^T Omega J - Omega|_F^2 / dim and its
      gradient, computed with the ``dim`` canonical basis vectors as probes (dim =
      3 * n_atoms * 2 probes). Exact up to round-off, but costs ``dim`` probes;
    - ``"high_k"``: the same estimator as in training with ``reference_probes`` (much
      larger than K) independent probes. Still stochastic, with variance reduced by
      ``reference_probes / K``: documented as an approximation;
    - ``"auto"``: ``"dense"`` if ``dim <= dense_max_dim``, otherwise ``"high_k"``.

    With ``data_loss_fn``, the data loss of the training on the same batch
    (``data_loss``) and, with ``data_gradient``, its gradient g_data: ``grad_norm_data``,
    ``grad_cosine_data_stochastic`` (cos(g_data, g_K)), ``grad_cosine_data_reference``
    (cos(g_data, g_ref)) and ``weighted_grad_ratio`` = ``symplectic_weight * |g_ref| /
    |g_data|``, how much the symplectic loss weighs in the update against the data loss
    (a negative cosine means that the two losses pull against each other).

    Nothing here steps the optimizer or changes parameters, ``.grad``, or the global
    random number generators (probes come from seeds derived from ``seed``, ``step``,
    ``split`` and K: fresh, independent probes for every diagnostic step).

    :param batch: the systems of the fixed diagnostic batch, or a function returning
        fresh ones (preferred: ``forward`` replaces ``system.positions`` in place).
    :param probe_gradients: also backpropagate every probe loss on its own (one backward
        per probe, and a ``K x n_params`` float32 matrix in CPU memory) and add
        :func:`probe_gradient_agreement` to each record: the cosines between the
        gradients of the individual probes (``probe_grad_cosines``, ...) and with the
        reference gradient. ``g_K`` is then the mean of these probe gradients.
    :param data_loss_fn: ``(systems, raw_predictions) -> loss``, the data loss of the
        training (on the targets of the same batch) evaluated on the predictions of the
        diagnostic forward, with its graph (NaN fields if it fails).
    :param data_gradient: also backpropagate the data loss (see above).
    :param symplectic_weight: the weight of the symplectic loss in the training, for
        ``weighted_grad_ratio``.
    :param normalization: the normalization of the loss, as in the training.
    :return: one dictionary of scalars per K (see :data:`DIAGNOSTIC_FIELDS`), plus
        ``probe_losses`` (list of the K individual losses) if ``return_probe_losses``.
    """
    Ks = [int(K) for K in Ks]
    if len(Ks) == 0 or min(Ks) < 1:
        raise ValueError(f"Ks must be positive integers, got {Ks}")
    if reference_mode not in ("auto", "dense", "high_k"):
        raise ValueError(f"Unknown reference_mode: {reference_mode}")
    reference_chunk_size = reference_chunk_size or chunk_size

    start_time = time.perf_counter()
    params = [p for p in model.parameters() if p.requires_grad]
    device = params[0].device
    memory_before = math.nan
    if device.type == "cuda":
        # what is already allocated (a leak of an earlier diagnostic shows up here)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        memory_before = float(torch.cuda.memory_allocated(device))

    with symplectic_forward(model, batch) as (systems, raw_predictions):
        # --- the data loss and its gradient, on the same forward
        data_loss = math.nan
        g_data: Optional[Tensor] = None
        if data_loss_fn is not None:
            # auxiliary: it must not take the symplectic diagnostics down with it
            try:
                data_loss_tensor = data_loss_fn(systems, raw_predictions)
                data_loss = float(data_loss_tensor.detach())
                if data_gradient and data_loss_tensor.requires_grad:
                    grads = torch.autograd.grad(
                        data_loss_tensor, params, retain_graph=True, allow_unused=True
                    )
                    g_data, _ = _flatten_gradient(
                        [None if g is None else g.detach().double() for g in grads],
                        params,
                    )
                del data_loss_tensor
            except Exception as error:
                warnings.warn(
                    f"Could not compute the data loss: {error!r}", stacklevel=2
                )

        q, p, q_next, p_next = phase_graph_from_predictions(
            model, systems, raw_predictions
        )
        dim = q.numel() + p.numel()
        scales: Optional[PhaseSpaceScales] = (
            phase_space_scales(model, systems, q)
            if resolve_normalization(normalization) == "data_scales"
            else None
        )

        # --- the K-probe stochastic estimators (what training uses)
        estimator_results = []
        for K in Ks:
            torch.manual_seed(_probe_seed(seed, step, split, K))
            chunks = calc_symplectic_probe_loss_chunks(
                q,
                p,
                q_next,
                p_next,
                n_probes=K,
                probe_dist=probe_distribution,
                chunk_size=min(chunk_size, K),
                with_parts=True,
                scales=scales,
            )
            if probe_gradients:
                losses, parts, grad_sums, G_probes = _loss_and_probe_gradients(
                    chunks, 1.0 / K, params
                )
            else:
                losses, parts, grad_sums = _loss_and_gradient(chunks, 1.0 / K, params)
                G_probes = None
            estimator_results.append(
                (K, losses, parts, *_flatten_gradient(grad_sums, params), G_probes)
            )

        # --- the reference
        mode = reference_mode
        if mode == "auto":
            mode = "dense" if dim <= dense_max_dim else "high_k"
        if mode == "dense":
            reference_n = dim
            reference_weight = 1.0  # exact value: the sum over the basis vectors
            reference_probe_fn: Optional[ProbeFn] = basis_probe_fn(q, p)
        else:
            reference_n = int(reference_probes)
            reference_weight = 1.0 / reference_n
            reference_probe_fn = None
            # independent from every K above
            torch.manual_seed(_probe_seed(seed, step, split, -1))
        chunks = calc_symplectic_probe_loss_chunks(
            q,
            p,
            q_next,
            p_next,
            n_probes=reference_n,
            probe_dist=probe_distribution,
            chunk_size=min(reference_chunk_size, reference_n),
            probe_fn=reference_probe_fn,
            with_parts=True,
            scales=scales,
        )
        reference_losses, reference_parts, reference_grad_sums = _loss_and_gradient(
            chunks, reference_weight, params
        )
        reference_value = float(reference_weight * reference_losses.sum())
        reference_q_rows, reference_p_rows = (
            float(reference_weight * v) for v in reference_parts.sum(dim=0)
        )
        g_ref, n_missing_ref = _flatten_gradient(reference_grad_sums, params)
        g_ref_norm = float(g_ref.norm())
        n_atoms = int(q.shape[0])
        neighbor_width = batch_neighbor_width(systems, neighbor_list_options(model))

    eps = 1e-30
    g_data_norm = math.nan if g_data is None else float(g_data.norm())

    def cosine_with_data(g: Tensor) -> float:
        if g_data is None:
            return math.nan
        return float(g_data @ g) / (g_data_norm * float(g.norm()) + eps)

    records: List[Dict[str, Any]] = []
    for K, losses, parts, g_K, n_missing_K, G_probes in estimator_results:
        mean = float(losses.mean())
        q_rows, p_rows = (float(v) for v in parts.mean(dim=0))
        probe_variance = float(losses.var(unbiased=True)) if K > 1 else math.nan
        estimator_variance = probe_variance / K
        g_K_norm = float(g_K.norm())
        record: Dict[str, Any] = {
            "step": int(step),
            "split": split,
            "K": K,
            "symp_mean": mean,
            "symp_probe_variance": probe_variance,
            "symp_estimator_variance": estimator_variance,
            "symp_stderr": math.sqrt(max(estimator_variance, 0.0))
            if estimator_variance == estimator_variance
            else math.nan,
            "symp_q_rows": q_rows,
            "symp_p_rows": p_rows,
            "grad_cosine": float(g_K @ g_ref) / (g_K_norm * g_ref_norm + eps),
            "grad_norm_stochastic": g_K_norm,
            "grad_norm_reference": g_ref_norm,
            "symp_reference": reference_value,
            "symp_reference_q_rows": reference_q_rows,
            "symp_reference_p_rows": reference_p_rows,
            "grad_rel_error": float((g_K - g_ref).norm()) / (g_ref_norm + eps),
            "reference_mode": mode,
            "reference_n_probes": reference_n,
            "normalization": normalization,
            "data_loss": data_loss,
            "grad_norm_data": g_data_norm,
            "grad_cosine_data_stochastic": cosine_with_data(g_K),
            "grad_cosine_data_reference": cosine_with_data(g_ref),
            "weighted_grad_ratio": symplectic_weight * g_ref_norm / (g_data_norm + eps)
            if g_data is not None
            else math.nan,
            "symplectic_weight": symplectic_weight,
            "n_samples": int(len(losses)),
            "n_params_without_grad_stochastic": n_missing_K,
            "n_params_without_grad_reference": n_missing_ref,
            "n_atoms": n_atoms,
        }
        if G_probes is not None:
            record.update(probe_gradient_agreement(G_probes, g_ref))
        if return_probe_losses:
            record["probe_losses"] = [float(v) for v in losses]
        records.append(record)

    elapsed = time.perf_counter() - start_time
    peak_memory = math.nan
    if device.type == "cuda":
        peak_memory = float(torch.cuda.max_memory_allocated(device))
    for record in records:
        record["diag_time_s"] = elapsed
        record["neighbor_width"] = neighbor_width
        record["memory_before_mib"] = memory_before / 2**20
        record["peak_memory_mib"] = peak_memory / 2**20
    return records


def run_symplectic_diagnostics(
    model: "MetatrainModel",
    batch: Union[Callable[[], List[System]], List[System]],
    K: int,
    reference_mode: ReferenceMode = "auto",
    step: int = 0,
    split: str = "train",
    **kwargs,
) -> Dict[str, Any]:
    """
    Diagnostics for a single probe count ``K``: the dictionary of metrics of
    :func:`run_symplectic_diagnostics_for_ks` (see there for the options and the
    definition of every quantity). Use that function directly to compare several K on
    the same parameters and batch, sharing the forward pass and the reference.
    """
    return run_symplectic_diagnostics_for_ks(
        model, batch, [K], reference_mode, step, split, **kwargs
    )[0]


# ------------------------------------------------------------------------------------
# Files and plots
# ------------------------------------------------------------------------------------


def append_diagnostic_record(
    csv_path: Path, jsonl_path: Path, record: Dict[str, Any]
) -> None:
    """
    Append one symplectic diagnostic record to a CSV file (scalars only) and to a JSON
    lines file (everything, including the individual probe losses, so that the
    statistics can be recomputed later). Written as the diagnostics are computed, so
    that they survive a job that is killed.
    """
    write_header = not csv_path.exists()
    fieldnames = list(DIAGNOSTIC_FIELDS)
    if not write_header:
        # appending to a file written by an older version (e.g. a restart in the same
        # directory): keep its columns, so that every row matches the header
        with open(csv_path, "r", newline="") as f:
            existing = next(csv.reader(f), None)
        if existing:
            fieldnames = existing
        else:
            write_header = True
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(record)
    with open(jsonl_path, "a") as f:
        f.write(json.dumps(record) + "\n")


def format_diagnostic_record(record: Dict[str, Any]) -> str:
    """One-line summary of a record, for the log."""
    probe_agreement = ""
    if "probe_grad_cos_mean" in record:
        probe_agreement = (
            f"probe-gradient cosine mean {record['probe_grad_cos_mean']:.3f} / min "
            f"{record['probe_grad_cos_min']:.3f}, "
        )
    data = f"data loss {record['data_loss']:.4e}"
    if record.get("grad_norm_data", math.nan) == record.get("grad_norm_data"):
        data += (
            f" (|g_data| {record['grad_norm_data']:.3e}, cos(g_data, g_ref) "
            f"{record['grad_cosine_data_reference']:.3f}, w|g_ref|/|g_data| "
            f"{record['weighted_grad_ratio']:.3e})"
        )
    total = record["symp_q_rows"] + record["symp_p_rows"]
    split_rows = (
        f"q-rows {record['symp_q_rows'] / total:.1%} / p-rows "
        f"{record['symp_p_rows'] / total:.1%}"
        if total > 0
        else "q-rows / p-rows n/a"
    )
    memory = ""
    if record.get("peak_memory_mib", math.nan) == record.get("peak_memory_mib"):
        memory = (
            f", batch of {record['n_atoms']} atoms padded to {record['neighbor_width']} "
            f"neighbors, GPU memory {record['memory_before_mib']:.0f} MiB before, peak "
            f"{record['peak_memory_mib']:.0f} MiB"
        )
    return (
        f"Symplectic diagnostics [{record['split']}] step {record['step']}, "
        f"K={record['K']}"
        f": loss {record['symp_mean']:.4e} "
        f"(SE {record['symp_stderr']:.2e}, probe variance "
        f"{record['symp_probe_variance']:.2e}, {split_rows}), reference "
        f"{record['symp_reference']:.4e}, {data}, "
        f"gradient cosine {record['grad_cosine']:.4f}, {probe_agreement}"
        f"|g_K| {record['grad_norm_stochastic']:.3e}, |g_ref| "
        f"{record['grad_norm_reference']:.3e} (reference: {record['reference_mode']}, "
        f"{record['reference_n_probes']} probes, {record['diag_time_s']:.1f} s)"
        f"{memory}"
    )


def _student_t_critical(dof: int, level: float = 0.975) -> Optional[float]:
    try:
        from scipy import stats

        return float(stats.t.ppf(level, dof))
    except Exception:
        return None


def plot_symplectic_diagnostics(
    records: List[Dict[str, Any]], path_prefix: str
) -> List[str]:
    """
    Plots of the diagnostics, one figure per split (``<path_prefix>_<split>.png``), one
    curve per K on the same axes:

    A. symplectic loss, with a band ``mean +- t(K-1) * SE`` (Student-t, 97.5% quantile,
       i.e. an *approximate* 95% interval that treats the probes as i.i.d.; plain
       ``+- 1 SE`` if scipy is not available), and the reference value;
    B. variance of the individual probe losses;
    C. variance of the K-probe averaged estimator (probe variance / K);
    D. gradient cosine similarity with the reference, with a line at 1;
    E. gradient norms of the stochastic estimators, of the reference and of the data
       loss;
    F. the two terms of the reference loss: q-rows and p-rows of the residual;
    G. the data loss against the symplectic loss: cosine between their gradients
       (left axis) and ``weight * |g_ref| / |g_data|`` (right axis, log).

    :return: the paths of the written figures.
    """
    import matplotlib
    import numpy as np

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths: List[str] = []
    for split in sorted({record["split"] for record in records}):
        split_records = [r for r in records if r["split"] == split]
        Ks = sorted({r["K"] for r in split_records})
        fig, axes = plt.subplots(2, 4, figsize=(22, 9))
        ax_loss, ax_pvar, ax_evar, ax_cos, ax_norm, ax_rows, ax_data, ax_text = (
            axes.flat
        )

        def series(K: int, key: str):
            rs = sorted((r for r in split_records if r["K"] == K), key=lambda r: r["step"])
            return np.array([r["step"] for r in rs]), np.array(
                [float(r.get(key, math.nan)) for r in rs], dtype=float
            )

        for i, K in enumerate(Ks):
            color = f"C{i}"
            style = dict(color=color, marker="o", ms=3, label=f"K={K}")
            steps, mean = series(K, "symp_mean")
            _, stderr = series(K, "symp_stderr")
            t = _student_t_critical(K - 1) if K > 1 else None
            half = (t if t is not None else 1.0) * stderr
            ax_loss.plot(steps, mean, **style)
            ax_loss.fill_between(steps, mean - half, mean + half, color=color, alpha=0.2)
            ax_pvar.plot(steps, series(K, "symp_probe_variance")[1], **style)
            ax_evar.plot(steps, series(K, "symp_estimator_variance")[1], **style)
            ax_cos.plot(steps, series(K, "grad_cosine")[1], **style)
            ax_norm.plot(
                steps,
                series(K, "grad_norm_stochastic")[1],
                **{**style, "label": f"|g_K|, K={K}"},
            )
            ax_data.plot(
                steps,
                series(K, "grad_cosine_data_stochastic")[1],
                **{**style, "label": f"cos(g_data, g_K), K={K}"},
            )

        K0 = Ks[0]
        steps, reference = series(K0, "symp_reference")
        ax_loss.plot(steps, reference, color="k", ls="--", label="reference value")
        steps, norm_ref = series(K0, "grad_norm_reference")
        ax_norm.plot(steps, norm_ref, color="k", ls="--", label="|g_ref|")
        steps, norm_data = series(K0, "grad_norm_data")
        if np.isfinite(norm_data).any():
            ax_norm.plot(steps, norm_data, color="tab:red", ls="-.", label="|g_data|")
        ax_cos.axhline(1.0, color="k", lw=0.8)
        ax_cos.axhline(0.0, color="gray", lw=0.5, ls=":")

        # F. the q-row and p-row terms of the reference loss
        steps, ref_q = series(K0, "symp_reference_q_rows")
        _, ref_p = series(K0, "symp_reference_p_rows")
        ax_rows.plot(steps, ref_q, color="C0", marker="o", ms=3, label="q-rows")
        ax_rows.plot(steps, ref_p, color="C1", marker="s", ms=3, label="p-rows")
        ax_rows.plot(steps, ref_q + ref_p, color="k", ls="--", label="total")

        # G. data loss vs symplectic loss
        steps, cos_data_ref = series(K0, "grad_cosine_data_reference")
        ax_data.plot(steps, cos_data_ref, color="k", ls="--", label="cos(g_data, g_ref)")
        ax_data.axhline(0.0, color="gray", lw=0.5, ls=":")
        ax_data.set_ylim(-1.05, 1.05)
        steps, ratio = series(K0, "weighted_grad_ratio")
        if np.isfinite(ratio).any() and (ratio > 0).any():
            ax_ratio = ax_data.twinx()
            ax_ratio.semilogy(
                steps, ratio, color="tab:red", lw=1.0, label="w |g_ref| / |g_data|"
            )
            ax_ratio.axhline(1.0, color="tab:red", lw=0.5, ls=":")
            ax_ratio.set_ylabel("w |g_ref| / |g_data|", color="tab:red")
            ax_ratio.legend(fontsize=8, loc="lower right")

        t_label = (
            "approx. 95% band: mean ± t(K-1)·SE"
            if _student_t_critical(1)
            else "band: mean ± 1 SE"
        )
        ax_loss.set_title(f"A. symplectic loss ({t_label})", fontsize=9)
        ax_pvar.set_title("B. variance of the individual probe losses")
        ax_evar.set_title("C. variance of the K-probe estimator (probe var / K)")
        ax_cos.set_title("D. gradient cosine with the reference")
        ax_norm.set_title("E. gradient norms")
        ax_rows.set_title("F. reference loss: q-rows and p-rows of the residual")
        ax_data.set_title("G. data loss vs symplectic loss gradients", fontsize=10)
        for ax in (ax_loss, ax_pvar, ax_evar, ax_norm, ax_rows):
            if ax.lines and any(
                (np.asarray(line.get_ydata(), dtype=float) > 0).any() for line in ax.lines
            ):
                ax.set_yscale("log")
        for ax in (ax_loss, ax_pvar, ax_evar, ax_cos, ax_norm, ax_rows, ax_data):
            ax.set_xlabel("optimizer step")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8, loc="upper left" if ax is ax_data else "best")

        last = sorted(split_records, key=lambda r: r["step"])[-1]
        ax_text.axis("off")
        ax_text.text(
            0.0,
            0.95,
            f"split: {split}\n"
            f"reference: {last['reference_mode']} ({last['reference_n_probes']} probes)\n"
            f"atoms in the fixed batch: {last['n_atoms']}\n"
            f"padded neighbor width: {last.get('neighbor_width', 'n/a')}\n"
            f"J v: double backward\n"
            f"diagnostic points: {len({r['step'] for r in split_records})}",
            va="top",
            fontsize=10,
        )
        fig.suptitle(f"stochastic symplectic loss diagnostics ({split} batch)")
        fig.tight_layout()
        path = f"{path_prefix}_{split}.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        paths.append(path)
    return paths


# ------------------------------------------------------------------------------------
# Driver used by the trainer
# ------------------------------------------------------------------------------------


class SymplecticDiagnostics:
    """
    Everything the trainer needs from the diagnostics, behind three calls:
    :meth:`step` (every optimizer step, before the parameters are updated),
    :meth:`finish` (after the last step) and the :attr:`records` it fills.

    The fixed diagnostic batches are the first structures of the first training and
    validation datasets, collated with the validation collate function (no rotation or
    other augmentation) and never shuffled. They are collated again at every use
    (deterministically) because ``forward`` replaces the positions of the systems in
    place.

    It does nothing on ranks other than 0, and a failure of a diagnostic is logged and
    disables the diagnostics, but never stops the training.

    :param hypers: the ``diagnostics`` trainer hypers (:class:`SymplecticDiagnosticsHypers`).
    :param model: the ``MetatrainModel`` (not wrapped in DistributedDataParallel).
    :param train_datasets: the training datasets: the fixed training batch is taken
        from the first one, the neighbor statistics cover all of them.
    :param val_datasets: the validation datasets (possibly empty), used the same way.
    :param collate_fn: the collate function without augmentation.
    :param loss_fn: the data loss of the training, evaluated on the diagnostic batches.
    :param per_structure_targets: the ``per_structure_targets`` of the training.
    :param loss_weight: the weight of the symplectic loss in the training.
    :param num_probes: the number of probes of the loss, the default of ``ks``.
    :param probe_distribution: the probe distribution of the loss.
    :param chunk_size: the chunk size of the loss, the default of the diagnostics'.
    :param normalization: the normalization of the loss
        (``weak_symplectic_normalization``).
    :param augmenter: the ``O3Augmenter`` of the training collate, for the rotation
        part of ``scaling_check``.
    :param batch_size: the batch size of the training, the default of the diagnostics'.
    :param checkpoint_dir: the default of ``output_dir``.
    """

    def __init__(
        self,
        hypers: Optional[Dict[str, Any]],
        *,
        model: "MetatrainModel",
        train_datasets: List[Any],
        val_datasets: List[Any],
        collate_fn: Callable,
        loss_fn: Callable,
        per_structure_targets: List[str],
        dtype: torch.dtype,
        device: torch.device,
        rank: int,
        loss_weight: float,
        num_probes: int,
        probe_distribution: ProbeDistribution,
        chunk_size: int,
        batch_size: int,
        checkpoint_dir: Union[str, Path],
        normalization: Normalization = "data_scales",
        augmenter: Optional[Any] = None,
    ) -> None:
        self.hypers = _resolve_hypers(hypers)
        h = self.hypers
        self.model = model
        self.dtype = dtype
        self.device = device
        self.rank = rank
        self.num_probes = num_probes
        self.probe_distribution = probe_distribution
        self.normalization = resolve_normalization(normalization)
        self.loss_chunk_size = chunk_size
        self.chunk_size = int(h["chunk_size"] or chunk_size)
        self.batch_size = int(h["batch_size"] or batch_size)
        self.train_batch_size = int(batch_size)
        self.ks = [int(k) for k in (h["ks"] or [num_probes])]
        self.every = int(h["every"] or 0)
        self.output_dir = Path(h["output_dir"] or checkpoint_dir)
        self.csv_path = self.output_dir / "symplectic_diagnostics.csv"
        self.jsonl_path = self.output_dir / "symplectic_diagnostics.jsonl"

        self._run_neighbor_info = bool(h["neighbor_info"])
        self._run_jacobian_sparsity = bool(h["jacobian_sparsity"])
        self._run_cost_benchmark = bool(h["cost_benchmark"])
        self._run_normalization_check = bool(h["normalization_check"])
        self._run_scaling_check = bool(h["scaling_check"])
        self._augmenter = augmenter
        # periodic dense Jacobians (own interval), on top of the one at the start
        self.jacobian_every = (
            int(h["jacobian_every"] or 0) if self._run_jacobian_sparsity else 0
        )
        self._last_jacobian_step: Optional[int] = None
        self.jacobian_csv_path = self.output_dir / "jacobian_sparsity.csv"
        self._neighbor_stats: Optional[Dict[str, Dict[str, Any]]] = None
        self._one_time_pending = (
            self._run_neighbor_info
            or self._run_jacobian_sparsity
            or self._run_cost_benchmark
            or self._run_normalization_check
            or self._run_scaling_check
        )

        # the datasets of the fixed batches (the first of each split)
        self._datasets = {"train": train_datasets[0]}
        if h["validation"] and len(val_datasets) > 0:
            self._datasets["validation"] = val_datasets[0]
        # all the datasets, for the statistics over the whole data
        self._train_datasets = list(train_datasets)
        self._all_datasets: Dict[str, List[Any]] = {"train": list(train_datasets)}
        if len(val_datasets) > 0:
            self._all_datasets["validation"] = list(val_datasets)
        self._collate_fn = collate_fn
        self._loss_fn = loss_fn
        self._per_structure_targets = per_structure_targets
        self._batches: Dict[str, Tuple[Callable, Callable]] = {}

        self.records: List[Dict[str, Any]] = []
        # weight of the symplectic loss at the current step, set by ``step``/``finish``
        self._symplectic_weight = float(loss_weight)
        self._last_step: Optional[int] = None
        self._skipped_splits: set = set()  # splits whose batch does not fit the device
        self._periodic_enabled = self.every > 0

        if self.rank == 0 and self.active:
            logging.info(
                "Symplectic diagnostics: "
                + (
                    f"every {self.every} steps, K={self.ks}, "
                    f"reference={h['reference']} (high-K: {h['reference_probes']} "
                    f"probes, dense if dim <= {h['dense_max_dim']}), "
                    f"validation={'validation' in self._datasets}, "
                    f"data gradient={bool(h['data_gradient'])}, "
                    f"normalization={self.normalization}, "
                    if self.every > 0
                    else "no periodic evaluation, "
                )
                + f"fixed batches of {self.batch_size} structures, "
                f"one-off checks: neighbors={self._run_neighbor_info}, "
                f"jacobian={self._run_jacobian_sparsity}"
                + (f" (every {self.jacobian_every} steps)" if self.jacobian_every else "")
                + f", cost_benchmark={self._run_cost_benchmark}"
                + f", normalization_check={self._run_normalization_check}"
                + f", scaling_check={self._run_scaling_check}"
            )

    @property
    def active(self) -> bool:
        return (
            self._periodic_enabled or self._one_time_pending or self.jacobian_every > 0
        )

    # --- fixed batches ---------------------------------------------------------------

    def _collate(self, samples: List[Any]):
        # collating inside isolated_rng leaves the training random numbers untouched
        with isolated_rng(self.device):
            batch = self._collate_fn(samples)
        systems, targets, extra_data = unpack_batch(batch)
        return batch_to(systems, targets, extra_data, dtype=self.dtype, device=self.device)

    def _fixed_batch(self, split: str) -> Tuple[Callable, Callable]:
        """``(make_systems, data_loss)`` of the fixed batch of ``split`` (cached)."""
        if split in self._batches:
            return self._batches[split]

        dataset = self._datasets[split]
        n_structures = min(self.batch_size, len(dataset))

        def collate():
            return self._collate([dataset[i] for i in range(n_structures)])

        def make_systems():
            return collate()[0]

        # the targets are not modified by the forward: collate them once
        _, fixed_targets, fixed_extra_data = collate()

        def data_loss(systems, raw_predictions):
            # the data loss of the training loop, on this batch (with its graph)
            predictions = average_by_num_atoms(
                raw_predictions, systems, self._per_structure_targets
            )
            targets = average_by_num_atoms(
                fixed_targets, systems, self._per_structure_targets
            )
            predictions = self.model.scaler.apply_scales(
                systems,
                predictions,
                remove=False,
                use_per_target_scales=False,
                use_per_property_scales=True,
            )
            return self._loss_fn(predictions, targets, fixed_extra_data)

        self._batches[split] = (make_systems, data_loss)
        return self._batches[split]

    def _neighbor_statistics(self) -> Dict[str, Dict[str, Any]]:
        if self._neighbor_stats is None:
            self._neighbor_stats = neighbor_statistics(
                self.model,
                self._all_datasets,
                self._collate_fn,
                n_probes=self.num_probes,
                chunk_size=self.loss_chunk_size,
                batch_size=self.batch_size,
            )
        return self._neighbor_stats

    def _benchmark_batch(self) -> Callable[[], List[System]]:
        """The systems of the cost benchmark: ``batch_size`` structures of the training
        data, the most connected ones with ``cost_benchmark_max_neighbors``."""
        if not self.hypers["cost_benchmark_max_neighbors"]:
            return self._fixed_batch("train")[0]
        worst = self._neighbor_statistics()["train"]["worst_structures"]
        samples = [
            self._train_datasets[dataset_index][index]
            for dataset_index, index, _, _ in worst[: self.batch_size]
        ]
        return lambda: self._collate(samples)[0]

    # --- one-off checks --------------------------------------------------------------

    def _record_jacobian(self, step: int, info: Dict[str, Any]) -> None:
        """Log, write to the CSV and plot one dense Jacobian computation."""
        E = info["symplectic_identities"]
        total = max(E["E1"] + 2 * E["E2"] + E["E3"], 1e-300)
        row = {
            "step": int(step),
            "exact_symplectic_loss": info["exact_symplectic_loss"],
            "E1": E["E1"],
            "E2": E["E2"],
            "E3": E["E3"],
            "J_zero_fraction": info["J_zero_fraction"],
            "M_zero_fraction": info["M_zero_fraction"],
            "n_atoms": info["n_atoms"],
            "dim": info["dim"],
        }
        self.output_dir.mkdir(parents=True, exist_ok=True)
        write_header = not self.jacobian_csv_path.exists()
        with open(self.jacobian_csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row), extrasaction="ignore")
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        logging.info(
            f"Jacobian sparsity step {step}: exact symplectic loss "
            f"{row['exact_symplectic_loss']:.4e} (E1 {E['E1'] / total:.1%}, E2 "
            f"{2 * E['E2'] / total:.1%}, E3 {E['E3'] / total:.1%}), exact zeros: J "
            f"{row['J_zero_fraction']:.1%}, M {row['M_zero_fraction']:.1%}"
        )
        if self.hypers["jacobian_plot"]:
            name = (
                f"jacobian_sparsity_step{step}.png"
                if self.jacobian_every > 0
                else "jacobian_sparsity.png"
            )
            path = plot_jacobian_sparsity(info, str(self.output_dir / name))
            logging.info(f"Jacobian sparsity heat maps saved in {path}")
        if self.hypers["jacobian_plot_pairs"]:
            name = (
                f"jacobian_pairs_step{step}.png"
                if self.jacobian_every > 0
                else "jacobian_pairs.png"
            )
            path = plot_jacobian_pairs(info, str(self.output_dir / name))
            logging.info(f"Jacobian atom-pair maps saved in {path}")
        self._last_jacobian_step = step

    def _compute_jacobian(self, step: int, verbose: bool) -> None:
        make_systems, _ = self._fixed_batch("train")
        try:
            with symplectic_forward(self.model, make_systems) as (systems, predictions):
                info = print_jacobian_sparsity(
                    self.model,
                    systems,
                    predictions,
                    max_dim=int(self.hypers["jacobian_max_dim"]),
                    chunk_size=int(self.hypers["jacobian_chunk_size"]),
                    verbose=verbose,
                )
        except Exception:
            self.jacobian_every = 0  # do not fail again at every interval
            raise
        if info is None:
            # too large: it will not work later either
            self.jacobian_every = 0
            return
        self._record_jacobian(step, info)

    def _run_probe_cost_benchmark(self) -> None:
        """Time and memory of the stages of the loss, and the batch size they allow."""
        h = self.hypers
        info = measure_probe_costs(
            self.model,
            self._benchmark_batch(),
            n_probes=int(h["cost_benchmark_probes"] or self.num_probes),
            probe_distribution=self.probe_distribution,
            fractions=[float(f) for f in h["cost_benchmark_fractions"]],
            repeats=int(h["cost_benchmark_repeats"]),
            chunk_sizes=[self.loss_chunk_size],  # the chunk size of the training
            normalization=self.normalization,
        )
        print_probe_costs(info)
        plan = plan_batch_size(
            info,
            batch_size=self.train_batch_size,
            chunk_size=self.loss_chunk_size,
            headroom=float(h["cost_benchmark_headroom"]),
        )
        if plan is not None:
            print_batch_size_plan(plan, info)

    def _run_one_time(self, step: int) -> None:
        if self._run_neighbor_info:
            # all the structures, not tied to a batch or to a forward pass
            def neighbor_info() -> None:
                self._neighbor_stats = print_neighbor_info(
                    self.model,
                    self._all_datasets,
                    self._collate_fn,
                    n_probes=self.num_probes,
                    chunk_size=self.loss_chunk_size,
                    batch_size=self.batch_size,
                )

            self._guarded(neighbor_info, periodic=False)
        if self._run_cost_benchmark:
            # first: it runs its own forwards, on a clean memory
            self._guarded(self._run_probe_cost_benchmark, periodic=False)
        if self._run_jacobian_sparsity:
            # which atom blocks of J and J^T Omega J - Omega are exactly zero
            self._guarded(lambda: self._compute_jacobian(step, True), periodic=False)
        if self._run_scaling_check:
            self._guarded(self._run_scaling_check_once, periodic=False)
        if self._run_normalization_check:
            self._guarded(self._run_normalization_check_once, periodic=False)

    def _run_scaling_check_once(self) -> None:
        """The scaling chain of q and p, on the fixed train batch."""
        dataset = self._datasets["train"]
        samples = [dataset[i] for i in range(min(self.batch_size, len(dataset)))]
        raw_collate = CollateFn(
            target_keys=list(self.model.dataset_info.targets.keys()), callables=[]
        )

        def make_raw_batch():
            with isolated_rng(self.device):
                batch = raw_collate(samples)
            systems, targets, extra = unpack_batch(batch)
            systems, targets, _ = batch_to(
                systems, targets, extra, dtype=self.dtype, device=self.device
            )
            return systems, targets

        def make_batch():
            systems, targets, _ = self._collate(samples)
            return systems, targets

        info = check_scaling_pipeline(
            self.model,
            make_raw_batch,
            make_batch,
            augmenter=self._augmenter,
            seed=int(self.hypers["seed"]),
        )
        print_scaling_pipeline(info)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / "scaling_check.json"
        with open(path, "w") as f:
            json.dump(info, f, indent=2)
        if info["passed"]:
            logging.info(f"Weak symplectic scaling check passed ({path})")
        else:
            logging.warning(
                f"Weak symplectic scaling check FAILED, see the table above and {path}"
            )

    def _run_normalization_check_once(self) -> None:
        """The normalized loss computed in independent ways, on the fixed train batch."""
        h = self.hypers
        make_systems, _ = self._fixed_batch("train")
        with symplectic_forward(self.model, make_systems) as (systems, predictions):
            dim = 6 * sum(len(system) for system in systems)
            if dim > int(h["jacobian_max_dim"]):
                print(
                    f"\n[weak symplectic] normalization check skipped: dim = {dim} > "
                    f"jacobian_max_dim = {h['jacobian_max_dim']} (lower "
                    "diagnostics.batch_size)\n",
                    flush=True,
                )
                return
            info = check_normalization(
                self.model,
                systems,
                predictions,
                n_probes=int(h["normalization_check_probes"]),
                probe_distribution=self.probe_distribution,
                chunk_size=self.chunk_size,
                jacobian_chunk_size=int(h["jacobian_chunk_size"]),
                seed=int(h["seed"]),
            )
        print_normalization_check(info)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / "normalization_check.json"
        with open(path, "w") as f:
            json.dump(info, f, indent=2)
        if info["passed"]:
            logging.info(f"Weak symplectic normalization check passed ({path})")
        else:
            logging.warning(
                "Weak symplectic normalization check FAILED: the normalized loss "
                f"computed in independent ways does not agree, see {path}"
            )

    # --- periodic diagnostics --------------------------------------------------------

    def _run_periodic(self, step: int) -> None:
        h = self.hypers
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for split in self._datasets:
            if split in self._skipped_splits:
                continue
            make_systems, data_loss_fn = self._fixed_batch(split)
            records = None
            # the chunk size of the diagnostics, then one probe at a time (the memory of a
            # chunk is the part that can be cut: the split is skipped if even that fails)
            for chunk_size in dict.fromkeys([self.chunk_size, 1]):
                self._free_memory()
                try:
                    records = run_symplectic_diagnostics_for_ks(
                        self.model,
                        make_systems,
                        self.ks,
                        h["reference"],
                        step,
                        split,
                        probe_distribution=self.probe_distribution,
                        chunk_size=chunk_size,
                        reference_probes=int(h["reference_probes"]),
                        dense_max_dim=int(h["dense_max_dim"]),
                        seed=int(h["seed"]),
                        probe_gradients=bool(h["probe_gradients"]),
                        data_loss_fn=data_loss_fn,
                        data_gradient=bool(h["data_gradient"]),
                        symplectic_weight=self._symplectic_weight,
                        normalization=self.normalization,
                    )
                    break
                except _OUT_OF_MEMORY:
                    logging.warning(
                        f"Symplectic diagnostics [{split}] step {step}: out of memory "
                        f"with chunk_size={chunk_size}"
                        + (", trying chunk_size=1" if chunk_size > 1 else "")
                    )
            # outside the except block: the traceback of the failure holds its graph
            self._free_memory()
            if records is None:
                logging.warning(
                    f"Symplectic diagnostics [{split}]: the fixed batch of "
                    f"{self.batch_size} structures does not fit on the device even one "
                    "probe at a time, so this split is skipped from now on (lower "
                    "`diagnostics.batch_size`); the other splits go on"
                )
                self._skipped_splits.add(split)
                continue
            for record in records:
                self.records.append(record)
                append_diagnostic_record(self.csv_path, self.jsonl_path, record)
                logging.info(format_diagnostic_record(record))
        self._last_step = step

    def _free_memory(self) -> None:
        """Release what a finished or failed diagnostic may still hold on the GPU:
        collect the garbage (graphs in reference cycles stay allocated until then) before
        returning the cached blocks."""
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def _guarded(self, function: Callable[[], Any], *, periodic: bool) -> None:
        try:
            function()
        except Exception:
            # the diagnostics must never take the training down with them
            if periodic:
                logging.exception(
                    "Symplectic diagnostics failed: the periodic diagnostics are "
                    "disabled for the rest of the training"
                )
                self._periodic_enabled = False
            else:
                logging.exception("A one-off symplectic diagnostic failed")
        # after the except block, so that the traceback (and the graph it holds) is gone
        self._free_memory()

    # --- called by the trainer -------------------------------------------------------

    def step(self, step: int, symplectic_weight: Optional[float] = None) -> None:
        """
        To be called before the optimizer step ``step`` (the diagnostics at step t see
        the parameters theta_t): runs the one-off checks the first time, and the
        periodic diagnostics every ``every`` steps.

        ``symplectic_weight`` is the weight of the symplectic loss at that step (it
        changes with a weight schedule); it is stored in the records.
        """
        if symplectic_weight is not None:
            self._symplectic_weight = float(symplectic_weight)
        if self.rank != 0 or not self.active:
            return
        if self._one_time_pending:
            self._one_time_pending = False
            self._run_one_time(step)
        if self._periodic_enabled and step % self.every == 0:
            self._guarded(lambda: self._run_periodic(step), periodic=True)
        if (
            self.jacobian_every > 0
            and step % self.jacobian_every == 0
            and self._last_jacobian_step != step
        ):
            self._guarded(lambda: self._compute_jacobian(step, False), periodic=False)

    def finish(self, step: int, symplectic_weight: Optional[float] = None) -> None:
        """
        To be called after the last optimizer step: the diagnostics of the final
        parameters (if not just done), and the plots of all of them.
        """
        if symplectic_weight is not None:
            self._symplectic_weight = float(symplectic_weight)
        if self.rank != 0:
            return
        if self._periodic_enabled and self._last_step != step:
            self._guarded(lambda: self._run_periodic(step), periodic=True)
        if self.jacobian_every > 0 and self._last_jacobian_step != step:
            self._guarded(lambda: self._compute_jacobian(step, False), periodic=False)
        if len(self.records) > 0:
            try:
                plot_paths = plot_symplectic_diagnostics(
                    self.records, str(self.output_dir / "symplectic_diagnostics")
                )
                logging.info(
                    "Symplectic diagnostics saved in "
                    f"{self.csv_path}, {self.jsonl_path} and {plot_paths}"
                )
            except Exception:
                logging.exception("Could not plot the symplectic diagnostics")

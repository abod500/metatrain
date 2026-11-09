import contextlib
import copy
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union, cast

import ase.data
import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DistributedSampler

from metatrain.composition import CompositionModel, train_or_load_composition_model
from metatrain.pet.modules.finetuning import (
    apply_finetuning_strategy,
    compute_stale_targets,
)
from metatrain.scaler import Scaler, train_or_load_scaler
from metatrain.utils.abc import (
    TrainerInterface,
    common_upgrade_checkpoint,
)
from metatrain.utils.additive import get_remove_additive_transform
from metatrain.utils.architectures import get_default_hypers
from metatrain.utils.augmentation import O3Augmenter
from metatrain.utils.data import (
    CollateFn,
    CombinedDataLoader,
    Dataset,
    DatasetInfo,
    build_train_dataloaders,
    build_val_dataloaders,
    get_num_workers,
    unpack_batch,
    validate_num_workers,
)
from metatrain.utils.distributed.distributed_data_parallel import (
    DistributedDataParallel,
)
from metatrain.utils.distributed.slurm import (
    initialize_slurm_nccl_process_group,
    resolve_distributed,
)
from metatrain.utils.evaluate_model import evaluate_model
from metatrain.utils.io import check_file_extension
from metatrain.utils.logging import ROOT_LOGGER, MetricLogger
from metatrain.utils.loss import LossAggregator, LossSpecification
from metatrain.utils.metrics import MAEAccumulator, RMSEAccumulator, get_selected_metric
from metatrain.utils.neighbor_lists import (
    get_requested_neighbor_lists,
    get_system_with_neighbor_lists_transform,
)
from metatrain.utils.per_atom import average_by_num_atoms
from metatrain.utils.scaler import get_remove_scale_transform
from metatrain.utils.transfer import batch_to
from metatrain.utils.wrapper import MetatrainModel

from . import checkpoints
from .documentation import ModelHypers, TrainerHypers
from .model import FlashMDWeakSymplectic
from .modules.additive import PositionAdditive
from .modules.symplectic_diagnostics import (
    SymplecticDiagnostics,
    species_scale_table,
    species_scaling_warning,
)
from .modules.weak_symplectic_loss import (
    num_loss_chunks,
    reference_action,
    resolve_normalization,
    resolve_weight_schedule_hypers,
    symplectic_loss_chunks,
    symplectic_weight_factor,
)


def _position_additive_dataset_info(dataset_info: DatasetInfo) -> DatasetInfo:
    """
    Get the dataset info for the ``PositionAdditive`` model, which only contains
    the position and momentum targets.

    :param dataset_info: The full dataset info.
    :return: The dataset info restricted to the valid targets.
    """
    return DatasetInfo(
        length_unit=dataset_info.length_unit,
        atomic_types=dataset_info.atomic_types,
        targets={
            target_name: target_info
            for target_name, target_info in dataset_info.targets.items()
            if PositionAdditive.is_valid_target(target_name, target_info)
        },
    )


def average_gradients_across_ranks(parameters: Any, world_size: int) -> None:
    """
    Average the ``.grad`` of the parameters over all the ranks, in place: what
    ``DistributedDataParallel`` does with its hooks (a sum over the ranks divided by the
    world size), done by hand.

    It is used when a single forward is followed by several backward passes (the
    symplectic loss is backpropagated chunk by chunk), which the hooks of DDP cannot
    handle. Every rank must have a gradient for the same parameters: the trainer adds
    ``0.0 * sum(p.sum())`` to the loss for that purpose.

    :param parameters: The parameters of the model.
    :param world_size: The number of ranks.
    """
    grads_by_kind: Dict[Any, List[torch.Tensor]] = {}
    for parameter in parameters:
        if parameter.requires_grad and parameter.grad is not None:
            kind = (parameter.grad.device, parameter.grad.dtype)
            grads_by_kind.setdefault(kind, []).append(parameter.grad)

    for grads in grads_by_kind.values():
        flat = torch.cat([grad.reshape(-1) for grad in grads])
        torch.distributed.all_reduce(flat)
        flat /= world_size
        offset = 0
        for grad in grads:
            grad.copy_(flat[offset : offset + grad.numel()].view_as(grad))
            offset += grad.numel()


def get_scheduler(
    optimizer: torch.optim.Optimizer,
    train_hypers: TrainerHypers,
    steps_per_epoch: int,
    start_epoch: int = 0,
) -> LambdaLR:
    """
    Get a learning-rate scheduler with linear warmup, then a quarter-cosine decay.

    :param optimizer: The optimizer for which to create the scheduler.
    :param train_hypers: The training hyperparameters.
    :param steps_per_epoch: The number of steps per epoch.
    :param start_epoch: The epoch the training (re)starts from. ``num_epochs`` epochs
        are run from there, so the schedule spans ``start_epoch + num_epochs`` epochs
        and the scheduler step restored from a checkpoint stays inside it.
    :return: The learning rate scheduler.
    """
    total_steps = (start_epoch + train_hypers["num_epochs"]) * steps_per_epoch
    warmup_steps = int(train_hypers["warmup_fraction"] * total_steps)
    # the learning rate decays from the peak to ``min_lr_ratio * peak`` along a quarter
    # period of a cosine (cos(theta), theta from 0 to pi/2), over the whole training
    min_lr_ratio = float(train_hypers["min_lr_ratio"])

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, warmup_steps))
        else:
            # Cosine decay
            progress = (current_step - warmup_steps) / float(
                max(1, total_steps - warmup_steps)
            )
            cosine_decay = math.cos(0.5 * math.pi * min(progress, 1.0))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    return scheduler


class Trainer(TrainerInterface[TrainerHypers, ModelHypers]):
    __checkpoint_version__ = 1

    def __init__(self, hypers: TrainerHypers) -> None:
        super().__init__(hypers)

        self.optimizer_state_dict: Optional[Dict[str, Any]] = None
        self.scheduler_state_dict: Optional[Dict[str, Any]] = None
        self.epoch: Optional[int] = None
        self.best_epoch: Optional[int] = None
        self.best_metric: Optional[float] = None
        self.best_model_state_dict: Optional[Dict[str, Any]] = None
        self.best_optimizer_state_dict: Optional[Dict[str, Any]] = None

        self.has_new_targets = False
        self._stale_finetune_targets: List[str] = []
        # records of the symplectic diagnostics of the last call to ``train``
        self.symplectic_diagnostics: List[Dict[str, Any]] = []

    def setup(
        self, model_hypers: ModelHypers, dataset_info: DatasetInfo
    ) -> MetatrainModel[FlashMDWeakSymplectic, Scaler]:
        model = FlashMDWeakSymplectic(hypers=model_hypers, dataset_info=dataset_info)

        # Set up additive models
        composition_model = CompositionModel.from_valid_targets(
            dataset_info, dataset_info.atomic_types
        )
        position_additive = PositionAdditive(
            hypers={"also_momenta": model_hypers["predict_momenta_as_difference"]},
            dataset_info=_position_additive_dataset_info(dataset_info),
        )
        additive_models = [composition_model, position_additive]

        # Initialize scaler
        scaler_hypers = get_default_hypers("scaler")["model"]
        scaler = Scaler.from_valid_targets(
            hypers=scaler_hypers, dataset_info=dataset_info
        )

        return MetatrainModel(
            core=model,
            additive_models=additive_models,
            scaler=scaler,
            dataset_info=dataset_info,
        )

    def restart(
        self,
        model: MetatrainModel[FlashMDWeakSymplectic, Scaler],
        dataset_info: DatasetInfo,
        model_hypers: ModelHypers,
    ) -> MetatrainModel[FlashMDWeakSymplectic, Scaler]:
        merged_info = model.dataset_info.union(dataset_info)
        self.has_new_targets = any(
            key not in model.dataset_info.targets for key in merged_info.targets
        )
        # Stale targets are only removed later, by ``apply_finetuning_strategy``
        self._stale_finetune_targets = compute_stale_targets(
            model.dataset_info.targets, dataset_info.targets
        )
        comp_model_info = DatasetInfo(
            length_unit=dataset_info.length_unit,
            atomic_types=dataset_info.atomic_types,
            targets={
                target_name: target_info
                for target_name, target_info in dataset_info.targets.items()
                if model.additive_models[0].is_valid_target(target_name, target_info)
            },
        )
        scaler_info = DatasetInfo(
            length_unit=dataset_info.length_unit,
            atomic_types=dataset_info.atomic_types,
            targets={
                target_name: target_info
                for target_name, target_info in dataset_info.targets.items()
                if model.scaler.is_valid_target(target_name, target_info)
            },
            extra_data=dataset_info.extra_data,
        )
        model.restart(
            dataset_info,
            additive_models_dataset_info=[
                comp_model_info,
                _position_additive_dataset_info(dataset_info),
            ],
            scaler_dataset_info=scaler_info,
            model_hypers=model_hypers,
        )
        return model

    def train(
        self,
        model: MetatrainModel[FlashMDWeakSymplectic, Scaler],
        dtype: torch.dtype,
        devices: List[torch.device],
        train_datasets: List[Union[Dataset, torch.utils.data.Subset]],
        val_datasets: List[Union[Dataset, torch.utils.data.Subset]],
        checkpoint_dir: str,
    ):
        assert dtype in FlashMDWeakSymplectic.__supported_dtypes__

        # --- weak symplectic loss settings -------------------------------------------
        weak_symplectic_weight = float(self.hypers["weak_symplectic_weight"])
        if weak_symplectic_weight < 0.0:
            raise ValueError(
                f"weak_symplectic_weight must be >= 0 (got {weak_symplectic_weight})"
            )
        weak_symplectic_num_probes = int(self.hypers["weak_symplectic_num_probes"])
        weak_symplectic_probe_distribution = self.hypers[
            "weak_symplectic_probe_distribution"
        ]
        weak_symplectic_normalization = resolve_normalization(
            self.hypers["weak_symplectic_normalization"]
        )
        if weak_symplectic_weight > 0.0 and weak_symplectic_num_probes < 1:
            # with no probes there would be no chunk to carry the data loss backward
            raise ValueError(
                "weak_symplectic_num_probes must be at least 1 when "
                f"weak_symplectic_weight > 0 (got {weak_symplectic_num_probes})"
            )
        # cosine schedule of the weight (``weak_symplectic_weight`` is its maximum):
        # off by default, see SymplecticWeightScheduleHypers
        weak_symplectic_schedule = resolve_weight_schedule_hypers(
            self.hypers["weak_symplectic_schedule"]
        )
        if weak_symplectic_schedule["enabled"] and weak_symplectic_weight == 0.0:
            logging.warning(
                "weak_symplectic_schedule is enabled but weak_symplectic_weight is 0: "
                "the schedule has no effect"
            )
        # number of probes whose graph is alive at the same time (memory/speed trade-off:
        # 1 = least memory, weak_symplectic_num_probes = all probes batched at once)
        weak_symplectic_chunk_size = max(
            1,
            min(
                int(self.hypers["weak_symplectic_chunk_size"]),
                max(1, weak_symplectic_num_probes),
            ),
        )
        n_symplectic_chunks = (
            num_loss_chunks(weak_symplectic_num_probes, weak_symplectic_chunk_size)
            if weak_symplectic_weight > 0.0
            else 0
        )
        logging.info(
            "Weak symplectic regularization: "
            f"weight={weak_symplectic_weight}, "
            f"K={weak_symplectic_num_probes}, "
            f"chunk_size={weak_symplectic_chunk_size}, "
            "J v by double backward, "
            f"probe_distribution={weak_symplectic_probe_distribution}, "
            f"normalization={weak_symplectic_normalization}"
        )
        if weak_symplectic_schedule["enabled"] and weak_symplectic_weight > 0.0:
            ratio = weak_symplectic_schedule["min_weight_ratio"]
            low = ratio * weak_symplectic_weight
            high = weak_symplectic_weight
            if weak_symplectic_schedule["mode"] == "decrease":
                low, high = high, low
            logging.info(
                "Weak symplectic weight schedule: cosine "
                f"{weak_symplectic_schedule['mode']} from {low} to {high} between "
                f"{weak_symplectic_schedule['start_fraction']:.0%} and "
                f"{weak_symplectic_schedule['end_fraction']:.0%} of the training steps"
            )

        # Set time step for the model
        if self.hypers["timestep"] is None:
            raise ValueError(
                "`timestep` must be provided for training a FlashMD model. This "
                "corresponds to the time step (in fs) that separates the current "
                "positions and momenta from those (in the future) that the model is "
                "asked to predict. This is typically the time step used in the MD "
                "simulations that generated the data times the number of MD steps that "
                "were skipped in order to create each training sample. For more "
                "details, see the FlashMD tutorial in the documentation."
            )
        else:
            model.core.set_timestep(self.hypers["timestep"])

        # Set masses for the model
        if len(self.hypers["masses"]) == 0:
            logging.info(
                "No atomic masses were provided. Default atomic masses from ASE will "
                "be registered into the model."
            )
        atomic_mass_dict = {
            atomic_number: self.hypers["masses"].get(
                atomic_number, ase.data.atomic_masses[atomic_number]
            )
            for atomic_number in model.dataset_info.atomic_types
        }
        model.core.set_masses(atomic_mass_dict)

        is_distributed = resolve_distributed(self.hypers.get("distributed"))
        is_finetune = self.hypers["finetune"]["read_from"] is not None

        if is_distributed:
            device, world_size, rank = initialize_slurm_nccl_process_group(
                self.hypers["distributed_port"]
            )
        else:
            rank = 0
            world_size = 1

        if is_distributed:
            if len(devices) > 1:
                raise ValueError(
                    "Requested distributed training with the `multi-gpu` device. "
                    " If you want to run distributed training with FlashMD, please "
                    "set `device` to cuda."
                )
        else:
            device = devices[0]
            # only one device, as we don't support non-distributed multi-gpu for now

        if is_distributed:
            logging.info(f"Training on {world_size} devices with dtype {dtype}")
        else:
            logging.info(f"Training on device {device} with dtype {dtype}")

        # Apply fine-tuning strategy if provided
        if is_finetune:
            assert self.hypers["finetune"]["read_from"] is not None  # for mypy
            # ``inherit_heads`` is a one-time weight-copy initialization step that
            # must only run when finetuning first starts (a fresh ``Trainer``), not
            # again when a restart resumes an already-started finetuning run (a
            # restarted ``Trainer`` has its ``optimizer_state_dict`` restored by
            # ``load_checkpoint``); otherwise it would clobber the head weights
            # trained so far with a fresh copy from the (possibly since-changed, or
            # already stale-pruned) source target.
            model = apply_finetuning_strategy(
                model,
                self.hypers["finetune"],
                apply_inherit_heads=self.optimizer_state_dict is None,
                stale_targets=self._stale_finetune_targets,
            )
            method = self.hypers["finetune"]["method"]
            num_params = sum(p.numel() for p in model.parameters())
            num_trainable_params = sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )

            logging.info(f"Applied finetuning strategy: {method}")
            logging.info(
                f"Number of trainable parameters: {num_trainable_params} "
                f"[{num_trainable_params / num_params:.2%} %]"
            )

        train_or_load_composition_model(
            composition_model=model.additive_models[0],
            atomic_baseline=self.hypers["atomic_baseline"],
            train_datasets=train_datasets,
            other_additive_models=list(model.additive_models[1:]),
            device=device,
            batch_size=self.hypers["batch_size"],
            is_distributed=is_distributed,
            checkpoint_dir=checkpoint_dir,
        )

        if self.hypers["scale_targets"]:
            if isinstance(self.hypers["fixed_scaling_weights"], str) and not isinstance(
                self.hypers["atomic_baseline"], str
            ):
                raise ValueError(
                    "Can't use a checkpoint for the scaler without providing"
                    "a checkpoint also for the composition model."
                )
            train_or_load_scaler(
                scaler=MetatrainModel(
                    core=model.scaler,
                    additive_models=[],
                    scaler=None,
                    dataset_info=model.dataset_info,
                ),
                train_datasets=train_datasets,
                additive_models=model.additive_models,
                device=device,
                batch_size=self.hypers["batch_size"],
                is_distributed=is_distributed,
                fixed_weights=self.hypers["fixed_scaling_weights"],
                per_structure_targets=self.hypers["per_structure_targets"],
                checkpoint_dir=checkpoint_dir,
            )

        logging.info("Setting up data loaders")

        if is_distributed:
            train_samplers = [
                DistributedSampler(
                    train_dataset,
                    num_replicas=world_size,
                    rank=rank,
                    shuffle=True,
                    drop_last=True,
                )
                for train_dataset in train_datasets
            ]
            val_samplers = [
                DistributedSampler(
                    val_dataset,
                    num_replicas=world_size,
                    rank=rank,
                    shuffle=False,
                    drop_last=False,
                )
                for val_dataset in val_datasets
            ]
        else:
            train_samplers = [None] * len(train_datasets)
            val_samplers = [None] * len(val_datasets)

        # Extract additive models and scaler and move them to CPU/float64 so they
        # can be used in the collate function.
        additive_models = copy.deepcopy(model.additive_models).to(
            dtype=torch.float64, device="cpu"
        )
        scaler = copy.deepcopy(model.scaler).to(dtype=torch.float64, device="cpu")

        # Create collate functions:
        dataset_info = model.dataset_info
        train_targets = dataset_info.targets
        extra_data_info = dataset_info.extra_data

        if weak_symplectic_weight > 0.0:
            missing_targets = {"position", "momentum"} - set(train_targets)
            if missing_targets:
                raise ValueError(
                    "Weak symplectic regularization requires both 'position' and "
                    f"'momentum' targets. Missing: {sorted(missing_targets)}"
                )

        rotational_augmenter = O3Augmenter(
            target_info_dict=train_targets, extra_data_info_dict=extra_data_info
        )
        requested_neighbor_lists = get_requested_neighbor_lists(model)
        collate_fn_train = CollateFn(
            target_keys=list(train_targets.keys()),
            callables=[
                rotational_augmenter.apply_random_augmentations,
                get_system_with_neighbor_lists_transform(requested_neighbor_lists),
                get_remove_additive_transform(additive_models, train_targets),
                get_remove_scale_transform(scaler),
            ],
        )
        collate_fn_val = CollateFn(
            target_keys=list(train_targets.keys()),
            callables=[  # no augmentation for validation
                get_system_with_neighbor_lists_transform(requested_neighbor_lists),
                get_remove_additive_transform(additive_models, train_targets),
                get_remove_scale_transform(scaler),
            ],
        )

        # Now move the whole model (including the composition model and scaler,
        # which were trained in float64 above) to the training device and dtype.
        model.to(device=device, dtype=dtype)

        if weak_symplectic_normalization == "data_scales" and {
            "position",
            "momentum",
        } <= set(train_targets):
            # the scales D of the normalization, from the (now fitted) scaler
            like = torch.zeros(1, dtype=dtype, device=device)
            rows = species_scale_table(model, like)
            logging.info(
                "Weak symplectic normalization (data_scales), per atomic type: "
                + "; ".join(
                    f"{r['atomic_type']} (mass {r['mass']:.4g}): sigma_q "
                    f"{r['sigma_q']:.4g}, sigma_p {r['sigma_p']:.4g}, "
                    f"sigma_q*sqrt(m) {r['sigma_q_sqrt_m']:.4g}, "
                    f"sigma_p/sqrt(m) {r['sigma_p_over_sqrt_m']:.4g}, "
                    f"Lambda/c {r['lambda_over_c']:.4f}"
                    for r in rows
                )
                + f"; reference action c = {reference_action(model, like):.4g}"
            )
            species_warning = species_scaling_warning(rows)
            if all(
                abs(r["sigma_q"] - 1.0) < 1e-12 and abs(r["sigma_p"] - 1.0) < 1e-12
                for r in rows
            ):
                logging.warning(
                    "Weak symplectic normalization: all the position and momentum "
                    "scales are 1 (fixed_scaling_weights of 1, or scale_targets: "
                    "false), so the normalized loss is the raw loss in physical units"
                )
            elif species_warning is not None:
                logging.warning(f"Weak symplectic normalization: {species_warning}")

        if self.hypers["num_workers"] is None:
            num_workers = get_num_workers()
            logging.info(
                "Number of workers for data-loading not provided and chosen "
                f"automatically. Using {num_workers} workers."
            )
        else:
            num_workers = self.hypers["num_workers"]
            validate_num_workers(num_workers)

        max_atoms = self.hypers["max_atoms_per_batch"]

        # Create dataloader for the training datasets:
        train_dataloaders, epoch_samplers = build_train_dataloaders(
            train_datasets=train_datasets,
            train_distributed_samplers=train_samplers,
            collate_fn_train=collate_fn_train,
            batch_size=self.hypers["batch_size"],
            max_atoms_per_batch=max_atoms,
            min_atoms_per_batch=self.hypers["min_atoms_per_batch"],
            num_workers=num_workers,
        )
        train_dataloader = CombinedDataLoader(train_dataloaders, shuffle=True)

        # Create dataloader for the validation datasets:
        val_dataloaders = build_val_dataloaders(
            val_datasets=val_datasets,
            val_distributed_samplers=val_samplers,
            collate_fn_val=collate_fn_val,
            batch_size=self.hypers["batch_size"],
            max_atoms_per_batch=max_atoms,
            num_workers=num_workers,
        )
        val_dataloader = CombinedDataLoader(val_dataloaders, shuffle=False)

        # the wrapper (MetatrainModel) and its core, without DDP
        mtt_model = model
        core_model = model.core
        core_model.set_weak_symplectic_enabled(weak_symplectic_weight > 0.0)

        if is_distributed:
            model = DistributedDataParallel(model, device_ids=[device])

        # Create a loss function:
        loss_hypers = cast(Dict[str, LossSpecification], self.hypers["loss"])  # mypy
        loss_fn = LossAggregator(
            targets=train_targets,
            config=loss_hypers,
        )
        logging.info("Using the following loss functions:")
        for name, info in loss_fn.metadata.items():
            logging.info(f"{name}:")
            main = {k: v for k, v in info.items() if k != "gradients"}
            logging.info(main)
            if "gradients" not in info or len(info["gradients"]) == 0:
                continue
            logging.info("With gradients:")
            for grad, ginfo in info["gradients"].items():
                logging.info(f"\t{name}::{grad}: {ginfo}")

        if self.hypers["weight_decay"] is not None:
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=self.hypers["learning_rate"],
                weight_decay=self.hypers["weight_decay"],
            )
        else:
            optimizer = torch.optim.Adam(
                model.parameters(), lr=self.hypers["learning_rate"]
            )

        if self.optimizer_state_dict is not None and not is_finetune:
            # try to load the optimizer state dict, but this is only possible
            # if there are no new targets in the model (new parameters)
            if not self.has_new_targets:
                optimizer.load_state_dict(self.optimizer_state_dict)

        # Create a learning rate scheduler
        start_epoch = 0 if self.epoch is None else self.epoch + 1
        lr_scheduler = get_scheduler(
            optimizer, self.hypers, len(train_dataloader), start_epoch
        )

        if self.scheduler_state_dict is not None and not is_finetune:
            # same as the optimizer, try to load the scheduler state dict
            if not self.has_new_targets:
                lr_scheduler.load_state_dict(self.scheduler_state_dict)

        # steps of the symplectic weight schedule: the same span as the learning-rate
        # schedule, counted from the start of the run (so a restart resumes it)
        schedule_total_steps = (start_epoch + self.hypers["num_epochs"]) * len(
            train_dataloader
        )
        schedule_first_step = start_epoch * len(train_dataloader)

        def scheduled_symplectic_weight(step: int) -> float:
            """The weight of the symplectic loss at optimizer step ``step`` of this
            call to ``train``."""
            return weak_symplectic_weight * symplectic_weight_factor(
                schedule_first_step + step,
                schedule_total_steps,
                weak_symplectic_schedule,
            )

        per_structure_targets = self.hypers["per_structure_targets"]

        # Diagnostics of the symplectic loss (on rank 0, apart from the optimization),
        # configured by the ``diagnostics`` hypers. They work with or without the
        # symplectic loss, running their own forward pass on fixed batches.
        diagnostics = SymplecticDiagnostics(
            self.hypers["diagnostics"],
            model=mtt_model,
            train_datasets=train_datasets,
            val_datasets=val_datasets,
            collate_fn=collate_fn_val,
            loss_fn=loss_fn,
            per_structure_targets=per_structure_targets,
            dtype=dtype,
            device=device,
            rank=rank,
            loss_weight=weak_symplectic_weight,
            num_probes=weak_symplectic_num_probes,
            probe_distribution=weak_symplectic_probe_distribution,
            chunk_size=weak_symplectic_chunk_size,
            normalization=weak_symplectic_normalization,
            augmenter=rotational_augmenter,
            batch_size=self.hypers["batch_size"],
            checkpoint_dir=checkpoint_dir,
        )
        self.symplectic_diagnostics = diagnostics.records

        # Log the initial learning rate:
        logging.info(f"Base learning rate: {self.hypers['learning_rate']}")

        # Train the model:
        if self.best_metric is None:
            self.best_metric = float("inf")
        logging.info("Starting training")
        epoch = start_epoch
        # number of optimizer steps taken in this call to ``train``: the "step" of the
        # symplectic diagnostics (diagnostics at step t use the parameters theta_t)
        global_step = 0
        # weight of the symplectic loss of the current step (logged per epoch)
        symplectic_weight = scheduled_symplectic_weight(global_step)
        # With the symplectic loss, one forward is followed by several backward passes
        # (one per chunk of probes). DDP expects a single backward per forward: it
        # decides at forward time that its gradient hooks must fire, and then fails
        # ("marked a variable ready twice") at the second backward. So in that case the
        # forward and all the backward passes run in ``no_sync()`` (the hooks do
        # nothing) and the gradients are averaged over the ranks by hand afterwards.
        manual_grad_sync = is_distributed and weak_symplectic_weight > 0.0

        for epoch in range(start_epoch, start_epoch + self.hypers["num_epochs"]):
            for sampler in epoch_samplers:
                sampler.set_epoch(epoch)
            train_rmse_calculator = RMSEAccumulator(self.hypers["log_separate_blocks"])
            val_rmse_calculator = RMSEAccumulator(self.hypers["log_separate_blocks"])
            if self.hypers["log_mae"]:
                train_mae_calculator = MAEAccumulator(
                    self.hypers["log_separate_blocks"]
                )
                val_mae_calculator = MAEAccumulator(self.hypers["log_separate_blocks"])

            train_loss = 0.0
            train_data_loss = 0.0
            train_symplectic_loss = 0.0
            for batch in train_dataloader:
                symplectic_weight = scheduled_symplectic_weight(global_step)
                # the loss is skipped when its scheduled weight is exactly 0 (the
                # training forward then also skips the phase-space leaves and uses the
                # fast attention); the same on all ranks, as it only depends on the step
                compute_symplectic = symplectic_weight > 0.0

                # diagnostics of the symplectic loss, apart from the optimization: they
                # only read the current parameters
                diagnostics.step(global_step, symplectic_weight=symplectic_weight)

                core_model.set_weak_symplectic_enabled(compute_symplectic)
                optimizer.zero_grad()

                systems, targets, extra_data = unpack_batch(batch)
                systems, targets, extra_data = batch_to(
                    systems, targets, extra_data, dtype=dtype, device=device
                )
                with (
                    model.no_sync() if manual_grad_sync else contextlib.nullcontext()
                ):
                    raw_predictions = evaluate_model(
                        model,
                        systems,
                        {key: train_targets[key] for key in targets.keys()},
                        is_training=True,
                    )

                if compute_symplectic:
                    # lazy: nothing but the phase-space graph is built here, the chunks
                    # are produced (and backpropagated) below
                    symplectic_chunks = symplectic_loss_chunks(
                        mtt_model,
                        systems,
                        raw_predictions,
                        n_probes=weak_symplectic_num_probes,
                        probe_distribution=weak_symplectic_probe_distribution,
                        chunk_size=weak_symplectic_chunk_size,
                        normalization=weak_symplectic_normalization,
                    )
                else:
                    symplectic_chunks = None

                # average by the number of atoms
                predictions = average_by_num_atoms(
                    raw_predictions, systems, per_structure_targets
                )
                targets = average_by_num_atoms(targets, systems, per_structure_targets)

                # Apply per-property scales to the predictions before loss computation.
                # The targets from the dataloader have only been scaled per-target, and
                # not per-property. This transformation only applies to targets with
                # per-property scales (i.e. multiple blocks or multiple properties), and
                # leaves the others unchanged.
                predictions = mtt_model.scaler.apply_scales(
                    systems,
                    predictions,
                    remove=False,
                    use_per_target_scales=False,  # never before loss
                    use_per_property_scales=True,
                )

                data_loss_batch = loss_fn(predictions, targets, extra_data)
                backward_data_loss = data_loss_batch

                if is_distributed:
                    # make sure all parameters contribute to the gradient calculation
                    # to make torch DDP happy (e.g. when a target's head is kept in
                    # the model but not part of the current run's targets)
                    backward_data_loss = backward_data_loss + 0.0 * sum(
                        p.sum() for p in model.parameters() if p.requires_grad
                    )

                if symplectic_chunks is None:
                    symplectic_loss_batch = None
                    with (
                        model.no_sync() if manual_grad_sync else contextlib.nullcontext()
                    ):
                        backward_data_loss.backward()
                else:
                    # The symplectic loss is backpropagated chunk by chunk: the graph of
                    # a chunk (kept alive by create_graph=True) is freed before the next
                    # chunk is built, so peak memory scales with the chunk size and not
                    # with the number of probes. The graph of the model forward is shared
                    # by all the chunks and the data loss, so it is retained until the
                    # last backward (which also carries the data loss).
                    symplectic_loss_batch = 0.0
                    for i, chunk in enumerate(symplectic_chunks):
                        is_last = i == n_symplectic_chunks - 1
                        chunk_loss = symplectic_weight * chunk
                        if is_last:
                            chunk_loss = chunk_loss + backward_data_loss
                        # with DDP, the gradients are not reduced by the hooks (see
                        # ``manual_grad_sync``): they are averaged over the ranks once,
                        # after the last backward
                        with (
                            model.no_sync()
                            if manual_grad_sync
                            else contextlib.nullcontext()
                        ):
                            chunk_loss.backward(retain_graph=not is_last)
                        symplectic_loss_batch = symplectic_loss_batch + chunk.detach()

                if manual_grad_sync:
                    average_gradients_across_ranks(model.parameters(), world_size)

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), self.hypers["grad_clip_norm"]
                )
                optimizer.step()
                global_step += 1
                lr_scheduler.step()

                # the logged losses are summed over the processes, so that
                # loss == data_loss + symplectic_weight * symplectic_loss for every
                # batch (with a scheduled weight the identity does not hold for the
                # epoch sums)
                symplectic_loss_value = (
                    0.0
                    if symplectic_loss_batch is None
                    else float(symplectic_loss_batch)
                )
                data_loss_value = data_loss_batch.detach().reshape(())
                loss_stats = torch.stack(
                    [
                        data_loss_value + symplectic_weight * symplectic_loss_value,
                        data_loss_value,
                        torch.tensor(
                            symplectic_loss_value,
                            dtype=data_loss_value.dtype,
                            device=data_loss_value.device,
                        ),
                    ]
                )
                if is_distributed:
                    torch.distributed.all_reduce(loss_stats)

                train_loss += loss_stats[0].item()
                train_data_loss += loss_stats[1].item()
                train_symplectic_loss += loss_stats[2].item()

                # Reapply scales and accumulate quantities for computing train metrics,
                # but only if this is an epoch to log
                if epoch == start_epoch or epoch % self.hypers["log_interval"] == 0:
                    scaled_predictions = mtt_model.scaler.apply_scales(
                        systems,
                        predictions,
                        remove=False,
                        use_per_target_scales=True,
                        use_per_property_scales=False,
                    )
                    scaled_targets = mtt_model.scaler.apply_scales(
                        systems,
                        targets,
                        remove=False,
                        use_per_target_scales=True,
                        use_per_property_scales=False,
                    )
                    train_rmse_calculator.update(
                        scaled_predictions, scaled_targets, extra_data
                    )
                    if self.hypers["log_mae"]:
                        train_mae_calculator.update(
                            scaled_predictions, scaled_targets, extra_data
                        )

            # Compute train metrics if they are to be logged this epoch:
            if epoch == start_epoch or epoch % self.hypers["log_interval"] == 0:
                finalized_train_info = train_rmse_calculator.finalize(
                    not_per_atom=["positions_gradients"] + per_structure_targets,
                    is_distributed=is_distributed,
                    device=device,
                )
                if self.hypers["log_mae"]:
                    finalized_train_info.update(
                        train_mae_calculator.finalize(
                            not_per_atom=["positions_gradients"]
                            + per_structure_targets,
                            is_distributed=is_distributed,
                            device=device,
                        )
                    )

            # the validation forward is in evaluation mode: no phase-space leaves
            with torch.set_grad_enabled(
                any(target_info.gradients for target_info in train_targets.values())
            ):  # keep gradients on if any of the targets require them
                val_loss = 0.0
                for batch in val_dataloader:
                    systems, targets, extra_data = unpack_batch(batch)
                    systems, targets, extra_data = batch_to(
                        systems, targets, extra_data, dtype=dtype, device=device
                    )
                    predictions = evaluate_model(
                        model,
                        systems,
                        {key: train_targets[key] for key in targets.keys()},
                        is_training=False,
                    )

                    # average by the number of atoms
                    predictions = average_by_num_atoms(
                        predictions, systems, per_structure_targets
                    )
                    targets = average_by_num_atoms(
                        targets, systems, per_structure_targets
                    )

                    # Apply per-property scales to the predictions before loss
                    # computation. The targets from the dataloader have only been scaled
                    # per-target, and not per-property. This transformation only applies
                    # to targets with per-property scales (i.e. multiple blocks or
                    # multiple properties), and leaves the others unchanged.
                    predictions = mtt_model.scaler.apply_scales(
                        systems,
                        predictions,
                        remove=False,
                        use_per_target_scales=False,
                        use_per_property_scales=True,
                    )

                    val_loss_batch = loss_fn(predictions, targets, extra_data)

                    if is_distributed:
                        # sum the loss over all processes
                        torch.distributed.all_reduce(val_loss_batch)
                    val_loss += val_loss_batch.item()

                    # Reapply scales and accumulate quantities for computing val
                    # metrics. This is done for every epoch as validation metrics are
                    # needed for model selection
                    scaled_predictions = mtt_model.scaler.apply_scales(
                        systems,
                        predictions,
                        remove=False,
                        use_per_target_scales=True,
                        use_per_property_scales=False,
                    )
                    scaled_targets = mtt_model.scaler.apply_scales(
                        systems,
                        targets,
                        remove=False,
                        use_per_target_scales=True,
                        use_per_property_scales=False,
                    )
                    val_rmse_calculator.update(
                        scaled_predictions, scaled_targets, extra_data
                    )
                    if self.hypers["log_mae"]:
                        val_mae_calculator.update(
                            scaled_predictions, scaled_targets, extra_data
                        )

            # Compute val metrics:
            finalized_val_info = val_rmse_calculator.finalize(
                not_per_atom=["positions_gradients"] + per_structure_targets,
                is_distributed=is_distributed,
                device=device,
            )
            if self.hypers["log_mae"]:
                finalized_val_info.update(
                    val_mae_calculator.finalize(
                        not_per_atom=["positions_gradients"] + per_structure_targets,
                        is_distributed=is_distributed,
                        device=device,
                    )
                )

            # Now we log the information:
            if epoch == start_epoch or epoch % self.hypers["log_interval"] == 0:
                finalized_train_info = {
                    "loss": train_loss,
                    "data_loss": train_data_loss,
                    "symplectic_loss": train_symplectic_loss,
                    # weight of the last step of the epoch
                    "symplectic_weight": symplectic_weight,
                    **finalized_train_info,
                }
            finalized_val_info = {
                "loss": val_loss,
                **finalized_val_info,
            }

            if epoch == start_epoch:
                metric_logger = MetricLogger(
                    log_obj=ROOT_LOGGER,
                    dataset_info=mtt_model.dataset_info,
                    initial_metrics=[finalized_train_info, finalized_val_info],
                    names=["training", "validation"],
                )
            if epoch % self.hypers["log_interval"] == 0:
                metric_logger.log(
                    metrics=[finalized_train_info, finalized_val_info],
                    epoch=epoch,
                    rank=rank,
                    learning_rate=optimizer.param_groups[0]["lr"],
                )

            val_metric = get_selected_metric(
                finalized_val_info, self.hypers["best_model_metric"]
            )
            if val_metric < self.best_metric:
                self.best_metric = val_metric
                self.best_model_state_dict = copy.deepcopy(mtt_model.state_dict())
                self.best_epoch = epoch
                self.best_optimizer_state_dict = copy.deepcopy(optimizer.state_dict())

            is_last_epoch = epoch == start_epoch + self.hypers["num_epochs"] - 1
            if epoch % self.hypers["checkpoint_interval"] == 0 or is_last_epoch:
                if is_distributed:
                    torch.distributed.barrier()
                self.optimizer_state_dict = optimizer.state_dict()
                self.scheduler_state_dict = lr_scheduler.state_dict()
                self.epoch = epoch
                if rank == 0:
                    self.save_checkpoint(
                        mtt_model,
                        Path(checkpoint_dir) / f"model_{epoch}.ckpt",
                    )

        # the diagnostics of the final parameters, and the plots of all of them
        diagnostics.finish(
            global_step, symplectic_weight=scheduled_symplectic_weight(global_step)
        )
        # the exported/evaluated model never builds the phase-space graph
        core_model.set_weak_symplectic_enabled(False)

        # prepare for the checkpoint that will be saved outside the function
        self.epoch = epoch
        self.optimizer_state_dict = optimizer.state_dict()
        self.scheduler_state_dict = lr_scheduler.state_dict()

        if is_distributed:
            torch.distributed.destroy_process_group()

    def save_checkpoint(
        self,
        model: MetatrainModel[FlashMDWeakSymplectic, Scaler],
        path: Union[str, Path],
    ) -> None:
        checkpoint = model.get_checkpoint(self.best_model_state_dict)
        checkpoint["core"]["epoch"] = self.epoch
        checkpoint["core"]["best_epoch"] = self.best_epoch
        checkpoint.update(
            {
                "trainer_ckpt_version": self.__checkpoint_version__,
                "train_hypers": self.hypers,
                "epoch": self.epoch,
                "optimizer_state_dict": self.optimizer_state_dict,
                "scheduler_state_dict": self.scheduler_state_dict,
                "best_epoch": self.best_epoch,
                "best_metric": self.best_metric,
                "best_optimizer_state_dict": self.best_optimizer_state_dict,
            }
        )
        torch.save(
            checkpoint,
            check_file_extension(path, ".ckpt"),
        )

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint: Dict[str, Any],
        hypers: TrainerHypers,
        context: Literal["restart", "finetune"],
    ) -> "Trainer":
        trainer = cls(hypers)
        trainer.optimizer_state_dict = checkpoint["optimizer_state_dict"]
        trainer.scheduler_state_dict = checkpoint["scheduler_state_dict"]
        if context == "restart":
            trainer.epoch = checkpoint["epoch"]
        else:
            assert context == "finetune"
            trainer.epoch = None  # interpreted as zero in the training loop
        trainer.best_epoch = checkpoint["best_epoch"]
        trainer.best_metric = checkpoint["best_metric"]
        trainer.best_model_state_dict = MetatrainModel.get_state_dict_from_checkpoint(
            checkpoint, "best_model_state_dict"
        )
        trainer.best_optimizer_state_dict = checkpoint["best_optimizer_state_dict"]

        return trainer

    @classmethod
    def upgrade_checkpoint(
        cls, checkpoint: dict[str, Any], version: Optional[int] = None
    ) -> dict[str, Any]:
        return common_upgrade_checkpoint(
            checkpoint, version, cls.__checkpoint_version__, "trainer", checkpoints
        )

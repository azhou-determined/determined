import determined as det
from determined import core
from determined import pytorch
from typing import Any, Callable, Dict, Iterator, List, Optional
import torch
import torch.nn as nn
from determined.common import check

import logging


class PyTorchTrainContext:
    def __init__(self):
        self._core = core.init()
        self.cluster_info = det.get_cluster_info()
        self.models = []  # type: List[nn.Module]
        self.distributed = self._core.distributed
        self._distributed_backend = det._DistributedBackend()
        self.device = self._init_device()
        self.optimizers = []  # type: List[torch.optim.Optimizer]
        self._wrapped_models = {}  # type: Dict[nn.Module, nn.Module]
        self.managed_training = True
        self._loss_ids = {}  # type: Dict[torch.Tensor, int]
        self._current_batch_idx = None  # type: Optional[int]
        self.aggregation_frequency = None  # type: Optional[int]
        self.average_gradients = False

    def _init_device(self) -> torch.device:
        self.n_gpus = self.cluster_info and len(self.cluster_info.gpu_uuids) or 0
        if self._core.distributed.size > 1:
            if self.n_gpus > 0:
                # We launch a horovod process per GPU. Each process
                # needs to bind to a unique GPU.
                device = torch.device("cuda", self._core.distributed.local_rank)
                torch.cuda.set_device(device)
            else:
                device = torch.device("cpu")
        elif self.n_gpus > 0:
            device = torch.device("cuda", 0)
        else:
            device = torch.device("cpu")
        check.is_not_none(device)
        return device

    def _should_communicate_and_update(self) -> bool:
        if self._current_batch_idx is None:
            raise det.errors.InternalException("Training hasn't started.")
        return (self._current_batch_idx + 1) % self.aggregation_frequency == 0

    def wrap_model(self, model: torch.nn.Module) -> torch.nn.Module:
        model = model.to(self.device)
        model_id = len(self.models)
        self._main_model = nn.Module()
        self._main_model.__setattr__(f"model_{model_id}", model)
        self.models.append(model)
        return model

    def backward(
        self,
        loss: torch.Tensor,
        gradient: Optional[torch.Tensor] = None,
        retain_graph: bool = False,
        create_graph: bool = False,
    ) -> None:
        loss.backward(  # type: ignore
            gradient=gradient, retain_graph=retain_graph, create_graph=create_graph
        )

    def wrap_optimizer(
        self,
        optimizer: torch.optim.Optimizer,
        backward_passes_per_step: int = 1,
        gradient_compression: bool = False,
    ) -> torch.optim.Optimizer:
        self.optimizers.append(optimizer)
        return optimizer

    @staticmethod
    def _average_gradients(parameters: Any, divisor: int) -> None:
        check.gt_eq(divisor, 1)
        if divisor == 1:
            return

        divisor_value = float(divisor)
        for p in filter(lambda param: param.grad is not None, parameters):
            p.grad.data.div_(divisor_value)

    def step_optimizer(
        self,
        optimizer: torch.optim.Optimizer,
        clip_grads: Optional[Callable[[Iterator], None]] = None,
        auto_zero_grads: bool = True,
        scaler: Optional[Any] = None,
    ) -> None:
        check.true(
            auto_zero_grads or self.aggregation_frequency == 1,
            "if optimizations.aggregation_frequency is larger than 1, "
            "you can only set auto_zero_grads to be true. ",
            )

        if not self._should_communicate_and_update():
            return

        parameters = [p for group in optimizer.param_groups for p in group.get("params", [])]

        if self.average_gradients:
            self._average_gradients(parameters=parameters, divisor=self.aggregation_frequency)

        if auto_zero_grads:
            optimizer.zero_grad()

    def __enter__(self):
        self._core.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._core.__exit__(exc_type, exc_val, exc_tb)


class Trainer:
    def __init__(self, trial: pytorch.PyTorchTrial, context: PyTorchTrainContext):
        self.trial = trial
        self.context = context
        self.core_context = self.context._core

    def train(
        self,
        max_epochs: Optional[int] = None,
        # OR
        max_batches: Optional[int] = None,
        min_checkpoint_period: int = 1,
        min_validation_period: int = 1,
        average_training_metrics: bool = True,
        average_aggregated_gradients: bool = True,
        aggregation_frequency: int = 2,
        searcher_metric="validation_loss",
        profiling=True,
        profiling_start=0,
        profiling_end=10,
        sync_timings=None,
        checkpoint_policy="best|all|none"
    ):
        assert (max_epochs is None) ^ (max_batches is None), "Either max_batches or max_epochs must be defined"

        # TODO: figure out a better way to do this.
        self.context.aggregation_frequency = aggregation_frequency

        # Get the minimum of checkpoint_period or validation_period for metrics reporting
        train_step_size = min(min_checkpoint_period, min_validation_period)

        train_data = self.trial.build_training_data_loader()
        num_replicas = self.core_context.distributed.size
        rank = self.core_context.distributed.rank

        training_loader = train_data.get_data_loader(
            repeat=False, num_replicas=num_replicas, rank=rank
        )

        # Set models to training mode
        for model in self.context.models:
            model.train()

        epochs = 0
        batches = 0

        # Report training has started
        self.core_context.train.set_status("training")

        while (max_epochs and epochs <= max_epochs) or (max_batches and batches <= max_batches):
            metrics = []
            for batch_idx, batch in enumerate(training_loader):
                self.context._current_batch_idx = batches
                training_metrics = self.trial.train_batch(batch, epochs, batch_idx)
                batches += 1
                if max_batches and batches == train_step_size:
                    self.core_context.train.report_training_metrics(
                        batches, training_metrics
                    )
                if max_batches and batches % min_validation_period == 0:
                    self.validate()
                metrics.append(training_metrics)

            epochs += 1
            if max_epochs and epochs == train_step_size:
                self.core_context.train.report_training_metrics(
                    epochs, metrics
                )
            if max_epochs and epochs % min_validation_period == 0:
                self.validate()

        return

    def validate(self):
        val_data = self.trial.build_validation_data_loader()
        num_replicas = self.core_context.distributed.size
        rank = self.core_context.distributed.rank

        val_loader = val_data.get_data_loader(
            repeat=False, num_replicas=num_replicas, rank=rank
        )

        # Set models to evaluation mode
        for model in self.context.models:
            model.eval()

        # Report training has started
        self.core_context.train.set_status("validating")

        for batch_idx, batch in enumerate(val_loader):
            val_metrics = self.trial.evaluate_batch(batch)
            print(f"Validation metrics: {val_metrics}")

        # Set models back to training mode
        for model in self.context.models:
            model.train()
        return


def init():
    context = PyTorchTrainContext()
    return context


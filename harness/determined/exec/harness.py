import argparse
import contextlib
import faulthandler
import logging
import sys
from typing import Iterator, Optional, Type

import determined as det
from determined import core, horovod, load
from determined.common.api import analytics, certs


@contextlib.contextmanager
def maybe_periodic_stacktraces(debug_enabled: bool) -> Iterator[None]:
    if debug_enabled:
        faulthandler.dump_traceback_later(30, repeat=True)
    try:
        yield
    finally:
        if debug_enabled:
            faulthandler.cancel_dump_traceback_later()


def main(train_entrypoint: str) -> int:
    info = det.get_cluster_info()
    assert info is not None, "must be run on-cluster"
    assert info.task_type == "TRIAL", f'must be run with task_type="TRIAL", not "{info.task_type}"'

    # TODO: refactor profiling to to not use the cli_cert.
    certs.cli_cert = certs.default_load(info.master_url)

    trial_class = load.trial_class_from_entrypoint(train_entrypoint)

    if info.container_rank == 0:
        try:
            analytics.send_analytics("trial_loaded", analytics.get_trial_analytics(trial_class))
        except Exception as e:
            logging.debug(f"Cannot send analytics: {e}")

    # We can't import pytorch directly because if running TfKerasTrials with an image that contains
    # both torch and keras, keras will throw exceptions due to unexpected CUDNN library versions.
    if hasattr(det, "pytorch") and issubclass(trial_class, det.pytorch.PyTorchTrial):
        return _run_pytorch_trial(trial_class, info)

    # TODO: Don't include EnvContext object in the future high-level APIs for PyTorch or Keras.
    # It was natural to create this big-blob-of-config object, but it was a mistake to pass it into
    # the lowest layers of the harness code; it's too large of an object to be easily mockable,
    # which is part of why building local training mode has always been a challenge.
    #
    # A better pattern is to pass in exactly the information that is necessary at each layer.  We
    # will use that pattern for the future high-level APIs, but it's not worth refactoring e.g. the
    # TFKerasTrialController or EstimatorTrialController to add that functionality, so for now we
    # continue with the legacy strategy.

    env = det.EnvContext(
        master_url=info.master_url,
        master_cert_file=info.master_cert_file,
        master_cert_name=info.master_cert_name,
        experiment_config=info.trial._config,
        hparams=info.trial.hparams,
        latest_checkpoint=info.latest_checkpoint,
        steps_completed=info.trial._steps_completed,
        use_gpu=bool(info.gpu_uuids),
        container_gpus=info.gpu_uuids,
        slot_ids=info.slot_ids,
        debug=info.trial._debug,
        det_trial_id=str(info.trial.trial_id),
        det_experiment_id=str(info.trial.experiment_id),
        det_agent_id=info.agent_id,
        det_cluster_id=info.cluster_id,
        trial_seed=info.trial.trial_seed,
        trial_run_id=info.trial._trial_run_id,
        allocation_id=info.allocation_id,
        managed_training=True,
        test_mode=False,
        on_cluster=True,
    )

    if hasattr(det, "keras") and issubclass(trial_class, det.keras.TFKerasTrial):
        return _run_keras_trial(trial_class, info, env)

    if hasattr(det, "estimator") and issubclass(trial_class, det.estimator.EstimatorTrial):
        return _run_estimator_trial(trial_class, info, env)

    if hasattr(det, "pytorch") and issubclass(trial_class, det.pytorch.deepspeed.DeepspeedTrial):
        return _run_deepspeed_trial(trial_class, info, env)

    return 0


def _run_keras_trial(
    trial_class: "Type[det.keras.TFKerasTrial]", info: det.ClusterInfo, env: det.EnvContext
) -> int:
    from determined import keras

    det.common.set_logger(env.debug)
    logging.debug("Starting harness.")
    with maybe_periodic_stacktraces(env.debug):
        # Initialize framework-specific details (dtrain framework, random seeds, etc).
        distributed_backend = det._DistributedBackend()
        keras.TFKerasTrialController.pre_execute_hook(env, distributed_backend)

        distributed = None
        if distributed_backend.use_horovod():
            distributed = core.DistributedContext.from_horovod(horovod.hvd)
        elif len(info.container_addrs) > 1 or len(info.slot_ids) > 1:
            raise ValueError(
                "In multi-slot tasks for TFKerasTrial, the determined.exec.harness module must be "
                "wrapped in the following launch layer: determined.launch.horovod."
            )

        with core.init(
            distributed=distributed,
            preempt_mode=core.PreemptMode.ChiefOnly,
            tensorboard_mode=core.TensorboardMode.MANUAL,
        ) as core_context:
            trial_context = keras.TFKerasTrialContext(core_context, env)

            trial_inst = trial_class(trial_context)

            logging.info(
                f"Creating {keras.TFKerasTrialController.__name__} with " f"{trial_class.__name__}."
            )
            controller = keras.TFKerasTrialController.from_trial(
                trial_inst=trial_inst,
                context=trial_context,
                env=env,
            )

            controller.run()
    return 0


def _run_estimator_trial(
    trial_class: "Type[det.estimator.EstimatorTrial]", info: det.ClusterInfo, env: det.EnvContext
) -> int:
    from determined import estimator

    det.common.set_logger(env.debug)
    logging.debug("Starting harness.")

    with maybe_periodic_stacktraces(env.debug):
        distributed_backend = det._DistributedBackend()
        estimator.EstimatorTrialController.pre_execute_hook(env, distributed_backend)

        distributed = None
        if distributed_backend.use_horovod():
            distributed = core.DistributedContext.from_horovod(horovod.hvd)
        elif len(info.container_addrs) > 1 or len(info.slot_ids) > 1:
            raise ValueError(
                "In multi-slot tasks for EstimatorTrial, the determined.exec.harness module must "
                "be wrapped in the following launch layer: determined.launch.horovod."
            )

        with core.init(
            distributed=distributed,
            preempt_mode=core.PreemptMode.ChiefOnly,
            tensorboard_mode=core.TensorboardMode.MANUAL,
        ) as core_context:
            trial_context = estimator.EstimatorTrialContext(core_context, env)
            trial_inst = trial_class(trial_context)

            logging.info(
                f"Creating {estimator.EstimatorTrialController.__name__} with "
                f"{trial_class.__name__}."
            )
            controller = estimator.EstimatorTrialController(
                estimator=trial_inst.build_estimator(),
                user_train_spec=trial_inst.build_train_spec(),
                val_spec=trial_inst.build_validation_spec(),
                serving_input_receiver_fns=trial_inst.build_serving_input_receiver_fns(),
                context=trial_context,
                env=env,
            )

            controller.run()
    return 0


def _run_deepspeed_trial(
    trial_class: "Type[det.pytorch.DeepspeedTrial]", info: det.ClusterInfo, env: det.EnvContext
) -> int:
    from determined.pytorch import deepspeed

    det.common.set_logger(env.debug)
    logging.debug("Starting harness.")

    with maybe_periodic_stacktraces(env.debug):
        distributed_backend = det._DistributedBackend()
        deepspeed.DeepSpeedTrialController.pre_execute_hook(env, distributed_backend)

        distributed = None
        if distributed_backend.use_deepspeed():
            distributed = core.DistributedContext.from_deepspeed()
        elif len(info.container_addrs) > 1 or len(info.slot_ids) > 1:
            raise ValueError(
                "In multi-slot tasks for DeepspeedTrial, the determined.exec.harness module must "
                "be wrapped in the following launch layer: determined.launch.deepspeed."
            )

        with core.init(
            distributed=distributed,
            preempt_mode=core.PreemptMode.ChiefOnly,
            tensorboard_mode=core.TensorboardMode.MANUAL,
        ) as core_context:
            trial_context = deepspeed.DeepSpeedTrialContext(core_context, env)
            trial_inst = trial_class(trial_context)

            logging.info(
                f"Creating {deepspeed.DeepSpeedTrialController.__name__} with "
                f"{trial_class.__name__}."
            )
            controller = deepspeed.DeepSpeedTrialController(
                trial_inst=trial_inst,
                context=trial_context,
                env=env,
            )

            controller.run()
    return 0


def _run_pytorch_trial(
    trial_class: "Type[det.pytorch.PyTorchTrial]",
    info: det.ClusterInfo,
) -> int:
    from determined import pytorch

    det.common.set_logger(info.trial._debug)

    logging.debug("Starting harness.")

    with maybe_periodic_stacktraces(info.trial._debug):
        with pytorch.init(
            hparams=info.trial.hparams,
            exp_conf=info.trial._config,
            aggregation_frequency=int(info.trial._config["optimizations"]["aggregation_frequency"]),
        ) as train_context:
            fp16_compression = bool(info.trial._config["optimizations"]["gradient_compression"])
            average_aggregated_gradients = bool(
                info.trial._config["optimizations"]["average_aggregated_gradients"]
            )

            train_context._set_default_gradient_compression(fp16_compression)
            train_context._set_default_average_aggregated_gradients(average_aggregated_gradients)
            train_context._set_is_pre_trainer()

            trial_inst = trial_class(train_context)

            if train_context.distributed.size > 1 and not train_context.distributed.rank == 0:
                log_level = logging.DEBUG if info.trial._debug else logging.WARNING
                logging.getLogger().setLevel(log_level)

            logging.info(
                f"Creating {pytorch._PyTorchTrialController.__name__} with {trial_class.__name__}."
            )

            trainer = pytorch.Trainer(trial_inst, train_context)

            trainer.configure_profiler(
                sync_timings=bool(info.trial._config["profiling"]["sync_timings"]),
                enabled=bool(info.trial._config["profiling"]["enabled"]),
                begin_on_batch=info.trial._config["profiling"]["begin_on_batch"],
                end_after_batch=info.trial._config["profiling"]["end_after_batch"],
            )

            if "global_batch_size" in info.trial.hparams:
                global_batch_size = int(
                    info.trial.hparams["global_batch_size"]
                )  # type: Optional[int]
            else:
                global_batch_size = None

            trainer.fit(
                checkpoint_period=pytorch.TrainUnit._from_values(
                    **info.trial._config["min_checkpoint_period"],
                    global_batch_size=global_batch_size,
                ),
                validation_period=pytorch.TrainUnit._from_values(
                    **info.trial._config["min_validation_period"],
                    global_batch_size=global_batch_size,
                ),
                reporting_period=pytorch.Batch(info.trial._config["scheduling_unit"]),
                checkpoint_policy=info.trial._config["checkpoint_policy"],
                latest_checkpoint=info.latest_checkpoint,
                step_zero_validation=info.trial._config["perform_initial_validation"],
                test_mode=False,
            )

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("train_entrypoint")
    args = parser.parse_args()
    sys.exit(main(args.train_entrypoint))

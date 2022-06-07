import contextlib
import logging
from typing import Type

from determined import core, horovod, load

import determined as det


class Trainer:
    def __init__(self, trial_class: Type[det.Trial]):
        self.trial_class = trial_class

    def build(self):
        return

    def __enter__(self):
        return

    @contextlib.contextmanager
    def train(self):
        info = det.get_cluster_info()
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
            det_trial_unique_port_offset=info.trial._unique_port_offset,
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

        controller_class = load.get_trial_controller_class(self.trial_class)

        distributed_backend = det._DistributedBackend()
        controller_class.pre_execute_hook(env, distributed_backend)

        distributed = None
        if distributed_backend.use_horovod():
            distributed = core.DistributedContext.from_horovod(horovod.hvd)
        elif distributed_backend.use_deepspeed():
            distributed = core.DistributedContext.from_deepspeed()
        elif distributed_backend.use_torch():
            distributed = core.DistributedContext.from_torch_distributed()
        elif len(info.container_addrs) > 1 or len(info.slot_ids) > 1:
            raise ValueError(
                "In multi-slot tasks, the determined.exec.harness module must not be invoked "
                "directly.  Instead, it must be wrapped in one of the following launch layers: "
                "determined.launch.horovod, determined.launch.deepspeed"
            )

        with core.init(
                distributed=distributed, preempt_mode=core.PreemptMode.ChiefOnly
        ) as core_context:
            trial_context = self.trial_class.trial_context_class(core_context, env)

            trial_inst = self.trial_class(trial_context)

            logging.info(f"Creating {controller_class.__name__} with {self.trial_class.__name__}.")
            controller = controller_class.from_trial(
                trial_inst=trial_inst,
                context=trial_context,
                env=env,
            )

            controller.run()

        return


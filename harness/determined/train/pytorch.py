import contextlib
import logging
from typing import Type, Any

from determined import core, horovod, load

import determined as det


class Trainer:
    def __init__(self, distributed_context: det.core.DistributedContext):
        self.distributed = distributed_context
        info = det.get_cluster_info()
        self.env = det.EnvContext(
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

    def build_trial(self, trial_class: Type[det.Trial]) -> det.Trial:
        trial_context = trial_class.trial_context_class(self.core_context, self.env)
        self.controller_class = load.get_trial_controller_class(trial_class)
        self.trial_context = trial_context
        return trial_class(trial_context)

    def __enter__(self):
        core_context = core.init(distributed=self.distributed, preempt_mode=core.PreemptMode.ChiefOnly)
        self.core_context = core_context.__enter__()
        return self

    def __exit__(self, typ: type, value: Exception, tb: Any) -> None:
        self.core_context.__exit__()
        return

    @contextlib.contextmanager
    def train(self, trial: det.Trial):
        controller = self.controller_class.from_trial(
            trial_inst=trial,
            context=self.trial_context,
            env=self.env,
        )
        controller.run()




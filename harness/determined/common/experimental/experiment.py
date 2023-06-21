import enum
import sys
import time
import warnings
from typing import Any, Dict, Iterator, List, Optional

from determined.common import api
from determined.common.api import bindings
from determined.common.experimental import checkpoint, trial


# Wrap the autogenerated bindings in something a little more ergonomic.
class ExperimentState(enum.Enum):
    UNSPECIFIED = bindings.experimentv1State.UNSPECIFIED.value
    ACTIVE = bindings.experimentv1State.ACTIVE.value
    PAUSED = bindings.experimentv1State.PAUSED.value
    STOPPING_COMPLETED = bindings.experimentv1State.STOPPING_COMPLETED.value
    STOPPING_CANCELED = bindings.experimentv1State.STOPPING_CANCELED.value
    STOPPING_ERROR = bindings.experimentv1State.STOPPING_ERROR.value
    COMPLETED = bindings.experimentv1State.COMPLETED.value
    CANCELED = bindings.experimentv1State.CANCELED.value
    ERROR = bindings.experimentv1State.ERROR.value
    DELETED = bindings.experimentv1State.DELETED.value
    DELETING = bindings.experimentv1State.DELETING.value
    DELETE_FAILED = bindings.experimentv1State.DELETE_FAILED.value
    STOPPING_KILLED = bindings.experimentv1State.STOPPING_KILLED.value
    QUEUED = bindings.experimentv1State.QUEUED.value
    PULLING = bindings.experimentv1State.PULLING.value
    STARTING = bindings.experimentv1State.STARTING.value
    RUNNING = bindings.experimentv1State.RUNNING.value

    def _to_bindings(self) -> bindings.experimentv1State:
        return bindings.experimentv1State(self.value)


class Experiment:
    """
    A class representing an Experiment object.

    An Experiment object is usually obtained from
    ``determined.experimental.client.create_experiment()``
    or ``determined.experimental.client.get_experiment()`` and contains helper methods that support
    querying the set of checkpoints associated with an experiment.

    Attributes:
        id: ID of experiment object in database.
        session: HTTP request session.
        config: (Mutable, Optional[Dict]) Experiment config for the experiment.
        state: (Mutable, Optional[experimentv1State) State of the experiment.

    Note:
        All attributes are cached by default.

        The `config` and `state` attributes are mutable and may be changed by methods that update
        these values, either automatically (eg. `wait()`) or explicitly with `reload()`.
    """

    def __init__(
        self,
        experiment_id: int,
        session: api.Session,
    ):
        self._id = experiment_id
        self._session = session

        # These properties may be mutable and will be set by _hydrate()
        self.config: Optional[Dict[str, Any]] = None
        self.state: Optional[bindings.experimentv1State] = None

    @property
    def id(self) -> int:
        return self._id

    def _hydrate(self, exp: bindings.v1Experiment) -> None:
        self.config = exp.config
        self.state = exp.state

    def reload(self) -> None:
        """
        Explicit refresh of cached properties.
        """
        resp = bindings.get_GetExperiment(self._session, experimentId=self.id).experiment
        self._hydrate(resp)

    def activate(self) -> None:
        bindings.post_ActivateExperiment(self._session, id=self._id)

    def archive(self) -> None:
        bindings.post_ArchiveExperiment(self._session, id=self._id)

    def cancel(self) -> None:
        bindings.post_CancelExperiment(self._session, id=self._id)

    def delete(self) -> None:
        """
        Delete an experiment and all its artifacts from persistent storage.

        You must be authenticated as admin to delete an experiment.
        """
        bindings.delete_DeleteExperiment(self._session, experimentId=self._id)

    def get_trials(
        self,
        sort_by: trial.TrialSortBy = trial.TrialSortBy.ID,
        order_by: trial.TrialOrderBy = trial.TrialOrderBy.ASCENDING,
    ) -> List[trial.Trial]:
        warnings.warn(
            "Experiment.get_trials() has been deprecated and will be removed in a future version."
            "Please call Experiment.list_trials() instead.",
            FutureWarning,
            stacklevel=2,
        )
        return list(self.list_trials(sort_by, order_by))

    def list_trials(
        self,
        sort_by: trial.TrialSortBy = trial.TrialSortBy.ID,
        order_by: trial.TrialOrderBy = trial.TrialOrderBy.ASCENDING,
    ) -> Iterator[trial.Trial]:
        """
        Get an iterator of :class:`~determined.experimental.Trial` instances
        representing trials for an experiment.

        Arguments:
            sort_by: Which field to sort by. See :class:`~determined.experimental.TrialSortBy`.
            order_by: Whether to sort in ascending or descending order. See
                :class:`~determined.experimental.TrialOrderBy`.

        Note:
            This method returns an Iterable type that lazily instantiates response objects. To
            fetch all models at once, call list(list_trials()).
        """

        def get_with_offset(offset: int) -> bindings.v1GetExperimentTrialsResponse:
            return bindings.get_GetExperimentTrials(
                self._session,
                experimentId=self._id,
                offset=offset,
                orderBy=bindings.v1OrderBy(order_by.value),
                sortBy=bindings.v1GetExperimentTrialsRequestSortBy(sort_by.value),
            )

        resps = api.read_paginated(get_with_offset)

        for r in resps:
            for t in r.trials:
                yield trial.Trial._from_bindings(t, self._session)

    def await_first_trial(self, interval: float = 0.1) -> trial.Trial:
        """
        Wait for the first trial to be started for this experiment.
        """
        while True:
            resp = bindings.get_GetExperimentTrials(
                self._session,
                experimentId=self._id,
                orderBy=bindings.v1OrderBy.ASC,
                sortBy=bindings.v1GetExperimentTrialsRequestSortBy.START_TIME,
            )
            if len(resp.trials) > 0:
                return trial.Trial._from_bindings(resp.trials[0], self._session)
            time.sleep(interval)

    def kill(self) -> None:
        bindings.post_KillExperiment(self._session, id=self._id)

    def pause(self) -> None:
        bindings.post_PauseExperiment(self._session, id=self._id)

    def unarchive(self) -> None:
        bindings.post_UnarchiveExperiment(self._session, id=self._id)

    def wait(self, interval: float = 5.0) -> ExperimentState:
        """
        Wait for the experiment to reach a complete or terminal state.

        Arguments:
            interval (int, optional): An interval time in seconds before checking
                next experiment state.
        """
        elapsed_time = 0.0
        while True:
            self.reload()
            if self.state in (
                bindings.experimentv1State.COMPLETED,
                bindings.experimentv1State.CANCELED,
                bindings.experimentv1State.DELETED,
                bindings.experimentv1State.ERROR,
            ):
                return ExperimentState(self.state.value)
            elif self.state == bindings.experimentv1State.PAUSED:
                raise ValueError(
                    f"Experiment {self.id} is in paused state. Make sure the experiment is active."
                )
            else:
                # ACTIVE, STOPPING_COMPLETED, etc.
                time.sleep(interval)
                elapsed_time += interval
                if elapsed_time % 60 == 0:
                    print(
                        f"Waiting for Experiment {self.id} to complete. "
                        f"Elapsed {elapsed_time / 60} minutes",
                        file=sys.stderr,
                    )

    def top_checkpoint(
        self,
        sort_by: Optional[str] = None,
        smaller_is_better: Optional[bool] = None,
    ) -> checkpoint.Checkpoint:
        """
        Return the :class:`~determined.experimental.Checkpoint` for this experiment that
        has the best validation metric, as defined by the ``sort_by`` and ``smaller_is_better``
        arguments.

        Arguments:
            sort_by (string, optional): The name of the validation metric to
                order checkpoints by. If this parameter is not specified, the metric
                defined in the experiment configuration ``searcher`` field will be used.

            smaller_is_better (bool, optional): Specifies whether to sort the
                metric above in ascending or descending order. If ``sort_by`` is unset,
                this parameter is ignored. By default, the value of ``smaller_is_better``
                from the experiment's configuration is used.
        """
        checkpoints = self.top_n_checkpoints(
            1, sort_by=sort_by, smaller_is_better=smaller_is_better
        )

        if not checkpoints:
            raise AssertionError("No checkpoints found for experiment {}".format(self.id))

        return checkpoints[0]

    def top_n_checkpoints(
        self,
        limit: int,
        sort_by: Optional[str] = None,
        smaller_is_better: Optional[bool] = None,
    ) -> List[checkpoint.Checkpoint]:
        """
        Return the N :class:`~determined.experimental.Checkpoint` instances with the best
        validation metrics, as defined by the ``sort_by`` and ``smaller_is_better``
        arguments. This method will return the best checkpoint from the
        top N best-performing distinct trials of the experiment. Only checkpoints in
        a ``COMPLETED`` state with a matching ``COMPLETED`` validation are considered.

        Arguments:
            limit (int): The maximum number of checkpoints to return.

            sort_by (string, optional): The name of the validation metric to use for
                sorting checkpoints. If this parameter is unset, the metric defined
                in the experiment configuration searcher field will be
                used.

            smaller_is_better (bool, optional): Specifies whether to sort the
                metric above in ascending or descending order. If ``sort_by`` is unset,
                this parameter is ignored. By default, the value of ``smaller_is_better``
                from the experiment's configuration is used.
        """

        def get_with_offset(offset: int) -> bindings.v1GetExperimentCheckpointsResponse:
            return bindings.get_GetExperimentCheckpoints(
                self._session,
                id=self._id,
                offset=offset,
                states=[bindings.checkpointv1State.COMPLETED],
            )

        resps = api.read_paginated(get_with_offset)

        checkpoints = [
            checkpoint.Checkpoint._from_bindings(c, self._session)
            for r in resps
            for c in r.checkpoints
        ]

        if not checkpoints:
            raise AssertionError("No checkpoint found for experiment {}".format(self.id))

        if not sort_by:
            training = checkpoints[0].training
            assert training
            config = training.experiment_config
            sb = config.get("searcher", {}).get("metric")
            if not isinstance(sb, str):
                raise ValueError(
                    "no searcher.metric found in experiment config; please provide a sort_by metric"
                )
            sort_by = sb
            smaller_is_better = config.get("searcher", {}).get("smaller_is_better", True)

        reverse = not smaller_is_better

        def key(ckpt: checkpoint.Checkpoint) -> Any:
            training = ckpt.training
            assert training
            metric = training.validation_metrics.get("avgMetrics") or {}
            metric = metric.get(sort_by)

            # Return a tuple that ensures checkpoints missing metrics appear last.
            if reverse:
                return metric is not None, metric
            else:
                return metric is None, metric

        checkpoints.sort(reverse=not smaller_is_better, key=key)

        # Ensure returned checkpoints are from distinct trials.
        t_ids = set()
        checkpoint_refs = []
        for ckpt in checkpoints:
            training = ckpt.training
            assert training
            if training.trial_id not in t_ids:
                checkpoint_refs.append(ckpt)
                t_ids.add(training.trial_id)

        return checkpoint_refs[:limit]

    def __repr__(self) -> str:
        return "Experiment(id={})".format(self.id)

    @classmethod
    def _from_bindings(
        cls, exp_bindings: bindings.v1Experiment, session: api.Session
    ) -> "Experiment":
        exp = cls(session=session, experiment_id=exp_bindings.id)
        exp._hydrate(exp_bindings)
        return exp

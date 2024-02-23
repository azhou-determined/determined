import abc
import logging
import queue
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List

import psutil

from determined import core
from determined.common import api, constants
from determined.common.api import bindings


try:
    import pynvml
except ImportError:
    pynvml = None

logger = logging.getLogger("determined.core")


class ProfilerContext:
    def __init__(
        self,
        session: api.Session,
        agent_id: str,
        trial_id: int,
        run_id: int,
        distributed: core.DistributedContext,
    ) -> None:
        self._session = session
        self._agent_id = agent_id
        self._trial_id = trial_id
        self._run_id = run_id
        # xxx: pass in train context here to use for metrics reporting API?
        self._distributed = distributed
        self._on = False

        self._collector = None
        self._shipper = None

    # xxx: other potential configs (collection interval, flush interval, max collection time)
    def on(self, interval: int = 1) -> None:
        if self._on:
            logger.warning(f"Profiler is already on.")
            return

        # Only enable on local chief workers.
        if self._distributed.local_rank != 0:
            return

        logger.info(f"Starting system metrics profiling.")

        metrics_queue = queue.Queue()
        self._collector = _Collector(
            metrics_queue=metrics_queue,
            collection_interval=interval,
        )
        self._shipper = _Shipper(
            session=self._session,
            agent_id=self._agent_id,
            trial_id=self._trial_id,
            run_id=self._run_id,
            metrics_queue=metrics_queue,
        )

        self._collector.start()
        self._shipper.start()
        self._on = True

    def off(self) -> None:
        if not self._on:
            logger.warning(f"Profiler is already off.")
            return
        logger.info(f"Stopping system metrics profiling.")
        self.close()
        self._on = False

    def close(self) -> None:
        if self._collector:
            self._collector.stop()
        if self._shipper:
            self._shipper.stop()


class DummyProfilerContext(ProfilerContext):
    def __init__(
        self,
    ) -> None:
        pass

    def on(self, interval: int = 1) -> None:
        pass

    def off(self) -> None:
        pass

    def close(self) -> None:
        pass


class _MetricGroupCollector(metaclass=abc.ABCMeta):
    group: str

    def __init__(self):
        pass

    @abc.abstractmethod
    def collect(self) -> Dict[str, Any]:
        pass


class _Network(_MetricGroupCollector):
    group = "network"

    def __init__(self) -> None:
        super().__init__()

        # Set initial values for throughput calculations.
        self._start_ts = time.time()
        self._start_vals = psutil.net_io_counters()

    def collect(self) -> Dict[str, Any]:
        ts = time.time()
        vals = psutil.net_io_counters()

        sent_thru = (vals.bytes_sent - self._start_vals.bytes_sent) / (ts - self._start_ts)
        recv_thru = (vals.bytes_recv - self._start_vals.bytes_recv) / (ts - self._start_ts)

        self._start_ts, self._start_vals = ts, vals

        return {
            "net_throughput_sent": sent_thru,
            "net_throughput_recv": recv_thru,
        }


class _Disk(_MetricGroupCollector):
    group = "disk"

    _disk_paths = ["/", constants.SHARED_FS_CONTAINER_PATH]

    def __init__(self) -> None:
        # Set initial values for throughput calculations.
        self._start_ts = time.time()
        self._start_vals = psutil.disk_io_counters()

        # Initialize accessible disk paths.
        self._paths = []
        for path in self._disk_paths:
            try:
                psutil.disk_usage(path)
                self._paths.append(path)
            except Exception:
                pass

        super().__init__()

    def collect(self) -> Dict[str, Any]:
        ts = time.time()
        vals = psutil.disk_io_counters()

        read_thru = (vals.read_bytes - self._start_vals.read_bytes) / (ts - self._start_ts)
        write_thru = (vals.write_bytes - self._start_vals.write_bytes) / (ts - self._start_ts)
        iops = (vals.read_count + vals.write_count) - (
            self._start_vals.read_count + self._start_vals.write_count
        ) / (ts - self._start_ts)
        self._start_ts, self._start_vals = ts, vals

        metrics = {
            "disk_iops": iops,
            "disk_throughput_read": read_thru,
            "disk_throughput_write": write_thru,
        }

        for path in self._paths:
            disk_usage = psutil.disk_usage(path)
            metrics.update({path: {"disk_util": disk_usage.percent}})
        return metrics


class _Memory(_MetricGroupCollector):
    group = "memory"

    def collect(self) -> Dict[str, Any]:
        free_mem_bytes = psutil.virtual_memory().available
        return {
            "free_memory": free_mem_bytes / 1e9,
        }


class _CPU(_MetricGroupCollector):
    group = "cpu"

    def collect(self) -> Dict[str, Any]:
        cpu_util = psutil.cpu_percent()
        return {
            "cpu_util_simple": cpu_util,
        }


class _GPU(_MetricGroupCollector):
    """
    Collects GPU metrics for all GPU devices on the machine.

    Returns `gpu_util` and `gpu_free_memory` metrics labeled by the device's UUID:

    .. code::
        {
            "GPU-UUID-A": {
                "gpu_util": 0.12,
                "gpu_free_memory": 5.662642176,
            },
            "GPU-UUID-B": {
                "gpu_util": 0.23,
                "gpu_free_memory": 6.552642176,
            },
        }
    """

    group = "gpu"

    def __init__(self):
        super().__init__()

        # xxx: map[uuid] -> handle
        self._pynvml_device_handles: Dict[str, Any] = {}

        if pynvml:
            self._init_pynvml()
        else:
            logging.warning(f"pynvml module not found. GPU metrics will not be collected.")

    def _init_pynvml(self) -> None:
        try:
            pynvml.nvmlInit()
            num_gpus = pynvml.nvmlDeviceGetCount()
            for i in range(num_gpus):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                uuid = pynvml.nvmlDeviceGetUUID(handle)
                pynvml.nvmlDeviceGetMemoryInfo(handle)
                pynvml.nvmlDeviceGetUtilizationRates(handle)
                self._pynvml_device_handles[uuid] = handle
        except pynvml.NVMLError as ne:
            logging.info(f"Error accessing NVML {ne}. GPU metrics will not be collected.")
        except Exception as e:
            logging.error(f"Error initializing pynvml: {e}")

    def collect(self) -> Dict[str, Any]:
        metrics = {}

        for uuid, handle in self._pynvml_device_handles.items():
            gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
            free_memory = pynvml.nvmlDeviceGetMemoryInfo(handle).free
            metrics.update(
                {
                    uuid: {
                        "gpu_util": gpu_util,
                        "gpu_free_memory": free_memory,
                    }
                }
            )
        return metrics


class _Metric:
    def __init__(
        self,
        group: str,
        metrics: Dict[str, Any],
        timestamp: datetime,
    ):
        self.group = group
        self.metrics = metrics
        self.timestamp = timestamp


class _ShutdownMessage:
    pass


class _Collector(threading.Thread):
    def __init__(
        self,
        metrics_queue: queue.Queue,
        collection_interval: int = 1,
        # xxx: figure out right time for this
        max_collection_s: int = 300,
    ):
        self._collection_interval = collection_interval
        self._should_exit = False
        self._last_flush_ts = None
        self._metrics_queue = metrics_queue
        self._max_collection_s = max_collection_s

        self._metric_collectors = [
            _GPU(),
            _CPU(),
            _Memory(),
            _Disk(),
            _Network(),
        ]
        self._should_exit = threading.Event()

        super().__init__(daemon=True)

    def run(self) -> None:
        collection_stop_ts = time.time() + self._max_collection_s
        while not self._should_exit.is_set():
            if time.time() > collection_stop_ts:
                # Max collection time reached. Notify queue consumers and exit.
                self._metrics_queue.put(_ShutdownMessage())
                return

            self._collect_metrics()
            self._should_exit.wait(self._collection_interval)

    def stop(self) -> None:
        self._should_exit.set()
        self._metrics_queue.put(_ShutdownMessage())

    def _collect_metrics(self) -> None:
        for collector in self._metric_collectors:
            timestamp = datetime.now(timezone.utc)
            metrics = collector.collect()
            if not metrics:
                continue
            group_metrics = _Metric(group=collector.group, metrics=metrics, timestamp=timestamp)
            self._metrics_queue.put(group_metrics)


class _Shipper(threading.Thread):
    def __init__(
        self,
        session: api.Session,
        agent_id: str,
        trial_id: int,
        run_id: int,
        metrics_queue: queue.Queue,
        flush_interval: int = 10,
    ):
        self._metrics_queue = metrics_queue
        self._should_exit = False
        self._flush_interval = flush_interval
        self._session = session
        self._agent_id = agent_id
        self._trial_id = trial_id
        self._run_id = run_id

        self._metrics: List[_Metric] = []

        super().__init__(daemon=True)

    def run(self) -> None:
        while True:
            next_flush_ts = time.time() + self._flush_interval
            # Pop messages from queue until the next scheduled flush time.
            while time.time() < next_flush_ts:
                deadline = next_flush_ts - time.time()
                try:
                    msg = self._metrics_queue.get(timeout=deadline)
                    if isinstance(msg, _ShutdownMessage):
                        # Received shutdown message, ship accumulated metrics and exit.
                        self._ship()
                        return

                    self._metrics.append(msg)
                except queue.Empty:
                    # Timeout reached, no messages in queue.
                    break

            self._ship()

    def stop(self) -> None:
        self._metrics_queue.put(_ShutdownMessage())

    def _ship(self) -> None:
        # xxx: change this to popleft from list?
        for metric in self._metrics:
            # Append agent ID to every metric reported.
            metrics = {self._agent_id: metric.metrics}
            self._report_metrics(group=metric.group, timestamp=metric.timestamp, metrics=metrics)
        self._metrics = []

    def _report_metrics(self, group: str, timestamp: datetime, metrics: Dict[str, Any]) -> None:
        v1metrics = bindings.v1Metrics(avgMetrics=metrics)
        v1TrialMetrics = bindings.v1TrialMetrics(
            metrics=v1metrics,
            trialId=self._trial_id,
            trialRunId=self._run_id,
            reportTime=timestamp.isoformat(),
        )
        body = bindings.v1ReportTrialMetricsRequest(metrics=v1TrialMetrics, group=group)
        bindings.post_ReportTrialMetrics(self._session, body=body, metrics_trialId=self._trial_id)

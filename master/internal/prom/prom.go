package prom

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"

	"github.com/determined-ai/determined/master/pkg/device"
)

var (
	// Gauge that maps tasks to their container IDs.
	containerIDToTaskID = promauto.NewGaugeVec(prometheus.GaugeOpts{
		Subsystem: "det",
		Name:      "container_id_task_id",
		Help:      `
Exposes mapping of container ID to task ID.

Task ID is the ID of the task within determined. This can be a little opaque but is shown
by 'det task list', which also provides a mapping to a more human-readable task name.

Container ID is Determined's internal identifier for a container or pod and appears
as a label on containers and metadata on pods (and thus in labels collected by monitoring tools).
This is useful to join in on metrics from those monitoring tools (e.g. cAdvisor).
`,
	}, []string{"container_id", "task_id"})

	containerIDToRuntimeID = promauto.NewGaugeVec(prometheus.GaugeOpts{
		Subsystem: "det",
		Name:      "container_id_runtime_container_id",
		Help:      "a mapping of the container ID to the container ID given be the runtime",
	}, []string{"container_runtime_id", "container_id"})

	gpuUUIDToContainerID = promauto.NewGaugeVec(prometheus.GaugeOpts{
		Subsystem: "det",
		Name: "gpu_uuid_container_id",
		Help: `
Exposes mapping of task name to container ID to GPU uuid.

Container ID is Determined's internal identifier for a container or pod and appears
as a label on containers or pods (and thus in container or pod monitoring tools) as
"ai.determined.container_id". This is useful to join in on metrics from those monitoring
tools (e.g. cAdvisor).

GPU UUID is the device ID as given by NVML (or nvidia-smi). This is useful to join in on
GPU metrics from other monitoring tools (e.g. DCGM).
`,
	}, []string{"gpu_uuid", "container_id"})

	// Reg is a prometheus registry containing all exported user-facing metrics.
	DetStateMetrics = prometheus.NewRegistry()
)

func init() {
	DetStateMetrics.MustRegister(containerIDToTaskID)
	DetStateMetrics.MustRegister(containerIDToRuntimeID)
	DetStateMetrics.MustRegister(gpuUUIDToContainerID)
}

// AssociateTaskContainer records the given task owns the given container.
func AssociateTaskContainer(tID string, cID string) {
	containerIDToTaskID.WithLabelValues(cID, tID).Inc()
}

// DisassociateTaskContainer records the given task no longer owns the given container.
func DisassociateTaskContainer(tID string, cID string) {
	containerIDToTaskID.WithLabelValues(cID, tID).Dec()
}

// AssociateTaskContainer associated the given Determined container ID with a runtime ID (e.g. Docker ID).
func AssociateContainerRuntimeID(cID string, dcID string) {
	containerIDToRuntimeID.WithLabelValues(dcID, cID).Inc()
}

// DisassociateTaskContainer records the given task no longer owns the given container.
func DisassociateContainerRuntimeID(cID string, dcID string) {
	containerIDToTaskID.WithLabelValues(cID, dcID).Dec()
}

// AssociateContainerGPUs records a usage of some devices by the specified a Determined container.
func AssociateContainerGPUs(cID string, ds ...device.Device) {
	for _, d := range ds {
		if d.Type == device.GPU {
			gpuUUIDToContainerID.
				WithLabelValues(d.UUID, cID).
				Inc()
		}
	}
}

// DisassociateContainerGPUs records a completion of usage of some devices by the specified a Determined container.
func DisassociateContainerGPUs(cID string, ds ...device.Device) {
	for _, d := range ds {
		if d.Type == device.GPU {
			gpuUUIDToContainerID.
				WithLabelValues(d.UUID, cID).
				Dec()
			//Need to Delete after prom has scraped the 0.
			//gpuUUIDToContainerID.
			//	// Note, theses labels are order-sensitive. Out of order is a memory leak.
			//	DeleteLabelValues(d.UUID, cID)
		}
	}
}

const (
	// Locations of expected exporters.
	// TODO: These should either be pre-installed on agent AMIs, or the
	// API should just return agent URLs instead and leave the conversion
	// to a file_sd_config to the user.
	cAdvisorExporter = ":8080"
	dcgmExporter     = ":9400"

	// The are extra labels added to metrics on scrape.
	detAgentIDLabel      = "det_agent_id"
	detResourcePoolLabel = "det_resource_pool"
)

// GetFileSDConfig returns a handle that on request returns a JSON blob in the format specified by
// https://prometheus.io/docs/prometheus/latest/configuration/configuration/#file_sd_config
// func GetFileSDConfig(system *actor.System) echo.HandlerFunc {
// 	return api.Route(func(c echo.Context) (interface{}, error) {
// 		type fileSDConfigEntry struct {
// 			Targets []string          `json:"targets"`
// 			Labels  map[string]string `json:"labels"`
// 		}
// 		summary := system.AskAt(actor.Addr("agents"), model.AgentsSummary{})
// 		var fileSDConfig []fileSDConfigEntry
// 		for _, a := range summary.Get().(model.AgentsSummary) {
// 			fileSDConfig = append(fileSDConfig, fileSDConfigEntry{
// 				// TODO: Maybe just expose what ports to scrape on agents as a config
// 				// (or maybe instead this really should just be a script that manipulates
// 				// the outputs of /api/v1/agents).
// 				Targets: []string{
// 					a.RemoteAddr + cAdvisorExporter,
// 					a.RemoteAddr + dcgmExporter,
// 				},
// 				Labels: map[string]string{
// 					detAgentIDLabel:      a.ID,
// 					detResourcePoolLabel: a.ResourcePool,
// 				},
// 			})
// 		}
// 		return fileSDConfig, nil
// 	})
// }

package db

import (
	"context"
	"github.com/determined-ai/determined/master/pkg/model"
	"github.com/determined-ai/determined/proto/pkg/trialv1"
)

// MetricPartitionType denotes what type the metric is. This is planned to be deprecated
// once we upgrade to pg11 and can use DEFAULT partitioning.
type MetricPartitionType string

const (
	// TrainingMetric designates metrics from training steps.
	TrainingMetric MetricPartitionType = "TRAINING"
	// ValidationMetric designates metrics from validation steps.
	ValidationMetric MetricPartitionType = "VALIDATION"
	// ProfilingMetric designates metric from profiling steps.
	ProfilingMetric MetricPartitionType = "PROFILING"
	// GenericMetric designates metrics from other sources.
	GenericMetric MetricPartitionType = "GENERIC"
)

// AddMetrics persists the given metrics to the database.
func (db *PgDB) AddMetrics(
	ctx context.Context, m *trialv1.TrialMetrics, mGroup model.MetricGroup,
) error {
	switch mGroup {
	case model.ProfilingMetricGroup:
		_, err := db.addTrialMetrics(ctx, m, mGroup)
		return err
	case model.TrainingMetricGroup:
	case model.ValidationMetricGroup:
	default:
		_, err := db.addTrialMetrics(ctx, m, mGroup)
		return err
	}
	return nil
}

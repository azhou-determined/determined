package trials

import (
	"context"
	"fmt"

	"github.com/uptrace/bun"

	"github.com/determined-ai/determined/master/internal/db"
	expauth "github.com/determined-ai/determined/master/internal/experiment"
	"github.com/determined-ai/determined/master/internal/grpcutil"
	"github.com/determined-ai/determined/proto/pkg/apiv1"
	"github.com/determined-ai/determined/proto/pkg/trialv1"
)

// TrialSourceInfoAPIServer is a dummy struct to do dependency injection to the api.
// This allows us to define apiServer functions in sub-modules.
type TrialSourceInfoAPIServer struct{}

// ReportTrialSourceInfo creates a TrialSourceInfo, which serves as a link between
// trials and checkpoints used for tracking purposes for fine tuning and inference.
func (a *TrialSourceInfoAPIServer) ReportTrialSourceInfo(
	ctx context.Context, req *apiv1.ReportTrialSourceInfoRequest,
) (*apiv1.ReportTrialSourceInfoResponse, error) {
	tsi := req.TrialSourceInfo
	curUser, _, err := grpcutil.GetUser(ctx)
	if err != nil {
		return nil, err
	}
	if err := CanGetTrialsExperimentAndCheckCanDoAction(ctx, int(tsi.TrialId), curUser,
		expauth.AuthZProvider.Get().CanEditExperiment); err != nil {
		return nil, err
	}
	resp, err := CreateTrialSourceInfo(ctx, tsi)
	return resp, err
}

// GetMetricsForTrialSourceInfoQuery takes in a bun.SelectQuery on the
// trial_source_infos table, and fetches the metrics for each of the connected trials.
func GetMetricsForTrialSourceInfoQuery(
	ctx context.Context, q *bun.SelectQuery,
	groupName *string,
) ([]*trialv1.MetricsReport, error) {
	trialIds := []struct {
		TrialID             int
		TrialSourceInfoType string
	}{}
	q = q.Column("trial_id", "trial_source_info_type")
	err := q.Scan(ctx, &trialIds)
	if err != nil {
		return nil, fmt.Errorf("failed to get trial source info %w", err)
	}

	curUser, _, err := grpcutil.GetUser(ctx)
	if err != nil {
		return nil, err
	}
	// TODO (Taylor): If we reach a point where this becomes a performance bottleneck
	// we should join on trial_source_infos -> trials -> experiments to get the
	// workspace_id and get permissions on those without checking each trial individually
	ret := []*trialv1.MetricsReport{}
	numMetricsLimit := 1000
	for _, val := range trialIds {
		if err := CanGetTrialsExperimentAndCheckCanDoAction(ctx, val.TrialID, curUser,
			expauth.AuthZProvider.Get().CanGetExperimentArtifacts); err != nil {
			// If the user can see the checkpoint, but not one of the inference
			// or fine tuning trials that points to it, simply don't show those
			// particular trials.
			continue
		}
		res, err := db.GetMetrics(ctx, val.TrialID, -1, numMetricsLimit, groupName)
		if err != nil {
			return nil, fmt.Errorf("failed to get metrics %w", err)
		}
		ret = append(ret, res...)
	}
	return ret, nil
}

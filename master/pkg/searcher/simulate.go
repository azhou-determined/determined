package searcher

import (
	"fmt"
	"github.com/determined-ai/determined/master/pkg/mathx"
	"github.com/determined-ai/determined/master/pkg/model"
	"github.com/determined-ai/determined/master/pkg/protoutils"
	"github.com/determined-ai/determined/master/pkg/schemas/expconf"
	"github.com/determined-ai/determined/proto/pkg/experimentv1"
	"github.com/pkg/errors"
	"math/rand"
)

// XXX: rewrite this whole thing

// ValidationFunction calculates the validation metric for the validation step.
type ValidationFunction func(random *rand.Rand, trialID, idx int) float64

// ConstantValidation returns the same validation metric for all validation steps.
func ConstantValidation(_ *rand.Rand, _, _ int) float64 { return 1 }

// RandomValidation returns a random validation metric for each validation step.
func RandomValidation(rand *rand.Rand, _, _ int) float64 { return rand.Float64() }

// TrialIDMetric returns the trialID as the metric for all validation steps.
func TrialIDMetric(_ *rand.Rand, trialID, _ int) float64 {
	return float64(trialID)
}

// SimulationResults holds all created trials and all executed workloads for each trial.
type SimulationResults map[model.RequestID][]ValidateAfter

type SearchSummary struct {
	Runs   map[int]SearchUnit
	Config expconf.SearcherConfig
}

type SearchUnit struct {
	Name      string
	Value     int
	Undefined bool
}

func (su SearchUnit) Proto() *experimentv1.SearchUnit {
	return &experimentv1.SearchUnit{
		Name:      su.Name,
		Value:     int32(su.Value),
		Undefined: su.Undefined,
	}
}

func (su SearchUnit) String() string {
	return fmt.Sprintf("%s(%d)", su.Name, su.Value)
}

func (s SearchSummary) Proto() *experimentv1.SearchSummary {
	runSummaries := make(map[int32]*experimentv1.SearchUnit)
	for k, v := range s.Runs {
		runSummaries[int32(k)] = v.Proto()
	}
	return &experimentv1.SearchSummary{
		Config: protoutils.ToStruct(s.Config),
		Runs:   runSummaries,
	}
}

// Simulate simulates the searcher.
func Simulate(conf expconf.SearcherConfig, hparams expconf.Hyperparameters) (SearchSummary, error) {
	searchSummary := SearchSummary{
		Runs:   make(map[int]SearchUnit),
		Config: conf,
	}
	switch {
	case conf.RawSingleConfig != nil:
		searchSummary.Runs[1] = SearchUnit{Undefined: true}
		return searchSummary, nil
	case conf.RawRandomConfig != nil:
		maxRuns := conf.RawRandomConfig.MaxTrials()
		searchSummary.Runs[maxRuns] = SearchUnit{Undefined: true}
		return searchSummary, nil
	case conf.RawGridConfig != nil:
		hparamGrid := NewHyperparameterGrid(hparams)
		searchSummary.Runs[len(hparamGrid)] = SearchUnit{Undefined: true}
		return searchSummary, nil
	case conf.RawAdaptiveASHAConfig != nil:
		ashaConfig := conf.RawAdaptiveASHAConfig
		brackets := makeBrackets(*ashaConfig)
		unitsPerRun := make(map[int]int)
		for _, bracket := range brackets {
			rungs := makeRungs(bracket.numRungs, ashaConfig.Divisor(), ashaConfig.Length().Units)
			rungRuns := bracket.maxTrials
			// For each rung, calculate number of runs that will be stopped before next rung.
			for i, rung := range rungs {
				rungUnits := int(rung.UnitsNeeded)
				runsContinued := mathx.Max(int(float64(rungRuns)/ashaConfig.Divisor()), 1)
				runsStopped := rungRuns - runsContinued
				if i == len(rungs)-1 {
					runsStopped = rungRuns
				}
				unitsPerRun[rungUnits] += runsStopped
				rungRuns = runsContinued
			}
		}
		for units, numRuns := range unitsPerRun {
			searchSummary.Runs[numRuns] = SearchUnit{
				Name:  string(ashaConfig.Length().Unit),
				Value: units,
			}
		}
		return searchSummary, nil
	default:
		return SearchSummary{}, errors.New("invalid searcher configuration")
	}
}

//func handleOperations(
//	pending map[model.RequestID][]Operation, requestIDs *[]model.RequestID, operations []Operation,
//) (bool, error) {
//	for _, operation := range operations {
//		switch op := operation.(type) {
//		case Create:
//			*requestIDs = append(*requestIDs, op.RequestID)
//			pending[op.RequestID] = []Operation{op}
//		case Requested:
//			pending[op.GetRequestID()] = append(pending[op.GetRequestID()], op)
//		case Shutdown:
//			return true, nil
//		default:
//			return false, errors.Errorf("unexpected operation: %T", operation)
//		}
//	}
//	return false, nil
//}
//
//func pickTrial(
//	random *rand.Rand, pending map[model.RequestID][]Operation, requestIDs []model.RequestID,
//	randomOrder bool,
//) (model.RequestID, error) {
//	// If randomOrder is false, then return the first id from requestIDs that has any operations
//	// pending.
//	if !randomOrder {
//		for _, requestID := range requestIDs {
//			operations := pending[requestID]
//			if len(operations) > 0 {
//				return requestID, nil
//			}
//		}
//		return model.RequestID{}, errors.New("tried to pick a trial when no trial had pending operations")
//	}
//
//	// If randomOrder is true, pseudo-randomly select a trial that has pending operations.
//	var candidates []model.RequestID
//	for requestID, operations := range pending {
//		if len(operations) > 0 {
//			candidates = append(candidates, requestID)
//		}
//	}
//	if len(candidates) == 0 {
//		return model.RequestID{}, errors.New("tried to pick a trial when no trial had pending operations")
//	}
//
//	// Map iteration order is nondeterministic, even for identical maps in the same run, so sort the
//	// candidates before selecting one.
//	sort.Slice(candidates, func(i, j int) bool {
//		return candidates[i].Before(candidates[j])
//	})
//
//	choice := random.Intn(len(candidates))
//	return candidates[choice], nil
//}

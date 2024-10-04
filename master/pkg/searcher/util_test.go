package searcher

import (
	"bytes"
	"fmt"
	"strconv"
	"strings"
	"testing"

	"github.com/pkg/errors"
	"gotest.tools/assert"

	"github.com/determined-ai/determined/master/pkg/check"

	"github.com/determined-ai/determined/master/pkg/schemas"
	"github.com/determined-ai/determined/master/pkg/schemas/expconf"
)

const defaultMetric = "metric"

func isExpected(actual, expected []ValidateAfter) bool {
	if len(actual) != len(expected) {
		return false
	}
	for i, act := range actual {
		if expected[i].Length != act.Length {
			return false
		}
	}
	return true
}

func checkSimulation(
	t *testing.T,
	method SearchMethod,
	params expconf.Hyperparameters,
	validation ValidationFunction,
	expected [][]ValidateAfter,
) {
	//search := NewSearcher(0, method, params)
	//actual, err := Simulate(search, new(int64), validation, true, defaultMetric)
	//assert.NilError(t, err)
	//
	//assert.Equal(t, len(actual.Results), len(expected))
	//for _, actualTrial := range actual.Results {
	//	found := false
	//	for i, expectedTrial := range expected {
	//		if isExpected(actualTrial, expectedTrial) {
	//			expected = append(expected[:i], expected[i+1:]...)
	//			found = true
	//			break
	//		}
	//	}
	//	if !found {
	//		t.Errorf("unexpected trial %+v not in %+v", actualTrial, expected)
	//	}
	//}
}

// checkReproducibility creates two searchers with the same seed and the given config, simulates
// them, and checks that they produce the same trials and the same sequence of workloads for each
// trial.
func checkReproducibility(
	t assert.TestingT, methodGen func() SearchMethod, hparams expconf.Hyperparameters, metric string,
) {
	//hparams = schemas.WithDefaults(hparams)
	//seed := int64(17)
	//searcher1 := NewSearcher(uint32(seed), methodGen(), hparams)
	//searcher2 := NewSearcher(uint32(seed), methodGen(), hparams)
	//
	//results1, err1 := Simulate(searcher1, &seed, ConstantValidation, true, metric)
	//assert.NilError(t, err1)
	//results2, err2 := Simulate(searcher2, &seed, ConstantValidation, true, metric)
	//assert.NilError(t, err2)
	//
	//assert.Equal(t, len(results1.Results), len(results2.Results),
	//	"searchers had different number of trials")
	//for requestID := range results1.Results {
	//	w1 := results1.Results[requestID]
	//	w2 := results2.Results[requestID]
	//
	//	assert.Equal(t, len(w1), len(w2), "trial had different numbers of workloads between searchers")
	//	for i := range w1 {
	//		// We want to ignore the start and end time fields, so check the rest individually.
	//		assert.Equal(t, w1[i], w2[i], "workload differed between searchers")
	//	}
	//}
}

func toOps(types string) (ops []ValidateAfter) {
	for _, unparsed := range strings.Fields(types) {
		count, err := strconv.ParseUint(unparsed[:len(unparsed)-1], 10, 64)
		if err != nil {
			panic(err)
		}
		switch unit := string(unparsed[len(unparsed)-1]); unit {
		case "R":
			ops = append(ops, ValidateAfter{Length: count})
		case "B":
			ops = append(ops, ValidateAfter{Length: count})
		case "E":
			ops = append(ops, ValidateAfter{Length: count})
		}
	}
	return ops
}

type predefinedTrial struct {
	Ops        []ValidateAfter
	ValMetrics []float64
	EarlyExit  *int
}

func newPredefinedTrial(ops []ValidateAfter, earlyExit *int, valMetrics []float64) predefinedTrial {
	return predefinedTrial{
		Ops:        ops,
		EarlyExit:  earlyExit,
		ValMetrics: valMetrics,
	}
}

func newEarlyExitPredefinedTrial(ops []ValidateAfter, valMetric float64) predefinedTrial {
	var valMetrics []float64
	for range ops {
		valMetrics = append(valMetrics, valMetric)
	}
	exitEarly := len(ops) - 1
	return newPredefinedTrial(ops, &exitEarly, valMetrics)
}

func newConstantPredefinedTrial(ops []ValidateAfter, valMetric float64) predefinedTrial {
	var valMetrics []float64
	for range ops {
		valMetrics = append(valMetrics, valMetric)
	}
	return newPredefinedTrial(ops, nil, valMetrics)
}

func (t *predefinedTrial) Train(length uint64, opIndex int) error {
	if opIndex >= len(t.Ops) {
		return errors.Errorf("ran out of expected ops trying to train")
	}
	op := t.Ops[opIndex]
	if op.Length != length {
		return errors.Errorf("wanted %v got %v", op.Length, length)
	}
	return nil
}

func (t *predefinedTrial) CheckComplete(opIndex int) error {
	return check.Equal(len(t.Ops), opIndex, "had ops %s left", t.Ops[opIndex:])
}

// checkValueSimulation will run a SearchMethod until completion, using predefinedTrials.
func checkValueSimulation(
	t *testing.T,
	method SearchMethod,
	params expconf.Hyperparameters,
	expectedTrials []predefinedTrial,
) error {
	//// Create requests are assigned a predefinedTrial in order.
	//var nextTrialID int
	//var pending []Operation
	//
	//trialIDs := map[model.RequestID]int{}
	//trialOpIdx := map[model.RequestID]int{}
	////trialEarlyExits := map[model.RequestID]bool{}
	//
	//ctx := context{
	//	rand:    nprand.New(0),
	//	hparams: params,
	//}
	//
	//ops, err := method.initialRuns(ctx)
	//if err != nil {
	//	return errors.Wrap(err, "initialOperations")
	//}
	//
	//pending = append(pending, ops...)
	//
	//for len(pending) > 0 {
	//	var requestID model.RequestID
	//	operation := pending[0]
	//	pending = pending[1:]
	//
	//	switch operation := operation.(type) {
	//	case Create:
	//		requestID = operation.RequestID
	//		if nextTrialID >= len(expectedTrials) {
	//			return errors.Errorf("search method created too many trials")
	//		}
	//		trialIDs[requestID] = nextTrialID
	//		trialOpIdx[requestID] = 0
	//
	//		ops, err = method.trialCreated(ctx, int32(nextTrialID))
	//		if err != nil {
	//			return errors.Wrap(err, "trialCreated")
	//		}
	//		nextTrialID++
	//
	//	//case ValidateAfter:
	//	//	requestID = operation.GetRequestID()
	//	//	if trialEarlyExits[requestID] {
	//	//		continue
	//	//	}
	//	//
	//	//	trialID := trialIDs[requestID]
	//	//	trial := expectedTrials[trialID]
	//	//	if trial.EarlyExit != nil && trialOpIdx[requestID] == *trial.EarlyExit {
	//	//		trialEarlyExits[requestID] = true
	//	//	}
	//	//	ops, err = simulateOperationComplete(ctx, method, trial, operation, trialOpIdx[requestID])
	//	//	if err != nil {
	//	//		return errors.Wrapf(err, "simulateOperationComplete for trial %v", trialID+1)
	//	//	}
	//	//	trialOpIdx[requestID]++
	//	//	if err = saveAndReload(method); err != nil {
	//	//		return errors.Wrap(err, "snapshot failed")
	//	//	}
	//
	//	case Close:
	//		requestID = operation.RequestID
	//		trialID := trialIDs[requestID]
	//		trial := expectedTrials[trialID]
	//		err = trial.CheckComplete(trialOpIdx[requestID])
	//		if err != nil {
	//			return errors.Wrapf(err, "trial %v closed before completion", trialID+1)
	//		}
	//
	//		ops, err = method.trialClosed(ctx, int32(trialID))
	//		if err != nil {
	//			return errors.Wrap(err, "trialClosed")
	//		}
	//
	//	default:
	//		return errors.Errorf("unexpected searcher operation: %T", operation)
	//	}
	//
	//	pending = append(pending, ops...)
	//}
	//
	//for requestID, trialID := range trialIDs {
	//	if err = expectedTrials[trialID].CheckComplete(trialOpIdx[requestID]); err != nil {
	//		return errors.Wrapf(err, "incomplete trial %v", trialID+1)
	//	}
	//}

	return nil
}

func runValueSimulationTestCases(t *testing.T, testCases []valueSimulationTestCase) {
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Apply WithDefaults in one place to make tests easyto write.
			config := schemas.WithDefaults(tc.config)
			hparams := schemas.WithDefaults(tc.hparams)
			method := NewSearchMethod(config)
			err := checkValueSimulation(t, method, hparams, tc.expectedTrials)
			assert.NilError(t, err)
		})
	}
}

type valueSimulationTestCase struct {
	name           string
	expectedTrials []predefinedTrial
	hparams        expconf.Hyperparameters
	config         expconf.SearcherConfig
}

func saveAndReload(method SearchMethod) error {
	// take the state back and forth through a round of serialization to test.
	if state, err := method.Snapshot(); err != nil {
		return err
	} else if err := method.Restore(state); err != nil {
		return err
	} else if state2, err := method.Snapshot(); err != nil { // Test restore is correct.
		return err
	} else if !bytes.Equal(state, state2) {
		unmarshaledState := method.Restore(state)
		unmarshaledState2 := method.Restore(state2)
		fmt.Printf("%+v\n", unmarshaledState)  //nolint: forbidigo
		fmt.Printf("%+v\n", unmarshaledState2) //nolint: forbidigo
		return errors.New("successive snapshots were not identical")
	}
	return nil
}

type TestSearchRunner struct {
	config   expconf.SearcherConfig
	searcher *Searcher
	method   SearchMethod
	runs     map[int32]testRun
	t        *testing.T
}

type testRun struct {
	id           int32
	hparams      HParamSample
	stopped      bool
	searchRunner *TestSearchRunner
}

func NewTestSearchRunner(t *testing.T, config expconf.SearcherConfig, hparams expconf.Hyperparameters) *TestSearchRunner {
	expSeed := uint32(102932948)
	method := NewSearchMethod(config)
	searcher := NewSearcher(expSeed, method, hparams)
	return &TestSearchRunner{t: t, config: config, searcher: searcher, method: method, runs: make(map[int32]testRun)}
}

func (sr *TestSearchRunner) start() ([]testRun, []testRun) {
	creates, err := sr.searcher.InitialRuns()
	assert.NilError(sr.t, err, "error getting initial runs")
	created, stopped := sr.handleActions(creates)
	return created, stopped
}

func (sr *TestSearchRunner) reportValidationMetric(runID int32, stepNum int, metricVal float64) ([]testRun, []testRun) {
	metrics := map[string]interface{}{
		sr.config.Metric(): metricVal,
	}
	if sr.config.RawAdaptiveASHAConfig != nil {
		timeMetric := string(sr.config.RawAdaptiveASHAConfig.Length().Unit)
		metrics[timeMetric] = float64(stepNum)
	}
	if sr.config.RawAsyncHalvingConfig != nil {
		timeMetric := string(sr.config.RawAsyncHalvingConfig.Length().Unit)
		metrics[timeMetric] = float64(stepNum)
	}
	actions, err := sr.searcher.ValidationCompleted(runID, metrics)
	assert.NilError(sr.t, err, "error completing validation")

	created, stopped := sr.handleActions(actions)

	return created, stopped
}

// run created, run stopped, error
func (sr *TestSearchRunner) handleActions(actions []Action) ([]testRun, []testRun) {
	var runsCreated []testRun
	var runsStopped []testRun

	for _, action := range actions {
		switch action := action.(type) {
		case Create:
			run := testRun{id: int32(len(sr.searcher.state.RunsCreated)), hparams: action.Hparams, searchRunner: sr}
			_, err := sr.searcher.RunCreated(run.id, action)
			assert.NilError(sr.t, err, "error creating run")

			sr.runs[run.id] = run
			runsCreated = append(runsCreated, run)
		case Stop:
			run := sr.runs[action.RunID]
			run.stopped = true
			sr.runs[action.RunID] = run
			runsStopped = append(runsStopped, run)
		}
	}
	return runsCreated, runsStopped
}

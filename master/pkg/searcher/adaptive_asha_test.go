//nolint:exhaustruct
package searcher

import (
	"github.com/stretchr/testify/require"
	"testing"

	"gotest.tools/assert"

	"github.com/determined-ai/determined/master/pkg/ptrs"
	"github.com/determined-ai/determined/master/pkg/schemas"
	"github.com/determined-ai/determined/master/pkg/schemas/expconf"
)

func TestBracketMaxTrials(t *testing.T) {
	assert.DeepEqual(t, getBracketMaxTrials(20, 3., []int{3, 2, 1}), []int{12, 5, 3})
	assert.DeepEqual(t, getBracketMaxTrials(50, 3., []int{4, 3}), []int{35, 15})
	assert.DeepEqual(t, getBracketMaxTrials(50, 4., []int{3, 2}), []int{37, 13})
	assert.DeepEqual(t, getBracketMaxTrials(100, 4., []int{4, 3, 2}), []int{70, 22, 8})
}

func TestBracketMaxConcurrentTrials(t *testing.T) {
	assert.DeepEqual(t, getBracketMaxConcurrentTrials(0, 3., []int{9, 3, 1}), []int{3, 3, 3})
	assert.DeepEqual(t, getBracketMaxConcurrentTrials(11, 3., []int{9, 3, 1}), []int{4, 4, 3})
	// We try to take advantage of the max degree of parallelism for the narrowest bracket.
	assert.DeepEqual(t, getBracketMaxConcurrentTrials(0, 4., []int{40, 10}), []int{10, 10})
}

func TestMakeBrackets(t *testing.T) {
	cases := []struct {
		conf        expconf.AdaptiveASHAConfig
		expBrackets []bracket
	}{
		{
			conf: expconf.AdaptiveASHAConfig{
				RawMode:                expconf.AdaptiveModePtr(expconf.StandardMode),
				RawMaxLength:           &expconf.LengthV0{Units: 100},
				RawMaxConcurrentTrials: ptrs.Ptr(2),
				RawMaxTrials:           ptrs.Ptr(10),
			},
			expBrackets: []bracket{
				{
					numRungs:            2,
					maxTrials:           7,
					maxConcurrentTrials: 1,
				},
				{
					numRungs:            1,
					maxTrials:           3,
					maxConcurrentTrials: 1,
				},
			},
		},
		{
			conf: expconf.AdaptiveASHAConfig{
				RawMode:                expconf.AdaptiveModePtr(expconf.ConservativeMode),
				RawMaxLength:           &expconf.LengthV0{Units: 1000},
				RawDivisor:             ptrs.Ptr(3.0),
				RawMaxConcurrentTrials: ptrs.Ptr(5),
				RawMaxTrials:           ptrs.Ptr(10),
			},
			expBrackets: []bracket{
				{
					numRungs:            3,
					maxTrials:           7,
					maxConcurrentTrials: 2,
				},
				{
					numRungs:            2,
					maxTrials:           2,
					maxConcurrentTrials: 2,
				},
				{
					numRungs:            1,
					maxTrials:           1,
					maxConcurrentTrials: 1,
				},
			},
		},
	}
	for _, c := range cases {
		brackets := makeBrackets(schemas.WithDefaults(c.conf))
		require.Equal(t, len(c.expBrackets), len(brackets))
		require.Equal(t, c.expBrackets, brackets)
	}
}

func modePtr(x expconf.AdaptiveMode) *expconf.AdaptiveMode {
	return &x
}

func TestAdaptiveASHASearcherReproducibility(t *testing.T) {
	conf := expconf.AdaptiveASHAConfig{
		RawMaxLength: ptrs.Ptr(expconf.NewLengthInBatches(6400)),
		RawMaxTrials: ptrs.Ptr(128),
	}
	conf = schemas.WithDefaults(conf)
	gen := func() SearchMethod { return newAdaptiveASHASearch(conf, true, "loss") }
	checkReproducibility(t, gen, nil, defaultMetric)
}

func TestAdaptiveASHAStoppingSearchMethod(t *testing.T) {
	testCases := []valueSimulationTestCase{
		{
			name: "smaller is better",
			expectedTrials: []predefinedTrial{
				newConstantPredefinedTrial(toOps("300B 900B"), 0.1),
				newConstantPredefinedTrial(toOps("300B"), 0.2),
				newConstantPredefinedTrial(toOps("300B"), 0.3),
				newConstantPredefinedTrial(toOps("900B"), 0.4),
				newConstantPredefinedTrial(toOps("900B"), 0.5),
			},
			config: expconf.SearcherConfig{
				RawSmallerIsBetter: ptrs.Ptr(true),
				RawAdaptiveASHAConfig: &expconf.AdaptiveASHAConfig{
					RawMaxLength: ptrs.Ptr(expconf.NewLengthInBatches(900)),
					RawMaxTrials: ptrs.Ptr(5),
					RawMode:      modePtr(expconf.StandardMode),
					RawMaxRungs:  ptrs.Ptr(2),
					RawDivisor:   ptrs.Ptr[float64](3),
					RawStopOnce:  ptrs.Ptr(true),
				},
			},
		},
		{
			name: "early exit -- smaller is better",
			expectedTrials: []predefinedTrial{
				newConstantPredefinedTrial(toOps("300B 900B"), 0.1),
				newEarlyExitPredefinedTrial(toOps("300B"), 0.2),
				newConstantPredefinedTrial(toOps("300B"), 0.3),
				newConstantPredefinedTrial(toOps("900B"), 0.4),
				newConstantPredefinedTrial(toOps("900B"), 0.5),
			},
			config: expconf.SearcherConfig{
				RawSmallerIsBetter: ptrs.Ptr(true),
				RawAdaptiveASHAConfig: &expconf.AdaptiveASHAConfig{
					RawMaxLength: ptrs.Ptr(expconf.NewLengthInBatches(900)),
					RawMaxTrials: ptrs.Ptr(5),
					RawMode:      modePtr(expconf.StandardMode),
					RawMaxRungs:  ptrs.Ptr(2),
					RawDivisor:   ptrs.Ptr[float64](3),
					RawStopOnce:  ptrs.Ptr(true),
				},
			},
		},
		{
			name: "smaller is not better",
			expectedTrials: []predefinedTrial{
				newConstantPredefinedTrial(toOps("300B 900B"), 0.1),
				newConstantPredefinedTrial(toOps("300B 900B"), 0.2),
				newConstantPredefinedTrial(toOps("300B 900B"), 0.3),
				newConstantPredefinedTrial(toOps("900B"), 0.4),
				newConstantPredefinedTrial(toOps("900B"), 0.5),
			},
			config: expconf.SearcherConfig{
				RawSmallerIsBetter: ptrs.Ptr(false),
				RawAdaptiveASHAConfig: &expconf.AdaptiveASHAConfig{
					RawMaxLength: ptrs.Ptr(expconf.NewLengthInBatches(900)),
					RawMaxTrials: ptrs.Ptr(5),
					RawMode:      modePtr(expconf.StandardMode),
					RawMaxRungs:  ptrs.Ptr(2),
					RawDivisor:   ptrs.Ptr[float64](3),
					RawStopOnce:  ptrs.Ptr(true),
				},
			},
		},
		{
			name: "early exit -- smaller is not better",
			expectedTrials: []predefinedTrial{
				newConstantPredefinedTrial(toOps("300B 900B"), 0.1),
				newEarlyExitPredefinedTrial(toOps("300B"), 0.2),
				newConstantPredefinedTrial(toOps("300B 900B"), 0.3),
				newConstantPredefinedTrial(toOps("900B"), 0.4),
				newConstantPredefinedTrial(toOps("900B"), 0.5),
			},
			config: expconf.SearcherConfig{
				RawSmallerIsBetter: ptrs.Ptr(false),
				RawAdaptiveASHAConfig: &expconf.AdaptiveASHAConfig{
					RawMaxLength: ptrs.Ptr(expconf.NewLengthInBatches(900)),
					RawMaxTrials: ptrs.Ptr(5),
					RawMode:      modePtr(expconf.StandardMode),
					RawMaxRungs:  ptrs.Ptr(2),
					RawDivisor:   ptrs.Ptr[float64](3),
					RawStopOnce:  ptrs.Ptr(true),
				},
			},
		},
	}

	runValueSimulationTestCases(t, testCases)
}

// Test an end-to-end flow.
func TestASHA(t *testing.T) {
	maxConcurrentTrials := 5
	maxTrials := 10
	divisor := 2.0
	maxTime := 1000

	config := expconf.SearcherConfig{
		RawAdaptiveASHAConfig: &expconf.AdaptiveASHAConfig{
			RawMaxTime:             &maxTime,
			RawDivisor:             &divisor,
			RawMaxConcurrentTrials: &maxConcurrentTrials,
			RawMaxTrials:           &maxTrials,
			RawTimeMetric:          ptrs.Ptr("batches"),
			RawMode:                modePtr(expconf.ConservativeMode),
		},
		RawMetric:          ptrs.Ptr("loss"),
		RawSmallerIsBetter: ptrs.Ptr(true),
	}
	config = schemas.WithDefaults(config)
	intHparam := &expconf.IntHyperparameter{RawMaxval: 10, RawCount: ptrs.Ptr(3)}
	hparams := expconf.Hyperparameters{
		"x": expconf.Hyperparameter{RawIntHyperparameter: intHparam},
	}

	// Create a new test searcher and verify brackets/rungs.
	testSearchRunner := NewTestSearchRunner(config, hparams)
	search := testSearchRunner.method.(*tournamentSearch)
	require.Equal(t, 4, len(search.subSearches))

	expectedRungs := []*rung{
		{UnitsNeeded: uint64(125)},
		{UnitsNeeded: uint64(250)},
		{UnitsNeeded: uint64(500)},
		{UnitsNeeded: uint64(1000)},
	}

	for i, searchMethod := range search.subSearches {
		ashaSearch := searchMethod.(*asyncHalvingStoppingSearch)
		require.Equal(t, expectedRungs[i:], ashaSearch.Rungs)
	}

	// Start the search, validate initial trials.
	runActions, err := testSearchRunner.start()
	require.NoError(t, err)
	require.Equal(t, maxConcurrentTrials, len(runActions.runsCreated))
	require.Equal(t, 0, len(runActions.runsStopped))

	//testSearchRunner.reportValidationMetric(0, )0
}

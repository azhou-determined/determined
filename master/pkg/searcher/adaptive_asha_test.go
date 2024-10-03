//nolint:exhaustruct
package searcher

import (
	"fmt"
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

// Test an end-to-end flow.
func TestAdaptiveASHA(t *testing.T) {
	maxConcurrentTrials := 5
	maxTrials := 10
	divisor := 3.0
	maxTime := 900
	config := expconf.SearcherConfig{
		RawAdaptiveASHAConfig: &expconf.AdaptiveASHAConfig{
			RawMaxTime:             &maxTime,
			RawDivisor:             &divisor,
			RawMaxConcurrentTrials: &maxConcurrentTrials,
			RawMaxTrials:           &maxTrials,
			RawTimeMetric:          ptrs.Ptr("batches"),
			RawMode:                modePtr(expconf.StandardMode),
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
	require.Equal(t, 2, len(search.subSearches))

	expectedRungs := []*rung{
		{UnitsNeeded: uint64(100)},
		{UnitsNeeded: uint64(300)},
		{UnitsNeeded: uint64(900)},
	}

	for i, searchMethod := range search.subSearches {
		ashaSearch := searchMethod.(*asyncHalvingStoppingSearch)
		require.Equal(t, expectedRungs[i:], ashaSearch.Rungs)
	}

	// Start the search, validate correct number of initial runs created across brackets.
	runsCreated, err := testSearchRunner.start()
	require.NoError(t, err)
	require.Equal(t, maxConcurrentTrials, len(runsCreated))

	bracketRuns := make(map[int][]int32)
	for rID, sID := range search.RunTable {
		bracketRuns[sID] = append(bracketRuns[sID], rID)
	}
	fmt.Printf("bracket runs %v\n", bracketRuns)
	require.Equal(t, 3, len(bracketRuns[0]))
	require.Equal(t, 2, len(bracketRuns[1]))

	// Bracket 1: [100, 300, 900]
	// Report progressively worse metrics for each run in first rung.
	// First run should continue.
	actions, err := testSearchRunner.reportValidationMetric(bracketRuns[0][0], 100, 3.0)
	require.NoError(t, err)
	require.Equal(t, 0, len(actions))
	// Second run should stop and create a new run.
	actions, err = testSearchRunner.reportValidationMetric(bracketRuns[0][1], 100, 4.0)
	require.NoError(t, err)
	require.Equal(t, 2, len(actions))

	// Third run should stop and create a new run.
	actions, err = testSearchRunner.reportValidationMetric(bracketRuns[0][2], 100, 5.0)
	require.NoError(t, err)
	require.Equal(t, 2, len(actions))

}

//nolint:exhaustruct
package searcher

import (
	"fmt"
	"testing"

	"github.com/determined-ai/determined/master/pkg/ptrs"
	"github.com/determined-ai/determined/master/pkg/schemas"
	"github.com/determined-ai/determined/master/pkg/schemas/expconf"
)

func TestRandomSearcherReproducibility(t *testing.T) {
	conf := expconf.RandomConfig{
		RawMaxTrials: ptrs.Ptr(4), RawMaxLength: ptrs.Ptr(expconf.NewLengthInBatches(300)),
	}
	conf = schemas.WithDefaults(conf)
	gen := func() SearchMethod { return newRandomSearch(conf) }
	checkReproducibility(t, gen, nil, defaultMetric)
}

func TestRandomSearchMethod(t *testing.T) {
	conf := expconf.SearcherConfig{
		RawRandomConfig: &expconf.RandomConfig{
			RawMaxTrials:           ptrs.Ptr(4),
			RawMaxConcurrentTrials: ptrs.Ptr(2),
			RawMaxLength:           ptrs.Ptr(expconf.NewLengthInBatches(300)),
		},
	}
	conf = schemas.WithDefaults(conf)
	intHparam := &expconf.IntHyperparameter{RawMaxval: 10, RawCount: ptrs.Ptr(3)}
	hparams := expconf.Hyperparameters{
		"x": expconf.Hyperparameter{RawIntHyperparameter: intHparam},
	}
	testSearchRunner := NewTestSearchRunner(t, conf, hparams)
	//search := testSearchRunner.method.(*randomSearch)
	actions, err := testSearchRunner.start()
	fmt.Printf("actions %v err %v\n", actions, err)
}

func TestSingleSearchMethod(t *testing.T) {
	testCases := []valueSimulationTestCase{
		{
			name: "test single search method",
			expectedTrials: []predefinedTrial{
				newConstantPredefinedTrial(toOps("500B"), .1),
			},
			config: expconf.SearcherConfig{
				RawSingleConfig: &expconf.SingleConfig{
					RawMaxLength: ptrs.Ptr(expconf.NewLengthInBatches(500)),
				},
			},
		},
	}

	runValueSimulationTestCases(t, testCases)
}

// Code generated by gen.py. DO NOT EDIT.

package expconf

import (
	"github.com/santhosh-tekuri/jsonschema/v2"

	"github.com/determined-ai/determined/master/pkg/schemas"
)

func (s SearcherConfigV0) Metric() string {
	if s.RawMetric == nil {
		panic("You must call WithDefaults on SearcherConfigV0 before .Metric")
	}
	return *s.RawMetric
}

func (s *SearcherConfigV0) SetMetric(val string) {
	s.RawMetric = &val
}

func (s SearcherConfigV0) SmallerIsBetter() bool {
	if s.RawSmallerIsBetter == nil {
		panic("You must call WithDefaults on SearcherConfigV0 before .SmallerIsBetter")
	}
	return *s.RawSmallerIsBetter
}

func (s *SearcherConfigV0) SetSmallerIsBetter(val bool) {
	s.RawSmallerIsBetter = &val
}

func (s SearcherConfigV0) SourceTrialID() *int {
	return s.RawSourceTrialID
}

func (s *SearcherConfigV0) SetSourceTrialID(val *int) {
	s.RawSourceTrialID = val
}

func (s SearcherConfigV0) SourceCheckpointUUID() *string {
	return s.RawSourceCheckpointUUID
}

func (s *SearcherConfigV0) SetSourceCheckpointUUID(val *string) {
	s.RawSourceCheckpointUUID = val
}

func (s SearcherConfigV0) GetUnionMember() interface{} {
	if s.RawSingleConfig != nil {
		return *s.RawSingleConfig
	}
	if s.RawRandomConfig != nil {
		return *s.RawRandomConfig
	}
	if s.RawGridConfig != nil {
		return *s.RawGridConfig
	}
	if s.RawAsyncHalvingConfig != nil {
		return *s.RawAsyncHalvingConfig
	}
	if s.RawAdaptiveASHAConfig != nil {
		return *s.RawAdaptiveASHAConfig
	}
	if s.RawPBTConfig != nil {
		return *s.RawPBTConfig
	}
	panic("no union member defined")
}

func (s SearcherConfigV0) ParsedSchema() interface{} {
	return schemas.ParsedSearcherConfigV0()
}

func (s SearcherConfigV0) SanityValidator() *jsonschema.Schema {
	return schemas.GetSanityValidator("http://determined.ai/schemas/expconf/v0/searcher.json")
}

func (s SearcherConfigV0) CompletenessValidator() *jsonschema.Schema {
	return schemas.GetCompletenessValidator("http://determined.ai/schemas/expconf/v0/searcher.json")
}

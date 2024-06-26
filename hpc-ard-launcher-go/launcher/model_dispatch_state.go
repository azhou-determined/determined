/*
Launcher API

The Launcher API is the execution layer for the Capsules framework.  It handles all the details of launching and monitoring runtime environments.

API version: 3.3.7
*/

// Code generated by OpenAPI Generator (https://openapi-generator.tech); DO NOT EDIT.

package launcher

import (
	"encoding/json"
	"fmt"
)

// DispatchState the model 'DispatchState'
type DispatchState string

// List of DispatchState
const (
	UNKNOWN DispatchState = "UNKNOWN"
	PENDING DispatchState = "PENDING"
	RUNNING DispatchState = "RUNNING"
	TERMINATING DispatchState = "TERMINATING"
	MISSING DispatchState = "MISSING"
	TERMINATED DispatchState = "TERMINATED"
	COMPLETED DispatchState = "COMPLETED"
	FAILED DispatchState = "FAILED"
)

// All allowed values of DispatchState enum
var AllowedDispatchStateEnumValues = []DispatchState{
	"UNKNOWN",
	"PENDING",
	"RUNNING",
	"TERMINATING",
	"MISSING",
	"TERMINATED",
	"COMPLETED",
	"FAILED",
}

func (v *DispatchState) UnmarshalJSON(src []byte) error {
	var value string
	err := json.Unmarshal(src, &value)
	if err != nil {
		return err
	}
	enumTypeValue := DispatchState(value)
	for _, existing := range AllowedDispatchStateEnumValues {
		if existing == enumTypeValue {
			*v = enumTypeValue
			return nil
		}
	}

	return fmt.Errorf("%+v is not a valid DispatchState", value)
}

// NewDispatchStateFromValue returns a pointer to a valid DispatchState
// for the value passed as argument, or an error if the value passed is not allowed by the enum
func NewDispatchStateFromValue(v string) (*DispatchState, error) {
	ev := DispatchState(v)
	if ev.IsValid() {
		return &ev, nil
	} else {
		return nil, fmt.Errorf("invalid value '%v' for DispatchState: valid values are %v", v, AllowedDispatchStateEnumValues)
	}
}

// IsValid return true if the value is valid for the enum, false otherwise
func (v DispatchState) IsValid() bool {
	for _, existing := range AllowedDispatchStateEnumValues {
		if existing == v {
			return true
		}
	}
	return false
}

// Ptr returns reference to DispatchState value
func (v DispatchState) Ptr() *DispatchState {
	return &v
}

type NullableDispatchState struct {
	value *DispatchState
	isSet bool
}

func (v NullableDispatchState) Get() *DispatchState {
	return v.value
}

func (v *NullableDispatchState) Set(val *DispatchState) {
	v.value = val
	v.isSet = true
}

func (v NullableDispatchState) IsSet() bool {
	return v.isSet
}

func (v *NullableDispatchState) Unset() {
	v.value = nil
	v.isSet = false
}

func NewNullableDispatchState(val *DispatchState) *NullableDispatchState {
	return &NullableDispatchState{value: val, isSet: true}
}

func (v NullableDispatchState) MarshalJSON() ([]byte, error) {
	return json.Marshal(v.value)
}

func (v *NullableDispatchState) UnmarshalJSON(src []byte) error {
	v.isSet = true
	return json.Unmarshal(src, &v.value)
}


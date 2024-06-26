/*
Launcher API

The Launcher API is the execution layer for the Capsules framework.  It handles all the details of launching and monitoring runtime environments.

API version: 3.3.7
*/

// Code generated by OpenAPI Generator (https://openapi-generator.tech); DO NOT EDIT.

package launcher

import (
	"encoding/json"
)

// ACLInfo struct for ACLInfo
type ACLInfo struct {
	Allowed *[]string `json:"allowed,omitempty"`
	Forbidden *[]string `json:"forbidden,omitempty"`
	AdditionalPropertiesField *map[string]interface{} `json:"additionalProperties,omitempty"`
}

// NewACLInfo instantiates a new ACLInfo object
// This constructor will assign default values to properties that have it defined,
// and makes sure properties required by API are set, but the set of arguments
// will change when the set of required properties is changed
func NewACLInfo() *ACLInfo {
	this := ACLInfo{}
	return &this
}

// NewACLInfoWithDefaults instantiates a new ACLInfo object
// This constructor will only assign default values to properties that have it defined,
// but it doesn't guarantee that properties required by API are set
func NewACLInfoWithDefaults() *ACLInfo {
	this := ACLInfo{}
	return &this
}

// GetAllowed returns the Allowed field value if set, zero value otherwise.
func (o *ACLInfo) GetAllowed() []string {
	if o == nil || o.Allowed == nil {
		var ret []string
		return ret
	}
	return *o.Allowed
}

// GetAllowedOk returns a tuple with the Allowed field value if set, nil otherwise
// and a boolean to check if the value has been set.
func (o *ACLInfo) GetAllowedOk() (*[]string, bool) {
	if o == nil || o.Allowed == nil {
		return nil, false
	}
	return o.Allowed, true
}

// HasAllowed returns a boolean if a field has been set.
func (o *ACLInfo) HasAllowed() bool {
	if o != nil && o.Allowed != nil {
		return true
	}

	return false
}

// SetAllowed gets a reference to the given []string and assigns it to the Allowed field.
func (o *ACLInfo) SetAllowed(v []string) {
	o.Allowed = &v
}

// GetForbidden returns the Forbidden field value if set, zero value otherwise.
func (o *ACLInfo) GetForbidden() []string {
	if o == nil || o.Forbidden == nil {
		var ret []string
		return ret
	}
	return *o.Forbidden
}

// GetForbiddenOk returns a tuple with the Forbidden field value if set, nil otherwise
// and a boolean to check if the value has been set.
func (o *ACLInfo) GetForbiddenOk() (*[]string, bool) {
	if o == nil || o.Forbidden == nil {
		return nil, false
	}
	return o.Forbidden, true
}

// HasForbidden returns a boolean if a field has been set.
func (o *ACLInfo) HasForbidden() bool {
	if o != nil && o.Forbidden != nil {
		return true
	}

	return false
}

// SetForbidden gets a reference to the given []string and assigns it to the Forbidden field.
func (o *ACLInfo) SetForbidden(v []string) {
	o.Forbidden = &v
}

// GetAdditionalPropertiesField returns the AdditionalPropertiesField field value if set, zero value otherwise.
func (o *ACLInfo) GetAdditionalPropertiesField() map[string]interface{} {
	if o == nil || o.AdditionalPropertiesField == nil {
		var ret map[string]interface{}
		return ret
	}
	return *o.AdditionalPropertiesField
}

// GetAdditionalPropertiesFieldOk returns a tuple with the AdditionalPropertiesField field value if set, nil otherwise
// and a boolean to check if the value has been set.
func (o *ACLInfo) GetAdditionalPropertiesFieldOk() (*map[string]interface{}, bool) {
	if o == nil || o.AdditionalPropertiesField == nil {
		return nil, false
	}
	return o.AdditionalPropertiesField, true
}

// HasAdditionalPropertiesField returns a boolean if a field has been set.
func (o *ACLInfo) HasAdditionalPropertiesField() bool {
	if o != nil && o.AdditionalPropertiesField != nil {
		return true
	}

	return false
}

// SetAdditionalPropertiesField gets a reference to the given map[string]interface{} and assigns it to the AdditionalPropertiesField field.
func (o *ACLInfo) SetAdditionalPropertiesField(v map[string]interface{}) {
	o.AdditionalPropertiesField = &v
}

func (o ACLInfo) MarshalJSON() ([]byte, error) {
	toSerialize := map[string]interface{}{}
	if o.Allowed != nil {
		toSerialize["allowed"] = o.Allowed
	}
	if o.Forbidden != nil {
		toSerialize["forbidden"] = o.Forbidden
	}
	if o.AdditionalPropertiesField != nil {
		toSerialize["additionalProperties"] = o.AdditionalPropertiesField
	}
	return json.Marshal(toSerialize)
}

type NullableACLInfo struct {
	value *ACLInfo
	isSet bool
}

func (v NullableACLInfo) Get() *ACLInfo {
	return v.value
}

func (v *NullableACLInfo) Set(val *ACLInfo) {
	v.value = val
	v.isSet = true
}

func (v NullableACLInfo) IsSet() bool {
	return v.isSet
}

func (v *NullableACLInfo) Unset() {
	v.value = nil
	v.isSet = false
}

func NewNullableACLInfo(val *ACLInfo) *NullableACLInfo {
	return &NullableACLInfo{value: val, isSet: true}
}

func (v NullableACLInfo) MarshalJSON() ([]byte, error) {
	return json.Marshal(v.value)
}

func (v *NullableACLInfo) UnmarshalJSON(src []byte) error {
	v.isSet = true
	return json.Unmarshal(src, &v.value)
}



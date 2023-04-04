// Code generated by mockery v2.23.1. DO NOT EDIT.

package mocks

import (
	actor "github.com/determined-ai/determined/master/pkg/actor"
	logger "github.com/determined-ai/determined/master/pkg/logger"

	mock "github.com/stretchr/testify/mock"

	sproto "github.com/determined-ai/determined/master/internal/sproto"

	tasks "github.com/determined-ai/determined/master/pkg/tasks"
)

// Resources is an autogenerated mock type for the Resources type
type Resources struct {
	mock.Mock
}

// Kill provides a mock function with given fields: _a0, _a1
func (_m *Resources) Kill(_a0 *actor.Context, _a1 logger.Context) {
	_m.Called(_a0, _a1)
}

// Start provides a mock function with given fields: _a0, _a1, _a2, _a3
func (_m *Resources) Start(_a0 *actor.Context, _a1 logger.Context, _a2 tasks.TaskSpec, _a3 sproto.ResourcesRuntimeInfo) error {
	ret := _m.Called(_a0, _a1, _a2, _a3)

	var r0 error
	if rf, ok := ret.Get(0).(func(*actor.Context, logger.Context, tasks.TaskSpec, sproto.ResourcesRuntimeInfo) error); ok {
		r0 = rf(_a0, _a1, _a2, _a3)
	} else {
		r0 = ret.Error(0)
	}

	return r0
}

// Summary provides a mock function with given fields:
func (_m *Resources) Summary() sproto.ResourcesSummary {
	ret := _m.Called()

	var r0 sproto.ResourcesSummary
	if rf, ok := ret.Get(0).(func() sproto.ResourcesSummary); ok {
		r0 = rf()
	} else {
		r0 = ret.Get(0).(sproto.ResourcesSummary)
	}

	return r0
}

type mockConstructorTestingTNewResources interface {
	mock.TestingT
	Cleanup(func())
}

// NewResources creates a new instance of Resources. It also registers a testing interface on the mock and a cleanup function to assert the mocks expectations.
func NewResources(t mockConstructorTestingTNewResources) *Resources {
	mock := &Resources{}
	mock.Mock.Test(t)

	t.Cleanup(func() { mock.AssertExpectations(t) })

	return mock
}

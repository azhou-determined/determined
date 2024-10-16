package config

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/determined-ai/determined/master/pkg/device"
	"github.com/determined-ai/determined/master/pkg/ptrs"
)

func TestDispatcherResourceManagerConfig_Validate(t *testing.T) {
	type fields struct {
		LauncherContainerRunType string
		JobProjectSource         *string
		SlotType                 *string
	}
	tests := []struct {
		name   string
		fields fields
		want   error
	}{
		{
			name:   "Default slot_type",
			fields: fields{LauncherContainerRunType: "singularity"},
			want:   nil,
		},
		{
			name: "cuda slot_type",
			fields: fields{
				LauncherContainerRunType: "singularity",
				SlotType:                 ptrs.Ptr("cuda"),
			},
			want: nil,
		},
		{
			name: "Invalid slot_type",
			fields: fields{
				LauncherContainerRunType: "singularity",
				SlotType:                 ptrs.Ptr("invalid-type"),
			},
			want: fmt.Errorf(
				"invalid slot_type 'invalid-type'.  Specify one of cuda, rocm, or cpu"),
		},
		{
			name:   "Invalid type case",
			fields: fields{LauncherContainerRunType: "invalid-type"},
			want:   fmt.Errorf("invalid launch container run type: 'invalid-type'"),
		},
		{
			name:   "singularity case",
			fields: fields{LauncherContainerRunType: "singularity"},
			want:   nil,
		},
		{
			name:   "podman case",
			fields: fields{LauncherContainerRunType: "podman"},
			want:   nil,
		},
		{
			name:   "enroot case",
			fields: fields{LauncherContainerRunType: "enroot"},
			want:   nil,
		},
		{
			name: "workspace case",
			fields: fields{
				LauncherContainerRunType: "enroot",
				JobProjectSource:         ptrs.Ptr("workspace"),
			},
			want: nil,
		},
		{
			name: "project case",
			fields: fields{
				LauncherContainerRunType: "enroot",
				JobProjectSource:         ptrs.Ptr("project"),
			},
			want: nil,
		},
		{
			name: "label case",
			fields: fields{
				LauncherContainerRunType: "enroot",
				JobProjectSource:         ptrs.Ptr("label"),
			},
			want: nil,
		},
		{
			name: "label: case",
			fields: fields{
				LauncherContainerRunType: "enroot",
				JobProjectSource:         ptrs.Ptr("label:something"),
			},
			want: nil,
		},
		{
			name: "invalid project source",
			fields: fields{
				LauncherContainerRunType: "enroot",
				JobProjectSource:         ptrs.Ptr("something-bad"),
			},
			want: fmt.Errorf(
				"invalid job_project_source value: 'something-bad'. " +
					"Specify one of project, workspace or label[:value]"),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := DispatcherResourceManagerConfig{
				LauncherContainerRunType: tt.fields.LauncherContainerRunType,
				JobProjectSource:         tt.fields.JobProjectSource,
				SlotType:                 (*device.Type)(tt.fields.SlotType),
			}
			if got := c.Validate(); got != nil {
				require.Equal(t, tt.want.Error(), got[0].Error(),
					"DispatcherResourceManagerConfig.Validate(%s) = %v, want %v", tt.name, got, tt.want)
			}
		})
	}
}

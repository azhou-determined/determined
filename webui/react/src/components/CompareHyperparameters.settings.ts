import { array, keyof, string, type, undefined as undefinedType, union } from 'io-ts';

import { SettingsConfig } from 'hooks/useSettings';
import { Metric, Scale } from 'types';

export interface CompareHyperparametersSettings {
  hParams: string[];
  metric?: Metric;
  scale: Scale;
}

export const settingsConfigForCompareHyperparameters = (
  hParams: string[],
  projectId: number,
): SettingsConfig<CompareHyperparametersSettings> => ({
  settings: {
    hParams: {
      defaultValue: hParams,
      skipUrlEncoding: true,
      storageKey: 'hParams',
      type: array(string),
    },
    metric: {
      defaultValue: undefined,
      skipUrlEncoding: true,
      storageKey: 'metric',
      type: union([undefinedType, type({ group: string, name: string })]),
    },
    scale: {
      defaultValue: Scale.Linear,
      storageKey: 'scale',
      type: keyof({ linear: null, log: null }),
    },
  },
  storagePath: `experiment-compare-hyperparameters-${projectId}`,
});

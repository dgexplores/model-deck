"""Measure detector behavior on seeded, explicitly synthetic batches."""
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.datasets import load_wine

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from monitoring import drift_report


def evaluate():
    manifest = json.loads((ROOT / 'artifacts/manifest.json').read_text())
    data = load_wine()
    reference = data.data[manifest['split']['train_indices']]
    stable_alerts, injected_alerts, correctly_localized = 0, 0, 0
    runs = 40
    for seed in range(2000, 2000 + runs):
        rng = np.random.default_rng(seed)
        batch = reference[rng.choice(len(reference), 100, replace=True)].copy()
        stable_alerts += drift_report(batch, reference, data.feature_names)['status'] == 'shift_detected'
        feature_index = seed % 13
        batch[:, feature_index] += 5 * reference[:, feature_index].std(ddof=1)
        result = drift_report(batch, reference, data.feature_names)
        injected_alerts += result['status'] == 'shift_detected'
        correctly_localized += result['shifted_features'] == [data.feature_names[feature_index]]
    return {
        'scenario': '100-row bootstrap batches from fixed training reference; paired injection shifts one feature by +5 reference standard deviations',
        'seed_start': 2000, 'runs_per_condition': runs, 'stable_batches_with_alert': stable_alerts,
        'injected_batches_with_alert': injected_alerts, 'injected_batches_only_expected_feature_flagged': correctly_localized,
        'reference_indices': manifest['split']['train_indices'],
        'source_sha256': manifest['provenance']['source_sha256'],
        'limitations': 'Synthetic stress check of one detector configuration, not a production false-alarm rate or evidence of superiority to SeamlessML. Repeated samples are drawn with replacement from the finite training reference. Large mean shifts are easy; joint-only changes, label shift and concept drift are not measured.',
    }


if __name__ == '__main__':
    report = evaluate()
    output = ROOT / 'docs/drift-evaluation.json'
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key not in ('reference_indices', 'source_sha256')}, indent=2))

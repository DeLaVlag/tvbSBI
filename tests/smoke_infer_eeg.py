"""Real GPU smoke path: pass the same CLI arguments as infer_eeg.py.

Example: python tests/smoke_infer_eeg.py --checkpoint MODEL --eeg EPOCHS
  --epoch-index 0 --condition EO --channel-order CHANNELS --output-dir NEW_DIR
No synthetic replacement of preprocessing, extraction, or sampling is used here.
"""
from pathlib import Path
import csv
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from tvbgpu import infer_eeg


def main():
    args = infer_eeg.parse_args()
    out = infer_eeg.run(args)
    metadata = json.loads((out / 'inference_metadata.json').read_text())
    samples = np.load(out / 'posterior_samples.npy')
    assert samples.shape == (args.num_samples, len(metadata['parameter_names']))
    assert np.isfinite(samples).all()
    assert (samples >= np.asarray(metadata['prior_low'])).all()
    assert (samples <= np.asarray(metadata['prior_high'])).all()
    for name, dim in [('raw',metadata['raw_feature_dimension']),
                      ('processed',metadata['final_feature_dimension'])]:
        features = np.load(out / f'eeg_features_{name}.npy')
        assert features.shape == (1,dim) and np.isfinite(features).all()
    with (out / 'posterior_summary.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert [row['parameter'] for row in rows[:7]] == metadata['parameter_names']
    assert (out / 'posterior_distributions.png').stat().st_size > 0
    print('PASS: real EEG inference smoke test')


if __name__ == '__main__':
    main()

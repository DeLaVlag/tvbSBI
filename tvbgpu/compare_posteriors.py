#!/usr/bin/env python3
"""Compare two saved EEG posteriors without loading checkpoints or rerunning inference."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from tvbgpu.analysis.posterior_validation import PARAMETER_ORDER


def load_run(directory):
    directory = Path(directory).resolve()
    metadata = json.loads((directory / 'inference_metadata.json').read_text())
    samples = np.load(directory / 'posterior_samples.npy', allow_pickle=False)
    if metadata['parameter_names'] != list(PARAMETER_ORDER):
        raise ValueError(f'{directory}: unexpected parameter order {metadata["parameter_names"]}')
    if (samples.ndim != 2 or samples.shape[1] != 7 or samples.shape[0] < 2
            or not np.isfinite(samples).all()):
        raise ValueError(f'{directory}: expected finite posterior samples (N>=2, 7), got {samples.shape}')
    if samples.shape[0] != metadata['num_samples']:
        raise ValueError(f'{directory}: samples shape {samples.shape} disagrees with num_samples={metadata["num_samples"]}')
    low, high = np.asarray(metadata['prior_low']), np.asarray(metadata['prior_high'])
    if (low.shape != (7,) or high.shape != (7,) or not np.isfinite([low, high]).all()
            or np.any(high <= low)):
        raise ValueError(f'{directory}: invalid prior bounds')
    if np.any(samples < low) or np.any(samples > high):
        raise ValueError(f'{directory}: posterior draws outside saved prior support')
    # Each fitted model may have a different mask, but its own metadata must be consistent.
    keep = np.asarray(metadata['feature_keep'])
    full_names, names = metadata['feature_names_full'], metadata['feature_names']
    if (keep.dtype != np.bool_ or keep.shape != (len(full_names),)
            or len(full_names) != metadata['raw_feature_dimension']
            or len(names) != metadata['final_feature_dimension']
            or [name for name, retained in zip(full_names, keep) if retained] != names):
        raise ValueError(f'{directory}: inconsistent feature names, mask or dimensions')
    if not metadata.get('checkpoint') or not metadata.get('eeg_input'):
        raise ValueError(f'{directory}: missing checkpoint or observation path')
    return samples, metadata


def verify_runs(small, full, expected_small=None, expected_full=None):
    # Exact metadata agreement is conservative: do not assume differently named files
    # contain identical EEG. Paths establish recorded provenance, not content identity.
    fields = ('eeg_input', 'input_format', 'parameter_names', 'prior_low', 'prior_high',
              'feature_pipeline_version', 'feature_config', 'preprocessing_config',
              'feature_names_full', 'raw_feature_representation')
    for key in fields:
        if key not in small or key not in full or small[key] != full[key]:
            raise ValueError(f'Incompatible runs: {key} differs or is missing')
    required_eeg = ('channel_names', 'sfreq', 'n_times', 'epoch_index', 'condition')
    selection_fields = ('start_sample', 'stop_sample', 'tmin', 'tmax', 'original_epoch_index',
                        'event', 'selected_original_indices', 'original_channel_names',
                        'resampled', 'original_sfreq', 'original_n_times', 'bad_channels',
                        'observed_eeg_projected', 'channel_selection_method')
    for key in required_eeg + selection_fields:
        a, b = small['eeg'], full['eeg']
        if key in required_eeg or key in a or key in b:
            if key not in a or key not in b or a[key] != b[key]:
                raise ValueError(f'Incompatible EEG observations: eeg.{key} differs or is missing')
    if small['checkpoint'] == full['checkpoint']:
        raise ValueError('Both runs record the same checkpoint path; small/full identity is ambiguous')
    for role, metadata, expected in (('small', small, expected_small), ('full', full, expected_full)):
        if expected is not None:
            # A basename checks the recorded filename; a path checks the full recorded path.
            actual = metadata['checkpoint'] if '/' in expected else Path(metadata['checkpoint']).name
            if actual != expected:
                raise ValueError(f'{role} checkpoint {metadata["checkpoint"]!r} does not match {expected!r}')
    return dict(same_recorded_observation=True, identical_parameter_order=True,
                compatible_feature_configuration=True,
                small_checkpoint=small['checkpoint'], full_checkpoint=full['checkpoint'],
                expected_small_checkpoint=expected_small, expected_full_checkpoint=expected_full,
                feature_masks_identical=small['feature_keep'] == full['feature_keep'],
                limitations=[
                    'EEG and checkpoint identity is based on recorded paths/metadata; no content hashes are saved.',
                    'Training simulation counts are not recorded by infer_eeg.py; plot labels are user supplied.',
                    'Saved PCA and normalization are model-specific and are not compared or reapplied.',
                    'Credible intervals describe posterior draws; this comparison does not establish calibration or recovery.'])


def compare_samples(small, full, low, high):
    rows = []
    for i, name in enumerate(PARAMETER_ORDER):
        span = float(high[i] - low[i])
        row = dict(parameter=name, prior_low=float(low[i]), prior_high=float(high[i]))
        for label, samples in (('small', small), ('full', full)):
            q = np.quantile(samples[:, i], [.025, .25, .5, .75, .975])
            row[label + '_num_samples'] = len(samples)
            for key, value in zip(('ci95_low', 'ci50_low', 'median', 'ci50_high', 'ci95_high'), q):
                row[label + '_' + key] = float(value)
                row[label + '_normalized_' + key] = float((value - low[i]) / span)
            row[label + '_width95'] = float(q[4] - q[0])
            row[label + '_normalized_width95'] = float((q[4] - q[0]) / span)
        row['small_full_width95_ratio'] = (row['small_width95'] / row['full_width95']
                                           if row['full_width95'] > 0 else None)
        row['normalized_median_displacement'] = (row['small_median'] - row['full_median']) / span
        row['absolute_normalized_median_displacement'] = abs(row['normalized_median_displacement'])
        rows.append(row)
    return rows


def plot_comparison(rows, destination, small_label, full_label, observation_label):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 7))
    for role, offset, color, label in (('small', -.14, '#d95f02', small_label),
                                        ('full', .14, '#377eb8', full_label)):
        for i, row in enumerate(rows):
            y = i + offset
            ax.hlines(y, row[role + '_normalized_ci95_low'], row[role + '_normalized_ci95_high'],
                      color=color, linewidth=1.5)
            ax.hlines(y, row[role + '_normalized_ci50_low'], row[role + '_normalized_ci50_high'],
                      color=color, linewidth=5)
            ax.plot(row[role + '_normalized_median'], y, 'o', color=color,
                    label=label if i == 0 else None)
    ax.set_yticks(range(len(rows)), [row['parameter'] for row in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel('Position within saved prior: (value − lower bound) / prior width')
    ax.set_title('Posterior comparison\n' + observation_label)
    ax.grid(axis='x', alpha=.25)
    ax.legend(loc='best')
    fig.text(.5, .02, 'Dot: median · Thick line: 50% interval · Thin line: 95% interval\n'
             'Noise remains in log10 space. Interval width does not establish accuracy or calibration.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .07, 1, 1))
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('small_dir', type=Path, help='Workshop inference directory')
    parser.add_argument('full_dir', type=Path, help='Full-model inference directory')
    parser.add_argument('--output-dir', type=Path, default=Path('.'), help='Existing output directory (default: current)')
    parser.add_argument('--expected-small-checkpoint', help='Expected recorded checkpoint basename or exact path')
    parser.add_argument('--expected-full-checkpoint', help='Expected recorded checkpoint basename or exact path')
    parser.add_argument('--small-label', default='Small workshop model')
    parser.add_argument('--full-label', default='Full pretrained model')
    return parser.parse_args(argv)


def run(args):
    if not args.output_dir.is_dir():
        raise ValueError(f'Output directory must already exist: {args.output_dir}')
    small, small_meta = load_run(args.small_dir)
    full, full_meta = load_run(args.full_dir)
    verification = verify_runs(small_meta, full_meta, args.expected_small_checkpoint, args.expected_full_checkpoint)
    for role, samples, metadata in (('Small', small, small_meta), ('Full', full, full_meta)):
        print(f'{role}: checkpoint={metadata["checkpoint"]}; samples={samples.shape}; EEG={metadata["eeg_input"]}')
    print('Verified matching recorded EEG selection, parameter order, priors and feature/preprocessing configuration.')
    for limitation in verification['limitations']:
        print('Note:', limitation)
    rows = compare_samples(small, full, np.asarray(small_meta['prior_low']), np.asarray(small_meta['prior_high']))
    csv_path = args.output_dir / 'posterior_comparison.csv'
    with csv_path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot_path = args.output_dir / 'posterior_comparison.png'
    observation_label = small_meta.get('observation_label') or Path(small_meta['eeg_input']).name
    plot_comparison(rows, plot_path, args.small_label, args.full_label, observation_label)
    report = dict(small_directory=str(args.small_dir.resolve()), full_directory=str(args.full_dir.resolve()),
                  small_samples_shape=list(small.shape), full_samples_shape=list(full.shape),
                  small_label=args.small_label, full_label=args.full_label, verification=verification,
                  small_metadata=small_meta, full_metadata=full_meta,
                  formulas=dict(normalized_position='(value - prior_low) / (prior_high - prior_low)',
                                width95='q97.5 - q2.5', width_ratio='small_width95 / full_width95; blank if full width is zero',
                                normalized_median_displacement='(small_median - full_median) / prior_width'))
    (args.output_dir / 'posterior_comparison_metadata.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(f'Saved {csv_path} and {plot_path}')
    return rows


def main(argv=None):
    run(parse_args(argv))


if __name__ == '__main__':
    main()

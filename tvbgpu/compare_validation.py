"""Compare existing consistency/predictive tests using compact saved JSON only."""
import argparse
import json
from pathlib import Path

import numpy as np


def load_comparison(directory):
    result = json.loads((Path(directory) / 'comparison.json').read_text())
    for test in ('consistency', 'predictive'):
        if test not in result['tests']:
            raise ValueError(f'{directory}: missing completed {test} test')
        values = result['tests'][test]
        theta, target = np.asarray(values['theta_true']), np.asarray(values['x_target'])
        distances = np.asarray(values['diffs' if test == 'consistency' else 'distances'])
        expected = (len(theta),) if test == 'consistency' else (len(theta), values['samples'])
        if (theta.shape != (len(theta), 7) or target.ndim != 2 or len(target) != len(theta)
                or distances.shape != expected or len(theta) == 0
                or not all(np.isfinite(x).all() for x in (theta, target, distances))):
            raise ValueError(f'{directory}: invalid {test} arrays')
        if test == 'predictive':
            baseline = np.asarray(values['baseline_distances'])
            if baseline.shape != expected or not np.isfinite(baseline).all():
                raise ValueError(f'{directory}: invalid baseline distances')
    return result


def verify_comparison(small, full):
    for key in ('shared_inputs_sha256', 'reference_sha256', 'seed', 'simulator_args'):
        if small.get(key) is None or small[key] != full.get(key):
            raise ValueError(f'Cannot compare: {key} differs or is missing; use the same --shared-inputs file')
    for test in ('consistency', 'predictive'):
        for key in ('theta_true', 'observation_indices', 'x_target', 'samples'):
            if small['tests'][test][key] != full['tests'][test][key]:
                raise ValueError(f'Cannot compare: {test}.{key} differs')
    if small['tests']['predictive']['baseline_distances'] != full['tests']['predictive']['baseline_distances']:
        raise ValueError('Cannot compare: predictive baselines differ')


def plot_comparison(small_directory, full_directory, output_dir=None):
    """Return two figures; no checkpoint, torch, SBI or simulator required."""
    import matplotlib.pyplot as plt
    small, full = load_comparison(small_directory), load_comparison(full_directory)
    verify_comparison(small, full)
    labels = [f'Small ({small["training_simulations"]} training simulations)',
              f'Full ({full["training_simulations"]} training simulations)']
    figures = {}
    for test in ('consistency', 'predictive'):
        fig, ax = plt.subplots(figsize=(8, 5))
        n = len(small['tests'][test]['theta_true'])
        positions = np.arange(n)
        for run, offset, color, label in zip((small, full), (-.12, .12), ('#d95f02', '#377eb8'), labels):
            values = run['tests'][test]
            if test == 'consistency':
                ax.scatter(positions + offset, values['diffs'], color=color, label=label)
            else:
                draws = np.asarray(values['distances'])
                for i, row in enumerate(draws):
                    ax.scatter(i + offset + np.linspace(-.06, .06, len(row)), row,
                               color=color, alpha=.65, label=label if i == 0 else None)
                    ax.plot([i + offset - .07, i + offset + .07], [np.median(row)] * 2, color=color)
        if test == 'predictive':
            baseline = np.median(small['tests'][test]['baseline_distances'], axis=1)
            ax.scatter(positions, baseline, marker='x', color='black', label='Random-row baseline median')
        ax.set_xticks(positions, [f'Case {i + 1}' for i in positions])
        ax.set_ylabel('L2 feature error (common reference normalization)')
        ax.set_title('Posterior-mean resimulation consistency' if test == 'consistency' else
                     'Posterior predictive draws (short lines: case medians)')
        ax.legend()
        ax.grid(axis='y', alpha=.2)
        fig.tight_layout()
        figures[test] = fig
    if output_dir is not None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        for test, fig in figures.items():
            fig.savefig(output / (test + '_comparison.png'), dpi=160)
    return figures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('small_directory')
    parser.add_argument('full_directory')
    parser.add_argument('--output-dir', default='validation_comparison')
    args = parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    plot_comparison(args.small_directory, args.full_directory, args.output_dir)


if __name__ == '__main__':
    main()

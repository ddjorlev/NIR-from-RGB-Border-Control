"""
Builds paper-ready comparison artifacts from the .npz files produced by
evaluate_protocol_A.py --save_npz:
  - analysis/figures/roc_curves_<backbone>[_<tag>].png   (full ROC, log-scale FPR)
  - analysis/figures/tpr_at_fpr_<backbone>[_<tag>].png    (grouped bars + 95% CI)
  - analysis/results_table[_<tag>].csv                    (flat table)

By default includes every recognized .npz file in analysis/npz/. Use
--stems to restrict to a subset (e.g. only v2 + baselines, or only v1 vs
v2) and --tag to name the output files distinctly so multiple report
variants can coexist.

Examples:
  python generate_report.py                                   # everything
  python generate_report.py --stems arcface_rgb_baseline,arcface_pix2pix,arcface_pid_v2,adaface_rgb_baseline,adaface_pix2pix,adaface_pid_v2 --tag v2_vs_baselines --title-suffix "(v2)"
  python generate_report.py --stems arcface_pid,arcface_pid_v2,adaface_pid,adaface_pid_v2 --tag v1_vs_v2 --title-suffix "(v1 vs v2)"
"""

import argparse
import glob
import os
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

NPZ_DIR = os.path.join(os.path.dirname(__file__), 'npz')
FIG_DIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# Map npz filename stem -> (backbone, display label). Extend as new methods
# (colleagues' models) are dropped into analysis/npz/ with --save_npz.
LABELS = {
    'arcface_rgb_baseline': ('arcface', 'RGB -> NIR (baseline, no translation)'),
    'arcface_pix2pix': ('arcface', 'Pix2Pix GAN'),
    'arcface_pid': ('arcface', 'PID Diffusion v1 (ours)'),
    'arcface_pid_v2': ('arcface', 'PID Diffusion (ours)'),
    'adaface_rgb_baseline': ('adaface', 'RGB -> NIR (baseline, no translation)'),
    'adaface_pix2pix': ('adaface', 'Pix2Pix GAN'),
    'adaface_pid': ('adaface', 'PID Diffusion v1 (ours)'),
    'adaface_pid_v2': ('adaface', 'PID Diffusion (ours)'),
}


def load_all(allowed_stems=None):
    results = {}
    for path in sorted(glob.glob(os.path.join(NPZ_DIR, '*.npz'))):
        stem = os.path.splitext(os.path.basename(path))[0]
        if stem not in LABELS:
            print(f"Skipping {path}: add an entry to LABELS in generate_report.py to include it")
            continue
        if allowed_stems is not None and stem not in allowed_stems:
            continue
        backbone, label = LABELS[stem]
        data = np.load(path, allow_pickle=True)
        results.setdefault(backbone, {})[label] = data
    return results


def write_table(results, tag):
    suffix = f"_{tag}" if tag else ""
    out_path = os.path.join(os.path.dirname(__file__), f'results_table{suffix}.csv')
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['backbone', 'method', 'fpr_target', 'mean_tpr_pct', 'ci_lower_pct', 'ci_upper_pct'])
        for backbone, methods in results.items():
            for label, data in methods.items():
                for fpr, mean_tpr, lo, hi in zip(data['fpr_targets'], data['mean_tpr'], data['ci_lower'], data['ci_upper']):
                    writer.writerow([backbone, label, fpr, f"{mean_tpr:.3f}", f"{lo:.3f}", f"{hi:.3f}"])
    print(f"Wrote {out_path}")


def plot_roc(results, tag, title_suffix):
    suffix = f"_{tag}" if tag else ""
    for backbone, methods in results.items():
        plt.figure(figsize=(6, 5))
        for label, data in methods.items():
            genuine = data['genuine_scores']
            impostor = data['impostor_scores']
            scores = np.concatenate([genuine, impostor])
            y_true = np.concatenate([np.ones_like(genuine), np.zeros_like(impostor)])
            fpr, tpr, _ = roc_curve(y_true, scores)
            fpr = np.clip(fpr, 1e-6, 1.0)  # avoid log(0)
            plt.plot(fpr, tpr, label=label)
        plt.xscale('log')
        plt.xlabel('False Positive Rate (log scale)')
        plt.ylabel('True Positive Rate')
        plt.title(f'Protocol A Verification ROC — {backbone} {title_suffix}'.strip())
        plt.legend(loc='lower right', fontsize=9)
        plt.grid(True, which='both', alpha=0.3)
        plt.tight_layout()
        out_path = os.path.join(FIG_DIR, f'roc_curves_{backbone}{suffix}.png')
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Wrote {out_path}")


def plot_bars(results, tag, title_suffix):
    suffix = f"_{tag}" if tag else ""
    for backbone, methods in results.items():
        fpr_targets = None
        labels = list(methods.keys())
        n_methods = len(labels)
        width = 0.8 / n_methods

        plt.figure(figsize=(7, 5))
        for i, label in enumerate(labels):
            data = methods[label]
            fpr_targets = data['fpr_targets']
            x = np.arange(len(fpr_targets)) + i * width
            means = data['mean_tpr']
            lo = data['mean_tpr'] - data['ci_lower']
            hi = data['ci_upper'] - data['mean_tpr']
            plt.bar(x, means, width=width, yerr=[lo, hi], capsize=3, label=label)

        fpr_str = [f"1e-{int(abs(np.log10(f)))}" for f in fpr_targets]
        plt.xticks(np.arange(len(fpr_targets)) + width * (n_methods - 1) / 2, fpr_str)
        plt.xlabel('FPR operating point')
        plt.ylabel('TPR (%)')
        plt.title(f'Protocol A Verification TPR@FPR — {backbone} {title_suffix} (95% CI)'.strip())
        plt.legend(fontsize=9)
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        out_path = os.path.join(FIG_DIR, f'tpr_at_fpr_{backbone}{suffix}.png')
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Wrote {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stems', default=None, help='Comma-separated npz stems to include (default: all recognized)')
    parser.add_argument('--tag', default=None, help='Suffix for output filenames, to keep report variants separate')
    parser.add_argument('--title-suffix', default='', help='Extra text appended to plot titles')
    args = parser.parse_args()

    allowed = set(args.stems.split(',')) if args.stems else None
    results = load_all(allowed)
    if not results:
        print("No matching .npz files found in analysis/npz/ for the requested stems.")
        return
    write_table(results, args.tag)
    plot_roc(results, args.tag, args.title_suffix)
    plot_bars(results, args.tag, args.title_suffix)


if __name__ == '__main__':
    main()

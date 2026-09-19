"""
Parses the SLURM training log (src/train.py's logging output) and plots
train/val loss vs epoch. Useful as a "does the model converge" figure for
the paper, and as evidence for the early-stopping decision if training is
cut short.

Usage: python analysis/plot_training_curve.py path/to/train_tufts_<jobid>.err
"""

import re
import sys
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

TRAIN_ONLY_RE = re.compile(r"Epoch (\d+): Train Loss = ([\d.]+)")
SUMMARY_EPOCH_RE = re.compile(r"Epoch (\d+) Summary:")
TRAIN_LOSS_RE = re.compile(r"Train Loss: ([\d.]+)")
VAL_LOSS_RE = re.compile(r"Val Loss: ([\d.]+)")


def parse_log(path):
    train_epochs, train_losses = [], []
    val_epochs, val_losses = [], []

    pending_epoch = None
    with open(path, 'r', errors='ignore') as f:
        for line in f:
            m = TRAIN_ONLY_RE.search(line)
            if m:
                train_epochs.append(int(m.group(1)))
                train_losses.append(float(m.group(2)))
                continue

            m = SUMMARY_EPOCH_RE.search(line)
            if m:
                pending_epoch = int(m.group(1))
                continue

            m = TRAIN_LOSS_RE.search(line)
            if m and pending_epoch is not None:
                train_epochs.append(pending_epoch)
                train_losses.append(float(m.group(1)))
                continue

            m = VAL_LOSS_RE.search(line)
            if m and pending_epoch is not None:
                val_epochs.append(pending_epoch)
                val_losses.append(float(m.group(1)))
                pending_epoch = None
                continue

    return train_epochs, train_losses, val_epochs, val_losses


def main():
    if len(sys.argv) != 2:
        print("Usage: python plot_training_curve.py <train_log.err>")
        sys.exit(1)

    log_path = sys.argv[1]
    train_epochs, train_losses, val_epochs, val_losses = parse_log(log_path)

    plt.figure(figsize=(7, 5))
    plt.plot(train_epochs, train_losses, label='Train loss', alpha=0.7, linewidth=1)
    plt.plot(val_epochs, val_losses, label='Val loss', marker='o', markersize=3)

    if val_losses:
        best_idx = min(range(len(val_losses)), key=lambda i: val_losses[i])
        plt.scatter([val_epochs[best_idx]], [val_losses[best_idx]], color='red', zorder=5,
                    label=f'Best val (epoch {val_epochs[best_idx]}, {val_losses[best_idx]:.4f})')

    plt.xlabel('Epoch')
    plt.ylabel('Loss (DDPM epsilon-MSE + physics term)')
    plt.title('PID Diffusion Training Curve (Tufts Faces)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..',
                            'archive_extract', 'dlord_rgbnir_verification', 'analysis', 'figures')
    out_dir = os.path.normpath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'training_curve.png')
    plt.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")


if __name__ == '__main__':
    main()

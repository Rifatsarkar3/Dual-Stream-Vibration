import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path


def plot_training_history(history, output_dir='outputs'):
    """Plot training curves."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].plot(history['train_loss'], label='Train Loss')
    axes[0, 0].plot(history['val_loss'], label='Val Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_title('Training & Validation Loss')

    axes[0, 1].plot(history['val_mae'], label='MAE')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_title('Validation MAE')

    axes[1, 0].plot(history['val_r2'], label='R²')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('R² Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_title('Validation R²')

    axes[1, 1].plot(history['gate_alpha'], label='Gate α (mean)')
    axes[1, 1].axhline(y=0.5, color='r', linestyle='--', label='Equal Weight')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Gate Value')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_title('Adaptive Fusion Gate α')

    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to {output_dir / 'training_curves.png'}")


def plot_predictions(targets, predictions, output_dir='outputs', title='RUL Predictions'):
    """Plot predicted vs. actual RUL."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].scatter(targets, predictions, alpha=0.6, s=20)
    min_val = min(targets.min(), predictions.min())
    max_val = max(targets.max(), predictions.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect')
    axes[0].set_xlabel('Ground Truth RUL')
    axes[0].set_ylabel('Predicted RUL')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title(title)

    errors = predictions - targets
    axes[1].hist(errors, bins=30, edgecolor='black', alpha=0.7)
    axes[1].axvline(x=0, color='r', linestyle='--', lw=2, label='Zero Error')
    axes[1].set_xlabel('Prediction Error')
    axes[1].set_ylabel('Frequency')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].set_title('Error Distribution')

    plt.tight_layout()
    plt.savefig(output_dir / 'predictions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Predictions plot saved to {output_dir / 'predictions.png'}")


def plot_noise_robustness(snr_levels, mae_values, rmse_values, r2_values, output_dir='outputs'):
    """Plot noise robustness across SNR levels."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(snr_levels, mae_values, marker='o', linewidth=2, markersize=8)
    axes[0].set_xlabel('Signal-to-Noise Ratio (dB)')
    axes[0].set_ylabel('MAE')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title('MAE vs. SNR')
    axes[0].invert_xaxis()

    axes[1].plot(snr_levels, rmse_values, marker='s', linewidth=2, markersize=8, color='orange')
    axes[1].set_xlabel('Signal-to-Noise Ratio (dB)')
    axes[1].set_ylabel('RMSE')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title('RMSE vs. SNR')
    axes[1].invert_xaxis()

    axes[2].plot(snr_levels, r2_values, marker='^', linewidth=2, markersize=8, color='green')
    axes[2].set_xlabel('Signal-to-Noise Ratio (dB)')
    axes[2].set_ylabel('R² Score')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_title('R² vs. SNR')
    axes[2].invert_xaxis()

    plt.tight_layout()
    plt.savefig(output_dir / 'noise_robustness.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Noise robustness plot saved to {output_dir / 'noise_robustness.png'}")


def plot_ablation_comparison(ablation_results, output_dir='outputs'):
    """Plot ablation study results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    models = list(ablation_results.keys())
    mae_values = [ablation_results[m].get('mae_mean', ablation_results[m].get('mae', 0)) for m in models]
    rmse_values = [ablation_results[m].get('rmse_mean', ablation_results[m].get('rmse', 0)) for m in models]
    r2_values = [ablation_results[m].get('r2_mean', ablation_results[m].get('r2', 0)) for m in models]

    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.bar(x - width, mae_values, width, label='MAE', alpha=0.8)
    ax.bar(x, rmse_values, width, label='RMSE', alpha=0.8)
    ax.bar(x + width, r2_values, width, label='R²', alpha=0.8)

    ax.set_xlabel('Model Configuration')
    ax.set_ylabel('Metric Value')
    ax.set_title('Ablation Study: Component Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'ablation_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Ablation comparison plot saved to {output_dir / 'ablation_comparison.png'}")


def plot_gate_distribution(gate_values, output_dir='outputs'):
    """Plot distribution of adaptive fusion gate values."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(gate_values, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    ax.axvline(x=gate_values.mean(), color='r', linestyle='--', lw=2, label=f'Mean: {gate_values.mean():.3f}')
    ax.axvline(x=0.5, color='g', linestyle='--', lw=2, label='Equal Weight (0.5)')
    ax.set_xlabel('Gate Value α (1D stream weight)')
    ax.set_ylabel('Frequency')
    ax.set_title('Adaptive Fusion Gate Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'gate_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Gate distribution plot saved to {output_dir / 'gate_distribution.png'}")

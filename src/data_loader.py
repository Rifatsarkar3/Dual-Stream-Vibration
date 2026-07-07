import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from pathlib import Path


class PronostiaDataset(Dataset):
    """PRONOSTIA bearing dataset with dual-stream preprocessing."""

    def __init__(self, data_dir, train=True, test_run=False, add_noise=False, snr=None):
        """
        Args:
            data_dir: Path to Processed_PRONOSTIA directory
            train: If True, use training data; else use test data
            test_run: If True, load only first 100 samples for testing
            add_noise: If True, add Gaussian noise at specified SNR
            snr: Signal-to-Noise Ratio in dB (if add_noise=True)
        """
        self.data_dir = Path(data_dir)
        self.train = train
        self.test_run = test_run
        self.add_noise = add_noise
        self.snr = snr
        self.samples = []

        self._load_data()

    def _load_data(self):
        """Load preprocessed vibration and spectrogram data."""
        if self.train:
            pattern = "train_*.npz"
        else:
            pattern = "test_*.npz"

        file_list = sorted(self.data_dir.glob(pattern))
        if not file_list:
            raise FileNotFoundError(f"No data files found in {self.data_dir}")

        for file_path in file_list:
            try:
                data = np.load(file_path, allow_pickle=True)
                vibration = data['vibration'].astype(np.float32)  # (N, 2560)
                spectrogram = data['spectrogram'].astype(np.float32)  # (N, freq_bins)
                rul = data['rul'].astype(np.float32)  # (N,)

                # Load each window as a separate sample
                n_samples = vibration.shape[0]
                for i in range(n_samples):
                    self.samples.append({
                        'vibration': vibration[i],
                        'spectrogram': spectrogram[i],
                        'rul': rul[i],
                        'file': file_path.name
                    })
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue

        if self.test_run and len(self.samples) > 100:
            self.samples = self.samples[:100]

        if not self.samples:
            raise ValueError(f"No valid samples loaded from {self.data_dir}")

    def _add_gaussian_noise(self, signal, snr_db):
        """Add Gaussian noise to achieve target SNR."""
        signal_power = np.mean(signal ** 2)
        snr_linear = 10 ** (snr_db / 10.0)
        noise_power = signal_power / snr_linear
        noise = np.random.normal(0, np.sqrt(noise_power), signal.shape)
        return signal + noise

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        vibration = sample['vibration'].copy()
        spectrogram = sample['spectrogram'].copy()
        rul = sample['rul'].copy()

        if self.add_noise and self.snr is not None:
            vibration = self._add_gaussian_noise(vibration, self.snr)

        # Convert to tensors
        vibration = torch.from_numpy(vibration).unsqueeze(0)  # (1, 2560)
        spectrogram = torch.from_numpy(spectrogram).unsqueeze(0)  # (1, 64, 64)

        # Swin-Tiny expects 3-channel input, so repeat the 2D spectrogram 3 times
        spectrogram = spectrogram.repeat(3, 1, 1)  # (3, 64, 64)

        rul = torch.tensor(rul, dtype=torch.float32)

        return vibration, spectrogram, rul


def create_dataloaders(
    data_dir,
    batch_size=32,
    num_workers=4,
    train_split=0.8,
    add_noise=False,
    snr=None,
    test_run=False
):
    """Create train and validation dataloaders."""

    train_dataset = PronostiaDataset(
        data_dir,
        train=True,
        test_run=test_run,
        add_noise=add_noise,
        snr=snr
    )

    test_dataset = PronostiaDataset(
        data_dir,
        train=False,
        test_run=test_run,
        add_noise=add_noise,
        snr=snr
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

    return train_loader, test_loader

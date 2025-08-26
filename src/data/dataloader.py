"""Data loading utilities for PPG glucose estimation."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from scipy.io import loadmat
from torch.utils.data import DataLoader, Dataset


class PPGDataset(Dataset):
    """PyTorch dataset for PPG signals and glucose labels.
    
    Examples:
        >>> dataset = PPGDataset(data_dir="data/", mode="train")
        >>> dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    """
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        mode: str = "train",
        transform: Optional[callable] = None,
        target_transform: Optional[callable] = None,
        window_length: int = 300,
    ):
        """Initialize PPG dataset.
        
        Args:
            data_dir: Directory containing PPG data
            mode: 'train', 'val', or 'test'
            transform: Optional transform for PPG windows
            target_transform: Optional transform for glucose labels
            window_length: Expected window length
        """
        self.data_dir = Path(data_dir)
        self.mode = mode
        self.transform = transform
        self.target_transform = target_transform
        self.window_length = window_length
        
        # Load data files
        self.samples = self._load_samples()
    
    def _load_samples(self) -> List[Dict]:
        """Load sample metadata from directory."""
        samples = []
        
        # Look for CSV or NPZ files
        data_path = self.data_dir / self.mode
        
        if data_path.exists():
            # Load preprocessed windows
            for window_file in sorted(data_path.glob("window_*.npz")):
                data = np.load(window_file)
                samples.append({
                    "ppg": data["ppg"],
                    "glucose": data["glucose"],
                    "subject_id": data.get("subject_id", 0),
                })
        else:
            # Load raw signals and labels
            signal_files = sorted(self.data_dir.glob("signal_*.csv"))
            label_files = sorted(self.data_dir.glob("label_*.csv"))
            
            for signal_file, label_file in zip(signal_files, label_files):
                # Load signal
                if signal_file.suffix == ".csv":
                    signal = pd.read_csv(signal_file).values.flatten()
                elif signal_file.suffix == ".mat":
                    signal = loadmat(signal_file)["signal"].flatten()
                else:
                    signal = np.load(signal_file)
                
                # Load label
                if label_file.suffix == ".csv":
                    label_data = pd.read_csv(label_file)
                    glucose = label_data["Glucose"].values[0]
                elif label_file.suffix == ".mat":
                    glucose = loadmat(label_file)["glucose"].item()
                else:
                    glucose = np.load(label_file).item()
                
                # Extract subject ID from filename
                subject_id = int(signal_file.stem.split("_")[1])
                
                # Create windows from signal
                n_windows = len(signal) // self.window_length
                for i in range(n_windows):
                    start = i * self.window_length
                    end = start + self.window_length
                    window = signal[start:end]
                    
                    samples.append({
                        "ppg": window,
                        "glucose": glucose,
                        "subject_id": subject_id,
                    })
        
        return samples
    
    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            (ppg_window, glucose_label)
        """
        sample = self.samples[idx]
        
        ppg = sample["ppg"]
        glucose = sample["glucose"]
        
        # Apply transforms
        if self.transform:
            ppg = self.transform(ppg)
        
        if self.target_transform:
            glucose = self.target_transform(glucose)
        
        # Convert to tensors
        ppg = torch.FloatTensor(ppg)
        glucose = torch.FloatTensor([glucose])
        
        return ppg, glucose


def load_ppg_data(
    data_path: Union[str, Path],
    file_format: str = "csv",
) -> Tuple[np.ndarray, float]:
    """Load a single PPG signal and glucose label.
    
    Args:
        data_path: Path to data file
        file_format: 'csv', 'mat', or 'npz'
        
    Returns:
        (ppg_signal, glucose_value)
    """
    data_path = Path(data_path)
    
    if file_format == "csv":
        df = pd.read_csv(data_path)
        if "ppg" in df.columns:
            ppg = df["ppg"].values
        else:
            ppg = df.iloc[:, 0].values
        
        if "glucose" in df.columns:
            glucose = df["glucose"].iloc[0]
        else:
            glucose = None
            
    elif file_format == "mat":
        mat_data = loadmat(data_path)
        ppg = mat_data.get("signal", mat_data.get("ppg")).flatten()
        glucose = mat_data.get("glucose", np.array([None])).item()
        
    elif file_format == "npz":
        data = np.load(data_path)
        ppg = data["ppg"]
        glucose = data.get("glucose", None)
        
    else:
        raise ValueError(f"Unknown file format: {file_format}")
    
    return ppg, glucose


def create_dataloaders(
    data_dir: Union[str, Path],
    batch_size: int = 32,
    num_workers: int = 4,
    train_split: float = 0.7,
    val_split: float = 0.15,
    test_split: float = 0.15,
    patient_wise_split: bool = True,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test dataloaders.
    
    Args:
        data_dir: Directory containing PPG data
        batch_size: Batch size for training
        num_workers: Number of data loading workers
        train_split: Training set proportion
        val_split: Validation set proportion
        test_split: Test set proportion
        patient_wise_split: Split by patient ID to avoid leakage
        seed: Random seed for splitting
        
    Returns:
        (train_loader, val_loader, test_loader)
    """
    # Load all data
    dataset = PPGDataset(data_dir, mode="all")
    
    if patient_wise_split:
        # Get unique subject IDs
        subject_ids = np.unique([s["subject_id"] for s in dataset.samples])
        np.random.seed(seed)
        np.random.shuffle(subject_ids)
        
        # Split subjects
        n_subjects = len(subject_ids)
        n_train = int(n_subjects * train_split)
        n_val = int(n_subjects * val_split)
        
        train_subjects = set(subject_ids[:n_train])
        val_subjects = set(subject_ids[n_train:n_train + n_val])
        test_subjects = set(subject_ids[n_train + n_val:])
        
        # Split samples
        train_samples = [s for s in dataset.samples if s["subject_id"] in train_subjects]
        val_samples = [s for s in dataset.samples if s["subject_id"] in val_subjects]
        test_samples = [s for s in dataset.samples if s["subject_id"] in test_subjects]
        
    else:
        # Random split
        n_samples = len(dataset)
        indices = np.arange(n_samples)
        np.random.seed(seed)
        np.random.shuffle(indices)
        
        n_train = int(n_samples * train_split)
        n_val = int(n_samples * val_split)
        
        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train + n_val]
        test_indices = indices[n_train + n_val:]
        
        train_samples = [dataset.samples[i] for i in train_indices]
        val_samples = [dataset.samples[i] for i in val_indices]
        test_samples = [dataset.samples[i] for i in test_indices]
    
    # Create dataset objects
    train_dataset = PPGDataset(data_dir, mode="train")
    train_dataset.samples = train_samples
    
    val_dataset = PPGDataset(data_dir, mode="val")
    val_dataset.samples = val_samples
    
    test_dataset = PPGDataset(data_dir, mode="test")
    test_dataset.samples = test_samples
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader, test_loader
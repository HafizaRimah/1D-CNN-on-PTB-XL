"""
Dataset loading, preprocessing, and inspection pipeline for PTB-XL ECG dataset.

Handles:
1. Downloading and reading ptbxl_database.csv and scp_statements.csv.
2. Mapping scp_codes to the 5 diagnostic superclasses (NORM, MI, STTC, CD, HYP).
3. Loading raw 12-lead signals using wfdb.rdsamp into shape (N, 12, 1000).
4. Standardizing each lead using training fold statistics (folds 1-8).
5. PyTorch Dataset and DataLoader construction with patient-aware splits.
6. 12-lead ECG plotting for visual inspection.
"""

import ast
import os
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    import wfdb
except ImportError:
    wfdb = None


# 5 Diagnostic superclasses in PTB-XL benchmark
SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]

# Standard 12-lead ordering in PTB-XL recordings
STANDARD_LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def download_ptbxl(data_dir: str = "data/ptbxl", sampling_rate: int = 100) -> None:
    """
    Downloads PTB-XL dataset files from PhysioNet if not already present.
    """
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    base_url = "https://physionet.org/files/ptb-xl/1.0.3/"

    # Metadata files
    meta_files = ["ptbxl_database.csv", "scp_statements.csv"]
    for file_name in meta_files:
        dest = data_path / file_name
        if not dest.exists():
            print(f"Downloading {file_name}...")
            urllib.request.urlretrieve(f"{base_url}{file_name}?download", dest)
            print(f"Saved {file_name} to {dest}")

    # Records zip file
    records_zip_name = f"records{sampling_rate}.zip"
    records_dir = data_path / f"records{sampling_rate}"
    if not records_dir.exists():
        records_zip_path = data_path / records_zip_name
        if not records_zip_path.exists():
            zip_url = f"https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip"
            print(f"Downloading PTB-XL dataset archive (~1.7GB)...")
            urllib.request.urlretrieve(zip_url, records_zip_path)

        print(f"Extracting {records_zip_name}...")
        with zipfile.ZipFile(records_zip_path, "r") as zip_ref:
            zip_ref.extractall(data_path)
        print("Extraction complete.")


def load_ptbxl_metadata(data_dir: str = "data/ptbxl") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads ptbxl_database.csv and scp_statements.csv, then maps diagnostic scp_codes
    into multi-hot indicator columns for the 5 superclasses:
    ['NORM', 'MI', 'STTC', 'CD', 'HYP'].

    Returns:
        df: DataFrame indexed by ecg_id with multi-hot superclass columns and strat_fold.
        scp_df: Statement definitions DataFrame.
    """
    data_path = Path(data_dir)
    db_file = data_path / "ptbxl_database.csv"
    scp_file = data_path / "scp_statements.csv"

    if not db_file.exists() or not scp_file.exists():
        raise FileNotFoundError(
            f"Metadata CSVs not found in {data_dir}. Run download_ptbxl() or place CSVs in {data_dir}."
        )

    # 1. Read metadata
    df = pd.read_csv(db_file, index_col="ecg_id")
    df.scp_codes = df.scp_codes.apply(ast.literal_eval)

    # 2. Read scp_statements and build mapping for diagnostic superclasses
    scp_df = pd.read_csv(scp_file, index_col=0)
    # diagnostic == 1 indicates diagnostic statement (as opposed to rhythm or form)
    diagnostic_statements = scp_df[scp_df.diagnostic == 1]
    code_to_class = diagnostic_statements.diagnostic_class.to_dict()

    # 3. Map individual SCP codes to superclasses
    def map_to_superclasses(scp_dict: Dict[str, float]) -> List[str]:
        classes = set()
        for code in scp_dict.keys():
            if code in code_to_class:
                super_cls = code_to_class[code]
                if super_cls in SUPERCLASSES:
                    classes.add(super_cls)
        return list(classes)

    df["diagnostic_superclass"] = df.scp_codes.apply(map_to_superclasses)

    # 4. Multi-hot encode the 5 superclasses
    for sc in SUPERCLASSES:
        df[sc] = df["diagnostic_superclass"].apply(lambda classes: int(sc in classes))

    print(f"Loaded {len(df)} ECG records.")
    print("Class distribution across entire dataset:")
    for sc in SUPERCLASSES:
        count = df[sc].sum()
        pct = (count / len(df)) * 100
        print(f"  {sc:5s}: {count:5d} ({pct:.1f}%)")

    return df, scp_df


def load_raw_signals(
    df: pd.DataFrame,
    data_dir: str = "data/ptbxl",
    sampling_rate: int = 100,
) -> np.ndarray:
    """
    Loads raw 12-lead ECG signals using wfdb.rdsamp.

    Input DataFrame must have 'filename_lr' (for 100 Hz) or 'filename_hr' (for 500 Hz).

    Returns:
        signals: numpy array of shape (N, 12, 1000) for PyTorch Conv1d
                 (N samples, 12 channels/leads, 1000 time steps).
    """
    if wfdb is None:
        raise ImportError("wfdb library is required to load PTB-XL signals. Install with `pip install wfdb`.")

    data_path = Path(data_dir)
    file_col = "filename_lr" if sampling_rate == 100 else "filename_hr"

    if file_col not in df.columns:
        raise KeyError(f"Expected column '{file_col}' in DataFrame.")

    signals = []
    print(f"Loading {len(df)} ECG records at {sampling_rate} Hz...")

    for rel_path in tqdm(df[file_col], desc="Reading ECG signals"):
        full_path = str(data_path / rel_path)
        # wfdb.rdsamp reads header and dat files, returning (signal_data, metadata)
        record_data, _ = wfdb.rdsamp(full_path)
        signals.append(record_data)

    # Raw signal shape from wfdb is (N, time_steps, leads) = (N, 1000, 12)
    signals = np.array(signals, dtype=np.float32)

    # Transpose to PyTorch standard (N, channels/leads, time_steps) = (N, 12, 1000)
    signals = np.transpose(signals, (0, 2, 1))

    print(f"Signals loaded into shape: {signals.shape} (N, leads, time_steps)")
    return signals


def standardize_per_lead(
    X: np.ndarray,
    train_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Standardises each lead using the training set's mean and standard deviation.

    X shape: (N, 12, 1000)
    train_mask: Boolean array of shape (N,) indicating training records.

    Returns:
        X_norm: Standardized signals of shape (N, 12, 1000).
        lead_mean: Array of shape (1, 12, 1) computed on training set.
        lead_std: Array of shape (1, 12, 1) computed on training set.
    """
    X_train = X[train_mask]

    # Compute mean and std per lead across samples and time: axis (0, 2)
    lead_mean = np.mean(X_train, axis=(0, 2), keepdims=True)  # (1, 12, 1)
    lead_std = np.std(X_train, axis=(0, 2), keepdims=True)    # (1, 12, 1)

    # Avoid division by zero
    lead_std = np.where(lead_std == 0, 1.0, lead_std)

    X_norm = (X - lead_mean) / lead_std
    return X_norm.astype(np.float32), lead_mean, lead_std


class PTBXLDataset(Dataset):
    """
    PyTorch Dataset for PTB-XL ECG records.
    Yields:
        signal: FloatTensor of shape (12, 1000)
        labels: FloatTensor of shape (5,) for multi-label binary targets
    """

    def __init__(self, signals: np.ndarray, labels: np.ndarray):
        self.signals = torch.from_numpy(signals).float()
        self.labels = torch.from_numpy(labels).float()

    def __len__(self) -> int:
        return len(self.signals)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.signals[idx], self.labels[idx]


def get_dataloaders(
    X: np.ndarray,
    y: np.ndarray,
    df: pd.DataFrame,
    batch_size: int = 64,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Constructs train, validation, and test DataLoaders according to PTB-XL strat_fold:
    - Train: folds 1-8
    - Val: fold 9
    - Test: fold 10

    Returns:
        train_loader, val_loader, test_loader
    """
    train_mask = (df.strat_fold <= 8).values
    val_mask = (df.strat_fold == 9).values
    test_mask = (df.strat_fold == 10).values

    train_ds = PTBXLDataset(X[train_mask], y[train_mask])
    val_ds = PTBXLDataset(X[val_mask], y[val_mask])
    test_ds = PTBXLDataset(X[test_mask], y[test_mask])

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    print(
        f"DataLoaders prepared -> Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}"
    )
    return train_loader, val_loader, test_loader


def plot_12lead_ecg(
    signal: np.ndarray,
    title: str = "12-Lead ECG",
    lead_names: Optional[List[str]] = None,
    sampling_rate: int = 100,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plots a 12-lead ECG in a clean 6x2 grid format.

    signal: array of shape (12, 1000) or (1000, 12).
    """
    if lead_names is None:
        lead_names = STANDARD_LEAD_NAMES

    # Ensure shape is (12, time_steps)
    if signal.shape[0] != 12 and signal.shape[1] == 12:
        signal = signal.T

    num_samples = signal.shape[1]
    time_axis = np.arange(num_samples) / sampling_rate

    fig, axes = plt.subplots(6, 2, figsize=(14, 10), sharex=True)
    axes = axes.flatten()

    for i in range(12):
        axes[i].plot(time_axis, signal[i], color="#1f77b4", linewidth=1.1)
        axes[i].set_title(lead_names[i], loc="left", fontsize=11, fontweight="bold", pad=2)
        axes[i].grid(True, linestyle="--", alpha=0.5)
        axes[i].set_ylabel("mV", fontsize=9)

    axes[-2].set_xlabel("Time (seconds)", fontsize=10)
    axes[-1].set_xlabel("Time (seconds)", fontsize=10)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"ECG plot saved to: {save_path}")

    return fig

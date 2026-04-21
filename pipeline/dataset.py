"""Data loading, splitting, normalization, and PyTorch dataset helpers."""

from __future__ import annotations

import json
import logging
import os

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from pipeline.config import (
    AUDIO_INPUT_DIM,
    BATCH_SIZE,
    FEAT_NAMES_PATH,
    RANDOM_SEED,
    SCALER_PATH,
    TEST_SPLIT,
    VAL_SPLIT,
    VIDEO_INPUT_DIM,
)

logger = logging.getLogger(__name__)


class DeceptionDataset(Dataset):
    """Wrap a `(N, 92)` feature matrix into video/audio tensors + labels."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        expected_dim = VIDEO_INPUT_DIM + AUDIO_INPUT_DIM
        if X.ndim != 2 or X.shape[1] != expected_dim:
            raise ValueError(f"Expected X to have shape (N, {expected_dim}), got {X.shape}")

        self.video = torch.tensor(X[:, :VIDEO_INPUT_DIM], dtype=torch.float32)
        self.audio = torch.tensor(X[:, VIDEO_INPUT_DIM:], dtype=torch.float32)
        self.labels = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.video[idx], self.audio[idx], self.labels[idx]


def load_features(feat_csv: str) -> tuple[np.ndarray, np.ndarray]:
    """Load `features.csv` and return `(X, y)` numpy arrays."""
    df = pd.read_csv(feat_csv)
    y = df["label"].to_numpy(dtype=int)
    X = df.drop(columns=["video_id", "label"]).to_numpy(dtype=np.float32)

    expected_dim = VIDEO_INPUT_DIM + AUDIO_INPUT_DIM
    if X.ndim != 2 or X.shape[1] != expected_dim:
        raise ValueError(
            f"{feat_csv} must contain {expected_dim} feature columns, found {X.shape}"
        )
    return X, y


def load_named_features(
    feat_csv: str,
    feat_names_path: str = FEAT_NAMES_PATH,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load features together with the ordered feature names."""
    X, y = load_features(feat_csv)
    with open(feat_names_path, "r", encoding="utf-8") as handle:
        names = json.load(handle)
    return X, y, names


def split_and_normalize(
    X: np.ndarray,
    y: np.ndarray,
    fit_scaler: bool = True,
    scaler: StandardScaler | None = None,
    scaler_save_path: str | None = None,
) -> dict:
    """
    Stratified 70/15/15 split and StandardScaler normalization.

    The scaler is fit on the training split only and saved when requested.
    """
    test_frac = TEST_SPLIT
    val_frac_of_remaining = VAL_SPLIT / (1.0 - TEST_SPLIT)

    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        test_size=test_frac,
        stratify=y,
        random_state=RANDOM_SEED,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=val_frac_of_remaining,
        stratify=y_temp,
        random_state=RANDOM_SEED,
    )

    if fit_scaler:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        save_path = scaler_save_path or SCALER_PATH
        joblib.dump(scaler, save_path)
        logger.info("Scaler saved to %s", save_path)
        print(f"Saved scaler to {save_path}")
    else:
        if scaler is None:
            raise ValueError("Must provide a fitted scaler when fit_scaler=False")
        X_train = scaler.transform(X_train)

    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    for array in (X_train, X_val, X_test):
        np.nan_to_num(array, copy=False, nan=0.0)

    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "scaler": scaler,
    }


def make_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = BATCH_SIZE,
    shuffle: bool = True,
    balance: bool = False,
) -> DataLoader:
    """Build a DataLoader from numpy arrays."""
    dataset = DeceptionDataset(X, y)

    sampler = None
    if balance:
        class_counts = np.bincount(y)
        sample_weights = 1.0 / class_counts[y]
        sampler = WeightedRandomSampler(
            weights=sample_weights.tolist(),
            num_samples=len(y),
            replacement=True,
        )
        shuffle = False

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        drop_last=False,
    )


def prepare_cross_dataset(
    source_csv: str,
    target_csv: str,
    few_shot_frac: float = 0.0,
    source_scaler_path: str | None = None,
) -> dict:
    """
    Normalize source and target datasets using the saved source training scaler.

    Official cross-dataset evaluation must reuse the persisted source scaler.
    """
    X_source, y_source = load_features(source_csv)
    X_target, y_target = load_features(target_csv)

    scaler_path = source_scaler_path or SCALER_PATH
    if not os.path.isfile(scaler_path):
        raise FileNotFoundError(
            f"Required source scaler not found at {scaler_path}. "
            "Official cross-dataset evaluation must reuse the saved training scaler."
        )

    scaler = joblib.load(scaler_path)
    logger.info("Loaded scaler from %s", scaler_path)

    X_source = scaler.transform(X_source)
    np.nan_to_num(X_source, copy=False, nan=0.0)

    result = {
        "X_source": X_source,
        "y_source": y_source,
        "scaler": scaler,
    }

    if 0.0 < few_shot_frac < 1.0:
        X_few, X_test, y_few, y_test = train_test_split(
            X_target,
            y_target,
            test_size=1.0 - few_shot_frac,
            stratify=y_target,
            random_state=RANDOM_SEED,
        )
        X_few = scaler.transform(X_few)
        X_test = scaler.transform(X_test)
        np.nan_to_num(X_few, copy=False, nan=0.0)
        np.nan_to_num(X_test, copy=False, nan=0.0)
        result.update(
            {
                "X_few": X_few,
                "y_few": y_few,
                "X_target": X_test,
                "y_target": y_test,
            }
        )
    else:
        X_target = scaler.transform(X_target)
        np.nan_to_num(X_target, copy=False, nan=0.0)
        result.update({"X_target": X_target, "y_target": y_target})

    return result

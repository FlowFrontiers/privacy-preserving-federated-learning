"""
Data handling utilities for federated learning experiments.
"""
import os
import logging
from typing import Tuple, List, Dict, Optional, Union, Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

logger = logging.getLogger(__name__)

class FederatedDataset(Dataset):
    """
    A dataset class that supports feature selection/suppression for federated learning.
    Handles both standard datasets (all features) and suppressed datasets (subset of features, zero-padded).
    """
    def __init__(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        feature_indices: Optional[List[int]] = None, 
        total_features: Optional[int] = None
    ):
        """
        Initialize dataset with optional feature selection.
        
        Args:
            X: Input features
            y: Labels
            feature_indices: Indices of features to use (None = use all)
            total_features: Total number of features in the full feature space
        """
        if feature_indices is None:
            self.X = torch.tensor(X, dtype=torch.float32)
        else:
            X_selected = np.zeros((len(X), total_features), dtype=np.float32)
            for i, idx in enumerate(feature_indices):
                X_selected[:, idx] = X[:, i]
            
            self.X = torch.tensor(X_selected, dtype=torch.float32)
        
        self.y = torch.tensor(y, dtype=torch.long)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_and_preprocess_data(
    file_path: str,
    feature_columns: List[str],
    target_column: str,
    scaler: Optional[StandardScaler] = None,
    label_encoder: Optional[LabelEncoder] = None
) -> Tuple[np.ndarray, np.ndarray, StandardScaler, LabelEncoder]:
    """
    Load and preprocess a dataset.
    
    Args:
        file_path: Path to the parquet file
        feature_columns: List of feature column names
        target_column: Name of the target column
        scaler: Optional pre-fitted scaler (for consistent scaling)
        label_encoder: Optional pre-fitted label encoder (for consistent labels)
        
    Returns:
        X: Preprocessed features
        y: Encoded labels
        scaler: Fitted standard scaler
        label_encoder: Fitted label encoder
    """
    # Load data
    df = pd.read_parquet(file_path, columns=feature_columns + [target_column])
    df = df.dropna()
    
    # Extract features and labels
    X = df[feature_columns].values.astype(np.float32)
    y_raw = df[target_column].values
    
    # Scale features
    if scaler is None:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    else:
        X = scaler.transform(X)
    
    # Encode labels
    if label_encoder is None:
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y_raw).astype(np.int32)
    else:
        y = label_encoder.transform(y_raw).astype(np.int32)
    
    return X, y, scaler, label_encoder


def prepare_federated_data(
    data_path: str,
    feature_columns: List[str],
    target_column: str,
    p1_path: Optional[str] = None,
    p2_path: Optional[str] = None,
    seed: int = 42,
    initial_split_ratio: float = 0.5,
    test_split_ratio: float = 0.1667,
    store_splits: bool = True,
    datasets_dir: str = "datasets"
) -> Dict[str, Any]:
    """
    Prepare data for federated learning experiments.
    
    This function can either:
    1. Load a single dataset and split it into P1/P2 parts
    2. Load pre-split P1/P2 datasets
    
    Args:
        data_path: Path to the full dataset
        feature_columns: List of feature column names
        target_column: Name of the target column
        p1_path: Optional path to Player 1's dataset
        p2_path: Optional path to Player 2's dataset
        seed: Random seed for reproducibility
        initial_split_ratio: Ratio for splitting into P1/P2
        test_split_ratio: Ratio for splitting each part into train/test
        store_splits: Whether to store the train/test splits
        datasets_dir: Directory to store generated datasets
        
    Returns:
        Dictionary containing train/test data for both parties
    """
    # Default paths if not provided but we want to check for pre-split data
    if p1_path is None:
        p1_path = os.path.join(datasets_dir, "p1.parquet")
    if p2_path is None:
        p2_path = os.path.join(datasets_dir, "p2.parquet")
    
    # Check if pre-split datasets exist
    if os.path.exists(p1_path) and os.path.exists(p2_path):
        logger.info(f"Loading pre-split datasets from {p1_path} and {p2_path}")
        # Load pre-split datasets
        X_p1, y_p1, scaler, label_encoder = load_and_preprocess_data(
            p1_path, feature_columns, target_column
        )
        X_p2, y_p2, _, _ = load_and_preprocess_data(
            p2_path, feature_columns, target_column, scaler, label_encoder
        )
        
        # Split each part into train/test
        X_p1_train, X_p1_test, y_p1_train, y_p1_test = train_test_split(
            X_p1, y_p1, test_size=test_split_ratio, stratify=y_p1, random_state=seed
        )
        
        X_p2_train, X_p2_test, y_p2_train, y_p2_test = train_test_split(
            X_p2, y_p2, test_size=test_split_ratio, stratify=y_p2, random_state=seed
        )
    elif os.path.exists(data_path):
        logger.info(f"Pre-split datasets not found. Loading and splitting dataset from {data_path}")
        # Load and process the full dataset
        X, y, scaler, label_encoder = load_and_preprocess_data(
            data_path, feature_columns, target_column
        )
        
        # First split: P1 and P2
        X_p1, X_p2, y_p1, y_p2 = train_test_split(
            X, y, test_size=initial_split_ratio, stratify=y, random_state=seed
        )
        
        # Second split: Train and Test for each part
        X_p1_train, X_p1_test, y_p1_train, y_p1_test = train_test_split(
            X_p1, y_p1, test_size=test_split_ratio, stratify=y_p1, random_state=seed
        )
        
        X_p2_train, X_p2_test, y_p2_train, y_p2_test = train_test_split(
            X_p2, y_p2, test_size=test_split_ratio, stratify=y_p2, random_state=seed
        )
        
        # Optionally store the splits for future use
        if store_splits:
            os.makedirs(datasets_dir, exist_ok=True)
            
            # Create DataFrames for storage
            df_p1 = pd.DataFrame(
                scaler.inverse_transform(X_p1), 
                columns=feature_columns
            )
            df_p1[target_column] = label_encoder.inverse_transform(y_p1)
            
            df_p2 = pd.DataFrame(
                scaler.inverse_transform(X_p2), 
                columns=feature_columns
            )
            df_p2[target_column] = label_encoder.inverse_transform(y_p2)
            
            # Store DataFrames to default locations
            default_p1_path = os.path.join(datasets_dir, "p1.parquet")
            default_p2_path = os.path.join(datasets_dir, "p2.parquet")
            
            df_p1.to_parquet(default_p1_path, index=False)
            df_p2.to_parquet(default_p2_path, index=False)
            
            logger.info(f"Stored split datasets to {default_p1_path} and {default_p2_path}")
    else:
        raise FileNotFoundError(f"Neither {data_path} nor both {p1_path} and {p2_path} exist")
    
    # Calculate number of classes
    # When using pre-split data, we need to determine number of classes from y_p1 and y_p2
    if 'y' not in locals():
        # Combine labels from both players to get all unique classes
        all_labels = np.concatenate([y_p1_train, y_p1_test, y_p2_train, y_p2_test])
        num_classes = len(np.unique(all_labels))
    else:
        # If we loaded the full dataset, we already have 'y'
        num_classes = len(np.unique(y))
    
    # Create object with the prepared data
    data = {
        'X_p1_train': X_p1_train, 
        'X_p1_test': X_p1_test, 
        'y_p1_train': y_p1_train, 
        'y_p1_test': y_p1_test,
        'X_p2_train': X_p2_train, 
        'X_p2_test': X_p2_test, 
        'y_p2_train': y_p2_train, 
        'y_p2_test': y_p2_test,
        'scaler': scaler,
        'label_encoder': label_encoder,
        'num_features': X_p1_train.shape[1],
        'num_classes': num_classes
    }
    
    return data


def create_dataloaders(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    batch_size: int = 256,
    feature_indices: Optional[List[int]] = None,
    total_features: Optional[int] = None,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and test data loaders.
    
    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels
        batch_size: Batch size
        feature_indices: Optional indices of features to use (for suppression)
        total_features: Optional total number of features (for suppression)
        seed: Random seed
        
    Returns:
        train_loader, test_loader
    """
    generator = torch.Generator().manual_seed(seed)
    
    if feature_indices is not None and total_features is not None:
        X_train_padded = np.zeros((len(X_train), total_features), dtype=np.float32)
        X_test_padded = np.zeros((len(X_test), total_features), dtype=np.float32)
        
        for i, idx in enumerate(feature_indices):
            X_train_padded[:, idx] = X_train[:, i]
            X_test_padded[:, idx] = X_test[:, i]
        
        train_dataset = TensorDataset(
            torch.tensor(X_train_padded, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long)
        )
        test_dataset = TensorDataset(
            torch.tensor(X_test_padded, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.long)
        )
    else:
        train_dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long)
        )
        test_dataset = TensorDataset(
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.long)
        )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_loader, test_loader


def get_feature_indices(feature_columns: List[str], selected_features: List[str]) -> List[int]:
    """
    Get the indices of selected features.
    
    Args:
        feature_columns: List of all feature column names
        selected_features: List of selected feature column names
        
    Returns:
        List of corresponding feature indices
    """
    return [feature_columns.index(feature) for feature in selected_features]
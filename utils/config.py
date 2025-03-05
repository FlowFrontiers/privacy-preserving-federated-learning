"""
Configuration module for federated learning experiments.
"""
import os
import argparse
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Union
import yaml
import json

# Default paths
DEFAULT_CONFIG_PATH = "config.yaml"
DEFAULT_RESULTS_DIR = "results"
DEFAULT_DATASETS_DIR = "datasets"
DEFAULT_LOGS_DIR = "logs"

@dataclass
class ExperimentConfig:
    """Configuration for federated learning experiments."""
    # Experiment identification
    experiment_name: str = "fl_experiment"
    experiment_type: str = "local"  # "local", "federated", "suppression", "differential_privacy"
    
    # Data parameters
    data_path: str = os.path.join(DEFAULT_DATASETS_DIR, "dataset.parquet")
    p1_path: Optional[str] = None  # Path to Player 1's dataset (if pre-split)
    p2_path: Optional[str] = None  # Path to Player 2's dataset (if pre-split)
    feature_columns: List[str] = field(default_factory=lambda: [
        'protocol',
        'bidirectional_min_ps',
        'bidirectional_mean_ps',
        'bidirectional_stddev_ps',
        'bidirectional_max_ps',
        'src2dst_stddev_ps',
        'src2dst_max_ps',
        'dst2src_min_ps',
        'dst2src_mean_ps',
        'dst2src_stddev_ps',
        'dst2src_max_ps',
        'bidirectional_stddev_piat_ms',
        'bidirectional_max_piat_ms',
        'bidirectional_rst_packets'
    ])
    target_column: str = 'application_name'
    
    # Dataset splitting parameters
    seed: int = 42
    initial_split_ratio: float = 0.5  # First split into P1/P2
    test_split_ratio: float = 0.1667  # Test split ratio for each participant

    # Training parameters
    batch_size: int = 256
    learning_rate: float = 0.001
    
    # Local training parameters
    local_epochs: int = 100
    
    # Federated learning parameters
    federated_rounds: int = 10
    client_epochs: int = 10  # Epochs per federated round
    
    # Privacy parameters
    # For suppression
    p1_features: Optional[List[str]] = None
    p2_features: Optional[List[str]] = None
    max_parallel_suppression_processes: int = 2  # Limit parallel suppression processes to avoid memory issues
    
    # For differential privacy
    noise_multiplier_p1: Optional[float] = None
    noise_multiplier_p2: Optional[float] = None
    noise_multiplier_p1_explicitly_set: bool = False  # Flag to indicate user explicitly set this, even if None
    noise_multiplier_p2_explicitly_set: bool = False  # Flag to indicate user explicitly set this, even if None
    max_grad_norm: float = 1.0
    max_epsilon: float = 10.0  # Maximum allowed privacy budget (epsilon)
    max_parallel_dp_processes: int = 2  # Limit parallel DP processes to avoid memory issues
    
    # Result paths
    results_dir: str = DEFAULT_RESULTS_DIR
    logs_dir: str = DEFAULT_LOGS_DIR
    save_models: bool = False
    checkpoint_interval: int = 1  # Save results every N rounds/epochs
    
    def __post_init__(self):
        """Create directories and validate configuration."""
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(DEFAULT_DATASETS_DIR, exist_ok=True)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        # Create the base dictionary
        config_dict = {}
        for k, v in self.__dict__.items():
            # Special handling for noise multiplier values to ensure they remain as floats or None
            if k in ['noise_multiplier_p1', 'noise_multiplier_p2'] and v is not None:
                config_dict[k] = float(v)
            else:
                config_dict[k] = v
        return config_dict
    
    def save(self, path: str) -> None:
        """Save configuration to file."""
        with open(path, 'w') as f:
            if path.endswith('.json'):
                json.dump(self.to_dict(), f, indent=4)
            else:
                yaml.dump(self.to_dict(), f, default_flow_style=False)
    
    @classmethod
    def from_file(cls, path: str) -> 'ExperimentConfig':
        """Load configuration from file."""
        with open(path, 'r') as f:
            if path.endswith('.json'):
                config_dict = json.load(f)
            else:
                config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    @classmethod
    def from_args(cls) -> 'ExperimentConfig':
        """Parse command line arguments into ExperimentConfig."""
        parser = argparse.ArgumentParser(description="Federated Learning Experiment Framework")
        parser.add_argument("--config", type=str, default=None, help="Path to configuration file")
        parser.add_argument("--experiment_name", type=str, help="Name of the experiment")
        parser.add_argument("--experiment_type", type=str, 
                            choices=["local", "federated", "suppression", "differential_privacy"],
                            help="Type of experiment to run")
        parser.add_argument("--data_path", type=str, help="Path to dataset file")
        parser.add_argument("--p1_path", type=str, help="Path to Player 1's dataset")
        parser.add_argument("--p2_path", type=str, help="Path to Player 2's dataset")
        parser.add_argument("--seed", type=int, help="Random seed")
        parser.add_argument("--local_epochs", type=int, help="Number of epochs for local training")
        parser.add_argument("--federated_rounds", type=int, help="Number of federated learning rounds")
        parser.add_argument("--client_epochs", type=int, help="Number of epochs per client per federated round")
        parser.add_argument("--noise_p1", type=str, help="Noise multiplier for Player 1 (use 'None' for no noise)")
        parser.add_argument("--noise_p2", type=str, help="Noise multiplier for Player 2 (use 'None' for no noise)")
        parser.add_argument("--results_dir", type=str, help="Directory to store results")
        
        args, _ = parser.parse_known_args()
        
        # Start with default config
        config = cls()
        
        # Update from config file if provided
        if args.config:
            file_config = cls.from_file(args.config)
            config = file_config
        
        # Update with command line arguments
        for key, value in vars(args).items():
            if value is not None and key != "config":
                if key == "noise_p1":
                    if value.lower() == "none":
                        config.noise_multiplier_p1 = None
                    else:
                        config.noise_multiplier_p1 = float(value)
                elif key == "noise_p2":
                    if value.lower() == "none":
                        config.noise_multiplier_p2 = None
                    else:
                        config.noise_multiplier_p2 = float(value)
                else:
                    setattr(config, key, value)
        
        return config


def parse_noise(value):
    """Parse noise parameter, converting 'None' to None."""
    if value is None or (isinstance(value, str) and value.lower() == "none"):
        return None
    try:
        return float(value)
    except ValueError:
        # Log an error message for debugging
        print(f"Error parsing noise value: '{value}' - not a valid float")
        return None


def get_experiment_filename(config: ExperimentConfig) -> str:
    """Generate a standardized filename for experiment results."""
    exp_type = config.experiment_type
    
    # Define subdirectories for each experiment type
    subdirs = {
        "local": "local_experiments",
        "federated": "federated_experiments",
        "suppression": "suppression_experiments",
        "differential_privacy": "dp_experiments"
    }
    
    # Get the appropriate subdirectory
    subdir = subdirs.get(exp_type, "")
    
    # Generate filenames based on experiment type
    if exp_type == "local":
        filename = f"local_baseline_s{config.seed}"
    elif exp_type == "federated":
        filename = f"federated_r{config.federated_rounds}_e{config.client_epochs}_s{config.seed}"
    elif exp_type == "suppression":
        p1_count = len(config.p1_features) if config.p1_features else len(config.feature_columns)
        p2_count = len(config.p2_features) if config.p2_features else len(config.feature_columns)
        filename = f"suppression_p1f{p1_count}_p2f{p2_count}_s{config.seed}"
    elif exp_type == "differential_privacy":
        p1_noise = config.noise_multiplier_p1 if config.noise_multiplier_p1 is not None else "None"
        p2_noise = config.noise_multiplier_p2 if config.noise_multiplier_p2 is not None else "None"
        filename = f"dp_p1n{p1_noise}_p2n{p2_noise}_s{config.seed}"
    else:
        filename = f"experiment_{config.experiment_name}"
    
    # For combined results (final results), we'll put them in the main results dir
    # For individual experiment results, put them in the appropriate subdirectory
    if "combined" in filename or "final" in filename:
        return filename
    else:
        return os.path.join(subdir, filename)
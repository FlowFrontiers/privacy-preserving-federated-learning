"""
Experiment runner for federated learning experiments.
"""
import os
import json
import time
import logging
import subprocess
from typing import Dict, List, Optional, Tuple, Any, Union
import multiprocessing
from datetime import datetime
from copy import deepcopy
from itertools import product
import traceback

import numpy as np
import torch
from torch.utils.data import DataLoader

from utils.config import ExperimentConfig, parse_noise, get_experiment_filename
from utils.data import prepare_federated_data, create_dataloaders, get_feature_indices
from utils.models import NeuralNetwork, create_model, train_model, evaluate_model, average_models

logger = logging.getLogger(__name__)

def setup_logger(config: ExperimentConfig) -> None:
    """
    Setup the logger for experiments.
    
    Args:
        config: Experiment configuration
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_name = f"{config.experiment_type}_{timestamp}"
    
    # Define subdirectories for each experiment type
    subdirs = {
        "local": "local_experiments",
        "federated": "federated_experiments",
        "suppression": "suppression_experiments",
        "differential_privacy": "dp_experiments"
    }
    
    # Get the appropriate subdirectory
    subdir = subdirs.get(config.experiment_type, "")
    
    # Create the subdirectory if it doesn't exist
    log_subdir = os.path.join(config.logs_dir, subdir)
    os.makedirs(log_subdir, exist_ok=True)
    
    # Set the log file path
    log_file = os.path.join(log_subdir, f"{experiment_name}.log")
    
    # Reset existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def save_results(results: Dict[str, Any], config: ExperimentConfig, filename: Optional[str] = None, is_intermediate: bool = False) -> str:
    """
    Save experiment results to a file.
    
    Args:
        results: Results to save
        config: Experiment configuration
        filename: Optional filename override
        is_intermediate: Whether this is an intermediate result file or per-case result
        
    Returns:
        Path to the saved results file
    """
    def convert_to_serializable(obj):
        """Convert numpy types to Python types for JSON serialization."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return obj
    
    # Define subdirectories for experiments with intermediate or per-case results
    subdirs = {
        "federated": "federated_experiments", 
        "suppression": "suppression_experiments",
        "differential_privacy": "dp_experiments"
    }
    
    # Generate filename if not provided
    if filename is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_filename = get_experiment_filename(config)
        filename = f"{base_filename}_{timestamp}.json"
    
    # Determine the appropriate directory path based on result type
    if is_intermediate and config.experiment_type in subdirs:
        # Intermediate results or per-case results go in experiment type subdirectory
        subdir = subdirs.get(config.experiment_type, "")
        dir_path = os.path.join(config.results_dir, subdir)
    else:
        # Final results and local results (which have no intermediate results)
        # go directly in the main results directory
        dir_path = config.results_dir
    
    # Create full file path
    file_path = os.path.join(dir_path, os.path.basename(filename))
    
    # Make sure the directory exists
    os.makedirs(dir_path, exist_ok=True)
    
    # Combine results with metadata
    results_with_meta = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'config': config.to_dict(),
        'results': results
    }
    
    # Convert to serializable format
    serializable_results = json.loads(
        json.dumps(results_with_meta, default=convert_to_serializable)
    )
    
    # Save to file
    with open(file_path, 'w') as f:
        json.dump(serializable_results, f, indent=4)
    
    # Log appropriately based on result type
    if is_intermediate:
        logger.info(f"Intermediate results saved to {file_path}")
    else:
        logger.info(f"Final results saved to {file_path}")
    
    return file_path


def load_results(file_path: str) -> Dict[str, Any]:
    """
    Load experiment results from a file.
    
    Args:
        file_path: Path to the results file
        
    Returns:
        Loaded results
    """
    with open(file_path, 'r') as f:
        return json.load(f)


def run_local_baseline(config: ExperimentConfig) -> Dict[str, Any]:
    """
    Run local baseline experiment.
    
    This experiment trains two separate models, one for each player,
    and evaluates them on both players' test sets.
    
    Args:
        config: Experiment configuration
        
    Returns:
        Dictionary with experiment results
    """
    setup_logger(config)
    start_time = time.time()
    
    logger.info("Starting local baseline experiment")
    logger.info(f"Configuration: {json.dumps(config.to_dict(), indent=2)}")
    
    # Set random seeds
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    
    # Prepare data
    data = prepare_federated_data(
        data_path=config.data_path,
        feature_columns=config.feature_columns,
        target_column=config.target_column,
        p1_path=config.p1_path,
        p2_path=config.p2_path,
        seed=config.seed,
        initial_split_ratio=config.initial_split_ratio,
        test_split_ratio=config.test_split_ratio
    )
    
    # Create dataloaders
    train_loader_p1, test_loader_p1 = create_dataloaders(
        data['X_p1_train'], data['X_p1_test'], 
        data['y_p1_train'], data['y_p1_test'],
        batch_size=config.batch_size,
        seed=config.seed
    )
    
    train_loader_p2, test_loader_p2 = create_dataloaders(
        data['X_p2_train'], data['X_p2_test'], 
        data['y_p2_train'], data['y_p2_test'],
        batch_size=config.batch_size,
        seed=config.seed
    )
    
    # Initialize models
    model_p1 = create_model(data['num_features'], data['num_classes'], config.seed)
    model_p2 = create_model(data['num_features'], data['num_classes'], config.seed)
    
    # Train model for P1
    logger.info("Training Model M1 (P1)...")
    training_start_time_p1 = time.time()
    model_p1, history_p1 = train_model(
        model_p1, 
        train_loader_p1, 
        test_loader_p1, 
        config.local_epochs, 
        config.learning_rate,
        client_id="P1",
        is_dp_experiment=False  # Not a DP experiment
    )
    training_time_p1 = time.time() - training_start_time_p1
    
    # Train model for P2
    logger.info("Training Model M2 (P2)...")
    training_start_time_p2 = time.time()
    model_p2, history_p2 = train_model(
        model_p2, 
        train_loader_p2, 
        test_loader_p2, 
        config.local_epochs, 
        config.learning_rate,
        client_id="P2",
        is_dp_experiment=False  # Not a DP experiment
    )
    training_time_p2 = time.time() - training_start_time_p2
    
    # Evaluate models
    logger.info("Evaluating both models...")
    # M1 on P1's test set
    loss_m1_p1, acc_m1_p1 = evaluate_model(model_p1, test_loader_p1)
    # M1 on P2's test set
    loss_m1_p2, acc_m1_p2 = evaluate_model(model_p1, test_loader_p2)
    # M2 on P1's test set
    loss_m2_p1, acc_m2_p1 = evaluate_model(model_p2, test_loader_p1)
    # M2 on P2's test set
    loss_m2_p2, acc_m2_p2 = evaluate_model(model_p2, test_loader_p2)
    
    # Log evaluation results in a consistent format
    logger.info(f"Final Results after training:")
    logger.info(f"Model M1 (P1) - P1 Test: Loss = {loss_m1_p1:.4f}, Accuracy = {acc_m1_p1:.4f}")
    logger.info(f"Model M1 (P1) - P2 Test: Loss = {loss_m1_p2:.4f}, Accuracy = {acc_m1_p2:.4f}")
    logger.info(f"Model M2 (P2) - P1 Test: Loss = {loss_m2_p1:.4f}, Accuracy = {acc_m2_p1:.4f}")
    logger.info(f"Model M2 (P2) - P2 Test: Loss = {loss_m2_p2:.4f}, Accuracy = {acc_m2_p2:.4f}")
    
    # Collect results
    total_time = time.time() - start_time
    
    results = {
        'models': {
            'M1': {
                'training': history_p1,
                'evaluation': {
                    'P1': {'loss': loss_m1_p1, 'accuracy': acc_m1_p1},
                    'P2': {'loss': loss_m1_p2, 'accuracy': acc_m1_p2}
                },
                'training_time': training_time_p1
            },
            'M2': {
                'training': history_p2,
                'evaluation': {
                    'P1': {'loss': loss_m2_p1, 'accuracy': acc_m2_p1},
                    'P2': {'loss': loss_m2_p2, 'accuracy': acc_m2_p2}
                },
                'training_time': training_time_p2
            }
        },
        'execution_time': {
            'total': total_time,
            'average_training_time': (training_time_p1 + training_time_p2) / 2
        }
    }
    
    # The final results have already been logged during evaluation
    
    # Save results - local results are not final combined results
    save_results(results, config)
    
    # Log completion
    logger.info(f"Local baseline experiment completed in {total_time:.2f}s")
    return results


def run_federated_learning(config: ExperimentConfig) -> Dict[str, Any]:
    """
    Run federated learning experiment.
    
    This experiment implements standard federated learning (FedAvg)
    without any privacy mechanisms.
    
    Args:
        config: Experiment configuration
        
    Returns:
        Dictionary with experiment results
    """
    setup_logger(config)
    start_time = time.time()
    
    logger.info("Starting federated learning experiment")
    logger.info(f"Configuration: {json.dumps(config.to_dict(), indent=2)}")
    
    # Set random seeds
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    
    # Prepare data
    data = prepare_federated_data(
        data_path=config.data_path,
        feature_columns=config.feature_columns,
        target_column=config.target_column,
        p1_path=config.p1_path,
        p2_path=config.p2_path,
        seed=config.seed,
        initial_split_ratio=config.initial_split_ratio,
        test_split_ratio=config.test_split_ratio
    )
    
    # Create dataloaders
    train_loader_p1, test_loader_p1 = create_dataloaders(
        data['X_p1_train'], data['X_p1_test'], 
        data['y_p1_train'], data['y_p1_test'],
        batch_size=config.batch_size,
        seed=config.seed
    )
    
    train_loader_p2, test_loader_p2 = create_dataloaders(
        data['X_p2_train'], data['X_p2_test'], 
        data['y_p2_train'], data['y_p2_test'],
        batch_size=config.batch_size,
        seed=config.seed
    )
    
    # Initialize global model
    global_model = create_model(data['num_features'], data['num_classes'], config.seed)
    
    # Initialize results storage
    round_results = []
    
    # Federated learning rounds
    for round_num in range(1, config.federated_rounds + 1):
        logger.info(f"Starting Federated Learning Round {round_num}/{config.federated_rounds}")
        round_start_time = time.time()
        
        # Initialize client models with global model parameters
        model_p1 = deepcopy(global_model)
        model_p2 = deepcopy(global_model)
        
        # Train client models
        model_p1, history_p1 = train_model(
            model_p1, 
            train_loader_p1, 
            test_loader_p1, 
            config.client_epochs, 
            config.learning_rate,
            client_id=f"P1 (Round {round_num})",
            is_dp_experiment=False  # Not a DP experiment
        )
        
        model_p2, history_p2 = train_model(
            model_p2, 
            train_loader_p2, 
            test_loader_p2, 
            config.client_epochs, 
            config.learning_rate,
            client_id=f"P2 (Round {round_num})",
            is_dp_experiment=False  # Not a DP experiment
        )
        
        # Aggregate models
        global_model = average_models([model_p1, model_p2])
        
        # Evaluate global model
        loss_p1, acc_p1 = evaluate_model(global_model, test_loader_p1)
        loss_p2, acc_p2 = evaluate_model(global_model, test_loader_p2)
        
        # Calculate round time
        round_time = time.time() - round_start_time
        
        # Store round results
        round_result = {
            'round': round_num,
            'time': round_time,
            'client_training': {
                'P1': history_p1,
                'P2': history_p2
            },
            'global_evaluation': {
                'P1': {'loss': loss_p1, 'accuracy': acc_p1},
                'P2': {'loss': loss_p2, 'accuracy': acc_p2}
            }
        }
        round_results.append(round_result)
        
        # Log results in a unified format
        logger.info(f"Round {round_num}/{config.federated_rounds} completed in {round_time:.2f}s")
        logger.info(f"Global Model - P1 Test: Loss = {loss_p1:.4f}, Accuracy = {acc_p1:.4f}")
        logger.info(f"Global Model - P2 Test: Loss = {loss_p2:.4f}, Accuracy = {acc_p2:.4f}")
        
        # Save intermediate results per round if needed
        if config.checkpoint_interval > 0 and round_num % config.checkpoint_interval == 0:
            intermediate_results = {
                'rounds': round_results[:round_num],
                'current_round': round_num,
                'execution_time_so_far': time.time() - start_time
            }
            # Save intermediate results 
            intermediate_filename = f"federated_intermediate_r{round_num}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            save_results(intermediate_results, config, intermediate_filename, is_intermediate=True)
    
    # Final evaluation - repeat the last round evaluation to be explicit
    logger.info("Final Global Model Evaluation:")
    logger.info(f"Global Model - P1 Test: Loss = {loss_p1:.4f}, Accuracy = {acc_p1:.4f}")
    logger.info(f"Global Model - P2 Test: Loss = {loss_p2:.4f}, Accuracy = {acc_p2:.4f}")
    
    # Calculate total time
    total_time = time.time() - start_time
    
    # Collect results
    results = {
        'rounds': round_results,
        'final_evaluation': {
            'P1': {'loss': loss_p1, 'accuracy': acc_p1},
            'P2': {'loss': loss_p2, 'accuracy': acc_p2}
        },
        'execution_time': total_time
    }
    
    # Save final federated results
    final_filename = f"federated_r{config.federated_rounds}_e{config.client_epochs}_s{config.seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    save_results(results, config, final_filename)
    
    # Log completion 
    logger.info(f"Federated learning experiment completed in {total_time:.2f}s")
    
    return results


def generate_feature_combinations(feature_columns: List[str]) -> List[List[str]]:
    """
    Generate all possible feature combinations for suppression experiments.
    
    Args:
        feature_columns: List of feature column names
        
    Returns:
        List of feature combinations
    """
    # Generate combinations from all features down to 1 feature
    return [feature_columns[:i] for i in range(len(feature_columns), 0, -1)]


def run_suppression_experiment(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run a single suppression experiment with specific feature sets.
    
    This function is designed to be called in a parallel process.
    
    Args:
        config_dict: Dictionary containing:
            - config: Experiment configuration
            - p1_features: Features to use for Player 1
            - p2_features: Features to use for Player 2
            - experiment_id: Unique identifier for this experiment
            - prepared_data: Pre-prepared data dictionary (optional)
            - logger: Logger instance to use (optional)
        
    Returns:
        Dictionary with experiment results
    """
    # Extract parameters from config dictionary
    config = ExperimentConfig(**config_dict.get('config'))
    p1_features = config_dict.get('p1_features')
    p2_features = config_dict.get('p2_features')
    experiment_id = config_dict.get('experiment_id')
    
    # Use provided logger or set up a new one
    logger = config_dict.get('logger')
    if logger is None:
        setup_logger(config)
        logger = logging.getLogger(__name__)
    
    # Only log the header if we're not supposed to skip it
    skip_header = config_dict.get('skip_header', False)
    if not skip_header:
        logger.info(f"Starting Suppression Experiment {experiment_id}")
        logger.info(f"P1 Features ({len(p1_features)}/{len(config.feature_columns)}): {p1_features}")
        logger.info(f"P2 Features ({len(p2_features)}/{len(config.feature_columns)}): {p2_features}")
    
    # Check if data is provided in the config dictionary
    prepared_data = config_dict.get('prepared_data')
    
    if prepared_data is None:
        # Need to load and prepare data from scratch
        logger.info("Loading and preparing data...")
        data = prepare_federated_data(
            data_path=config.data_path,
            feature_columns=config.feature_columns,
            target_column=config.target_column,
            p1_path=config.p1_path,
            p2_path=config.p2_path,
            seed=config.seed,
            initial_split_ratio=config.initial_split_ratio,
            test_split_ratio=config.test_split_ratio
        )
    else:
        # Use provided data
        data = prepared_data
    
    # Get feature indices - ensure proper handling if indices length doesn't match
    p1_feature_indices = get_feature_indices(config.feature_columns, p1_features)
    p2_feature_indices = get_feature_indices(config.feature_columns, p2_features)
    
    # Validation to ensure indices are valid
    for idx in p1_feature_indices:
        if idx >= data['X_p1_train'].shape[1]:
            raise ValueError(f"Feature index {idx} out of bounds for P1 data with {data['X_p1_train'].shape[1]} features")
    
    for idx in p2_feature_indices:
        if idx >= data['X_p2_train'].shape[1]:
            raise ValueError(f"Feature index {idx} out of bounds for P2 data with {data['X_p2_train'].shape[1]} features")
            
    # Create suppressed datasets
    X_p1_train_sup = data['X_p1_train'][:, p1_feature_indices]
    X_p1_test_sup = data['X_p1_test'][:, p1_feature_indices]
    X_p2_train_sup = data['X_p2_train'][:, p2_feature_indices]
    X_p2_test_sup = data['X_p2_test'][:, p2_feature_indices]
    
    total_features = len(config.feature_columns)
    
    # Create dataloaders with proper feature mapping
    train_loader_p1, test_loader_p1 = create_dataloaders(
        X_p1_train_sup, X_p1_test_sup, 
        data['y_p1_train'], data['y_p1_test'],
        batch_size=config.batch_size,
        feature_indices=p1_feature_indices,
        total_features=total_features,
        seed=config.seed
    )
    
    train_loader_p2, test_loader_p2 = create_dataloaders(
        X_p2_train_sup, X_p2_test_sup, 
        data['y_p2_train'], data['y_p2_test'],
        batch_size=config.batch_size,
        feature_indices=p2_feature_indices,
        total_features=total_features,
        seed=config.seed
    )
    
    # Initialize global model
    global_model = create_model(total_features, data['num_classes'], config.seed)
    
    experiment_results = {
        'experiment_id': experiment_id,
        'p1_features': p1_features,
        'p2_features': p2_features,
        'rounds': [],
        'final_evaluation': {}
    }
    
    # Federated learning rounds
    for round_num in range(1, config.federated_rounds + 1):
        round_start_time = time.time()
        
        # Initialize local models
        model_p1 = deepcopy(global_model)
        model_p2 = deepcopy(global_model)
        
        # Train local models
        model_p1, history_p1 = train_model(
            model_p1, 
            train_loader_p1, 
            test_loader_p1, 
            config.client_epochs, 
            config.learning_rate,
            client_id=f"P1 (Round {round_num})",
            logger=logger
        )
        
        model_p2, history_p2 = train_model(
            model_p2, 
            train_loader_p2, 
            test_loader_p2, 
            config.client_epochs, 
            config.learning_rate,
            client_id=f"P2 (Round {round_num})",
            logger=logger
        )
        
        # Aggregate models
        global_model = average_models([model_p1, model_p2])
        
        # Evaluate global model
        loss_p1, acc_p1 = evaluate_model(global_model, test_loader_p1)
        loss_p2, acc_p2 = evaluate_model(global_model, test_loader_p2)
        
        round_time = time.time() - round_start_time
        
        # Store round results
        round_results = {
            'round': round_num,
            'time': round_time,
            'client_training': {
                'P1': history_p1,
                'P2': history_p2
            },
            'global_evaluation': {
                'P1': {'loss': loss_p1, 'accuracy': acc_p1},
                'P2': {'loss': loss_p2, 'accuracy': acc_p2}
            }
        }
        experiment_results['rounds'].append(round_results)
        
        logger.info(f"Round {round_num} completed in {round_time:.2f}s")
        logger.info(f"Global Model - P1 Test: Loss = {loss_p1:.4f}, Accuracy = {acc_p1:.4f}")
        logger.info(f"Global Model - P2 Test: Loss = {loss_p2:.4f}, Accuracy = {acc_p2:.4f}")
    
    # Final evaluation
    final_loss_p1, final_acc_p1 = evaluate_model(global_model, test_loader_p1)
    final_loss_p2, final_acc_p2 = evaluate_model(global_model, test_loader_p2)
    
    # Log final results in a consistent format
    logger.info(f"Final Results after {config.federated_rounds} rounds:")
    logger.info(f"Global Model - P1 Test: Loss = {final_loss_p1:.4f}, Accuracy = {final_acc_p1:.4f}")
    logger.info(f"Global Model - P2 Test: Loss = {final_loss_p2:.4f}, Accuracy = {final_acc_p2:.4f}")
    # Note: Suppression experiment doesn't have epsilon values like DP
    
    experiment_results['final_evaluation'] = {
        'P1': {'loss': final_loss_p1, 'accuracy': final_acc_p1},
        'P2': {'loss': final_loss_p2, 'accuracy': final_acc_p2}
    }
    
    return experiment_results


def run_suppression_subprocess(config_dict: Dict[str, Any]) -> None:
    """
    Run a suppression experiment in a subprocess.
    
    Args:
        config_dict: Experiment configuration and parameters dictionary
    """
    # Extract parameters for logging
    config = ExperimentConfig(**config_dict.get('config'))
    p1_features = config_dict.get('p1_features')
    p2_features = config_dict.get('p2_features')
    experiment_id = config_dict.get('experiment_id')
    
    # Set up experiment-specific logging in the suppression_experiments subdirectory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_subdir = os.path.join(config.logs_dir, "suppression_experiments")
    os.makedirs(log_subdir, exist_ok=True)
    log_filename = f"suppression_exp{experiment_id}_p1f{len(p1_features)}_p2f{len(p2_features)}_{timestamp}.log"
    log_path = os.path.join(log_subdir, log_filename)
    
    # Configure logging specifically for this subprocess
    subprocess_logger = logging.getLogger(f"experiment_{experiment_id}")
    subprocess_logger.handlers = []  # Remove any existing handlers
    subprocess_logger.setLevel(logging.INFO)
    
    # Add file handler
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    subprocess_logger.addHandler(file_handler)
    
    # Add a header to the log file with experiment details
    subprocess_logger.info(f"Starting Suppression Experiment {experiment_id}")
    subprocess_logger.info(f"P1 Features ({len(p1_features)}/{len(config.feature_columns)}): {p1_features}")
    subprocess_logger.info(f"P2 Features ({len(p2_features)}/{len(config.feature_columns)}): {p2_features}")
    
    # Override the logger in the config_dict to use in the experiment
    config_dict['logger'] = subprocess_logger
    
    # Remove any startup logs to avoid duplication
    if 'experiment_id' in config_dict:
        config_dict['skip_header'] = True
    
    try:
        # Run experiment
        result = run_suppression_experiment(config_dict)
        
        # Don't log again - the results are already shown before this point
        
        # Create filename for this experiment
        filename = f"exp_{experiment_id}_p1f{len(p1_features)}_p2f{len(p2_features)}.json"
        
        # Use save_results with is_intermediate flag to place in subdirectory
        file_path = save_results(result, config, filename, is_intermediate=True)
        
        subprocess_logger.info(f"Experiment {experiment_id} results saved to {file_path}")
    except Exception as e:
        subprocess_logger.error(f"Error in experiment {experiment_id}: {str(e)}")
        subprocess_logger.error(traceback.format_exc())


def run_suppression(config: ExperimentConfig) -> Dict[str, Any]:
    """
    Run feature suppression experiments.
    
    This experiment explores the impact of suppressing different feature
    combinations in a federated learning setting.
    
    Args:
        config: Experiment configuration
        
    Returns:
        Dictionary with experiment results
    """
    setup_logger(config)
    start_time = time.time()
    
    logger.info("Starting feature suppression experiments")
    logger.info(f"Configuration: {json.dumps(config.to_dict(), indent=2)}")
    
    # Set random seeds
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    
    # Prepare data once to share across processes
    logger.info("Preparing data (will be shared across experiments)...")
    data = prepare_federated_data(
        data_path=config.data_path,
        feature_columns=config.feature_columns,
        target_column=config.target_column,
        p1_path=config.p1_path,
        p2_path=config.p2_path,
        seed=config.seed,
        initial_split_ratio=config.initial_split_ratio,
        test_split_ratio=config.test_split_ratio
    )
    
    # Generate feature combinations
    if config.p1_features is not None and config.p2_features is not None:
        # Use predefined feature sets
        feature_combinations_p1 = [config.p1_features]
        feature_combinations_p2 = [config.p2_features]
    else:
        # Generate all combinations
        feature_combinations = generate_feature_combinations(config.feature_columns)
        feature_combinations_p1 = feature_combinations
        feature_combinations_p2 = feature_combinations
    
    # Calculate total experiments
    experiment_combinations = list(product(feature_combinations_p1, feature_combinations_p2))
    total_experiments = len(experiment_combinations)
    logger.info(f"Running {total_experiments} suppression experiments")
    
    # Create a pool of workers with safety limit for memory usage
    max_system_cores = multiprocessing.cpu_count()
    available_cores = min(max_system_cores, total_experiments)
    pool_size = min(available_cores, config.max_parallel_suppression_processes)
    logger.info(f"Using {pool_size} parallel processes (limited by max_parallel_suppression_processes={config.max_parallel_suppression_processes})")
    
    # Create directory for results
    results_dir = os.path.join(config.results_dir, "suppression_experiments")
    os.makedirs(results_dir, exist_ok=True)
    
    # Prepare experiment configs
    experiment_configs = []
    for i, (p1_features, p2_features) in enumerate(experiment_combinations):
        experiment_id = f"{i+1}"
        
        # Create a config dictionary for this experiment
        exp_config = {
            'config': config.to_dict(),
            'p1_features': p1_features,
            'p2_features': p2_features,
            'experiment_id': experiment_id,
            'prepared_data': data  # Share the prepared data across processes
        }
        
        experiment_configs.append(exp_config)
    
    # Run experiments in parallel
    with multiprocessing.Pool(processes=pool_size) as pool:
        pool.map(run_suppression_subprocess, experiment_configs)
    
    # Calculate total time
    total_time = time.time() - start_time
    
    # Combine results from all experiments
    combined_results = {
        'experiments': [],
        'execution_time': total_time
    }
    
    # Load results from individual files
    result_files = [f for f in os.listdir(results_dir) if f.startswith("exp_") and f.endswith(".json")]
    
    for file in sorted(result_files, key=lambda x: int(x.split('_')[1].split('p1f')[0])):
        file_path = os.path.join(results_dir, file)
        try:
            with open(file_path, 'r') as f:
                experiment_result = json.load(f)
                combined_results['experiments'].append(experiment_result)
        except Exception as e:
            logger.error(f"Error loading experiment result from {file_path}: {str(e)}")
    
    # Generate timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # First log the final combined results summary
    logger.info(f"Final Combined Results Summary:")
    logger.info(f"Processed {len(combined_results['experiments'])} suppression experiment combinations")
    
    # Create a single final results file in the main results directory
    final_filename = f"suppression_final_{timestamp}.json"
    save_results(combined_results, config, final_filename)
    
    # Log completion 
    logger.info(f"All suppression experiments completed in {total_time:.2f}s")
    
    return combined_results


def create_serializable_dp_results(round_results, experiment_id, noise_p1, noise_p2):
    """
    Create a serializable version of DP experiment results, avoiding circular references and NaN values.
    
    Args:
        round_results: Original round results list
        experiment_id: ID for this experiment
        noise_p1: Noise multiplier for P1
        noise_p2: Noise multiplier for P2
        
    Returns:
        Dict with serializable results
    """
    # Create minimal result object with only what's needed
    result = {
        'experiment_id': experiment_id,
        'noise_multiplier_p1': float(noise_p1) if noise_p1 is not None else None,
        'noise_multiplier_p2': float(noise_p2) if noise_p2 is not None else None,
        'rounds': []
    }
    
    logger = logging.getLogger(__name__)
    
    # Copy only needed data from each round
    for r in round_results:
        try:
            # Basic round information
            round_copy = {
                'round': int(r['round']),
                'time': float(r['time']),
                'test_metrics': {
                    'p1': {
                        'loss': float(r['test_metrics']['p1']['loss']),
                        'accuracy': float(r['test_metrics']['p1']['accuracy'])
                    },
                    'p2': {
                        'loss': float(r['test_metrics']['p2']['loss']),
                        'accuracy': float(r['test_metrics']['p2']['accuracy'])
                    }
                }
            }
            
            # Add gradient metrics if available (from client history)
            if 'client_training' in r:
                # Add gradient metrics from both clients (average them)
                dp_metrics = {
                    'avg_grad_norm': [],
                    'avg_noise_norm': [],
                    'snr': []
                }
                
                for client, history in r['client_training'].items():
                    if 'avg_grad_norm' in history:
                        for i, val in enumerate(history['avg_grad_norm']):
                            if i >= len(dp_metrics['avg_grad_norm']):
                                dp_metrics['avg_grad_norm'].append(0)
                            dp_metrics['avg_grad_norm'][i] += float(val) / 2
                            
                    if 'noise_norms' in history:
                        for i, val in enumerate(history['noise_norms']):
                            if i >= len(dp_metrics['avg_noise_norm']):
                                dp_metrics['avg_noise_norm'].append(0)
                            dp_metrics['avg_noise_norm'][i] += float(val) / 2
                            
                    if 'snr' in history:
                        for i, val in enumerate(history['snr']):
                            if i >= len(dp_metrics['snr']):
                                dp_metrics['snr'].append(0)
                            dp_metrics['snr'][i] += float(val) / 2
                
                # Add metrics to output if they have values
                if dp_metrics['avg_grad_norm']:
                    round_copy['dp_metrics'] = {
                        'avg_grad_norm': dp_metrics['avg_grad_norm'],
                        'avg_noise_norm': dp_metrics['avg_noise_norm'],
                        'snr': dp_metrics['snr']
                    }
            
            # Add epsilon values if available
            if 'epsilons' in r:
                # Handle nested structure
                epsilon_copy = {}
                cumulative_epsilons = {}
                
                for player, eps_list in r['epsilons'].items():
                    # Convert all values to float and filter out non-finite values
                    valid_eps = []
                    running_sum = 0.0
                    cumulative_eps = []
                    
                    for e in eps_list:
                        try:
                            e_float = float(e)
                            if not np.isfinite(e_float):
                                logger.warning(f"Skipping non-finite epsilon value: {e}")
                                continue
                            valid_eps.append(e_float)
                            running_sum += e_float
                            cumulative_eps.append(running_sum)
                        except (ValueError, TypeError) as err:
                            logger.warning(f"Skipping non-convertible epsilon value: {e}, error: {err}")
                    
                    epsilon_copy[player] = valid_eps
                    cumulative_epsilons[player] = cumulative_eps
                
                round_copy['epsilons'] = epsilon_copy
                round_copy['cumulative_epsilons'] = cumulative_epsilons
            
            result['rounds'].append(round_copy)
            
        except Exception as e:
            logger.error(f"Error processing round result: {e}")
            # Create simplified round data
            backup_round = {
                'round': r.get('round', 0),
                'time': float(r.get('time', 0)),
                'error': str(e)
            }
            result['rounds'].append(backup_round)
    
    return result


def run_dp_experiment(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run a differential privacy experiment with specific noise levels.
    
    This function should not be used directly, but through run_differential_privacy.
    
    Args:
        config_dict: Dictionary containing:
            - config: Experiment configuration with noise levels
            - logger: Custom logger for this experiment (optional)
        
    Returns:
        Dictionary with experiment results
    """
    # Extract config and logger from config_dict
    if isinstance(config_dict, ExperimentConfig):
        config = config_dict
        log = logging.getLogger(__name__)
    else:
        # Create the config object - avoid ** unpacking to better control conversion
        config = ExperimentConfig()
        
        # Copy each attribute individually to ensure proper type conversion
        for key, value in config_dict.items():
            if key == 'noise_multiplier_p1' and value is not None:
                try:
                    setattr(config, key, float(value))
                except (ValueError, TypeError):
                    setattr(config, key, value)
            elif key == 'noise_multiplier_p2' and value is not None:
                try:
                    setattr(config, key, float(value))
                except (ValueError, TypeError):
                    setattr(config, key, value)
            else:
                setattr(config, key, value)
        
        log = config_dict.get('logger', logging.getLogger(__name__))
    
    # Force the noise values to be proper types
    if hasattr(config, 'noise_multiplier_p1') and config.noise_multiplier_p1 is not None:
        try:
            config.noise_multiplier_p1 = float(config.noise_multiplier_p1)
        except (ValueError, TypeError):
            pass
    
    if hasattr(config, 'noise_multiplier_p2') and config.noise_multiplier_p2 is not None:
        try:
            config.noise_multiplier_p2 = float(config.noise_multiplier_p2)
        except (ValueError, TypeError):
            pass
    
    # Generate a unique experiment ID
    noise_p1 = "None" if config.noise_multiplier_p1 is None else str(config.noise_multiplier_p1)
    noise_p2 = "None" if config.noise_multiplier_p2 is None else str(config.noise_multiplier_p2)
    experiment_id = f"dp_p1n{noise_p1}_p2n{noise_p2}"
    
    log.info(f"Starting Differential Privacy Experiment: {experiment_id}")
    log.info(f"P1 Noise Multiplier: {noise_p1}, P2 Noise Multiplier: {noise_p2}")
    
    try:
        # Prepare data
        data = prepare_federated_data(
            data_path=config.data_path,
            feature_columns=config.feature_columns,
            target_column=config.target_column,
            p1_path=config.p1_path,
            p2_path=config.p2_path,
            seed=config.seed,
            initial_split_ratio=config.initial_split_ratio,
            test_split_ratio=config.test_split_ratio
        )
        
        # Create dataloaders
        log.info("Creating dataloaders...")
        train_loader_p1, test_loader_p1 = create_dataloaders(
            data['X_p1_train'], data['X_p1_test'], 
            data['y_p1_train'], data['y_p1_test'],
            batch_size=config.batch_size,
            seed=config.seed
        )
        
        train_loader_p2, test_loader_p2 = create_dataloaders(
            data['X_p2_train'], data['X_p2_test'], 
            data['y_p2_train'], data['y_p2_test'],
            batch_size=config.batch_size,
            seed=config.seed
        )
        
        # Initialize global model
        global_model = create_model(data['num_features'], data['num_classes'], config.seed)
        
        # Initialize results
        round_results = []
        
        # Convert noise multipliers to proper types for consistency
        noise_p1 = float(config.noise_multiplier_p1) if config.noise_multiplier_p1 is not None else None
        noise_p2 = float(config.noise_multiplier_p2) if config.noise_multiplier_p2 is not None else None
        
        # Initialize global privacy accounting
        cumulative_epsilon_p1 = 0.0
        cumulative_epsilon_p2 = 0.0
        
        # Log experiment details
        log.info(f"Running experiment with {config.federated_rounds} rounds, {config.client_epochs} epochs per client")
        
        # Federated learning rounds
        for round_num in range(1, config.federated_rounds + 1):
            log.info(f"Starting Round {round_num}/{config.federated_rounds}")
            round_start_time = time.time()
            
            # Create fresh model instances
            model_p1 = NeuralNetwork(input_dim=data['num_features'], num_classes=data['num_classes'])
            model_p2 = NeuralNetwork(input_dim=data['num_features'], num_classes=data['num_classes'])
            
            # Copy model parameters from global model
            model_p1.load_state_dict(global_model.state_dict())
            model_p2.load_state_dict(global_model.state_dict())
            
            # Train client models with DP
            log.info(f"Training client P1 (Round {round_num}) with noise_multiplier={noise_p1}")
            
            model_p1, history_p1 = train_model(
                model_p1,
                train_loader_p1, 
                test_loader_p1, 
                config.client_epochs, 
                config.learning_rate,
                noise_multiplier=noise_p1,
                max_grad_norm=config.max_grad_norm,
                client_id=f"P1 (Round {round_num})",
                logger=log,
                is_dp_experiment=True  # Flag this as a DP experiment
            )
            
            log.info(f"Training client P2 (Round {round_num}) with noise_multiplier={noise_p2}")
            
            model_p2, history_p2 = train_model(
                model_p2,
                train_loader_p2, 
                test_loader_p2, 
                config.client_epochs, 
                config.learning_rate,
                noise_multiplier=noise_p2,
                max_grad_norm=config.max_grad_norm,
                client_id=f"P2 (Round {round_num})",
                logger=log,
                is_dp_experiment=True  # Flag this as a DP experiment
            )
            
            # Track privacy budget from this round's training
            if 'epsilons' in history_p1 and history_p1['epsilons']:
                round_epsilon_p1 = history_p1['epsilons'][-1]
                cumulative_epsilon_p1 += round_epsilon_p1
                log.info(f"P1 privacy budget: round ε = {round_epsilon_p1:.4f}, cumulative ε = {cumulative_epsilon_p1:.4f}")
                
            if 'epsilons' in history_p2 and history_p2['epsilons']:
                round_epsilon_p2 = history_p2['epsilons'][-1]
                cumulative_epsilon_p2 += round_epsilon_p2
                log.info(f"P2 privacy budget: round ε = {round_epsilon_p2:.4f}, cumulative ε = {cumulative_epsilon_p2:.4f}")
            
            # Secure aggregation with noise weighting
            # When clients have different noise levels, weight models inversely to noise magnitude
            if noise_p1 is not None and noise_p2 is not None and noise_p1 != noise_p2:
                # Inverse noise weighting for better privacy preservation
                weight_p1 = 1.0 / noise_p1 if noise_p1 > 0 else 1.0
                weight_p2 = 1.0 / noise_p2 if noise_p2 > 0 else 1.0
                total_weight = weight_p1 + weight_p2
                
                # Normalize weights
                weight_p1 /= total_weight
                weight_p2 /= total_weight
                
                log.info(f"Using secure aggregation with weights: P1={weight_p1:.4f}, P2={weight_p2:.4f}")
                
                # Create a new model for secure aggregation
                global_model = create_model(data['num_features'], data['num_classes'], config.seed)
                
                # Weighted average of parameters
                with torch.no_grad():
                    for p_global, p1, p2 in zip(global_model.parameters(), 
                                               model_p1.parameters(), 
                                               model_p2.parameters()):
                        p_global.data = weight_p1 * p1.data + weight_p2 * p2.data
            else:
                # Standard model averaging when noise levels are equal
                global_model = average_models([model_p1, model_p2])
            
            # Evaluate global model
            loss_p1, acc_p1 = evaluate_model(global_model, test_loader_p1)
            loss_p2, acc_p2 = evaluate_model(global_model, test_loader_p2)
            
            # Calculate round time
            round_time = time.time() - round_start_time
            
            # Prepare round metrics
            round_metrics = {
                'round': round_num,
                'time': round_time,
                'noise_multiplier_p1': noise_p1,
                'noise_multiplier_p2': noise_p2,
                'client_training': {
                    'P1': history_p1,
                    'P2': history_p2
                },
                'test_metrics': {
                    'p1': {'loss': loss_p1, 'accuracy': acc_p1},
                    'p2': {'loss': loss_p2, 'accuracy': acc_p2}
                }
            }
            
            # Add epsilon values if available
            for client, history in [('p1', history_p1), ('p2', history_p2)]:
                if 'epsilons' in history and history['epsilons']:
                    if 'epsilons' not in round_metrics:
                        round_metrics['epsilons'] = {}
                    round_metrics['epsilons'][client] = history['epsilons']
            
            # Store round results
            round_results.append(round_metrics)
            
            # Log detailed results in consistent format
            log.info(f"Round {round_num}/{config.federated_rounds} completed in {round_time:.2f}s")
            log.info(f"Global Model - P1 Test: Loss = {loss_p1:.4f}, Accuracy = {acc_p1:.4f}")
            log.info(f"Global Model - P2 Test: Loss = {loss_p2:.4f}, Accuracy = {acc_p2:.4f}")
            
            # Print privacy summary if available
            if 'epsilons' in round_metrics:
                p1_eps = round_metrics['epsilons'].get('p1', [])
                p2_eps = round_metrics['epsilons'].get('p2', [])
                if p1_eps:
                    log.info(f"P1 Privacy Budget: ε = {p1_eps[-1]:.2f}" if p1_eps[-1] < 1000 else "P1 Privacy Budget: ε > 1000")
                if p2_eps:
                    log.info(f"P2 Privacy Budget: ε = {p2_eps[-1]:.2f}" if p2_eps[-1] < 1000 else "P2 Privacy Budget: ε > 1000")
        
        # Create serializable results using our helper function
        serializable_results = create_serializable_dp_results(
            round_results, experiment_id, 
            noise_p1, 
            noise_p2
        )
        
        # Format noise levels for filename in a clean way
        noise_p1_str = "None" if noise_p1 is None else str(noise_p1)
        noise_p2_str = "None" if noise_p2 is None else str(noise_p2)
        
        # Create the filename
        filename = f"results_p1_{noise_p1_str}_p2_{noise_p2_str}.json"
        
        # Save in dp_experiments directory
        dp_dir = os.path.join(config.results_dir, "dp_experiments")
        os.makedirs(dp_dir, exist_ok=True)
        file_path = os.path.join(dp_dir, filename)
        
        try:
            with open(file_path, 'w') as f:
                json.dump(serializable_results, f, indent=4)
            log.info(f"Results saved to {file_path}")
        except Exception as e:
            log.error(f"Error saving results: {e}")
        
        # Return the serializable results for further processing
        return serializable_results
    
    except Exception as e:
        log.error(f"Error in experiment {experiment_id}: {str(e)}")
        log.error(traceback.format_exc())
        return {
            'experiment_id': experiment_id,
            'error': str(e),
            'traceback': traceback.format_exc()
        }


def run_dp_subprocess(config_dict: Dict[str, Any]) -> None:
    """
    Run a DP experiment in a subprocess.
    
    Args:
        config_dict: Experiment configuration dictionary
    """
    # First make a copy of the config_dict to avoid modifying the original
    modified_config = config_dict.copy()
    
    # Check and convert noise values directly
    for key in ['noise_multiplier_p1', 'noise_multiplier_p2']:
        if key in modified_config and modified_config[key] is not None:
            try:
                # Convert to float if it's not None
                modified_config[key] = float(modified_config[key])
            except (ValueError, TypeError):
                pass
    
    # Now setup logging with the original config dict to keep proper paths
    config = ExperimentConfig(**config_dict)
    noise_p1 = "None" if modified_config.get('noise_multiplier_p1') is None else modified_config.get('noise_multiplier_p1')
    noise_p2 = "None" if modified_config.get('noise_multiplier_p2') is None else modified_config.get('noise_multiplier_p2')
    
    # Set up experiment-specific logging in the dp_experiments subdirectory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_subdir = os.path.join(config.logs_dir, "dp_experiments")
    os.makedirs(log_subdir, exist_ok=True)
    log_filename = f"dp_p1n{noise_p1}_p2n{noise_p2}_{timestamp}.log"
    log_path = os.path.join(log_subdir, log_filename)
    
    # Configure logging specifically for this subprocess
    subprocess_logger = logging.getLogger(f"dp_experiment_{noise_p1}_{noise_p2}")
    subprocess_logger.handlers = []  # Remove any existing handlers
    subprocess_logger.setLevel(logging.INFO)
    
    # Add file handler
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    subprocess_logger.addHandler(file_handler)
    
    # Add a header to the log file with experiment details - this is all that's needed
    # The dp_experiment() function will add its own detailed header
    
    # Store the logger and the fixed noise values in the modified_config for the experiment
    modified_config['logger'] = subprocess_logger
    
    try:
        # Run experiment with the modified config that has float noise values
        run_dp_experiment(modified_config)
        subprocess_logger.info(f"Differential Privacy Experiment completed successfully")
    except Exception as e:
        subprocess_logger.error(f"Error in DP experiment with noise P1={noise_p1}, P2={noise_p2}: {str(e)}")
        subprocess_logger.error(traceback.format_exc())


def run_differential_privacy(config: ExperimentConfig) -> Dict[str, Any]:
    """
    Run differential privacy experiments.
    
    This experiment explores the impact of different noise levels in
    a federated learning setting with differential privacy.
    
    Args:
        config: Experiment configuration
        
    Returns:
        Dictionary with experiment results
    """
    setup_logger(config)
    start_time = time.time()
    
    logger.info("Starting differential privacy experiments")
    
    # Define noise levels
    # Check if noise values were explicitly provided via command line
    #noise_p1_provided = hasattr(config, 'noise_multiplier_p1_explicitly_set') or config.noise_multiplier_p1 is not None
    noise_p1_provided = hasattr(config, 'noise_multiplier_p1_explicitly_set') and config.noise_multiplier_p1_explicitly_set
    #noise_p2_provided = hasattr(config, 'noise_multiplier_p2_explicitly_set') or config.noise_multiplier_p2 is not None
    noise_p2_provided = hasattr(config, 'noise_multiplier_p2_explicitly_set') and config.noise_multiplier_p2_explicitly_set

    # If either noise multiplier was explicitly provided (even None), use those specific values
    if noise_p1_provided or noise_p2_provided:
        # User specified noise levels, use those exact values (including None)
        noise_levels_p1 = [config.noise_multiplier_p1]
        noise_levels_p2 = [config.noise_multiplier_p2]
        logger.info(f"Using user-specified noise levels: P1={config.noise_multiplier_p1}, P2={config.noise_multiplier_p2}")
    else:
        # No noise levels specified at all, use the defaults
        noise_levels_p1 = [None, 0.5, 1.0, 1.5, 2.0]
        noise_levels_p2 = [None, 0.5, 1.0, 1.5, 2.0]
        logger.info(f"No noise levels specified, using default combinations")
    
    # Generate all combinations
    combinations = list(product(noise_levels_p1, noise_levels_p2))
    logger.info(f"Running {len(combinations)} differential privacy experiments")
    
    # Create a pool of workers with safety limit for memory usage
    max_system_cores = multiprocessing.cpu_count()
    available_cores = min(max_system_cores, len(combinations))
    pool_size = min(available_cores, config.max_parallel_dp_processes)
    
    logger.info(f"Using {pool_size} parallel processes (limited by max_parallel_dp_processes={config.max_parallel_dp_processes})")
    
    # Prepare experiment configs
    experiment_configs = []
    for noise_p1, noise_p2 in combinations:
        # Create a new config for this combination
        exp_config = deepcopy(config)
        exp_config.noise_multiplier_p1 = noise_p1
        exp_config.noise_multiplier_p2 = noise_p2
        
        # Add to list
        config_dict = exp_config.to_dict()
        experiment_configs.append(config_dict)
    
    # Run experiments in parallel - store the configs for later use
    combinations_run = [(cfg['noise_multiplier_p1'], cfg['noise_multiplier_p2']) for cfg in experiment_configs]
    with multiprocessing.Pool(processes=pool_size) as pool:
        pool.map(run_dp_subprocess, experiment_configs)
    
    # Calculate total time
    total_time = time.time() - start_time
    
    # Combine results from individual result files
    dp_experiments_dir = os.path.join(config.results_dir, "dp_experiments")
    combined_results = {
        'experiments': [],
        'execution_time': total_time
    }
    
    # Make sure directory exists first
    os.makedirs(dp_experiments_dir, exist_ok=True)
    
    # Find all result files
    result_files = [f for f in os.listdir(dp_experiments_dir) if f.startswith("results_p1_") and f.endswith(".json")]
    
    # If we found result files, load them
    if result_files:
        logger.info(f"Loading {len(result_files)} DP experiment result files")
        for file in result_files:
            file_path = os.path.join(dp_experiments_dir, file)
            try:
                with open(file_path, 'r') as f:
                    experiment_results = json.load(f)
                    combined_results['experiments'].append(experiment_results)
            except Exception as e:
                logger.error(f"Error loading file {file_path}: {e}")
    else:
        # Fallback - add experiment summaries directly if no files were found
        logger.warning("No individual DP experiment result files found, creating summary only")
        for noise_p1, noise_p2 in combinations_run:
            exp_summary = {
                'noise_multiplier_p1': noise_p1,
                'noise_multiplier_p2': noise_p2,
                'success': True  # Flag to indicate this experiment was run
            }
            combined_results['experiments'].append(exp_summary)
    
    # Create a single final results file in the final directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # First log the final combined results summary
    logger.info(f"Final DP Combined Results Summary:")
    logger.info(f"Processed {len(combined_results['experiments'])} DP experiment combinations")
    
    # Save to the main results directory
    final_filename = f"dp_final_{timestamp}.json"
    save_results(combined_results, config, final_filename)
    
    # Log completion
    logger.info(f"All differential privacy experiments completed in {total_time:.2f}s")
    
    return combined_results


def run_experiment(config: ExperimentConfig) -> Dict[str, Any]:
    """
    Run an experiment based on configuration.
    
    Args:
        config: Experiment configuration
        
    Returns:
        Dictionary with experiment results
    """
    if config.experiment_type == "local":
        return run_local_baseline(config)
    elif config.experiment_type == "federated":
        return run_federated_learning(config)
    elif config.experiment_type == "suppression":
        return run_suppression(config)
    elif config.experiment_type == "differential_privacy":
        return run_differential_privacy(config)
    else:
        raise ValueError(f"Unknown experiment type: {config.experiment_type}")


def create_heatmaps_suppression(results_file: str, output_dir: str = None) -> None:
    """
    Create heatmaps from suppression experiment results.
    
    Args:
        results_file: Path to the results file
        output_dir: Directory to save the heatmaps (default: same as results_file)
    """
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    # Only process combined results files for efficiency
    is_combined_file = "suppression_final_results" in results_file or "combined" in results_file
    
    # Skip individual experiment files unless explicitly requested
    if not is_combined_file and "exp_" in os.path.basename(results_file):
        return
    
    # Only print loading message for the combined results file
    if is_combined_file:
        print(f"Loading suppression results from {results_file}")
    
    # Load results
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    # Extract experiment results
    experiments = None
    config = {}
    
    # First check for the most common structure where data is wrapped in results
    if 'results' in data and 'experiments' in data['results']:
        experiments = data['results']['experiments']
        config = data['config']
    # Then check for direct experiments array (less common)
    elif 'experiments' in data:
        experiments = data['experiments']
        config = data.get('config', {})
    # Check for timestamp + results format
    elif 'timestamp' in data and 'results' in data:
        if 'experiments' in data['results']:
            experiments = data['results']['experiments']
        else:
            # Support for legacy format where results is a single experiment
            experiments = [data['results']]
        config = data['config']
    # Check for direct experiment data (individual experiment file)
    elif 'experiment_id' in data and ('p1_features' in data or 'final_evaluation' in data):
        experiments = [data]
    
    if experiments is None:
        if is_combined_file:
            print(f"Warning: Unrecognized result format in {results_file}")
            print(f"Available keys: {list(data.keys())}")
        return
    
    # Determine total number of features
    if isinstance(config, dict) and 'feature_columns' in config:
        total_features = len(config['feature_columns'])
    elif experiments and 'p1_features' in experiments[0]:
        total_features = max(
            max(len(exp.get('p1_features', [])) for exp in experiments),
            max(len(exp.get('p2_features', [])) for exp in experiments)
        )
    else:
        if is_combined_file:
            print(f"Warning: Could not determine feature count in {results_file}")
        total_features = 14  # Default based on the dataset
            
    if is_combined_file:
        print(f"Processing {len(experiments)} experiments with {total_features} total features")
    
    # Initialize matrices for accuracies
    p1_accuracies = np.zeros((total_features, total_features))
    p2_accuracies = np.zeros((total_features, total_features))
    
    # Process each experiment
    for exp in experiments:
        # Try to find feature information - it might be at different locations
        p1_features = None
        p2_features = None
        
        # Check if there's a nested results field in the experiment
        if 'results' in exp and isinstance(exp['results'], dict):
            exp_data = exp['results']  # Use the nested results content
        else:
            exp_data = exp  # Use the experiment directly
        
        # Direct feature access in the proper location
        if 'p1_features' in exp_data and 'p2_features' in exp_data:
            p1_features = exp_data['p1_features']
            p2_features = exp_data['p2_features']
        # Features might be in a nested structure
        elif 'experiment_id' in exp_data and isinstance(exp_data['experiment_id'], str) and exp_data['experiment_id'].startswith('exp_'):
            # Try to parse from experiment ID (exp_ID_p1fX_p2fY format)
            try:
                id_parts = exp_data['experiment_id'].split('_')
                for part in id_parts:
                    if part.startswith('p1f'):
                        p1_count = int(part[3:])
                        p1_features = config['feature_columns'][:p1_count] if 'feature_columns' in config else []
                    elif part.startswith('p2f'):
                        p2_count = int(part[3:])
                        p2_features = config['feature_columns'][:p2_count] if 'feature_columns' in config else []
            except Exception as e:
                pass
        
        # If still can't find features, try to extract from filename or ID field
        if p1_features is None or p2_features is None:
            # Try experiment ID field if it exists
            exp_id = exp_data.get('experiment_id')
            if exp_id is not None:
                # Sometimes experiment_id is just a number - convert to string
                exp_id_str = str(exp_id)
                
                # Try to parse ID with format in log: suppression_exp42_p1f12_p2f1 
                if isinstance(exp_id, (int, str)):
                    try:
                        # Find corresponding file in suppression_experiments
                        supp_dir = os.path.join(os.path.dirname(results_file), 'suppression_experiments')
                        if os.path.exists(supp_dir):
                            matching_files = [f for f in os.listdir(supp_dir) 
                                            if f.endswith('.json') and f"{exp_id}" in f]
                            if matching_files:
                                file = matching_files[0]
                                # Extract feature counts from filename
                                if '_p1f' in file and '_p2f' in file:
                                    parts = file.split('_')
                                    for i, part in enumerate(parts):
                                        if part.startswith('p1f') and i+1 < len(parts):
                                            p1_count = int(part[3:])
                                            p1_features = config['feature_columns'][:p1_count] if 'feature_columns' in config else []
                                        elif part.startswith('p2f'):
                                            p2_count = int(part[3:])
                                            p2_features = config['feature_columns'][:p2_count] if 'feature_columns' in config else []
                    except Exception as e:
                        pass
        
        # If we still don't have feature information, skip this experiment
        if p1_features is None or p2_features is None:
            print(f"Warning: Experiment missing feature information for ID {exp.get('experiment_id', 'unknown')}")
            continue
            
        p1_feature_count = len(p1_features)
        p2_feature_count = len(p2_features)
        
        # Ensure counts are valid
        if p1_feature_count <= 0 or p1_feature_count > total_features or p2_feature_count <= 0 or p2_feature_count > total_features:
            print(f"Warning: Invalid feature counts: P1={p1_feature_count}, P2={p2_feature_count}")
            continue
        
        # Adjust indices
        col_idx = total_features - p1_feature_count  # x-axis (P1)
        row_idx = p2_feature_count - 1  # y-axis (P2)
        
        # Extract accuracies from final evaluation, handling different possible formats
        try:
            if 'final_evaluation' in exp_data:
                p1_accuracies[row_idx, col_idx] = exp_data['final_evaluation']['P1']['accuracy']
                p2_accuracies[row_idx, col_idx] = exp_data['final_evaluation']['P2']['accuracy']
            elif 'rounds' in exp_data and exp_data['rounds']:
                # Take the last round's results
                last_round = exp_data['rounds'][-1]
                if 'global_evaluation' in last_round:
                    p1_accuracies[row_idx, col_idx] = last_round['global_evaluation']['P1']['accuracy']
                    p2_accuracies[row_idx, col_idx] = last_round['global_evaluation']['P2']['accuracy']
                elif 'test_metrics' in last_round:
                    p1_accuracies[row_idx, col_idx] = last_round['test_metrics']['p1']['accuracy']
                    p2_accuracies[row_idx, col_idx] = last_round['test_metrics']['p2']['accuracy']
        except (KeyError, IndexError, TypeError) as e:
            print(f"Warning: Could not extract accuracies from experiment: {e}")
    
    # Create output directory if needed
    if output_dir is None:
        output_dir = os.path.dirname(results_file)
    os.makedirs(output_dir, exist_ok=True)
    
    # Get feature names if available
    feature_names = []
    feature_abbreviations = []
    if isinstance(config, dict) and 'feature_columns' in config:
        feature_names = config['feature_columns']
        # Create abbreviations for feature names (max 15 chars)
        feature_abbreviations = [f"{i+1}. {name[:15]}" for i, name in enumerate(feature_names)]
    
    # If no feature names available, use numbers
    if not feature_abbreviations:
        feature_abbreviations = [f"Feature {i+1}" for i in range(total_features)]
    
    # Create labels for X and Y axes - use original orientation
    x_labels = [str(i) for i in range(total_features, 0, -1)]  # Features used by P1, from total to 1
    y_labels = [str(i) for i in range(1, total_features + 1)]  # Features used by P2, from 1 to total
    
    # Common settings for heatmaps - use original colormap
    heatmap_kwargs = {
        'cmap': 'YlOrRd',  # Original colormap
        'vmin': 0.2,
        'vmax': 1.0,
        'annot': True,
        'fmt': '.2f',
        'cbar_kws': {'label': 'Accuracy'}
    }
    
    # Create separate figure for P1's accuracy
    fig1, ax1 = plt.subplots(figsize=(12, 10))
    
    # Create P1's heatmap - no flipping needed, just use the matrix as is
    sns.heatmap(p1_accuracies, ax=ax1, **heatmap_kwargs)
    
    # Set proper tick labels - they're already in the right order
    ax1.set_xticklabels(x_labels)
    ax1.set_yticklabels(y_labels)
    
    ax1.set_title("P1's Test Accuracy by Feature Suppression", pad=20, fontsize=16)
    ax1.set_xlabel("Number of features used by P1", fontsize=12)
    ax1.set_ylabel("Number of features used by P2", fontsize=12)
    
    # Add feature name legend if available - commented out to hide it
    # if feature_names:
    #     legend_text = "Feature Index Legend:\n" + "\n".join(feature_abbreviations)
    #     props = dict(boxstyle='round', facecolor='white', alpha=0.7)
    #     plt.figtext(1.02, 0.5, legend_text, fontsize=8, 
    #                 bbox=props, verticalalignment='center')
    
    # Adjust layout and save
    fig1.tight_layout()
    output_file = os.path.join(output_dir, "suppression_heatmap_p1.png")
    fig1.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.close(fig1)
    
    # Create separate figure for P2's accuracy
    fig2, ax2 = plt.subplots(figsize=(12, 10))
    
    # Create P2's heatmap - no flipping needed
    sns.heatmap(p2_accuracies, ax=ax2, **heatmap_kwargs)
    
    # Set proper tick labels
    ax2.set_xticklabels(x_labels)
    ax2.set_yticklabels(y_labels)
    
    ax2.set_title("P2's Test Accuracy by Feature Suppression", pad=20, fontsize=16)
    ax2.set_xlabel("Number of features used by P1", fontsize=12)
    ax2.set_ylabel("Number of features used by P2", fontsize=12)
    
    # Add feature name legend if available - commented out to hide it
    # if feature_names:
    #     legend_text = "Feature Index Legend:\n" + "\n".join(feature_abbreviations)
    #     props = dict(boxstyle='round', facecolor='white', alpha=0.7)
    #     plt.figtext(1.02, 0.5, legend_text, fontsize=8, 
    #                 bbox=props, verticalalignment='center')
    
    # Adjust layout and save
    fig2.tight_layout()
    output_file = os.path.join(output_dir, "suppression_heatmap_p2.png")
    fig2.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.close(fig2)
    
    logger.info(f"Suppression heatmaps saved to {output_file}")


def create_heatmaps_dp(results_file: str, output_dir: str = None) -> None:
    """
    Create heatmaps from differential privacy experiment results.
    
    Args:
        results_file: Path to the results file
        output_dir: Directory to save the heatmaps (default: same as results_file)
    """
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    # Only print loading message for final results files to reduce noise
    if "dp_final_" in results_file:
        print(f"Loading DP results from {results_file}")
    # Load results
    with open(results_file, 'r') as f:
        combined_data = json.load(f)
    
    # Extract experiment results - handle different possible formats
    if 'results' in combined_data and 'experiments' in combined_data['results']:
        # Format with results wrapper
        experiments = combined_data['results']['experiments']
    elif 'experiments' in combined_data:
        # Direct format with experiments array
        experiments = combined_data['experiments']
    elif 'timestamp' in combined_data and 'results' in combined_data:
        # Current format with results wrapper
        if 'experiments' in combined_data['results']:
            experiments = combined_data['results']['experiments']
        else:
            # Single experiment results
            experiments = [combined_data['results']]
    elif 'experiment_id' in combined_data and 'rounds' in combined_data:
        # This is a single experiment result file
        experiments = [combined_data]
    elif 'rounds' in combined_data:
        # Another possible format for a single experiment
        experiments = [combined_data]
    else:
        print(f"Warning: Unrecognized result format in {results_file}")
        print(f"Available keys: {list(combined_data.keys())}")
        return
    
    # Only print processing message for final results
    if "dp_final_" in results_file:
        print(f"Processing {len(experiments)} DP experiments")
    
    # Extract and process the records
    records = []
    for result in experiments:
        try:
            # Check if there's a nested results field in the result
            if 'results' in result and isinstance(result['results'], dict):
                exp_data = result['results']  # Use the nested results content
            else:
                exp_data = result  # Use the result directly
                
            # Extract noise multipliers
            nm_p1 = exp_data.get('noise_multiplier_p1')
            nm_p2 = exp_data.get('noise_multiplier_p2')
            
            # Convert "None" strings to None for consistent display
            if isinstance(nm_p1, str) and nm_p1.lower() == "none":
                nm_p1 = None
            if isinstance(nm_p2, str) and nm_p2.lower() == "none":
                nm_p2 = None
            
            # Only process if we have rounds data
            if 'rounds' in exp_data and exp_data['rounds']:
                final_round = exp_data['rounds'][-1]  # last element in rounds list
                
                # Extract accuracy metrics, handling different formats
                if 'test_metrics' in final_round:
                    p1_acc = final_round["test_metrics"]["p1"]["accuracy"]
                    p2_acc = final_round["test_metrics"]["p2"]["accuracy"]
                elif 'global_evaluation' in final_round:
                    p1_acc = final_round["global_evaluation"]["P1"]["accuracy"]
                    p2_acc = final_round["global_evaluation"]["P2"]["accuracy"]
                else:
                    print(f"Warning: Could not find test metrics in final round")
                    continue
                    
                # Extract epsilon values if available
                record = {
                    "noise_multiplier_p1": nm_p1,
                    "noise_multiplier_p2": nm_p2,
                    "final_round": final_round.get("round", 0),
                    "final_accuracy_p1": p1_acc,
                    "final_accuracy_p2": p2_acc
                }
                
                # Try to extract epsilon values
                if 'epsilons' in final_round:
                    if 'p1' in final_round['epsilons'] and final_round['epsilons']['p1']:
                        record["final_epsilon_p1"] = final_round['epsilons']['p1'][-1]
                    if 'p2' in final_round['epsilons'] and final_round['epsilons']['p2']:
                        record["final_epsilon_p2"] = final_round['epsilons']['p2'][-1]
                
                # Try to extract cumulative epsilon values (from our enhanced tracking)
                if 'cumulative_epsilons' in final_round:
                    if 'p1' in final_round['cumulative_epsilons'] and final_round['cumulative_epsilons']['p1']:
                        record["cumulative_epsilon_p1"] = final_round['cumulative_epsilons']['p1'][-1]
                    if 'p2' in final_round['cumulative_epsilons'] and final_round['cumulative_epsilons']['p2']:
                        record["cumulative_epsilon_p2"] = final_round['cumulative_epsilons']['p2'][-1]
                
                # Record the data point
                records.append(record)
            else:
                print(f"Warning: No rounds data for experiment with noise P1={nm_p1}, P2={nm_p2}")
        except (KeyError, IndexError, TypeError) as e:
            print(f"Warning: Error processing experiment: {e}")
    
    # Create a DataFrame
    df_final = pd.DataFrame(records)
    
    # Create output directory if needed
    if output_dir is None:
        output_dir = os.path.dirname(results_file)
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a DataFrame
    df_final = pd.DataFrame(records)
    
    # Check if we have enough data to create heatmaps
    if len(df_final) < 2:
        # Only print warning for combined results file to reduce noise
        if "combined_results" in results_file:
            print(f"Not enough data to create heatmaps: found only {len(df_final)} datapoints")
        return
    
    # Create output directory if needed
    if output_dir is None:
        output_dir = os.path.dirname(results_file)
    os.makedirs(output_dir, exist_ok=True)
    
    # Create heatmaps for accuracy of P1 and P2 if we have data for both
    for column, title in zip(
        ['final_accuracy_p1', 'final_accuracy_p2'], 
        ['P1 Test Accuracy Heatmap', 'P2 Test Accuracy Heatmap']
    ):
        if column not in df_final.columns:
            print(f"Column {column} not found in results")
            continue
            
        # Make sure we have at least 2 unique values for each axis
        if len(df_final['noise_multiplier_p1'].unique()) < 2 or len(df_final['noise_multiplier_p2'].unique()) < 2:
            print(f"Need at least 2 unique values for each noise multiplier to create heatmap")
            
            # Create a simpler plot instead
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # If we have multiple P2 values but only one P1 value
            if len(df_final['noise_multiplier_p1'].unique()) == 1 and len(df_final['noise_multiplier_p2'].unique()) > 1:
                sns.barplot(x='noise_multiplier_p2', y=column, data=df_final, color='#fd8d3c', ax=ax)
                ax.set_title(f"{title} (Fixed P1={df_final['noise_multiplier_p1'].iloc[0]})")
                ax.set_xlabel('P2 Noise Multiplier')
                ax.set_ylabel('Accuracy')
            
            # If we have multiple P1 values but only one P2 value
            elif len(df_final['noise_multiplier_p2'].unique()) == 1 and len(df_final['noise_multiplier_p1'].unique()) > 1:
                sns.barplot(x='noise_multiplier_p1', y=column, data=df_final, color='#fd8d3c', ax=ax)
                ax.set_title(f"{title} (Fixed P2={df_final['noise_multiplier_p2'].iloc[0]})")
                ax.set_xlabel('P1 Noise Multiplier')
                ax.set_ylabel('Accuracy')
            
            # Otherwise just plot all points
            else:
                ax.bar(range(len(df_final)), df_final[column], color='#fd8d3c')
                ax.set_xticks(range(len(df_final)))
                ax.set_xticklabels([f"P1={p1}, P2={p2}" for p1, p2 in zip(df_final['noise_multiplier_p1'], df_final['noise_multiplier_p2'])], rotation=45)
                ax.set_title(title)
                ax.set_ylabel('Accuracy')
            
            fig.tight_layout()
            output_file = os.path.join(output_dir, f"dp_plot_{column}.png")
            fig.savefig(output_file, bbox_inches='tight', dpi=300)
            plt.close(fig)
            
            logger.info(f"DP plot saved to {output_file}")
            continue
            
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Create pivot table for heatmap
        try:
            heatmap_data = df_final.pivot(
                index='noise_multiplier_p2', 
                columns='noise_multiplier_p1', 
                values=column
            )
            
            # Sort index in descending order for better visualization
            heatmap_data = heatmap_data.sort_index(ascending=False)
            
            # Create heatmap with YlOrRd colormap
            sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap='YlOrRd', ax=ax)
            ax.set_title(title)
            ax.set_xlabel('P1 Noise Multiplier')
            ax.set_ylabel('P2 Noise Multiplier')
            
            # Save figure
            output_file = os.path.join(output_dir, f"dp_heatmap_{column}.png")
            fig.savefig(output_file, bbox_inches='tight', dpi=300)
            plt.close(fig)
            
            logger.info(f"DP heatmap saved to {output_file}")
        except Exception as e:
            print(f"Error creating heatmap: {e}")
            print("Data points available:")
            for i, row in df_final.iterrows():
                print(f"  P1={row['noise_multiplier_p1']}, P2={row['noise_multiplier_p2']}, {column}={row.get(column, 'N/A')}")
                
    # Create privacy-utility tradeoff plots if we have cumulative epsilon values
    # if 'cumulative_epsilon_p1' in df_final.columns or 'cumulative_epsilon_p2' in df_final.columns:
    #     print("Creating privacy-utility tradeoff plots")
        
    #     try:
    #         fig, ax = plt.subplots(figsize=(10, 6))
            
    #         # Plot P1 privacy-utility tradeoff if data available
    #         if 'cumulative_epsilon_p1' in df_final.columns and 'final_accuracy_p1' in df_final.columns:
    #             sns.scatterplot(
    #                 x='cumulative_epsilon_p1', 
    #                 y='final_accuracy_p1', 
    #                 data=df_final, 
    #                 label='P1',
    #                 s=100,
    #                 ax=ax
    #             )
                
    #             # Add P1 noise level annotations
    #             for _, row in df_final.iterrows():
    #                 if pd.notna(row.get('cumulative_epsilon_p1')) and pd.notna(row.get('final_accuracy_p1')):
    #                     ax.annotate(
    #                         f"n={row['noise_multiplier_p1']}",
    #                         (row['cumulative_epsilon_p1'], row['final_accuracy_p1']),
    #                         xytext=(5, 5),
    #                         textcoords='offset points'
    #                     )
            
    #         # Plot P2 privacy-utility tradeoff if data available
    #         if 'cumulative_epsilon_p2' in df_final.columns and 'final_accuracy_p2' in df_final.columns:
    #             sns.scatterplot(
    #                 x='cumulative_epsilon_p2', 
    #                 y='final_accuracy_p2', 
    #                 data=df_final, 
    #                 label='P2',
    #                 s=100,
    #                 ax=ax
    #             )
                
    #             # Add P2 noise level annotations
    #             for _, row in df_final.iterrows():
    #                 if pd.notna(row.get('cumulative_epsilon_p2')) and pd.notna(row.get('final_accuracy_p2')):
    #                     ax.annotate(
    #                         f"n={row['noise_multiplier_p2']}",
    #                         (row['cumulative_epsilon_p2'], row['final_accuracy_p2']),
    #                         xytext=(5, 5),
    #                         textcoords='offset points'
    #                     )
            
    #         ax.set_xlabel('Cumulative Privacy Budget (ε)')
    #         ax.set_ylabel('Final Test Accuracy')
    #         ax.set_title('Privacy-Utility Tradeoff')
    #         ax.grid(True, linestyle='--', alpha=0.7)
            
    #         # Save the figure
    #         tradeoff_file = os.path.join(output_dir, "dp_privacy_utility_tradeoff.png")
    #         fig.savefig(tradeoff_file, bbox_inches='tight', dpi=300)
    #         plt.close(fig)
            
    #         logger.info(f"Privacy-utility tradeoff plot saved to {tradeoff_file}")
            
    #     except Exception as e:
    #         print(f"Error creating privacy-utility tradeoff plot: {e}")

    # Create privacy-utility tradeoff plots if we have cumulative epsilon values
    if 'cumulative_epsilon_p1' in df_final.columns or 'cumulative_epsilon_p2' in df_final.columns:
        print("Creating privacy-utility tradeoff plots")
        
        try:
            # Create figure for the main plot with annotations
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Create a dictionary to track placed annotations for collision detection
            placed_annotations = {}
            
            # Define a function to adjust annotation positions to avoid overlap
            def adjust_position(x, y, placed_annotations, ax):
                """Find a suitable position for text annotation that doesn't overlap with existing ones"""
                # Initial offset
                offsets = [(5, 5), (5, -15), (-15, 5), (-15, -15), (15, 15), (15, -25), (-25, 15), (-25, -25)]
                
                # Convert to display coordinates
                display_coords = ax.transData.transform((x, y))
                
                # Approximate text size (adjust as needed)
                text_size = (30, 15)
                
                for dx, dy in offsets:
                    # Position in display coordinates
                    text_pos = (display_coords[0] + dx, display_coords[1] + dy)
                    
                    # Check for collisions with existing annotations
                    collision = False
                    for pos, size in placed_annotations.values():
                        if (abs(text_pos[0] - pos[0]) < (text_size[0] + size[0])/2 and 
                            abs(text_pos[1] - pos[1]) < (text_size[1] + size[1])/2):
                            collision = True
                            break
                    
                    if not collision:
                        # Convert back to data coordinates
                        inv_pos = ax.transData.inverted().transform(text_pos)
                        # Store this position
                        placed_annotations[(x, y)] = (text_pos, text_size)
                        return (dx, dy)
                
                # If all positions are occupied, use a larger offset
                return (25, 25)
            
            # Collect data for table
            data_for_table = []
            
            # Plot P1 privacy-utility tradeoff if data available
            if 'cumulative_epsilon_p1' in df_final.columns and 'final_accuracy_p1' in df_final.columns:
                # Identify points for P1
                p1_points = df_final[pd.notna(df_final['cumulative_epsilon_p1']) & 
                                   pd.notna(df_final['final_accuracy_p1'])].copy()
                
                # Plot P1 points
                sns.scatterplot(
                    x='cumulative_epsilon_p1', 
                    y='final_accuracy_p1', 
                    data=p1_points, 
                    label='P1',
                    s=100,
                    ax=ax
                )
                
                # Add P1 noise level annotations with adjusted positions
                for idx, row in p1_points.iterrows():
                    x, y = row['cumulative_epsilon_p1'], row['final_accuracy_p1']
                    offset = adjust_position(x, y, placed_annotations, ax)
                    ax.annotate(
                        f"n={row['noise_multiplier_p1']:.2f}",
                        (x, y),
                        xytext=offset,
                        textcoords='offset points',
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                        fontsize=8
                    )
                    
                    # Add to table data
                    data_for_table.append({
                        'Participant': 'P1',
                        'Epsilon': row['cumulative_epsilon_p1'],
                        'Accuracy': row['final_accuracy_p1'],
                        'Noise': row['noise_multiplier_p1']
                    })
            
            # Plot P2 privacy-utility tradeoff if data available
            if 'cumulative_epsilon_p2' in df_final.columns and 'final_accuracy_p2' in df_final.columns:
                # Identify points for P2
                p2_points = df_final[pd.notna(df_final['cumulative_epsilon_p2']) & 
                                   pd.notna(df_final['final_accuracy_p2'])].copy()
                
                # Plot P2 points
                sns.scatterplot(
                    x='cumulative_epsilon_p2', 
                    y='final_accuracy_p2', 
                    data=p2_points, 
                    label='P2',
                    s=100,
                    ax=ax
                )
                
                # Add P2 noise level annotations with adjusted positions
                for idx, row in p2_points.iterrows():
                    x, y = row['cumulative_epsilon_p2'], row['final_accuracy_p2']
                    offset = adjust_position(x, y, placed_annotations, ax)
                    ax.annotate(
                        f"n={row['noise_multiplier_p2']:.2f}",
                        (x, y),
                        xytext=offset,
                        textcoords='offset points',
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                        fontsize=8
                    )
                    
                    # Add to table data
                    data_for_table.append({
                        'Participant': 'P2',
                        'Epsilon': row['cumulative_epsilon_p2'],
                        'Accuracy': row['final_accuracy_p2'],
                        'Noise': row['noise_multiplier_p2']
                    })
            
            ax.set_xlabel('Cumulative Privacy Budget (ε)')
            ax.set_ylabel('Final Test Accuracy')
            ax.set_title('Privacy-Utility Tradeoff')
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Add a legend and ensure it doesn't overlap with the data
            plt.legend(loc='best')
            
            # Adjust the margins to make sure labels fit
            plt.tight_layout()
            
            # Save the figure
            tradeoff_file = os.path.join(output_dir, "dp_privacy_utility_tradeoff.png")
            fig.savefig(tradeoff_file, bbox_inches='tight', dpi=300)
            plt.close(fig)
            
            # Create a dataframe from the collected table data
            df_table = pd.DataFrame(data_for_table)
            if not df_table.empty:
                # Sort by participant, epsilon, and accuracy
                df_table = df_table.sort_values(['Participant', 'Epsilon', 'Accuracy'])
                
                # Round values for display
                df_table_display = df_table.copy()
                df_table_display['Epsilon'] = df_table_display['Epsilon'].round(3)
                df_table_display['Accuracy'] = df_table_display['Accuracy'].round(3)
                df_table_display['Noise'] = df_table_display['Noise'].round(3)
                
                # Save data table as CSV
                table_file = os.path.join(output_dir, "dp_privacy_utility_data.csv")
                df_table.to_csv(table_file, index=False)
                logger.info(f"Privacy-utility data table saved to {table_file}")
                
                # Create a second figure for a visual table
                fig_table, ax_table = plt.subplots(figsize=(10, max(6, len(df_table_display) * 0.4)))
                ax_table.axis('off')
                
                # Create a table
                table = ax_table.table(
                    cellText=df_table_display.values,
                    colLabels=df_table_display.columns,
                    cellLoc='center',
                    loc='center'
                )
                
                # Style the table
                table.auto_set_font_size(False)
                table.set_fontsize(10)
                table.scale(1.2, 1.5)
                
                # Color the header
                for i in range(len(df_table_display.columns)):
                    table[(0, i)].set_facecolor('#e6e6e6')
                    table[(0, i)].set_text_props(weight='bold')
                
                # Color P1 and P2 rows differently
                for i in range(len(df_table_display)):
                    if df_table_display.iloc[i]['Participant'] == 'P1':
                        table[(i+1, 0)].set_facecolor('#e6f3ff')
                    else:
                        table[(i+1, 0)].set_facecolor('#ffe6e6')
                
                fig_table.tight_layout()
                
                # Save the table figure
                table_fig_file = os.path.join(output_dir, "dp_privacy_utility_table.png")
                fig_table.savefig(table_fig_file, bbox_inches='tight', dpi=300)
                plt.close(fig_table)
                
                logger.info(f"Privacy-utility table figure saved to {table_fig_file}")
            
            logger.info(f"Privacy-utility tradeoff plot saved to {tradeoff_file}")
            
        except Exception as e:
            print(f"Error creating privacy-utility tradeoff plot: {e}")
            logger.error(f"Error creating privacy-utility tradeoff plot: {e}")


def analyze_results(results_dir: str, output_dir: str = None) -> None:
    """
    Analyze and visualize experimental results.
    
    Args:
        results_dir: Directory containing experiment results
        output_dir: Directory to save analysis results (default: results_dir/analysis)
    """
    if output_dir is None:
        output_dir = os.path.join(results_dir, "analysis")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Analyzing results in directory: {results_dir}")
    print(f"Saving analysis to: {output_dir}")
    
    # Find result files - prioritize combined result files
    result_files = {
        'local': [],
        'federated': [],
        'suppression': [],
        'dp': []
    }
    
    # Check if the directory exists
    if not os.path.exists(results_dir):
        print(f"Error: Results directory {results_dir} does not exist")
        return
    
    # Look for suppression final results in the main results directory
    suppression_final_file = None
    for file in os.listdir(results_dir):
        if file.endswith('.json') and ('suppression_final_' in file):
            suppression_final_file = os.path.join(results_dir, file)
            result_files['suppression'].append(suppression_final_file)
            print(f"Found suppression final results file: {suppression_final_file}")
            break
    
    # Check for suppression results in subdirectory if we didn't find a combined file
    if not suppression_final_file:
        suppression_dir = os.path.join(results_dir, "suppression_experiments")
        if os.path.exists(suppression_dir) and os.path.isdir(suppression_dir):
            # Look for combined files first
            suppression_combined = None
            for file in os.listdir(suppression_dir):
                if file.endswith('.json') and ('combined' in file or 'final' in file):
                    suppression_combined = os.path.join(suppression_dir, file)
                    result_files['suppression'].append(suppression_combined)
                    print(f"Found combined suppression results file in subdirectory: {suppression_combined}")
                    break
            
            # Only if no combined file was found, add individual files (but we'll skip them later)
            if not suppression_combined:
                exp_files = []
                for file in os.listdir(suppression_dir):
                    if file.endswith('.json') and file.startswith('exp_'):
                        exp_files.append(os.path.join(suppression_dir, file))
                
                if exp_files:
                    print(f"No combined suppression results file found. Found {len(exp_files)} individual experiment files.")
                    # Since we're not using individual files (our update to create_heatmaps_suppression skips them),
                    # just add the first file to indicate we have suppression results
                    result_files['suppression'].append(exp_files[0])
    
    # Look for DP final results in the main results directory
    dp_final_file = None
    for file in os.listdir(results_dir):
        if file.endswith('.json') and ('dp_final_' in file):
            dp_final_file = os.path.join(results_dir, file)
            result_files['dp'].append(dp_final_file)
            print(f"Found DP final results file: {dp_final_file}")
            break
            
    # Check for DP results in subdirectory if no final file found
    dp_dir = os.path.join(results_dir, "dp_experiments")
    if os.path.exists(dp_dir) and os.path.isdir(dp_dir) and not dp_final_file:
        # First look for combined_results.json which has all experiments
        combined_file = os.path.join(dp_dir, "combined_results.json")
        if os.path.exists(combined_file):
            result_files['dp'].append(combined_file)
            print(f"Found combined DP results file: {combined_file}")
        else:
            # If no combined file exists, use individual result files
            individual_files = []
            for file in os.listdir(dp_dir):
                if file.endswith('.json') and "results_p" in file:
                    individual_files.append(os.path.join(dp_dir, file))
            
            if individual_files:
                print(f"No combined DP results file found. Using {len(individual_files)} individual DP result files.")
                result_files['dp'].extend(individual_files)
    
    # Look for local experiments in their subdirectory
    local_dir = os.path.join(results_dir, "local_experiments")
    if os.path.exists(local_dir) and os.path.isdir(local_dir):
        for file in os.listdir(local_dir):
            if file.endswith('.json') and 'local_baseline' in file:
                file_path = os.path.join(local_dir, file)
                result_files['local'].append(file_path)
                
    # Look for federated experiments in their subdirectory
    federated_dir = os.path.join(results_dir, "federated_experiments")
    if os.path.exists(federated_dir) and os.path.isdir(federated_dir):
        for file in os.listdir(federated_dir):
            if file.endswith('.json') and 'federated' in file:
                file_path = os.path.join(federated_dir, file)
                result_files['federated'].append(file_path)
    
    # Also check main directory for any results that might still be there
    for file in os.listdir(results_dir):
        if file.endswith('.json'):
            file_path = os.path.join(results_dir, file)
            # Skip files we've already processed
            if file_path in result_files['suppression'] or file_path in result_files['dp']:
                continue
                
            # Try to determine experiment type from the filename
            if file.startswith('local_baseline'):
                result_files['local'].append(file_path)
            elif file.startswith('federated'):
                result_files['federated'].append(file_path)
            elif ('dp' in file.lower() or 'differential' in file.lower() or 'privacy' in file.lower()) and not dp_final_file:
                # Only add if we don't already have DP final results
                result_files['dp'].append(file_path)
            else:
                # Try to determine from file content as a last resort
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    
                    if 'config' in data and 'experiment_type' in data['config']:
                        exp_type = data['config']['experiment_type'].lower()
                        if 'local' in exp_type:
                            result_files['local'].append(file_path)
                        elif 'federated' in exp_type:
                            result_files['federated'].append(file_path)
                        # Skip checking for suppression and DP here - we've already handled those
                except Exception as e:
                    print(f"Error reading file {file_path}: {e}")
    
    # Print summary of found files
    for exp_type, files in result_files.items():
        if files:
            print(f"Found {len(files)} {exp_type} experiment result files")
    
    # Process suppression results - only combined file or nothing
    if result_files['suppression']:
        try:
            # Just use the first file (which should be the combined file if available)
            create_heatmaps_suppression(result_files['suppression'][0], output_dir)
        except Exception as e:
            print(f"Error creating suppression heatmaps: {e}")
    
    # Process DP results
    if result_files['dp']:
        for file in result_files['dp']:
            try:
                create_heatmaps_dp(file, output_dir)
            except Exception as e:
                print(f"Error creating DP heatmaps for {file}: {e}")
    
    # More analysis can be added here
    
    print(f"Analysis complete. Results saved to {output_dir}")
    logger.info(f"Analysis results saved to {output_dir}")
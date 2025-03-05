#!/usr/bin/env python3
"""
Main entry point for running federated learning experiments.
"""
import os
import sys
import argparse
import logging
from datetime import datetime

from utils.config import ExperimentConfig, parse_noise
from utils.experiment import run_experiment, analyze_results

def create_example_config():
    """Create an example configuration file."""
    config = ExperimentConfig()
    config_path = "example_config.yaml"
    config.save(config_path)
    print(f"Example configuration saved to {config_path}")
    
    return config_path

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Federated Learning Experiments")
    
    # Basic commands
    parser.add_argument("--create-config", action="store_true", 
                        help="Create an example configuration file")
    parser.add_argument("--analyze", type=str, metavar="RESULTS_DIR",
                        help="Analyze experiment results in the specified directory")
    
    # Configuration options
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--experiment-type", type=str, 
                        choices=["local", "federated", "suppression", "differential_privacy"],
                        help="Type of experiment to run")
    
    # Data options
    parser.add_argument("--data-path", type=str, help="Path to the dataset")
    parser.add_argument("--p1-path", type=str, help="Path to Player 1's dataset")
    parser.add_argument("--p2-path", type=str, help="Path to Player 2's dataset")
    
    # Model options
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--local-epochs", type=int, help="Number of epochs for local training")
    parser.add_argument("--federated-rounds", type=int, help="Number of federated learning rounds")
    parser.add_argument("--client-epochs", type=int, help="Number of epochs per client per federated round")
    
    # Privacy options
    parser.add_argument("--noise-p1", type=str, 
                        help="Noise multiplier for Player 1 (use 'None' for no noise)")
    parser.add_argument("--noise-p2", type=str,
                        help="Noise multiplier for Player 2 (use 'None' for no noise)")
    parser.add_argument("--max-epsilon", type=float,
                        help="Maximum allowed privacy budget (epsilon)")
    parser.add_argument("--max-dp-processes", type=int, 
                        help="Maximum number of parallel processes for DP experiments")
    parser.add_argument("--max-suppression-processes", type=int,
                        help="Maximum number of parallel processes for suppression experiments")
    
    # Output options
    parser.add_argument("--results-dir", type=str, help="Directory to store results")
    
    args = parser.parse_args()
    
    # Handle special commands
    if args.create_config:
        create_example_config()
        return
    
    if args.analyze:
        analyze_results(args.analyze)
        return
    
    # Load configuration
    if args.config:
        config = ExperimentConfig.from_file(args.config)
    else:
        config = ExperimentConfig()
    
    # Override with command line arguments
    if args.experiment_type:
        config.experiment_type = args.experiment_type
    if args.data_path:
        config.data_path = args.data_path
    if args.p1_path:
        config.p1_path = args.p1_path
    if args.p2_path:
        config.p2_path = args.p2_path
    if args.seed:
        config.seed = args.seed
    if args.local_epochs:
        config.local_epochs = args.local_epochs
    if args.federated_rounds:
        config.federated_rounds = args.federated_rounds
    if args.client_epochs:
        config.client_epochs = args.client_epochs
    if args.noise_p1:
        config.noise_multiplier_p1 = parse_noise(args.noise_p1)
        config.noise_multiplier_p1_explicitly_set = True
    if args.noise_p2:
        config.noise_multiplier_p2 = parse_noise(args.noise_p2)
        config.noise_multiplier_p2_explicitly_set = True
    if args.max_dp_processes:
        config.max_parallel_dp_processes = args.max_dp_processes
    if args.max_suppression_processes:
        config.max_parallel_suppression_processes = args.max_suppression_processes
    if args.max_epsilon:
        config.max_epsilon = args.max_epsilon
    if args.results_dir:
        config.results_dir = args.results_dir
    
    # Create directories
    os.makedirs(config.results_dir, exist_ok=True)
    os.makedirs(config.logs_dir, exist_ok=True)
    
    # Run experiment
    results = run_experiment(config)
    
    print(f"Experiment completed. Results saved to {config.results_dir}")

if __name__ == "__main__":
    main()
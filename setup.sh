#!/bin/bash
# Install required dependencies for the federated learning experiments

echo "Installing dependencies..."
pip install torch scikit-learn pandas numpy matplotlib seaborn pyyaml opacus pyarrow

echo "Creating necessary directories..."
mkdir -p datasets
mkdir -p results
mkdir -p logs

echo "Setup complete! You can now run experiments with:"
echo "  python run_experiment.py --create-config"
echo "  python run_experiment.py --config example_config.yaml"
echo "  python run_experiment.py --experiment-type local"
# Federated Learning Privacy Experiments Framework

This framework provides a unified way to run and analyze federated learning experiments with different privacy mechanisms. It includes four main types of experiments:

1. **Local Baseline**: Train separate models on each client's data without federation
2. **Federated Learning**: Standard federated learning without privacy mechanisms
3. **Feature Suppression**: Privacy through selective feature hiding in federated learning
4. **Differential Privacy**: Privacy through noise addition in federated learning

## Requirements

- Python 3.8+
- PyTorch
- Opacus (for differential privacy)
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- pyyaml
- pyarrow

## Installation

```bash
pip install torch scikit-learn pandas numpy matplotlib seaborn pyyaml opacus pyarrow
```

## Directory Structure

```
.
├── README.md                # This file
├── run_experiment.py        # Main entry point for experiments
├── utils/                   # Utility modules
│   ├── config.py            # Configuration handling
│   ├── data.py              # Data preparation and loading
│   ├── models.py            # Neural network models
│   └── experiment.py        # Experiment runners
├── datasets/                # Directory for datasets (created automatically)
├── results/                 # Experiment results (created automatically)
│   ├── federated_experiments/ # Only intermediate federated results
│   ├── suppression_experiments/ # Only per-case suppression results
│   └── dp_experiments/      # Only per-case DP experiment results
└── logs/                    # Experiment logs (created automatically)
    ├── local_experiments/   # Logs for local experiments
    ├── federated_experiments/ # Logs for federated experiments
    ├── suppression_experiments/ # Logs for suppression experiments
    └── dp_experiments/      # Logs for differential privacy experiments
```

### Result File Organization

- **All final results** are stored directly in the `results/` directory (including local results)
- **Federated learning**: Final results in main `results/` directory and intermediate checkpoints in `results/federated_experiments/`
- **Suppression experiments**: Final combined results in main `results/` directory (e.g., `suppression_final_*.json`) and per-case results in `results/suppression_experiments/`
- **Differential privacy**: Final combined results in main `results/` directory (e.g., `dp_final_*.json`) and per-case results in `results/dp_experiments/`

## Usage

### Running Experiments

You can run experiments in different ways:

#### 1. Using Command Line Arguments

```bash
# Run a local baseline experiment
python run_experiment.py --experiment-type local
python run_experiment.py --experiment-type local --seed 42

# Run a federated learning experiment
python run_experiment.py --experiment-type federated
python run_experiment.py --experiment-type federated --federated-rounds 10 --client-epochs 10

# Run a suppression experiment
python run_experiment.py --experiment-type suppression

# Run a suppression experiment with limited parallel processes (for memory control)
python run_experiment.py --experiment-type suppression --max-suppression-processes 4

# Run a differential privacy experiment with specific noise levels
python run_experiment.py --experiment-type differential_privacy
python run_experiment.py --experiment-type differential_privacy --noise-p1 0.5 --noise-p2 1.0

# Run DP experiments with limited parallel processes (to control memory usage)
python run_experiment.py --experiment-type differential_privacy --max-dp-processes 5
```

#### 2. Using Configuration Files

First, create an example configuration file:

```bash
python run_experiment.py --create-config
```

Then, edit the configuration file (`example_config.yaml`) to set your parameters. Run the experiment with:

```bash
python run_experiment.py --config example_config.yaml
```

### Analyzing Results

To create visualizations and analyze experiment results:

```bash
python run_experiment.py --analyze results
```

This will create heatmaps and other visualizations based on the available results in the specified directory.

## Configuration Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| experiment_name | Name for the experiment | "fl_experiment" |
| experiment_type | Type of experiment: "local", "federated", "suppression", "differential_privacy" | "local" |
| data_path | Path to the main dataset | "datasets/dataset.parquet" |
| p1_path | Path to Player 1's dataset (if pre-split) | None |
| p2_path | Path to Player 2's dataset (if pre-split) | None |
| feature_columns | List of feature column names | [multiple network features] |
| target_column | Name of the target column | "application_name" |
| seed | Random seed for reproducibility | 42 |
| initial_split_ratio | Ratio for splitting into P1/P2 | 0.5 |
| test_split_ratio | Ratio for splitting each part into train/test | 0.1667 |
| batch_size | Batch size for training | 256 |
| learning_rate | Learning rate | 0.001 |
| local_epochs | Number of epochs for local baseline training | 100 |
| federated_rounds | Number of federated learning rounds | 10 |
| client_epochs | Number of epochs per client per federated round | 10 |
| p1_features | Features to use for Player 1 (suppression) | None (all) |
| p2_features | Features to use for Player 2 (suppression) | None (all) |
| max_parallel_suppression_processes | Maximum number of parallel processes for suppression experiments | 2 |
| noise_multiplier_p1 | Noise multiplier for Player 1 (DP) | None |
| noise_multiplier_p2 | Noise multiplier for Player 2 (DP) | None |
| max_grad_norm | Maximum gradient norm for DP | 1.0 |
| max_parallel_dp_processes | Maximum number of parallel processes for DP experiments | 2 |
| results_dir | Directory to store results | "results" |
| logs_dir | Directory to store logs | "logs" |

## Experiment Flow

### 1. Local Baseline

1. Split data between two players (P1 and P2) if not pre-split
2. Train model M1 on P1's training data
3. Train model M2 on P2's training data
4. Evaluate both models on both players' test data
5. Save results including training histories and evaluation metrics

### 2. Federated Learning

1. Split data between two players (P1 and P2) if not pre-split
2. Initialize global model
3. For each federated round:
   - Distribute global model to both players
   - Each player trains the model on their local data
   - Aggregate updated models into a new global model
   - Evaluate global model on both players' test data
4. Save results for each round and final evaluation

### 3. Feature Suppression

1. Split data between two players (P1 and P2) if not pre-split
2. For each combination of feature sets:
   - Create suppressed datasets for each player
   - Run federated learning with these suppressed datasets
   - Track performance across all feature combinations
3. Generate heatmaps to visualize impact of suppression

### 4. Differential Privacy

1. Split data between two players (P1 and P2) if not pre-split
2. For each combination of noise levels:
   - Run federated learning with DP-SGD at specified noise levels
   - Track privacy guarantees (epsilon values) and model performance
   - Control memory usage by limiting parallel processes
3. Generate heatmaps to visualize privacy-utility tradeoff

## Results Analysis

The framework provides tools to analyze and visualize experiment results, including:

- Heatmaps for suppression experiments showing accuracy vs. feature counts
- Heatmaps for differential privacy experiments showing accuracy vs. noise levels
- Other visualizations and analysis functions can be added as needed

## License

This project is open source and available under the MIT License.
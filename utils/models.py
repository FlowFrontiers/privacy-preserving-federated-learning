"""
Model definitions for federated learning experiments.
"""
import logging
import time
from typing import Dict, Tuple, List, Optional, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from opacus import PrivacyEngine

logger = logging.getLogger(__name__)

class NeuralNetwork(nn.Module):
    """Neural network model for federated learning experiments."""
    
    def __init__(self, input_dim: int, num_classes: int, seed: int = 42):
        """
        Initialize a neural network model.
        
        Args:
            input_dim: Input dimension
            num_classes: Number of classes
            seed: Random seed for reproducibility
        """
        super(NeuralNetwork, self).__init__()
        torch.manual_seed(seed)
        
        # Calculate hidden layer sizes
        hidden_layer1_neurons = int((2 / 3) * input_dim + num_classes)
        hidden_layer2_neurons = input_dim
        
        # Define model architecture
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_layer1_neurons),
            nn.ReLU(),
            nn.Linear(hidden_layer1_neurons, hidden_layer2_neurons),
            nn.ReLU(),
            nn.Linear(hidden_layer2_neurons, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        return self.model(x)


def create_model(input_dim: int, num_classes: int, seed: int = 42) -> NeuralNetwork:
    """
    Create a new neural network model.
    
    Args:
        input_dim: Input dimension
        num_classes: Number of classes
        seed: Random seed for reproducibility
        
    Returns:
        Initialized neural network model
    """
    return NeuralNetwork(input_dim, num_classes, seed)


def average_models(models: List[NeuralNetwork]) -> NeuralNetwork:
    """
    Average multiple models' parameters (FedAvg).
    
    Args:
        models: List of models to average
        
    Returns:
        A new model with averaged parameters
    """
    # Determine device from first model
    device = next(models[0].parameters()).device
    
    # Get input and output dimensions directly from the first model
    if hasattr(models[0], '_module'):
        # Handle Opacus-modified model
        module = models[0]._module
        input_dim = module.model[0].in_features
        output_dim = module.model[-1].out_features
    else:
        # Regular model
        input_dim = models[0].model[0].in_features
        output_dim = models[0].model[-1].out_features
    
    # Create a new model with same random seed for reproducibility
    global_model = NeuralNetwork(input_dim, output_dim, seed=42)
    global_model.to(device)
    
    # Simplified parameter averaging
    with torch.no_grad():
        # Initialize dict to store parameter sums
        param_dict = {name: torch.zeros_like(param) for name, param in global_model.named_parameters()}
        
        # Sum parameters from all models
        for model in models:
            if hasattr(model, '_module'):
                # For Opacus-modified models
                for name, param in model.named_parameters():
                    if name.startswith('_module.'):
                        clean_name = name[8:]  # Remove '_module.' prefix
                        param_dict[clean_name] += param.data
            else:
                # For regular models
                for name, param in model.named_parameters():
                    param_dict[name] += param.data
        
        # Average and update global model parameters
        for name, param in global_model.named_parameters():
            param.data = param_dict[name] / len(models)
    
    return global_model



def train_model(
    model: NeuralNetwork, 
    train_loader: DataLoader, 
    val_loader: DataLoader,
    epochs: int, 
    learning_rate: float, 
    noise_multiplier: Optional[float] = None,
    max_grad_norm: float = 1.0,
    device: str = "cpu",
    client_id: str = "Unknown",
    logger: Optional[logging.Logger] = None,
    is_dp_experiment: bool = False  # Flag to indicate if this is a DP experiment
) -> Tuple[NeuralNetwork, Dict[str, Any]]:
    """
    Train a model with or without differential privacy.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Number of training epochs
        learning_rate: Learning rate
        noise_multiplier: Noise multiplier for DP (None = no DP)
        max_grad_norm: Maximum gradient norm for DP
        device: Device to use for training
        client_id: Identifier for the client (for logging)
        logger: Optional logger instance
        is_dp_experiment: Flag to indicate if this is a DP experiment
        
    Returns:
        Trained model and training history
    """
    # Initialize optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    log = logger or logging.getLogger(__name__)
    
    # Initialize history with basic metrics
    history = {
        "loss": [], 
        "accuracy": [], 
        "val_loss": [], 
        "val_accuracy": []
    }
    
    # Only add DP-specific metrics for DP experiments
    if is_dp_experiment or noise_multiplier is not None:
        history.update({
            "avg_grad_norm": [],
            "noise_norms": [],
            "snr": []
        })
    
    privacy_engine = None
    final_epsilon = None
    
    # Setup differential privacy if needed
    if noise_multiplier is not None:
        try:
            noise_multiplier = float(noise_multiplier)
            
            history["epsilons"] = []
            privacy_engine = PrivacyEngine()
            
            # Make delta dataset-dependent for stronger privacy guarantees
            # Delta should be less than 1/N for meaningful DP
            dataset_size = len(train_loader.dataset)
            delta = min(1e-5, 1.0 / (10 * dataset_size))  # Use 1/10N for added security
            
            # Initialize privacy engine
            model, optimizer, train_loader = privacy_engine.make_private(
                module=model,
                optimizer=optimizer,
                data_loader=train_loader,
                noise_multiplier=noise_multiplier,
                max_grad_norm=max_grad_norm,
                clipping="flat",
                poisson_sampling=True
            )
            
            # Simple log for DP setup
            log.info(f"{client_id} - DP setup: noise={noise_multiplier}, max_grad_norm={max_grad_norm}")
            
        except Exception as e:
            log.error(f"Error applying DP: {e}")
            log.error(f"DP parameters: noise_multiplier={noise_multiplier}, max_grad_norm={max_grad_norm}")
            raise
    
    # Training loop
    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        grad_norms = []
        noise_norms = []
        
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Track gradient norms for DP analysis - without redundant clipping
            if noise_multiplier is not None:
                # Calculate gradient norm without redundant clipping (Opacus already handles this)
                grad_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        grad_norm += p.grad.norm(2).item() ** 2
                grad_norm = grad_norm ** 0.5
                grad_norms.append(grad_norm)
                
                # Calculate noise norm for signal-to-noise ratio
                noise_std = noise_multiplier * max_grad_norm
                noise_norm = noise_std * np.sqrt(sum(p.numel() for p in model.parameters() if p.requires_grad))
                noise_norms.append(noise_norm)
            
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        # Calculate metrics
        avg_loss = running_loss / total
        accuracy = correct / total
        
        # Calculate metrics
        if is_dp_experiment or noise_multiplier is not None:
            # Calculate DP-specific metrics
            avg_grad_norm = np.mean(grad_norms) if grad_norms else 0.0
            avg_noise_norm = np.mean(noise_norms) if noise_norms else 0.0
            snr = avg_grad_norm / avg_noise_norm if avg_noise_norm > 0 else float('inf')
        
        # Evaluate on validation set (not part of training)
        val_loss, val_accuracy = evaluate_model(model, val_loader, criterion, device)
        
        end_time = time.time()
        epoch_time = end_time - start_time
        
        # Update history with basic metrics
        history["loss"].append(avg_loss)
        history["accuracy"].append(accuracy)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)
        
        # Only add DP metrics for DP experiments
        if is_dp_experiment or noise_multiplier is not None:
            history["avg_grad_norm"].append(avg_grad_norm)
            history["noise_norms"].append(avg_noise_norm)
            history["snr"].append(snr)
        
        # Prepare a unified log message with common metrics
        log_message = (f"{client_id} - Epoch {epoch+1}/{epochs} | "
                      f"Loss: {avg_loss:.4f} | Acc: {accuracy:.4f} | "
                      f"Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.4f}")
        
        # Add gradient metrics to log if DP is enabled
        if is_dp_experiment or noise_multiplier is not None:
            log_message += f" | Grad Norm: {avg_grad_norm:.4f} | Noise: {avg_noise_norm:.4f} | SNR: {snr:.4f}"
        
        # Calculate and display epsilon - simplified logic
        if noise_multiplier is not None and privacy_engine is not None:
            epsilon = privacy_engine.get_epsilon(delta=delta)
            history["epsilons"].append(float(epsilon))
            log_message += f" | ε: {epsilon:.2f} (δ={delta:.1e})"
            # Save for final budget calculation
            final_epsilon = float(epsilon)
        
        # Always include timing information
        log_message += f" | Time: {epoch_time:.2f}s"
        
        # Log the unified message
        log.info(log_message)
    
    # Final statement about privacy budget - simplified
    if noise_multiplier is not None and privacy_engine is not None and final_epsilon is not None:
        log.info(f"{client_id} - Final Privacy Budget: ε = {final_epsilon:.4f} (δ={delta:.1e})")
    
    return model, history


def evaluate_model(
    model: NeuralNetwork,
    data_loader: DataLoader,
    criterion: nn.Module = None,
    device: str = "cpu"
) -> Tuple[float, float]:
    """
    Evaluate a model on a dataset.
    
    Args:
        model: Model to evaluate
        data_loader: Data loader for evaluation
        criterion: Loss criterion (if None, use CrossEntropyLoss)
        device: Device to use for evaluation
        
    Returns:
        (loss, accuracy)
    """
    model.eval()
    
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return running_loss / total, correct / total
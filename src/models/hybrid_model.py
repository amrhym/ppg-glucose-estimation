"""Hybrid CNN-GRU model for glucose estimation from PPG signals."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class ModelConfig:
    """Configuration for hybrid CNN-GRU model."""
    
    # Input parameters
    input_length: int = 300  # 10s at 30Hz
    
    # CNN Branch A (small kernels)
    cnn_small_kernels: List[int] = None
    cnn_small_channels: List[int] = None
    
    # CNN Branch B (large kernels)
    cnn_large_kernels: List[int] = None
    cnn_large_channels: List[int] = None
    
    # GRU parameters
    gru_layers: int = 2
    gru_hidden: int = 128
    gru_bidirectional: bool = True
    
    # Dense layers
    dense_dims: List[int] = None
    
    # Regularization
    dropout: float = 0.5
    l2_weight: float = 0.01
    
    def __post_init__(self):
        if self.cnn_small_kernels is None:
            self.cnn_small_kernels = [3, 5]
        if self.cnn_small_channels is None:
            self.cnn_small_channels = [64, 128]
        if self.cnn_large_kernels is None:
            self.cnn_large_kernels = [11, 15]
        if self.cnn_large_channels is None:
            self.cnn_large_channels = [64, 128]
        if self.dense_dims is None:
            self.dense_dims = [256, 128, 64]


class CNNBranch(nn.Module):
    """1D CNN branch for feature extraction."""
    
    def __init__(
        self,
        kernels: List[int],
        channels: List[int],
        dropout: float = 0.5,
    ):
        super().__init__()
        
        layers = []
        in_channels = 1
        
        for i, (kernel_size, out_channels) in enumerate(zip(kernels, channels)):
            # Conv block
            layers.append(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    padding=kernel_size // 2,
                )
            )
            layers.append(nn.BatchNorm1d(out_channels))
            layers.append(nn.ReLU())
            
            # Pooling every other layer
            if i % 2 == 1:
                layers.append(nn.MaxPool1d(2))
            
            # Dropout
            if dropout > 0:
                layers.append(nn.Dropout1d(dropout))
            
            in_channels = out_channels
        
        self.conv_layers = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through CNN branch."""
        # x: (batch, length) -> (batch, 1, length)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        # Conv layers
        x = self.conv_layers(x)
        
        # Global pooling
        x = self.global_pool(x)
        
        # Flatten
        x = x.squeeze(-1)
        
        return x


class GRUBranch(nn.Module):
    """GRU branch for temporal modeling."""
    
    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.5,
        bidirectional: bool = True,
    ):
        super().__init__()
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )
        
        self.output_size = hidden_size * (2 if bidirectional else 1)
        
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through GRU branch."""
        # x: (batch, length) -> (batch, length, 1)
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        
        # GRU
        output, hidden = self.gru(x)
        
        # Use last output
        if self.gru.bidirectional:
            # Concatenate forward and backward final states
            hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            hidden = hidden[-1]
        
        return hidden


class HybridCNNGRU(nn.Module):
    """Hybrid CNN-GRU model for glucose estimation.
    
    Architecture:
        - Branch A: 1D-CNN with small kernels (fine morphology)
        - Branch B: 1D-CNN with large kernels (global shape)
        - Branch C: GRU layers (temporal dynamics)
        - Concatenate branches -> Dense layers -> Glucose output
        
    Examples:
        >>> config = ModelConfig(input_length=300)
        >>> model = HybridCNNGRU(config)
        >>> ppg = torch.randn(32, 300)  # batch of 32 windows
        >>> glucose = model(ppg)
    """
    
    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__()
        
        self.config = config or ModelConfig()
        
        # Branch A: Small kernel CNN
        self.cnn_small = CNNBranch(
            kernels=self.config.cnn_small_kernels,
            channels=self.config.cnn_small_channels,
            dropout=self.config.dropout,
        )
        
        # Branch B: Large kernel CNN
        self.cnn_large = CNNBranch(
            kernels=self.config.cnn_large_kernels,
            channels=self.config.cnn_large_channels,
            dropout=self.config.dropout,
        )
        
        # Branch C: GRU
        self.gru_branch = GRUBranch(
            input_size=1,
            hidden_size=self.config.gru_hidden,
            num_layers=self.config.gru_layers,
            dropout=self.config.dropout,
            bidirectional=self.config.gru_bidirectional,
        )
        
        # Calculate concatenated size
        concat_size = (
            self.config.cnn_small_channels[-1]
            + self.config.cnn_large_channels[-1]
            + self.gru_branch.output_size
        )
        
        # Dense layers
        dense_layers = []
        in_features = concat_size
        
        for out_features in self.config.dense_dims:
            dense_layers.append(nn.Linear(in_features, out_features))
            dense_layers.append(nn.ReLU())
            if self.config.dropout > 0:
                dense_layers.append(nn.Dropout(self.config.dropout))
            in_features = out_features
        
        # Output layer
        dense_layers.append(nn.Linear(in_features, 1))
        
        self.dense = nn.Sequential(*dense_layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through model.
        
        Args:
            x: Input PPG windows of shape (batch, length)
            
        Returns:
            Glucose predictions of shape (batch, 1)
        """
        # Branch A: Small kernel CNN
        branch_a = self.cnn_small(x)
        
        # Branch B: Large kernel CNN
        branch_b = self.cnn_large(x)
        
        # Branch C: GRU
        branch_c = self.gru_branch(x)
        
        # Concatenate branches
        combined = torch.cat([branch_a, branch_b, branch_c], dim=1)
        
        # Dense layers
        output = self.dense(combined)
        
        return output
    
    def get_num_parameters(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def get_num_trainable_parameters(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
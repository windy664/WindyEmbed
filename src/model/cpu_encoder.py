"""
WindyEmbed CPU-Optimized Encoder

Designed for:
- E5 Xeon (no GPU)
- <5ms inference on CPU
- Minimal memory usage
- High quality embeddings

Architecture options:
1. TextCNN - Fastest on CPU (recommended)
2. BiLSTM - Good balance of speed and quality
3. MLP - Simplest, fastest

NO Transformer - too slow on CPU!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List
from dataclasses import dataclass


@dataclass
class CPUEncoderConfig:
    """Configuration for CPU-optimized encoder."""

    vocab_size: int = 32000
    embedding_dim: int = 128
    hidden_dim: int = 256
    output_dim: int = 256
    max_seq_len: int = 128
    dropout: float = 0.1
    architecture: str = "cnn"  # "cnn", "bilstm", "mlp"

    # CNN specific
    kernel_sizes: List[int] = None  # [3, 4, 5]
    num_filters: int = 128

    # BiLSTM specific
    num_layers: int = 2
    bidirectional: bool = True

    def __post_init__(self):
        if self.kernel_sizes is None:
            self.kernel_sizes = [3, 4, 5]


class TextCNNEncoder(nn.Module):
    """
    TextCNN Encoder - Fastest on CPU!

    Uses multiple 1D convolutions with different kernel sizes
    to capture n-gram features. Very CPU-friendly because:
    - Convolution is parallelizable
    - No sequential dependency
    - Simple matrix operations

    Expected performance:
    - E5-2680 v4: ~2ms per embedding
    - i7-1185G7: ~3ms per embedding
    """

    def __init__(self, config: CPUEncoderConfig):
        super().__init__()
        self.config = config

        # Embedding layer
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim, padding_idx=0)

        # Multiple 1D convolutions with different kernel sizes
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=config.embedding_dim,
                out_channels=config.num_filters,
                kernel_size=k,
                padding=k // 2  # Same padding
            )
            for k in config.kernel_sizes
        ])

        # Output projection
        total_filters = config.num_filters * len(config.kernel_sizes)
        self.projection = nn.Sequential(
            nn.Linear(total_filters, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.output_dim),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len] (1=valid, 0=padding)

        Returns:
            embeddings: [batch_size, output_dim]
        """
        # Embedding: [batch, seq_len, embed_dim]
        x = self.embedding(input_ids)

        # Apply attention mask to zero out padding
        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1).float()

        # Transpose for Conv1d: [batch, embed_dim, seq_len]
        x = x.transpose(1, 2)

        # Apply multiple convolutions
        conv_outputs = []
        for conv in self.convs:
            # Conv + ReLU + MaxPool
            c = conv(x)  # [batch, num_filters, seq_len]
            c = F.relu(c)
            c = F.adaptive_max_pool1d(c, 1).squeeze(-1)  # [batch, num_filters]
            conv_outputs.append(c)

        # Concatenate all conv outputs
        x = torch.cat(conv_outputs, dim=1)  # [batch, total_filters]

        # Project to output dimension
        x = self.projection(x)

        # Normalize
        x = F.normalize(x, p=2, dim=-1)

        return x


class BiLSTMEncoder(nn.Module):
    """
    BiLSTM Encoder - Good balance of speed and quality.

    Uses bidirectional LSTM to capture sequential patterns.
    Reasonably fast on CPU, better at long-range dependencies than CNN.

    Expected performance:
    - E5-2680 v4: ~5ms per embedding
    - i7-1185G7: ~8ms per embedding
    """

    def __init__(self, config: CPUEncoderConfig):
        super().__init__()
        self.config = config

        # Embedding layer
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim, padding_idx=0)

        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=config.embedding_dim,
            hidden_size=config.hidden_dim // 2,  # Divide by 2 for bidirectional
            num_layers=config.num_layers,
            batch_first=True,
            bidirectional=config.bidirectional,
            dropout=config.dropout if config.num_layers > 1 else 0,
        )

        # Output projection
        self.projection = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.output_dim),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]

        Returns:
            embeddings: [batch_size, output_dim]
        """
        # Embedding
        x = self.embedding(input_ids)

        # Pack padded sequences for efficiency
        if attention_mask is not None:
            lengths = attention_mask.sum(dim=1).cpu()
            x = nn.utils.rnn.pack_padded_sequence(
                x, lengths, batch_first=True, enforce_sorted=False
            )

        # BiLSTM
        lstm_out, (hidden, cell) = self.lstm(x)

        # Unpack if needed
        if attention_mask is not None:
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)

        # Use last hidden state (concatenate forward and backward)
        # hidden shape: [num_layers * num_directions, batch, hidden_dim//2]
        forward_hidden = hidden[-2]  # Last forward layer
        backward_hidden = hidden[-1]  # Last backward layer
        combined = torch.cat([forward_hidden, backward_hidden], dim=1)  # [batch, hidden_dim]

        # Project
        x = self.projection(combined)

        # Normalize
        x = F.normalize(x, p=2, dim=-1)

        return x


class MLPEncoder(nn.Module):
    """
    MLP Encoder - Simplest and fastest.

    Pure feed-forward network with bag-of-words.
    Extremely fast on CPU, but may lose word order information.

    Expected performance:
    - E5-2680 v4: ~1ms per embedding
    - i7-1185G7: ~2ms per embedding
    """

    def __init__(self, config: CPUEncoderConfig):
        super().__init__()
        self.config = config

        # Embedding layer
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim, padding_idx=0)

        # MLP layers
        self.mlp = nn.Sequential(
            nn.Linear(config.embedding_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.output_dim),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]

        Returns:
            embeddings: [batch_size, output_dim]
        """
        # Embedding
        x = self.embedding(input_ids)  # [batch, seq_len, embed_dim]

        # Apply mask
        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1).float()

        # Average pooling (bag-of-words)
        if attention_mask is not None:
            x = x.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True).clamp(min=1)
        else:
            x = x.mean(dim=1)

        # MLP
        x = self.mlp(x)

        # Normalize
        x = F.normalize(x, p=2, dim=-1)

        return x


def create_encoder(config: CPUEncoderConfig) -> nn.Module:
    """Create encoder based on architecture choice."""
    if config.architecture == "cnn":
        return TextCNNEncoder(config)
    elif config.architecture == "bilstm":
        return BiLSTMEncoder(config)
    elif config.architecture == "mlp":
        return MLPEncoder(config)
    else:
        raise ValueError(f"Unknown architecture: {config.architecture}")


class WindyEmbedModel(nn.Module):
    """
    Complete WindyEmbed model with projection head for training.
    """

    def __init__(self, config: CPUEncoderConfig, projection_dim: int = 128):
        super().__init__()
        self.config = config
        self.encoder = create_encoder(config)

        # Projection head for contrastive learning
        self.projection = nn.Sequential(
            nn.Linear(config.output_dim, config.output_dim),
            nn.ReLU(),
            nn.Linear(config.output_dim, projection_dim),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple:
        """
        Forward pass with projection.

        Returns:
            embeddings: [batch_size, output_dim]
            projected: [batch_size, projection_dim]
        """
        embeddings = self.encoder(input_ids, attention_mask)
        projected = self.projection(embeddings)
        projected = F.normalize(projected, p=2, dim=-1)
        return embeddings, projected

    def get_encoder(self) -> nn.Module:
        """Get encoder without projection head."""
        return self.encoder
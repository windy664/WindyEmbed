"""
MiniEmbed Encoder - Lightweight Transformer for Embeddings

A tiny BERT-like model optimized for:
- Small size (10M-50M params)
- Fast inference on CPU
- High quality embeddings
- Easy ONNX export

Architecture:
- 6 Transformer layers
- 256 hidden dimension
- 4 attention heads
- Mean pooling (not CLS token)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, Tuple
from dataclasses import dataclass


@dataclass
class EncoderConfig:
    """Configuration for MiniEmbed encoder."""

    vocab_size: int = 32000
    hidden_dim: int = 256
    num_layers: int = 6
    num_heads: int = 4
    ffn_ratio: float = 4.0
    dropout: float = 0.1
    max_seq_len: int = 128
    pooling: str = "mean"  # "mean" or "cls"

    def __post_init__(self):
        assert self.hidden_dim % self.num_heads == 0, \
            "hidden_dim must be divisible by num_heads"


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention."""

    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_dim // config.num_heads
        self.scale = math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.k_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.v_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.out_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        # Apply mask if provided
        if mask is not None:
            # mask: [batch, seq_len] where 1 = valid, 0 = padding
            mask = mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq_len]
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        # Softmax
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Weighted sum
        attn_output = torch.matmul(attn_weights, v)

        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, -1)
        output = self.out_proj(attn_output)

        return output


class TransformerLayer(nn.Module):
    """Single Transformer layer with pre-norm."""

    def __init__(self, config: EncoderConfig):
        super().__init__()

        # Self-attention
        self.norm1 = nn.LayerNorm(config.hidden_dim)
        self.attention = MultiHeadAttention(config)

        # Feed-forward
        self.norm2 = nn.LayerNorm(config.hidden_dim)
        ffn_dim = int(config.hidden_dim * config.ffn_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(ffn_dim, config.hidden_dim),
            nn.Dropout(config.dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self-attention with residual
        residual = x
        x = self.norm1(x)
        x = self.attention(x, mask)
        x = residual + x

        # FFN with residual
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x

        return x


class MiniEmbedEncoder(nn.Module):
    """
    MiniEmbed Encoder

    A lightweight Transformer encoder for producing text embeddings.

    Args:
        config: EncoderConfig instance
    """

    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.config = config

        # Token embedding
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_dim)

        # Position embedding
        self.pos_embedding = nn.Embedding(config.max_seq_len, config.hidden_dim)

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(config) for _ in range(config.num_layers)
        ])

        # Final layer norm
        self.norm = nn.LayerNorm(config.hidden_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

        # Special init for embeddings
        nn.init.normal_(self.embedding.weight, mean=0, std=0.02)
        nn.init.normal_(self.pos_embedding.weight, mean=0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Mask for valid tokens [batch_size, seq_len]

        Returns:
            Token embeddings [batch_size, seq_len, hidden_dim]
        """
        batch_size, seq_len = input_ids.shape

        # Get device
        device = input_ids.device

        # Token embedding
        x = self.embedding(input_ids)

        # Add positional embedding
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        x = x + self.pos_embedding(positions)

        # Apply dropout
        x = F.dropout(x, p=self.config.dropout, training=self.training)

        # Apply Transformer layers
        for layer in self.layers:
            x = layer(x, attention_mask)

        # Final layer norm
        x = self.norm(x)

        return x

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Get sentence embeddings.

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Mask for valid tokens [batch_size, seq_len]

        Returns:
            Sentence embeddings [batch_size, hidden_dim]
        """
        # Get token embeddings
        token_embeddings = self.forward(input_ids, attention_mask)

        if self.config.pooling == "mean":
            # Mean pooling
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)

            # Expand mask to match embedding dimensions
            mask = attention_mask.unsqueeze(-1).float()  # [batch, seq_len, 1]

            # Sum embeddings where mask is 1
            sum_embeddings = torch.sum(token_embeddings * mask, dim=1)

            # Count valid tokens
            count = torch.sum(mask, dim=1).clamp(min=1e-9)

            # Average
            embeddings = sum_embeddings / count

        elif self.config.pooling == "cls":
            # CLS token pooling
            embeddings = token_embeddings[:, 0, :]

        else:
            raise ValueError(f"Unknown pooling: {self.config.pooling}")

        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=-1)

        return embeddings

    def get_num_params(self) -> Dict[str, int]:
        """Get parameter count statistics."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        embedding = sum(p.numel() for p in self.embedding.parameters())
        pos_embedding = sum(p.numel() for p in self.pos_embedding.parameters())

        return {
            'total': total,
            'trainable': trainable,
            'embedding': embedding + pos_embedding,
            'encoder': total - embedding - pos_embedding,
        }


class MiniEmbedModel(nn.Module):
    """
    Complete MiniEmbed model with projection head for training.

    Adds a projection head on top of the encoder for contrastive learning.
    This head is removed during inference.
    """

    def __init__(self, config: EncoderConfig, projection_dim: int = 128):
        super().__init__()
        self.config = config
        self.encoder = MiniEmbedEncoder(config)

        # Projection head for contrastive learning
        self.projection = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, projection_dim),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with projection.

        Returns:
            embeddings: Sentence embeddings [batch_size, hidden_dim]
            projected: Projected embeddings for contrastive loss [batch_size, projection_dim]
        """
        # Get sentence embeddings
        embeddings = self.encoder.encode(input_ids, attention_mask)

        # Project for contrastive learning
        projected = self.projection(embeddings)
        projected = F.normalize(projected, p=2, dim=-1)

        return embeddings, projected

    def get_encoder(self) -> MiniEmbedEncoder:
        """Get the encoder without projection head (for inference)."""
        return self.encoder
"""
ONNX Export for MiniEmbed

Exports the trained model to ONNX format for:
- Java inference (ONNX Runtime)
- Cross-platform deployment
- No Python dependency at inference time
"""

import torch
import os
import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def export_to_onnx(
    model,
    output_path: str,
    max_seq_len: int = 128,
    opset_version: int = 14,
):
    """
    Export MiniEmbed model to ONNX format.

    Args:
        model: MiniEmbedEncoder instance
        output_path: Path to save ONNX model
        max_seq_len: Maximum sequence length
        opset_version: ONNX opset version
    """
    model.eval()

    # Create dummy inputs
    batch_size = 1
    dummy_input_ids = torch.randint(0, 1000, (batch_size, max_seq_len))
    dummy_attention_mask = torch.ones(batch_size, max_seq_len, dtype=torch.long)

    # Export
    logger.info(f"Exporting model to {output_path}...")

    torch.onnx.export(
        model,
        (dummy_input_ids, dummy_attention_mask),
        output_path,
        opset_version=opset_version,
        input_names=['input_ids', 'attention_mask'],
        output_names=['embeddings'],
        dynamic_axes={
            'input_ids': {0: 'batch_size', 1: 'sequence_length'},
            'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
            'embeddings': {0: 'batch_size'},
        },
    )

    logger.info(f"Model exported to {output_path}")

    # Get file size
    file_size_mb = os.path.getsize(output_path) / 1e6
    logger.info(f"Model size: {file_size_mb:.2f} MB")


def export_tokenizer_config(
    tokenizer,
    output_dir: str,
):
    """Export tokenizer configuration for Java."""
    config = {
        'vocab_size': len(tokenizer),
        'max_seq_len': 128,
        'do_lower_case': True,
    }

    # Save vocab
    vocab_path = os.path.join(output_dir, 'vocab.txt')
    with open(vocab_path, 'w', encoding='utf-8') as f:
        for token, idx in sorted(tokenizer.vocab.items(), key=lambda x: x[1]):
            f.write(f"{token}\n")

    # Save config
    config_path = os.path.join(output_dir, 'tokenizer_config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)

    logger.info(f"Tokenizer config exported to {output_dir}")


def export_model_config(
    config,
    output_dir: str,
):
    """Export model configuration for Java."""
    model_config = {
        'vocab_size': config.vocab_size,
        'hidden_dim': config.hidden_dim,
        'num_layers': config.num_layers,
        'num_heads': config.num_heads,
        'max_seq_len': config.max_seq_len,
        'pooling': config.pooling,
    }

    config_path = os.path.join(output_dir, 'model_config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(model_config, f, indent=2)

    logger.info(f"Model config exported to {output_dir}")
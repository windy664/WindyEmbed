#!/usr/bin/env python3
"""
Train WindyEmbed Model

Trains a lightweight embedding model using contrastive learning.

Usage:
    python scripts/train.py --config configs/windyembed-10m.yaml
    python scripts/train.py --data data/pairs.json --epochs 5
"""

import argparse
import json
import os
import sys
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import logging
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.encoder import MiniEmbedModel, MiniEmbedEncoder, EncoderConfig
from src.training.contrastive import ContrastiveLoss, MultipleNegativesRankingLoss

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


class TextPairDataset(Dataset):
    """Dataset of text pairs for contrastive learning."""

    def __init__(self, data_path: str, tokenizer, max_len: int = 128):
        self.tokenizer = tokenizer
        self.max_len = max_len

        # Load data
        with open(data_path, 'r', encoding='utf-8') as f:
            self.pairs = json.load(f)

        logger.info(f"Loaded {len(self.pairs)} text pairs")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]

        # Tokenize both texts
        ids_a = self.tokenizer.encode(pair['text_a'], self.max_len)
        ids_b = self.tokenizer.encode(pair['text_b'], self.max_len)

        # Create attention masks
        mask_a = [1 if t != 0 else 0 for t in ids_a]
        mask_b = [1 if t != 0 else 0 for t in ids_b]

        return {
            'input_ids_a': torch.tensor(ids_a, dtype=torch.long),
            'attention_mask_a': torch.tensor(mask_a, dtype=torch.long),
            'input_ids_b': torch.tensor(ids_b, dtype=torch.long),
            'attention_mask_b': torch.tensor(mask_b, dtype=torch.long),
        }


class SimpleTokenizer:
    """Simple character-level tokenizer."""

    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size

    def encode(self, text: str, max_len: int) -> list:
        # Simple byte-level encoding
        bytes_text = text.lower().encode('utf-8')
        ids = [b + 1 for b in bytes_text]  # +1 to avoid 0 (padding)
        ids = ids[:max_len]
        ids = ids + [0] * (max_len - len(ids))
        return ids


def create_sample_data(output_path: str):
    """Create sample training data."""
    pairs = [
        {"text_a": "What is machine learning?", "text_b": "Machine learning is a subset of AI"},
        {"text_a": "How does Python work?", "text_b": "Python is an interpreted programming language"},
        {"text_a": "What is a neural network?", "text_b": "A neural network is inspired by the brain"},
        {"text_a": "Explain deep learning", "text_b": "Deep learning uses multiple layers of neurons"},
        {"text_a": "What is NLP?", "text_b": "Natural language processing deals with text"},
        {"text_a": "How to learn programming?", "text_b": "Start with Python and practice daily"},
        {"text_a": "What is Java?", "text_b": "Java is a popular object-oriented language"},
        {"text_a": "What is an API?", "text_b": "API stands for application programming interface"},
        {"text_a": "Explain databases", "text_b": "Databases store and organize structured data"},
        {"text_a": "What is cloud computing?", "text_b": "Cloud computing provides on-demand resources"},
        # Add more pairs for training
        {"text_a": "The cat sat on the mat", "text_b": "A feline rested on the rug"},
        {"text_a": "The sun is bright today", "text_b": "It is a sunny day"},
        {"text_a": "I love programming", "text_b": "Coding is my passion"},
        {"text_a": "The movie was great", "text_b": "I enjoyed the film"},
        {"text_a": "The food was delicious", "text_b": "The meal tasted wonderful"},
    ]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(pairs, f, indent=2)

    logger.info(f"Created {len(pairs)} sample pairs at {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Train MiniEmbed model")
    parser.add_argument("--config", type=str, default="configs/miniembed-10m.yaml",
                       help="Config file")
    parser.add_argument("--data", type=str, default=None,
                       help="Training data (JSON file with text pairs)")
    parser.add_argument("--epochs", type=int, default=5,
                       help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=8,
                       help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4,
                       help="Learning rate")
    parser.add_argument("--output", type=str, default="checkpoints/embed",
                       help="Output directory")
    args = parser.parse_args()

    # Create sample data if not provided
    if args.data is None:
        args.data = "data/sample_pairs.json"
        create_sample_data(args.data)

    # Load config
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {}

    # Create tokenizer
    tokenizer = SimpleTokenizer(vocab_size=config.get('model', {}).get('vocab_size', 32000))

    # Create dataset
    dataset = TextPairDataset(args.data, tokenizer, max_len=128)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Create model
    model_config = EncoderConfig(
        vocab_size=config.get('model', {}).get('vocab_size', 32000),
        hidden_dim=config.get('model', {}).get('hidden_dim', 256),
        num_layers=config.get('model', {}).get('num_layers', 6),
        num_heads=config.get('model', {}).get('num_heads', 4),
        max_seq_len=128,
    )

    model = MiniEmbedModel(model_config, projection_dim=128)

    # Print model info
    params = model.encoder.get_num_params()
    logger.info(f"Model: {params['total']:,} parameters ({params['total']/1e6:.1f}M)")

    # Training setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    criterion = ContrastiveLoss(temperature=0.05)

    # Training loop
    logger.info("Starting training...")
    start_time = time.time()

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        num_batches = 0

        for batch in dataloader:
            # Move to device
            ids_a = batch['input_ids_a'].to(device)
            mask_a = batch['attention_mask_a'].to(device)
            ids_b = batch['input_ids_b'].to(device)
            mask_b = batch['attention_mask_b'].to(device)

            # Forward pass
            emb_a, proj_a = model(ids_a, mask_a)
            emb_b, proj_b = model(ids_b, mask_b)

            # Compute loss
            loss = criterion(proj_a, proj_b)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        # Log progress
        avg_loss = total_loss / num_batches
        elapsed = time.time() - start_time
        logger.info(f"Epoch {epoch+1}/{args.epochs}: loss={avg_loss:.4f}, time={elapsed:.1f}s")

    # Save model
    os.makedirs(args.output, exist_ok=True)
    torch.save({
        'model_state_dict': model.encoder.state_dict(),
        'config': model_config,
    }, os.path.join(args.output, 'model.pt'))

    logger.info(f"Model saved to {args.output}")

    # Export to ONNX
    logger.info("Exporting to ONNX...")
    from src.export.onnx_export import export_to_onnx, export_model_config

    onnx_path = os.path.join(args.output, 'model.onnx')
    export_to_onnx(model.encoder, onnx_path)
    export_model_config(model_config, args.output)

    logger.info("="*60)
    logger.info("Training Complete!")
    logger.info("="*60)
    logger.info(f"Model: {params['total']:,} parameters")
    logger.info(f"ONNX: {onnx_path}")
    logger.info(f"Use in Java: new MiniEmbed(\"{onnx_path}\")")
    logger.info("="*60)


if __name__ == "__main__":
    main()
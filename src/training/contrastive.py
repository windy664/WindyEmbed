"""
Contrastive Learning for Embedding Models

Trains the model to:
- Pull similar texts closer in embedding space
- Push different texts apart in embedding space

This is how we learn high-quality embeddings without massive datasets.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for training embedding models.

    Uses InfoNCE (Noise Contrastive Estimation) loss,
    which is the standard for contrastive learning.

    For a batch of (anchor, positive) pairs:
    - Pull anchor and positive together
    - Push anchor away from all other positives (negatives)
    """

    def __init__(self, temperature: float = 0.05):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        embeddings_a: torch.Tensor,
        embeddings_b: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute contrastive loss.

        Args:
            embeddings_a: Embeddings of first texts [batch_size, dim]
            embeddings_b: Embeddings of second texts [batch_size, dim]

        Returns:
            Scalar loss
        """
        # Normalize embeddings
        embeddings_a = F.normalize(embeddings_a, p=2, dim=-1)
        embeddings_b = F.normalize(embeddings_b, p=2, dim=-1)

        # Compute similarity matrix
        # sim[i][j] = similarity(embeddings_a[i], embeddings_b[j])
        similarity = torch.matmul(embeddings_a, embeddings_b.T) / self.temperature

        # Labels: diagonal is positive (i matches i)
        batch_size = embeddings_a.shape[0]
        labels = torch.arange(batch_size, device=embeddings_a.device)

        # Cross-entropy loss
        loss_a2b = F.cross_entropy(similarity, labels)
        loss_b2a = F.cross_entropy(similarity.T, labels)

        # Average both directions
        loss = (loss_a2b + loss_b2a) / 2

        return loss


class MultipleNegativesRankingLoss(nn.Module):
    """
    Multiple Negatives Ranking Loss (MNR Loss)

    A more efficient contrastive loss that uses all non-paired
    samples as negatives (in-batch negatives).

    This is the standard loss for training sentence embeddings.
    """

    def __init__(self, temperature: float = 0.05):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        embeddings_a: torch.Tensor,
        embeddings_b: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute MNR loss.

        Args:
            embeddings_a: Embeddings of anchors [batch_size, dim]
            embeddings_b: Embeddings of positives [batch_size, dim]

        Returns:
            Scalar loss
        """
        # Normalize
        embeddings_a = F.normalize(embeddings_a, p=2, dim=-1)
        embeddings_b = F.normalize(embeddings_b, p=2, dim=-1)

        # Similarity matrix
        similarity = torch.matmul(embeddings_a, embeddings_b.T) / self.temperature

        # Labels: each anchor matches its positive
        labels = torch.arange(embeddings_a.shape[0], device=embeddings_a.device)

        # Cross-entropy (each anchor should match its positive)
        loss = F.cross_entropy(similarity, labels)

        return loss


class TripletLoss(nn.Module):
    """
    Triplet Loss

    Trains with (anchor, positive, negative) triplets.
    Anchor should be closer to positive than to negative.
    """

    def __init__(self, margin: float = 0.2):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute triplet loss.

        Args:
            anchor: Anchor embeddings [batch_size, dim]
            positive: Positive embeddings [batch_size, dim]
            negative: Negative embeddings [batch_size, dim]

        Returns:
            Scalar loss
        """
        # Normalize
        anchor = F.normalize(anchor, p=2, dim=-1)
        positive = F.normalize(positive, p=2, dim=-1)
        negative = F.normalize(negative, p=2, dim=-1)

        # Compute distances
        pos_dist = 1 - torch.sum(anchor * positive, dim=-1)
        neg_dist = 1 - torch.sum(anchor * negative, dim=-1)

        # Triplet loss with margin
        loss = F.relu(pos_dist - neg_dist + self.margin)

        return loss.mean()


class CosineSimilarityLoss(nn.Module):
    """
    Cosine Similarity Loss

    Simple loss that maximizes cosine similarity for positive pairs
    and minimizes it for negative pairs.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        embeddings_a: torch.Tensor,
        embeddings_b: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute cosine similarity loss.

        Args:
            embeddings_a: First embeddings [batch_size, dim]
            embeddings_b: Second embeddings [batch_size, dim]
            labels: 1 for similar, 0 for dissimilar [batch_size]

        Returns:
            Scalar loss
        """
        # Normalize
        embeddings_a = F.normalize(embeddings_a, p=2, dim=-1)
        embeddings_b = F.normalize(embeddings_b, p=2, dim=-1)

        # Cosine similarity
        similarity = torch.sum(embeddings_a * embeddings_b, dim=-1)

        # MSE loss against labels
        loss = F.mse_loss(similarity, labels)

        return loss
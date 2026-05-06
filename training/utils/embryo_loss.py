"""
EmbryoLoss — CE only with Soft Label support (Ordinal Regression)

  L = CrossEntropy(logits, soft_targets)

  - Standard Soft Labels: Gaussian distribution over class indices to teach the model
    the sequential nature of embryo stages (ordinality).
  - Normal CrossEntropy: Used if ordinal_sigma = 0.

Usage:
    from utils.embryo_loss import EmbryoLoss
    loss_fn = EmbryoLoss(num_classes=7, ordinal_sigma=0.5)
    loss = loss_fn(output, target)
"""

import torch
import torch.nn as nn
from torch import Tensor
import math


class EmbryoLoss(nn.Module):
    """
    CE loss with Soft Labels for Ordinal Stages and optional per-class weights.

    Args:
        num_classes:     Number of stage classes (7 or 9)
        label_smoothing: For CE loss (default: 0.1, used if ordinal_sigma=0)
        class_weights:   Per-class weight tensor (default: None)
        ordinal_sigma:   Sigma for Gaussian soft labels (default: 0.0 - disabled)
    """
    def __init__(self, num_classes: int = 7,
                 label_smoothing: float = 0.1,
                 class_weights: torch.Tensor = None,
                 ordinal_sigma: float = 0.0,
                 **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.ordinal_sigma = ordinal_sigma
        
        # If ordinal_sigma > 0, we don't use label_smoothing in CrossEntropy 
        # because the soft labels already provide smoothing.
        ce_smoothing = label_smoothing if ordinal_sigma == 0 else 0.0
        self.ce = nn.CrossEntropyLoss(
            label_smoothing=ce_smoothing,
            weight=class_weights,
        )
        self.last_losses: dict = {'ce': 0.}

    def _generate_soft_labels(self, targets: Tensor) -> Tensor:
        """
        Generate Gaussian soft labels for ordinal stages.
        Input:  (B,) long targets
        Output: (B, num_classes) soft labels
        """
        B = targets.size(0)
        device = targets.device
        
        # indices: (1, num_classes)
        indices = torch.arange(self.num_classes, device=device).unsqueeze(0).float()
        # targets: (B, 1)
        targets_f = targets.unsqueeze(1).float()
        
        # Gaussian distribution: exp(-(i-j)^2 / (2 * sigma^2))
        dist = torch.exp(-(indices - targets_f)**2 / (2 * self.ordinal_sigma**2))
        
        # Normalize to sum to 1.0 along class dimension
        soft_labels = dist / dist.sum(dim=1, keepdim=True)
        return soft_labels

    def forward(self, output, targets: Tensor, **kwargs) -> Tensor:
        # Support legacy tuple output (logits, count_pred) — just take logits
        if isinstance(output, (tuple, list)):
            logits = output[0]
        else:
            logits = output

        # Prepare targets
        if self.ordinal_sigma > 0:
            # Generate soft labels: (B,) -> (B, num_classes)
            soft_targets = self._generate_soft_labels(targets)
            # CrossEntropyLoss supports (N, C) soft targets since Torch 1.10+
            loss = self.ce(logits.float(), soft_targets)
        else:
            # Standard hard targets
            loss = self.ce(logits.float(), targets)

        self.last_losses = {'ce': loss.item()}
        return loss

    def extra_repr(self) -> str:
        return f"num_classes={self.num_classes}, ordinal_sigma={self.ordinal_sigma}"


# ---------------------------------------------------------------------------
# Aliases and Factory
# ---------------------------------------------------------------------------
EmbryoCompositionLoss = EmbryoLoss

def build_embryo_loss(num_classes: int = 7,
                      label_smoothing: float = 0.1,
                      class_weights: torch.Tensor = None,
                      ordinal_sigma: float = 0.0,
                      **kwargs) -> EmbryoLoss:
    """Factory. Silently ignores legacy kwargs."""
    return EmbryoLoss(
        num_classes=num_classes,
        label_smoothing=label_smoothing,
        class_weights=class_weights,
        ordinal_sigma=ordinal_sigma,
    )


if __name__ == "__main__":
    # Test 1: Standard CE
    loss_fn = EmbryoLoss(num_classes=7, ordinal_sigma=0.0)
    logits = torch.randn(2, 7)
    targets = torch.tensor([1, 4])
    loss = loss_fn(logits, targets)
    print(f"Standard CE: {loss.item():.4f}")

    # Test 2: Soft Labels (Ordinal)
    loss_fn_soft = EmbryoLoss(num_classes=7, ordinal_sigma=0.5)
    soft_targets = loss_fn_soft._generate_soft_labels(targets)
    print(f"Soft targets for labels {targets.tolist()}:")
    print(soft_targets)
    
    loss_soft = loss_fn_soft(logits, targets)
    print(f"Ordinal Soft Loss: {loss_soft.item():.4f}")
    print("PASS")

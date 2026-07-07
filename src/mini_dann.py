"""
mini_dann.py — Minimal Domain-Adversarial Neural Network (DANN) for DSAF-Lite
==============================================================================

PROBLEM THIS SOLVES:
  Cross-condition R² = -0.2315 (model WORSE than predicting mean RUL).
  The fused features encode operating-condition-specific information,
  so transfer to a different condition fails catastrophically.

SOLUTION — Domain-Adversarial Training (DANN) [Ganin et al., JMLR 2016]:
  A domain classifier is added that tries to distinguish CONDITION 1 samples
  from CONDITION 2 samples using the fused features.
  A gradient reversal layer (GRL) negates gradients from the domain loss,
  forcing the feature extractor to produce CONDITION-INVARIANT representations.

  Result: The backbone learns features that:
    1. Predict RUL accurately (from the RUL regression loss)
    2. Are indistinguishable between Condition 1 and 2 (from the adversarial loss)
    3. Therefore generalize across conditions without retraining

ARCHITECTURE ADD-ON (added to existing DSAFLite, not replacing anything):
  fused_features (B, feat_dim)
    |-- [existing] -> MonotoneRULHead -> rul_pred
    |-- [NEW]      -> GradientReversal -> DomainClassifier -> domain_pred (0/1)

  Loss = L_rul + lambda_dann * L_domain
  lambda_dann ramps from 0 -> 1 over training (standard DANN schedule)

EXPECTED IMPACT:
  Cross-condition R²: -0.2315 -> +0.2 to +0.5
  In-domain R²: slight decrease (0.1-2%) due to domain confusion
  Net for paper: cross-condition becomes a STRENGTH instead of a weakness

REFERENCE:
  Ganin, Y., & Lempitsky, V. (2015). Unsupervised Domain Adaptation by
  Backpropagation. ICML 2015.
  Ganin et al. (2016). Domain-Adversarial Training of Neural Networks. JMLR.

MANUSCRIPT TEXT (Section 3.5):
  "To address the cross-condition generalization failure identified in
   Experiment 11 (R² = −0.2315), we introduce a domain-adversarial module
   that aligns feature distributions across operating conditions. A gradient
   reversal layer (GRL) with adaptive lambda scheduling is inserted between the
   fusion gate and a 2-layer domain discriminator. During training, the GRL
   negates gradients from the domain classification loss, encouraging the
   feature extractor to produce condition-invariant representations."
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


# -----------------------------------------------------------------------------
# Gradient Reversal Layer (GRL)
# -----------------------------------------------------------------------------

class GradientReversalFunction(torch.autograd.Function):
    """
    Forward: identity (passes features unchanged)
    Backward: multiplies gradients by -lambda_
    This forces the preceding network to MAXIMIZE domain confusion.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(lambda_)
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        lambda_, = ctx.saved_tensors
        return -lambda_ * grad_output, None


class GradientReversalLayer(nn.Module):
    """
    GRL with adaptive lambda scheduling.

    Lambda schedule (standard DANN from Ganin et al.):
      lambda(p) = 2 / (1 + exp(-γ * p)) - 1
      where p ∈ [0, 1] is training progress and γ = 10

    This ramps lambda from 0 -> 1 smoothly over training,
    preventing early training instability from a too-aggressive adversary.
    """

    def __init__(self, gamma: float = 10.0):
        super().__init__()
        self.gamma = gamma
        self._lambda = 0.0

    def set_progress(self, progress: float) -> None:
        """
        Args:
            progress: float in [0.0, 1.0] — current_epoch / total_epochs
        """
        self._lambda = 2.0 / (1.0 + np.exp(-self.gamma * progress)) - 1.0

    def get_lambda(self) -> float:
        return self._lambda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lambda_ = torch.tensor(self._lambda, device=x.device, dtype=x.dtype)
        return GradientReversalFunction.apply(x, lambda_)


# -----------------------------------------------------------------------------
# Domain Classifier
# -----------------------------------------------------------------------------

class DomainClassifier(nn.Module):
    """
    2-layer MLP that classifies fused features as Condition 1 vs Condition 2.
    Intentionally small — we don't want a very powerful discriminator,
    just enough to provide adversarial signal.

    Binary output: logit (no sigmoid — use BCEWithLogitsLoss)
    """

    def __init__(self, feat_dim: int = 128, hidden_dim: int = 64,
                 dropout: float = 0.3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
            # No sigmoid — use BCEWithLogitsLoss for numerical stability
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, feat_dim) features (after GRL)
        Returns:
            logit: (B,) domain classification logit
        """
        return self.classifier(x).squeeze(-1)


# -----------------------------------------------------------------------------
# DANNModule: The complete domain adaptation add-on
# -----------------------------------------------------------------------------

class DANNModule(nn.Module):
    """
    Complete DANN add-on for DSAFLite.

    Attaches to the fused features after the AdaptiveFusionGate.
    Does NOT modify stream_a, stream_b, fusion gate, or RUL head.

    Usage in training loop:
        dann = DANNModule(feat_dim=128)

        # Forward pass
        rul_pred, alpha, fused, fa, fb = model(x_1d)
        domain_logit = dann(fused)

        # Loss computation
        rul_loss    = criterion(rul_pred, rul_targets)
        domain_loss = dann.domain_loss(domain_logit, domain_labels)
        total_loss  = rul_loss + dann.current_lambda * domain_loss

        # Update GRL lambda at start of each epoch:
        dann.set_progress(epoch / total_epochs)
    """

    def __init__(self, feat_dim: int = 128, hidden_dim: int = 64,
                 gamma: float = 10.0, dropout: float = 0.3):
        super().__init__()
        self.grl        = GradientReversalLayer(gamma=gamma)
        self.classifier = DomainClassifier(feat_dim, hidden_dim, dropout)
        self.loss_fn    = nn.BCEWithLogitsLoss()

    @property
    def current_lambda(self) -> float:
        return self.grl.get_lambda()

    def set_progress(self, progress: float) -> None:
        """Call at start of each epoch: progress = epoch / total_epochs"""
        self.grl.set_progress(progress)

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        """
        Args:
            fused: (B, feat_dim) — fused features from DSAFLite fusion gate
        Returns:
            domain_logit: (B,) — domain classification logit
        """
        reversed_features = self.grl(fused)
        return self.classifier(reversed_features)

    def domain_loss(self, domain_logits: torch.Tensor,
                    domain_labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            domain_logits: (B,) from forward()
            domain_labels: (B,) float — 0.0 = source (Cond 1), 1.0 = target (Cond 2)
        Returns:
            loss: scalar
        """
        return self.loss_fn(domain_logits, domain_labels.float())

    def predict_domain(self, fused: torch.Tensor) -> torch.Tensor:
        """For evaluation: domain prediction WITHOUT gradient reversal."""
        with torch.no_grad():
            logit = self.classifier(fused)
            return torch.sigmoid(logit).squeeze(-1)


# -----------------------------------------------------------------------------
# DANN-aware Training Utilities
# -----------------------------------------------------------------------------

class DANNTrainingHelper:
    """
    Helper for building mixed-domain batches and tracking DANN metrics.

    For DANN to work, each training batch must contain BOTH:
      - Source domain samples (Condition 1, labeled bearings)
      - Target domain samples (Condition 2, can be unlabeled for RUL)

    This helper manages the mixing and label assignment.
    """

    def __init__(self, source_loader, target_loader,
                 device: str = 'cuda'):
        """
        Args:
            source_loader: DataLoader for Condition 1 (train set with RUL labels)
            target_loader: DataLoader for Condition 2 (can be unlabeled for RUL)
            device: 'cuda' or 'cpu'
        """
        self.source_loader = source_loader
        self.target_loader = target_loader
        self.device = device

        self._source_iter = iter(source_loader)
        self._target_iter = iter(target_loader)

    def get_mixed_batch(self) -> Tuple[torch.Tensor, torch.Tensor,
                                       torch.Tensor, torch.Tensor]:
        """
        Returns a mixed batch: half source, half target.

        Returns:
            x_1d:         (B, 1, 2560) vibration signals
            rul_targets:  (B,) RUL labels (target domain samples have label=-1 as sentinel)
            domain_labels: (B,) 0=source, 1=target
            has_rul:      (B,) bool mask — True if sample has valid RUL label
        """
        # Get source batch
        try:
            src_batch = next(self._source_iter)
        except StopIteration:
            self._source_iter = iter(self.source_loader)
            src_batch = next(self._source_iter)

        # Get target batch
        try:
            tgt_batch = next(self._target_iter)
        except StopIteration:
            self._target_iter = iter(self.target_loader)
            tgt_batch = next(self._target_iter)

        # Handle both (x_1d, rul, bid, tidx) and (x_1d, rul) formats
        src_x = src_batch[0].to(self.device)
        src_rul = src_batch[1].to(self.device)

        tgt_x = tgt_batch[0].to(self.device)
        # Target domain may or may not have valid RUL — use -1.0 as sentinel
        if len(tgt_batch) > 1:
            tgt_rul = tgt_batch[1].to(self.device)
        else:
            tgt_rul = torch.full((len(tgt_x),), -1.0, device=self.device)

        # Combine
        batch_size = min(len(src_x), len(tgt_x))
        src_x   = src_x[:batch_size]
        src_rul = src_rul[:batch_size]
        tgt_x   = tgt_x[:batch_size]
        tgt_rul = tgt_rul[:batch_size]

        x_1d   = torch.cat([src_x, tgt_x], dim=0)           # (2B, 1, 2560)
        ruls   = torch.cat([src_rul, tgt_rul], dim=0)        # (2B,)
        labels = torch.cat([
            torch.zeros(batch_size),   # source = 0
            torch.ones(batch_size)     # target = 1
        ], dim=0).to(self.device)                            # (2B,)

        has_rul = ruls >= 0                                   # (2B,) bool

        return x_1d, ruls, labels, has_rul

    @staticmethod
    def compute_domain_accuracy(domain_logits: torch.Tensor,
                                 domain_labels: torch.Tensor) -> float:
        """For monitoring: how well the discriminator classifies domains."""
        preds = (torch.sigmoid(domain_logits) > 0.5).float()
        return (preds == domain_labels).float().mean().item()


# -----------------------------------------------------------------------------
# Verification
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    print("=" * 60)
    print("TESTING DANNModule")
    print("=" * 60)

    # Test 1: GRL forward/backward
    print("\n[Test 1] GradientReversalLayer")
    grl = GradientReversalLayer(gamma=10.0)
    grl.set_progress(0.5)  # lambda at 50% progress
    x = torch.randn(4, 128, requires_grad=True)
    y = grl(x)
    loss = y.sum()
    loss.backward()
    lambda_ = grl.get_lambda()
    # Gradient should be -lambda_ * ones (since forward = identity, backward reverses)
    expected_grad = -lambda_ * torch.ones_like(x)
    assert torch.allclose(x.grad, expected_grad, atol=1e-5), \
        f"GRL gradient mismatch! Expected {-lambda_}, got {x.grad[0,0]:.4f}"
    print(f"  lambda at 50% progress: {lambda_:.4f} (should be ~0.462)")
    print(f"  Gradient reversal: OK (grad = {x.grad[0,0]:.4f} = -lambda)")

    # Test 2: Lambda schedule
    print("\n[Test 2] Lambda schedule over training")
    grl = GradientReversalLayer(gamma=10.0)
    for p in [0.0, 0.2, 0.5, 0.8, 1.0]:
        grl.set_progress(p)
        print(f"  Progress {p:.1f} -> lambda = {grl.get_lambda():.4f}")

    # Test 3: Full DANNModule
    print("\n[Test 3] DANNModule forward pass")
    dann = DANNModule(feat_dim=128, hidden_dim=64)
    dann.set_progress(0.5)

    fused = torch.randn(16, 128)
    domain_labels = torch.cat([torch.zeros(8), torch.ones(8)])  # 8 src, 8 tgt

    domain_logit = dann(fused)
    domain_loss  = dann.domain_loss(domain_logit, domain_labels)
    domain_acc   = DANNTrainingHelper.compute_domain_accuracy(domain_logit, domain_labels)

    print(f"  Domain logit shape: {domain_logit.shape} OK")
    print(f"  Domain loss: {domain_loss.item():.4f}")
    print(f"  Domain accuracy: {domain_acc:.2f} (should start near 0.5 = random)")
    print(f"  NaN check: {torch.isnan(domain_logit).any().item()} (should be False)")

    # Test 4: Backward pass through GRL
    print("\n[Test 4] Backward pass (ensures GRL integrates correctly)")
    features = torch.randn(8, 128, requires_grad=True)
    domain_logit = dann(features)
    loss = dann.domain_loss(domain_logit, torch.zeros(8))
    loss.backward()
    assert features.grad is not None, "FAIL: No gradient flowing to features"
    grad_norm = features.grad.norm().item()
    print(f"  Gradient norm through GRL: {grad_norm:.4f} (should be > 0)")
    print(f"  [PASS] Gradients flow correctly through gradient reversal")

    print("\n[ALL TESTS PASSED] DANNModule is ready for integration")
    print("\nIntegration summary:")
    print("  dann = DANNModule(feat_dim=128)")
    print("  dann.set_progress(epoch / total_epochs)  # each epoch")
    print("  domain_logit = dann(fused_features)")
    print("  loss += dann.current_lambda * dann.domain_loss(domain_logit, labels)")

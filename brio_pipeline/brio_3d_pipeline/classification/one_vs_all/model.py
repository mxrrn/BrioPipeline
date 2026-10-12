"""
One-vs-all component classifier: MobileNetV3-small backbone, K independent
binary heads instead of a K-way softmax.

Architecture note: this is a single nn.Linear(in_features, K) producing K
raw logits — that IS the K heads. A Linear layer's outputs are computed
independently of each other (no normalization across them), which is what
makes this one-vs-all rather than softmax: at training time each output is
trained against its own binary target (BCEWithLogitsLoss, see train.py), and
at inference each is sigmoid'd independently, so a crop can score "no" on
every head at once (see ONE_VS_ALL_PLAN.md, Section 2). There is no
dedicated "_bg" head — background crops act as negative examples for every
real-class head instead; "none of the above" falls out of every head saying
no, not from a 30th head saying yes.

The backbone/head-replacement pattern mirrors component_classifier.py's
ComponentClassifier exactly (keep the pretrained 576->1024 projection,
replace only the final projection layer) for consistency with the
already-proven architecture in this codebase.
"""
import torch.nn as nn
from torchvision import models


class OneVsAllClassifier(nn.Module):
    def __init__(self, classes: list[str], pretrained: bool = True):
        super().__init__()
        self.classes = list(classes)
        weights = "IMAGENET1K_V1" if pretrained else None
        self.backbone = models.mobilenet_v3_small(weights=weights)
        in_features = self.backbone.classifier[-1].in_features
        self.backbone.classifier[-1] = nn.Linear(in_features, len(self.classes))

    def forward(self, x):
        """Returns (B, K) raw logits — NOT sigmoid'd. Apply sigmoid at loss
        (BCEWithLogitsLoss does this internally) or inference time."""
        return self.backbone(x)

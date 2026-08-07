import torch
import torch.nn as nn
import torch.nn.functional as F


class ECGCNNBaseline(nn.Module):
    """
    Small 1D-CNN for ECG's 187-sample beat waveform -- exploits the
    sequential/local structure that the dense baselines (MLPBaseline,
    LayerMatchedMLP) ignore by treating the beat as an unordered feature
    vector. Global-average-pooled head keeps the parameter count from being
    dominated by a flatten->FC layer, as is standard in compact CNN design.
    """

    # "Mixed precision" keeps the first and last layers float32 (see
    # compress_model_mixed's mixed_layers param) -- this model has no "fc1",
    # so the default ("fc1", "out") would silently quantize every conv layer.
    MIXED_LAYERS = ("conv1", "out")

    def __init__(self, num_classes=5):
        super().__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv1d(1, 16, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(32, 48, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.out = nn.Linear(48, max(1, num_classes))

    def forward(self, x):
        x = x.unsqueeze(1)  # (batch, 187) -> (batch, 1, 187)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = x.mean(dim=2)  # global average pool -> (batch, 48)
        x = self.out(x)
        if self.num_classes == 1:
            return torch.sigmoid(x)
        return x

    def size_bytes(self):
        return sum(p.nelement() * p.element_size() for p in self.parameters())


class CompactHARMLP(nn.Module):
    """
    Plain, small MLP for HAPT's 561 pre-extracted features. Not shaped to
    match DendriticNetwork's widths -- that's MLPBaseline/LayerMatchedMLP's
    job as internal ablation controls. This is a literature-representative
    "typical compact HAR classifier" baseline instead: a 1D-CNN doesn't
    apply here since these are pre-extracted statistical features with no
    spatial/temporal locality to convolve over (see docs/experiment_log.md).
    """

    def __init__(self, input_dim, num_classes=12):
        super().__init__()
        self.num_classes = num_classes
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 16)
        self.out = nn.Linear(16, max(1, num_classes))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        if self.num_classes == 1:
            return torch.sigmoid(x)
        return x

    def size_bytes(self):
        return sum(p.nelement() * p.element_size() for p in self.parameters())

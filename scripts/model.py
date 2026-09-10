"""
Shared model definition for the STGCN emergency rerouting project.

This is the SINGLE source of truth for the model architecture. Every script
that needs to train, load, or run this model should import it from here
instead of redefining the class. This prevents architecture drift between
files (e.g., training script and inference script silently disagreeing on
hidden_dimensions and producing a checkpoint-loading mismatch).
"""
import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import TGCN


class EmergencyRoutingSTGCN(torch.nn.Module):
    def __init__(self, node_features, hidden_dimensions):
        super(EmergencyRoutingSTGCN, self).__init__()
        # NOTE: reverted from a 2-layer TGCN stack. Tested both; the 2-layer
        # version showed no improvement over this single-layer version
        # (Test MAE 0.2961 vs 0.2926, both scaled) -- documented as a
        # negative ablation result rather than pursued further, since
        # forecasting accuracy is not this project's core contribution.
        self.tgcn = TGCN(in_channels=node_features, out_channels=hidden_dimensions)
        self.linear = torch.nn.Linear(hidden_dimensions, 1)

    def forward(self, x, edge_index, edge_weight):
        h = None
        for t in range(x.size(2)):
            x_t = x[:, :, t]
            h = self.tgcn(x_t, edge_index, edge_weight, h)
        h = F.relu(h)
        predicted_delta = self.linear(h)

        # Residual/skip connection to the persistence baseline: the model
        # predicts the CHANGE from the last known observed value, and we add
        # that delta back on. This means an undertrained delta head defaults
        # toward persistence rather than an arbitrary absolute-value guess.
        last_known_speed = x[:, 0, -1].unsqueeze(1)
        return last_known_speed + predicted_delta
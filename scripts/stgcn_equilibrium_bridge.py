"""
STGCN -> Wardrop Equilibrium bridge.

This is the piece that finally connects the two previously-separate
workstreams: the trained STGCN forecasting model, and the Wardrop
equilibrium engine. Until now, equilibrium free-flow times were derived
once from METR-LA's static edge_attr and never changed. This module lets
the STGCN's LIVE predictions dynamically adjust those free-flow times,
representing "what the network looks like under a specific forecasted
condition" (e.g. right before/during a simulated failure) rather than a
single fixed baseline.

MODELING ASSUMPTION (state explicitly in the paper):
METR-LA speeds are Z-score normalized with no straightforward way to
recover real mph from this loader (already established in earlier
diagnostic work). Rather than attempting a fragile unscaling, we use a
RELATIVE congestion signal: for each node, compare the STGCN's predicted
Z-score speed to that node's own TYPICAL (mean, ~0 by construction of
Z-scoring) speed. If predicted speed is BELOW typical, outgoing edges from
that node get proportionally slower. If predicted speed is AT OR ABOVE
typical, no bonus is applied (capped at multiplier=1.0) -- free-flow time
already represents the best-case scenario, so we don't reward "unusually
fast" predictions with faster-than-free-flow travel times.

This is a deliberately simple, defensible first version. A more precise
version would use real speed units and posted speed limits; that data
does not exist in METR-LA and would need to come from an external source
(e.g. the Delhi OSM/GTFS extension discussed for Project 3).
"""

import torch

from model import EmergencyRoutingSTGCN
import config


# --- Congestion multiplier parameters ---
# How strongly a speed deficit below typical translates into slower travel
# time. 0.5 means: 1 full Z-score unit below typical -> 50% slower.
CONGESTION_SCALE = 0.5
# Hard ceiling so a single extreme prediction can't produce an absurd
# multiplier (e.g. 10x slower) -- 3.0 means travel time can at most triple.
CONGESTION_CAP = 3.0


def load_trained_model(device=None):
    """Loads the trained (single-layer, residual-connection) STGCN checkpoint."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EmergencyRoutingSTGCN(
        node_features=config.NODE_FEATURES,
        hidden_dimensions=config.HIDDEN_DIMENSIONS
    ).to(device)
    model.load_state_dict(torch.load(config.CHECKPOINT_PATH, map_location=device, weights_only=True))
    model.eval()
    return model, device


def compute_node_typical_speeds(all_snapshots, num_nodes):
    """
    Computes each node's TYPICAL (mean) speed across the dataset, in the
    same Z-score space the model operates in. By construction of Z-score
    normalization this should be close to 0 for most nodes, but computing
    it directly (rather than assuming exactly 0) is more robust -- some
    sensors may have been normalized with slightly different effective
    statistics, or subset/train-only normalization could shift this.
    """
    node_speed_sums = torch.zeros(num_nodes)
    count = 0
    for snapshot in all_snapshots:
        # feature index 0 = speed (confirmed earlier), use the most recent
        # timestep of each snapshot's input window as one sample
        node_speed_sums += snapshot.x[:, 0, -1]
        count += 1
    return node_speed_sums / count


def get_congestion_multipliers(model, snapshot, typical_speeds, device):
    """
    Runs the STGCN forward pass on one snapshot and converts its predicted
    speeds into a per-node congestion multiplier.

    Returns: dict {node_index: multiplier}, multiplier >= 1.0
    """
    with torch.no_grad():
        x = snapshot.x.to(device)
        edge_index = snapshot.edge_index.to(device)
        edge_attr = snapshot.edge_attr.to(device)
        predicted_speeds = model(x, edge_index, edge_attr).squeeze(1).cpu()

    multipliers = {}
    for node in range(len(predicted_speeds)):
        predicted = predicted_speeds[node].item()
        typical = typical_speeds[node].item()
        deficit = max(0.0, typical - predicted)  # positive only if SLOWER than typical
        multiplier = 1.0 + deficit * CONGESTION_SCALE
        multipliers[node] = min(multiplier, CONGESTION_CAP)

    return multipliers


def apply_congestion_multipliers(edges, multipliers):
    """
    Applies per-node congestion multipliers to edge free-flow times.
    Consistent with the convention already used in the original (now
    replaced) Phase 3 code: an edge's cost is scaled by its DESTINATION
    node's predicted condition -- i.e. "how bad is traffic where this road
    is taking you", not the origin. This does NOT mutate the original
    edges list -- returns a new list of Edge-like objects with adjusted
    free_flow_time, so the un-adjusted baseline network remains available
    for comparison.
    """
    import copy
    adjusted_edges = []
    for e in edges:
        new_edge = copy.copy(e)
        multiplier = multipliers.get(e.destination, 1.0)
        new_edge.free_flow_time = e.free_flow_time * multiplier
        new_edge.flow = 0.0  # reset -- this is a fresh network state
        adjusted_edges.append(new_edge)
    return adjusted_edges
"""
PHASE 3g: STGCN-Driven Dynamic Equilibrium

The final integration piece. Instead of solving Wardrop equilibrium once
on static free-flow times (03b/03c/03d/03f), this uses the TRAINED STGCN's
live predictions to set free-flow times dynamically, then solves
equilibrium on that STGCN-informed network. Compared against the static
baseline, this shows whether/how much the forecasting model's predictions
actually change the equilibrium outcome -- the concrete demonstration that
STGCN and Wardrop equilibrium are genuinely coupled, not two independent
pieces of code that happen to share a folder.
"""

import torch
from torch_geometric_temporal.dataset import METRLADatasetLoader
from torch_geometric_temporal.signal import temporal_signal_split

from network_builder import build_metrla_network, load_metrla_topology, build_edges, find_largest_scc
from equilibrium import TrafficNetwork, solve_wardrop_equilibrium
from stgcn_equilibrium_bridge import (
    load_trained_model, compute_node_typical_speeds,
    get_congestion_multipliers, apply_congestion_multipliers
)
import config


def main():
    print("=" * 70)
    print("PHASE 3g: STGCN-Driven Dynamic Equilibrium")
    print("=" * 70)

    # --- 1. Load trained model and dataset ---
    print("\n--- [1/4] Loading trained STGCN model and dataset ---")
    model, device = load_trained_model()
    print(f"Model loaded on device: {device}")

    loader = METRLADatasetLoader()
    dataset = loader.get_dataset(
        num_timesteps_in=config.NUM_TIMESTEPS_IN,
        num_timesteps_out=config.NUM_TIMESTEPS_OUT
    )
    _, test_dataset = temporal_signal_split(dataset, train_ratio=config.TRAIN_RATIO)
    test_snapshots = list(test_dataset)
    print(f"Loaded {len(test_snapshots)} test snapshots")

    # --- 2. Compute each node's typical speed (for the congestion multiplier baseline) ---
    print("\n--- [2/4] Computing per-node typical speeds ---")
    num_nodes = test_snapshots[0].x.shape[0]
    typical_speeds = compute_node_typical_speeds(test_snapshots, num_nodes)
    print(f"Typical speed range across nodes: min={typical_speeds.min():.4f}, "
          f"max={typical_speeds.max():.4f}, mean={typical_speeds.mean():.4f}")
    print("(Should be close to 0 -- these are Z-score normalized speeds, and mean")
    print(" speed by construction of Z-scoring should center near 0.)")

    # --- 3. Pick one test snapshot, run the model, get congestion multipliers ---
    print("\n--- [3/4] Running STGCN on a test snapshot ---")
    snapshot = test_snapshots[0]  # a specific point in time from the held-out test set
    multipliers = get_congestion_multipliers(model, snapshot, typical_speeds, device)

    multiplier_values = list(multipliers.values())
    print(f"Congestion multiplier range across nodes: min={min(multiplier_values):.4f}, "
          f"max={max(multiplier_values):.4f}, mean={sum(multiplier_values)/len(multiplier_values):.4f}")
    congested_nodes = sum(1 for m in multiplier_values if m > 1.01)
    print(f"Nodes predicted to be meaningfully slower than typical: {congested_nodes}/{num_nodes}")

    # --- 4. Solve equilibrium TWICE: static baseline vs. STGCN-informed ---
    print("\n--- [4/4] Comparing STATIC baseline vs. STGCN-INFORMED equilibrium ---\n")

    # Static baseline (same as 03b -- fixed edge_attr-derived free-flow times)
    static_network, num_nodes_check, largest_scc = build_metrla_network(verbose=False)
    solve_wardrop_equilibrium(static_network, max_iterations=1000,
                               convergence_threshold=1e-5, verbose=False)
    static_total_time = static_network.total_system_travel_time()
    print(f"Static baseline total system travel time: {static_total_time:.2f}")

    # STGCN-informed: same topology and SAME od_demand, but free-flow times
    # adjusted by the model's live predictions before solving.
    edge_index, edge_attr, _ = load_metrla_topology()
    raw_edges = build_edges(edge_index, edge_attr)
    dynamic_edges = apply_congestion_multipliers(raw_edges, multipliers)
    dynamic_network = TrafficNetwork(edges=dynamic_edges, od_demand=static_network.od_demand)

    solve_wardrop_equilibrium(dynamic_network, max_iterations=1000,
                               convergence_threshold=1e-5, verbose=False)
    dynamic_total_time = dynamic_network.total_system_travel_time()
    print(f"STGCN-informed total system travel time:  {dynamic_total_time:.2f}")

    delta = dynamic_total_time - static_total_time
    pct_change = (delta / static_total_time) * 100
    print(f"\nChange from STGCN predictions: {delta:+.2f} ({pct_change:+.4f}%)")

    if abs(pct_change) > 0.01:
        print(f"\n✅ The STGCN's predictions measurably changed the equilibrium outcome --")
        print(f"   confirming the forecasting model and equilibrium engine are genuinely")
        print(f"   coupled, not independent components.")
    else:
        print(f"\nNOTE: change is very small. This could mean (a) this specific snapshot")
        print(f"happened to predict close-to-typical conditions everywhere, or (b) the")
        print(f"congestion multiplier scale (CONGESTION_SCALE in stgcn_equilibrium_bridge.py)")
        print(f"is too conservative. Try a different snapshot index, or increase CONGESTION_SCALE.")


if __name__ == "__main__":
    main()
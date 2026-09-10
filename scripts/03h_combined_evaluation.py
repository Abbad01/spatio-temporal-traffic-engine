"""
PHASE 3h: Combined Evaluation

Ties together every previously-separate mechanism into one evaluation loop:
  1. STGCN predicts congestion for a given point in time (varies per trial).
  2. An infrastructure failure is simulated (varies per trial).
  3. Background equilibrium is solved on the STGCN-informed, post-failure network.
  4. The emergency vehicle is routed both naively and equilibrium-aware.
  5. Price of priority is computed (capacity reduction on the emergency route).

This is run across many trials and aggregated -- this is the evaluation the
paper's title actually claims, as opposed to the individually-validated
mechanisms tested in 03b-03g.

DESIGN CHOICE: background OD demand is held FIXED (config.OD_GENERATION_SEED)
across all trials. Only the STGCN snapshot (time) and the failure edge vary
per trial. This isolates the effects of interest -- if OD demand also
varied randomly per trial, it would be harder to attribute changes in
outcome to the failure/forecast rather than to a different demand pattern.
"""

import random
import statistics
import csv
import os
import networkx as nx

from network_builder import load_metrla_topology, build_edges, find_largest_scc, generate_od_demand
from equilibrium import TrafficNetwork, solve_wardrop_equilibrium
from stgcn_equilibrium_bridge import (
    load_trained_model, compute_node_typical_speeds, get_congestion_multipliers,
    apply_congestion_multipliers
)
import config

from torch_geometric_temporal.dataset import METRLADatasetLoader
from torch_geometric_temporal.signal import temporal_signal_split

NUM_COMBINED_TRIALS = 100


def get_route_and_true_cost(network, cost_graph, origin, destination):
    path = nx.shortest_path(cost_graph, source=origin, target=destination, weight='weight')
    true_cost = 0.0
    edge_lookup = {(e.origin, e.destination): e for e in network.edges}
    for i in range(len(path) - 1):
        edge = edge_lookup[(path[i], path[i + 1])]
        true_cost += edge.bpr_cost()
    return path, true_cost


def find_diverging_emergency_od(network):
    """Search top-10 highest-demand OD pairs for one where naive and
    equilibrium-aware routing genuinely differ (see 03f for rationale)."""
    candidates = sorted(network.od_demand.items(), key=lambda item: item[1], reverse=True)
    naive_graph = nx.DiGraph()
    for e in network.edges:
        naive_graph.add_edge(e.origin, e.destination, weight=e.free_flow_time)
    equilibrium_graph = network.build_cost_graph()

    for (origin, destination), demand in candidates[:10]:
        naive_path, naive_cost = get_route_and_true_cost(network, naive_graph, origin, destination)
        aware_path, aware_cost = get_route_and_true_cost(network, equilibrium_graph, origin, destination)
        if naive_path != aware_path:
            return origin, destination, naive_cost, aware_cost, aware_path, True

    origin, destination = candidates[0][0]
    naive_path, naive_cost = get_route_and_true_cost(network, naive_graph, origin, destination)
    aware_path, aware_cost = get_route_and_true_cost(network, equilibrium_graph, origin, destination)
    return origin, destination, naive_cost, aware_cost, aware_path, False


def compute_edge_betweenness():
    """
    Computes edge betweenness centrality once on the intact network topology.
    Betweenness measures how many shortest paths across the whole network pass
    through each edge -- a measure of STRUCTURAL importance, independent of how
    much traffic currently uses it. Computed once and reused across trials
    because it depends only on topology (which is identical every trial), not
    on flow or forecast.
    """
    edge_index, edge_attr, _ = load_metrla_topology()
    edges = build_edges(edge_index, edge_attr)
    topo = nx.DiGraph()
    for e in edges:
        topo.add_edge(e.origin, e.destination)
    return nx.edge_betweenness_centrality(topo)


def run_combined_trial(model, device, typical_speeds, test_snapshots, trial_seed,
                        edge_betweenness=None):
    result = {"trial_seed": trial_seed, "valid": False}
    rng = random.Random(trial_seed)

    # --- STGCN: pick a different point in time per trial ---
    snapshot_index = rng.randrange(len(test_snapshots))
    snapshot = test_snapshots[snapshot_index]
    multipliers = get_congestion_multipliers(model, snapshot, typical_speeds, device)

    # --- Build STGCN-informed network (no failure yet) ---
    edge_index, edge_attr, num_nodes = load_metrla_topology()
    raw_edges = build_edges(edge_index, edge_attr)
    dynamic_edges = apply_congestion_multipliers(raw_edges, multipliers)

    largest_scc = find_largest_scc(dynamic_edges)
    od_demand = generate_od_demand(largest_scc)  # fixed seed by default -- same demand every trial

    baseline_network = TrafficNetwork(edges=dynamic_edges, od_demand=od_demand)
    try:
        solve_wardrop_equilibrium(baseline_network, max_iterations=1000,
                                   convergence_threshold=1e-5, verbose=False)
    except RuntimeError:
        return result  # disconnected OD pair even before failure -- skip trial

    baseline_time = baseline_network.total_system_travel_time()

    # --- Pick and apply a random failure (from edges carrying flow) ---
    used_edges = [e for e in baseline_network.edges if e.flow > 1e-6]
    if not used_edges:
        return result
    failed_edge = rng.choice(used_edges)
    failure_edges = {(failed_edge.origin, failed_edge.destination)}

    raw_edges_2 = build_edges(edge_index, edge_attr, excluded_edges=failure_edges)
    dynamic_edges_2 = apply_congestion_multipliers(raw_edges_2, multipliers)
    failure_network = TrafficNetwork(edges=dynamic_edges_2, od_demand=od_demand)

    try:
        solve_wardrop_equilibrium(failure_network, max_iterations=1000,
                                   convergence_threshold=1e-5, verbose=False)
    except RuntimeError:
        result["disconnected_by_failure"] = True
        return result

    failure_time = failure_network.total_system_travel_time()
    failure_pct_change = (failure_time - baseline_time) / baseline_time * 100

    # --- Emergency vehicle routing on the POST-FAILURE, STGCN-informed network ---
    origin, destination, naive_cost, aware_cost, aware_path, found_divergent = \
        find_diverging_emergency_od(failure_network)
    routing_improvement_pct = ((naive_cost - aware_cost) / naive_cost * 100) if naive_cost > 0 else 0.0

    # --- Price of priority: reduce capacity along the emergency route, re-solve ---
    priority_route_edges = set(zip(aware_path[:-1], aware_path[1:]))
    raw_edges_3 = build_edges(edge_index, edge_attr, excluded_edges=failure_edges)
    dynamic_edges_3 = apply_congestion_multipliers(raw_edges_3, multipliers)
    for e in dynamic_edges_3:
        if (e.origin, e.destination) in priority_route_edges:
            e.capacity = e.capacity * (1 - config.EMERGENCY_LANE_CAPACITY_REDUCTION)
    priority_network = TrafficNetwork(edges=dynamic_edges_3, od_demand=od_demand)

    try:
        solve_wardrop_equilibrium(priority_network, max_iterations=1000,
                                   convergence_threshold=1e-5, verbose=False)
        priority_time = priority_network.total_system_travel_time()
        price_of_priority_pct = (priority_time - failure_time) / failure_time * 100
    except RuntimeError:
        price_of_priority_pct = None

    result.update({
        "valid": True,
        "snapshot_index": snapshot_index,
        "failed_edge": (failed_edge.origin, failed_edge.destination),
        "failed_edge_betweenness": (edge_betweenness or {}).get(
            (failed_edge.origin, failed_edge.destination), None),
        "failed_edge_utilization": failed_edge.flow / failed_edge.capacity,
        "baseline_time": baseline_time,
        "failure_time": failure_time,
        "failure_pct_change": failure_pct_change,
        "emergency_od": (origin, destination),
        "found_divergent_route": found_divergent,
        "routing_improvement_pct": routing_improvement_pct,
        "price_of_priority_pct": price_of_priority_pct,
    })
    return result


def main():
    print("=" * 70)
    print(f"PHASE 3h: Combined Evaluation ({NUM_COMBINED_TRIALS} trials)")
    print("=" * 70)

    print("\n--- Loading model and dataset ---")
    model, device = load_trained_model()
    loader = METRLADatasetLoader()
    dataset = loader.get_dataset(
        num_timesteps_in=config.NUM_TIMESTEPS_IN,
        num_timesteps_out=config.NUM_TIMESTEPS_OUT
    )
    _, test_dataset = temporal_signal_split(dataset, train_ratio=config.TRAIN_RATIO)
    test_snapshots = list(test_dataset)
    num_nodes = test_snapshots[0].x.shape[0]
    typical_speeds = compute_node_typical_speeds(test_snapshots, num_nodes)
    print(f"Loaded {len(test_snapshots)} test snapshots, model ready.")

    print("\n--- Computing edge betweenness centrality (once, reused across trials) ---")
    edge_betweenness = compute_edge_betweenness()
    print(f"Computed betweenness for {len(edge_betweenness)} edges.\n")

    results = []
    for i in range(NUM_COMBINED_TRIALS):
        print(f"[Trial {i+1}/{NUM_COMBINED_TRIALS}] seed={i}...")
        r = run_combined_trial(model, device, typical_speeds, test_snapshots,
                                trial_seed=i, edge_betweenness=edge_betweenness)
        if r["valid"]:
            price_str = f"{r['price_of_priority_pct']:+.3f}%" if r['price_of_priority_pct'] is not None else "N/A"
            print(f"  Failure impact: {r['failure_pct_change']:+.3f}% | "
                  f"Routing improvement: {r['routing_improvement_pct']:+.3f}% | "
                  f"Price of priority: {price_str}")
            results.append(r)
        else:
            print(f"  Skipped (disconnected scenario)")

    # --- Save results to CSV so figures and re-checks don't require re-running ---
    os.makedirs("results", exist_ok=True)
    csv_path = "results/combined_evaluation.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "trial_seed", "snapshot_index", "failed_edge", "failed_edge_betweenness",
            "failed_edge_utilization", "baseline_time", "failure_time",
            "failure_pct_change", "emergency_od", "found_divergent_route",
            "routing_improvement_pct", "price_of_priority_pct"
        ])
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k) for k in writer.fieldnames})
    print(f"\nResults saved to {csv_path}")

    print(f"\n{'='*70}")
    print("AGGREGATE RESULTS")
    print(f"{'='*70}")
    print(f"Valid trials: {len(results)} / {NUM_COMBINED_TRIALS}")

    if len(results) >= 2:
        failure_changes = [r["failure_pct_change"] for r in results]
        routing_improvements = [r["routing_improvement_pct"] for r in results]
        priority_prices = [r["price_of_priority_pct"] for r in results if r["price_of_priority_pct"] is not None]
        divergent_count = sum(1 for r in results if r["found_divergent_route"])

        print(f"\nFailure impact on STGCN-informed total system travel time:")
        print(f"  Mean: {statistics.mean(failure_changes):+.4f}% | "
              f"Median: {statistics.median(failure_changes):+.4f}% | "
              f"Stdev: {statistics.stdev(failure_changes):.4f}%")
        print(f"  Min: {min(failure_changes):+.4f}% | Max: {max(failure_changes):+.4f}%")

        print(f"\nEquilibrium-aware vs naive emergency routing improvement:")
        print(f"  Mean: {statistics.mean(routing_improvements):+.4f}% | "
              f"Median: {statistics.median(routing_improvements):+.4f}%")
        print(f"  Trials with a genuinely divergent route found: {divergent_count}/{len(results)}")

        if priority_prices:
            stdev_str = f" | Stdev: {statistics.stdev(priority_prices):.4f}%" if len(priority_prices) > 1 else ""
            print(f"\nPrice of Priority:")
            print(f"  Mean: {statistics.mean(priority_prices):+.4f}% | "
                  f"Median: {statistics.median(priority_prices):+.4f}%{stdev_str}")

        # --- The key structural finding, now measured across all trials ---
        print(f"\n{'='*70}")
        print("STRUCTURAL CRITICALITY vs. TRAFFIC LOAD as predictors of failure impact")
        print(f"{'='*70}")
        paired = [(r["failed_edge_betweenness"], r["failed_edge_utilization"],
                   r["failure_pct_change"]) for r in results
                  if r["failed_edge_betweenness"] is not None]

        if len(paired) >= 3:
            betweenness_vals = [p[0] for p in paired]
            utilization_vals = [p[1] for p in paired]
            impact_vals = [p[2] for p in paired]

            try:
                corr_betweenness = statistics.correlation(betweenness_vals, impact_vals)
                corr_utilization = statistics.correlation(utilization_vals, impact_vals)
                print(f"Correlation of failure impact with:")
                print(f"  Edge BETWEENNESS centrality (structural): r = {corr_betweenness:+.4f}")
                print(f"  Edge CAPACITY UTILIZATION (traffic load): r = {corr_utilization:+.4f}")
                print(f"  (n = {len(paired)} trials)")
                if abs(corr_betweenness) > abs(corr_utilization):
                    print(f"\n=> Betweenness is the STRONGER predictor of failure impact.")
                    print(f"   This supports the structural-criticality finding across the full")
                    print(f"   sample, rather than resting on individual example edges.")
                else:
                    print(f"\n=> Capacity utilization is the stronger predictor in this sample.")
                    print(f"   Note: this differs from the single-edge comparison in 03e --")
                    print(f"   report the full-sample correlation, which is the more reliable result.")
            except statistics.StatisticsError as err:
                print(f"Could not compute correlation: {err}")
                print("(Likely cause: zero variance in one variable across trials.)")
        else:
            print("Not enough trials with recorded betweenness for correlation analysis.")
    else:
        print("Not enough valid trials for meaningful statistics.")


if __name__ == "__main__":
    main()
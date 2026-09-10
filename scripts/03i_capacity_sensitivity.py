"""
PHASE 3i: Capacity Sensitivity Analysis

Addresses the most serious ADDRESSABLE limitation of this work: capacity is
assumed uniform (config.DEFAULT_EDGE_CAPACITY) across all 1,722 edges,
because METR-LA provides no lane-count or road-class data from which real
capacities could be derived.

The question a reviewer will ask: "Do your conclusions depend on that
arbitrary capacity value?"

This script answers it empirically by re-running the combined evaluation at
several capacity levels and comparing whether the QUALITATIVE conclusions
hold:
  - Is Price of Priority still consistently positive?
  - Is failure impact still right-skewed (mean > median)?
  - Does the relative ordering of predictors (utilization vs. betweenness)
    remain stable?

Note on interpretation: absolute magnitudes SHOULD change with capacity --
that is expected and not a problem. Lower capacity means edges congest
sooner, so BPR costs rise faster and every effect amplifies. What matters
is whether the DIRECTION and RELATIVE ORDERING of findings survive. If they
do, the conclusions are robust to the assumption even though the specific
numbers are not.
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

# Capacity levels to test. The baseline (1000) sits in the middle so we
# probe both a more-congested and a less-congested regime.
CAPACITY_LEVELS = [500, 1000, 2000]

# Fewer trials per capacity level than the main 100-trial run, since we are
# testing robustness of direction rather than estimating precise magnitudes,
# and this runs the whole pipeline once per capacity level.
TRIALS_PER_LEVEL = 30


def get_route_and_true_cost(network, cost_graph, origin, destination):
    path = nx.shortest_path(cost_graph, source=origin, target=destination, weight='weight')
    true_cost = 0.0
    edge_lookup = {(e.origin, e.destination): e for e in network.edges}
    for i in range(len(path) - 1):
        edge = edge_lookup[(path[i], path[i + 1])]
        true_cost += edge.bpr_cost()
    return path, true_cost


def find_diverging_emergency_od(network):
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


def build_edges_with_capacity(edge_index, edge_attr, capacity, excluded_edges=None):
    """build_edges() uses config.DEFAULT_EDGE_CAPACITY; here we override it
    per-edge after construction so we can sweep capacity without mutating
    global config state (which would leak between capacity levels)."""
    edges = build_edges(edge_index, edge_attr, excluded_edges=excluded_edges)
    for e in edges:
        e.capacity = capacity
    return edges


def run_trial_at_capacity(model, device, typical_speeds, test_snapshots,
                           trial_seed, capacity, edge_betweenness):
    result = {"valid": False}
    rng = random.Random(trial_seed)

    snapshot = test_snapshots[rng.randrange(len(test_snapshots))]
    multipliers = get_congestion_multipliers(model, snapshot, typical_speeds, device)

    edge_index, edge_attr, _ = load_metrla_topology()
    raw_edges = build_edges_with_capacity(edge_index, edge_attr, capacity)
    dynamic_edges = apply_congestion_multipliers(raw_edges, multipliers)
    for e in dynamic_edges:
        e.capacity = capacity  # apply_congestion_multipliers copies edges; re-assert

    largest_scc = find_largest_scc(dynamic_edges)
    od_demand = generate_od_demand(largest_scc)

    baseline_network = TrafficNetwork(edges=dynamic_edges, od_demand=od_demand)
    try:
        solve_wardrop_equilibrium(baseline_network, max_iterations=1000,
                                   convergence_threshold=1e-5, verbose=False)
    except RuntimeError:
        return result
    baseline_time = baseline_network.total_system_travel_time()

    used_edges = [e for e in baseline_network.edges if e.flow > 1e-6]
    if not used_edges:
        return result
    failed_edge = rng.choice(used_edges)
    failure_edges = {(failed_edge.origin, failed_edge.destination)}

    raw_edges_2 = build_edges_with_capacity(edge_index, edge_attr, capacity, excluded_edges=failure_edges)
    dynamic_edges_2 = apply_congestion_multipliers(raw_edges_2, multipliers)
    for e in dynamic_edges_2:
        e.capacity = capacity
    failure_network = TrafficNetwork(edges=dynamic_edges_2, od_demand=od_demand)

    try:
        solve_wardrop_equilibrium(failure_network, max_iterations=1000,
                                   convergence_threshold=1e-5, verbose=False)
    except RuntimeError:
        return result
    failure_time = failure_network.total_system_travel_time()
    failure_pct_change = (failure_time - baseline_time) / baseline_time * 100

    origin, destination, naive_cost, aware_cost, aware_path, found_divergent = \
        find_diverging_emergency_od(failure_network)
    routing_improvement_pct = ((naive_cost - aware_cost) / naive_cost * 100) if naive_cost > 0 else 0.0

    priority_route_edges = set(zip(aware_path[:-1], aware_path[1:]))
    raw_edges_3 = build_edges_with_capacity(edge_index, edge_attr, capacity, excluded_edges=failure_edges)
    dynamic_edges_3 = apply_congestion_multipliers(raw_edges_3, multipliers)
    for e in dynamic_edges_3:
        e.capacity = capacity * (1 - config.EMERGENCY_LANE_CAPACITY_REDUCTION) \
            if (e.origin, e.destination) in priority_route_edges else capacity
    priority_network = TrafficNetwork(edges=dynamic_edges_3, od_demand=od_demand)

    try:
        solve_wardrop_equilibrium(priority_network, max_iterations=1000,
                                   convergence_threshold=1e-5, verbose=False)
        price_of_priority_pct = (priority_network.total_system_travel_time() - failure_time) / failure_time * 100
    except RuntimeError:
        price_of_priority_pct = None

    result.update({
        "valid": True,
        "capacity": capacity,
        "failure_pct_change": failure_pct_change,
        "routing_improvement_pct": routing_improvement_pct,
        "price_of_priority_pct": price_of_priority_pct,
        "found_divergent_route": found_divergent,
        "failed_edge_betweenness": edge_betweenness.get(
            (failed_edge.origin, failed_edge.destination), None),
        "failed_edge_utilization": failed_edge.flow / failed_edge.capacity,
    })
    return result


def main():
    print("=" * 70)
    print("PHASE 3i: Capacity Sensitivity Analysis")
    print(f"Capacity levels: {CAPACITY_LEVELS} | {TRIALS_PER_LEVEL} trials each")
    print("=" * 70)

    model, device = load_trained_model()
    loader = METRLADatasetLoader()
    dataset = loader.get_dataset(
        num_timesteps_in=config.NUM_TIMESTEPS_IN,
        num_timesteps_out=config.NUM_TIMESTEPS_OUT
    )
    _, test_dataset = temporal_signal_split(dataset, train_ratio=config.TRAIN_RATIO)
    test_snapshots = list(test_dataset)
    typical_speeds = compute_node_typical_speeds(test_snapshots, test_snapshots[0].x.shape[0])

    edge_index, edge_attr, _ = load_metrla_topology()
    topo = nx.DiGraph()
    for e in build_edges(edge_index, edge_attr):
        topo.add_edge(e.origin, e.destination)
    edge_betweenness = nx.edge_betweenness_centrality(topo)
    print("Model, data, and betweenness ready.\n")

    all_results = []
    summary_by_capacity = {}

    for capacity in CAPACITY_LEVELS:
        print(f"\n{'='*70}")
        print(f"CAPACITY = {capacity}")
        print(f"{'='*70}")
        level_results = []
        for i in range(TRIALS_PER_LEVEL):
            r = run_trial_at_capacity(model, device, typical_speeds, test_snapshots,
                                       trial_seed=i, capacity=capacity,
                                       edge_betweenness=edge_betweenness)
            if r["valid"]:
                level_results.append(r)
                all_results.append(r)
            if (i + 1) % 10 == 0:
                print(f"  ...{i+1}/{TRIALS_PER_LEVEL} trials done ({len(level_results)} valid)")

        if len(level_results) < 3:
            print(f"  Too few valid trials at capacity {capacity} to summarise.")
            continue

        failures = [r["failure_pct_change"] for r in level_results]
        prices = [r["price_of_priority_pct"] for r in level_results if r["price_of_priority_pct"] is not None]
        divergent = sum(1 for r in level_results if r["found_divergent_route"])

        betw = [r["failed_edge_betweenness"] for r in level_results if r["failed_edge_betweenness"] is not None]
        util = [r["failed_edge_utilization"] for r in level_results if r["failed_edge_betweenness"] is not None]
        impacts = [r["failure_pct_change"] for r in level_results if r["failed_edge_betweenness"] is not None]

        corr_b = corr_u = None
        if len(betw) >= 3:
            try:
                corr_b = statistics.correlation(betw, impacts)
                corr_u = statistics.correlation(util, impacts)
            except statistics.StatisticsError:
                pass

        summary_by_capacity[capacity] = {
            "n": len(level_results),
            "failure_mean": statistics.mean(failures),
            "failure_median": statistics.median(failures),
            "price_mean": statistics.mean(prices) if prices else None,
            "price_all_positive": all(p > 0 for p in prices) if prices else None,
            "divergent_rate": divergent / len(level_results),
            "corr_betweenness": corr_b,
            "corr_utilization": corr_u,
        }

        s = summary_by_capacity[capacity]
        print(f"\n  Valid trials: {s['n']}/{TRIALS_PER_LEVEL}")
        print(f"  Failure impact  -- mean: {s['failure_mean']:+.4f}%  median: {s['failure_median']:+.4f}%")
        print(f"  Price of priority -- mean: {s['price_mean']:+.4f}%" if s['price_mean'] is not None else "  Price of priority: N/A")
        print(f"  Price positive in all trials: {s['price_all_positive']}")
        print(f"  Divergent route rate: {s['divergent_rate']*100:.1f}%")
        if corr_b is not None:
            print(f"  Correlation with impact -- betweenness: r={corr_b:+.4f} | utilization: r={corr_u:+.4f}")

    # --- Save and summarise ---
    os.makedirs("results", exist_ok=True)
    with open("results/capacity_sensitivity.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "capacity", "failure_pct_change", "routing_improvement_pct",
            "price_of_priority_pct", "found_divergent_route",
            "failed_edge_betweenness", "failed_edge_utilization"
        ])
        writer.writeheader()
        for r in all_results:
            writer.writerow({k: r.get(k) for k in writer.fieldnames})
    print(f"\nPer-trial results saved to results/capacity_sensitivity.csv")

    print(f"\n{'='*70}")
    print("ROBUSTNESS SUMMARY -- do conclusions hold across capacity assumptions?")
    print(f"{'='*70}\n")

    header = (f"{'Capacity':<10} {'Fail mean':<12} {'Fail med':<12} {'Price mean':<12} "
              f"{'All +ve':<9} {'Diverge%':<10} {'r_betw':<10} {'r_util':<10}")
    print(header)
    print("-" * len(header))

    for cap in CAPACITY_LEVELS:
        if cap not in summary_by_capacity:
            continue
        s = summary_by_capacity[cap]
        fail_mean = f"{s['failure_mean']:+.4f}%"
        fail_med = f"{s['failure_median']:+.4f}%"
        price_mean = f"{s['price_mean']:+.4f}%" if s['price_mean'] is not None else "N/A"
        all_pos = str(s['price_all_positive'])
        diverge = f"{s['divergent_rate']*100:.1f}"
        r_betw = f"{s['corr_betweenness']:+.4f}" if s['corr_betweenness'] is not None else "N/A"
        r_util = f"{s['corr_utilization']:+.4f}" if s['corr_utilization'] is not None else "N/A"

        print(f"{cap:<10} {fail_mean:<12} {fail_med:<12} {price_mean:<12} "
              f"{all_pos:<9} {diverge:<10} {r_betw:<10} {r_util:<10}")

    print("\nInterpretation guide:")
    print("  - Absolute magnitudes SHOULD vary with capacity (lower capacity = more congestion).")
    print("  - What matters for robustness: is Price of Priority still positive everywhere?")
    print("    Is failure impact still right-skewed (mean > median)? Does the ordering of")
    print("    r_betw vs r_util stay consistent? If yes, conclusions are robust to the")
    print("    capacity assumption even though specific numbers are not.")


if __name__ == "__main__":
    main()
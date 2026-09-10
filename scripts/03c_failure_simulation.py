"""
PHASE 3c: Infrastructure Failure Simulation

Solves Wardrop equilibrium TWICE on the same METR-LA network with the same
OD demand: once with the network intact (baseline), once with one or more
edges removed (simulated failure -- e.g. a collapsed bridge, a closed
road). Compares the two equilibria to quantify the impact of the failure
on total system travel time and on individual OD pairs.

This is the "chaos" step from the original project plan, but done properly
this time: instead of a single Dijkstra call with static predicted speeds,
background traffic actually RE-EQUILIBRATES around the failure, the way
real traffic does.
"""

import random

from network_builder import build_metrla_network
from equilibrium import solve_wardrop_equilibrium
import config


def pick_random_failure_edges(edges, num_failures=1, seed=None):
    """
    Randomly selects edges to remove, simulating infrastructure failure.
    Only selects from edges that currently carry meaningful flow in some
    reference solve -- picking a totally unused edge to "fail" wouldn't
    produce an interesting scenario. For a first version, we just pick
    randomly from all edges; a more targeted version (failing the busiest
    edges, or spatially correlated failures per the equity/monsoon framing
    from earlier discussions) is a natural extension once this baseline
    works.
    """
    seed = config.OD_GENERATION_SEED + 1 if seed is None else seed  # different seed than OD demand
    rng = random.Random(seed)
    chosen = rng.sample(edges, min(num_failures, len(edges)))
    return {(e.origin, e.destination) for e in chosen}


def run_failure_scenario(num_failures=1, seed=None, verbose=True, target_edges=None):
    """
    Runs one baseline-vs-failure comparison and returns a results dict.

    target_edges: optional explicit set of (origin, dest) tuples to fail,
                  instead of randomly selecting them. Used for deliberately
                  testing high-utilization "worst case" edges rather than
                  relying on random sampling to find them.

    Returns a dict with keys: baseline_time, failure_time, delta, pct_change,
    failure_edges, disconnected (bool), or None if the scenario resulted in
    a fully disconnected OD pair (no valid comparison possible).
    """
    if verbose:
        print("=" * 70)
        print("PHASE 3c: Infrastructure Failure Simulation")
        print("=" * 70)

    # --- Step 1: Solve BASELINE equilibrium (network intact) ---
    if verbose:
        print("\n--- [1/3] Solving BASELINE equilibrium (network intact) ---\n")
    baseline_network, num_nodes, baseline_scc = build_metrla_network(verbose=verbose)
    solve_wardrop_equilibrium(baseline_network, max_iterations=1000,
                               convergence_threshold=1e-5, verbose=False)
    baseline_total_time = baseline_network.total_system_travel_time()
    if verbose:
        print(f"\nBaseline total system travel time: {baseline_total_time:.2f}")

    # --- Step 2: Pick failure edge(s) ---
    if target_edges is not None:
        failure_edges = target_edges
    else:
        used_baseline_edges = [e for e in baseline_network.edges if e.flow > 1e-6]
        failure_edges = pick_random_failure_edges(used_baseline_edges, num_failures=num_failures, seed=seed)

    if verbose:
        print(f"\n--- [2/3] Simulating failure: removing edge(s) {failure_edges} ---")
        for (o, d) in failure_edges:
            matching = next((e for e in baseline_network.edges if e.origin == o and e.destination == d), None)
            if matching:
                print(f"  Failed edge ({o}->{d}): carried {matching.flow:.2f} flow at baseline "
                      f"({matching.flow/matching.capacity*100:.1f}% of capacity)")

    # --- Step 3: Solve POST-FAILURE equilibrium ---
    if verbose:
        print("\n--- [3/3] Solving POST-FAILURE equilibrium ---\n")
    failure_network, _, failure_scc = build_metrla_network(excluded_edges=failure_edges, verbose=verbose)

    try:
        solve_wardrop_equilibrium(failure_network, max_iterations=1000,
                                   convergence_threshold=1e-5, verbose=False)
    except RuntimeError as err:
        if verbose:
            print(f"\n⚠️  Failure disconnected at least one OD pair entirely: {err}")
        return {
            "baseline_time": baseline_total_time,
            "failure_time": None,
            "delta": None,
            "pct_change": None,
            "failure_edges": failure_edges,
            "disconnected": True,
        }

    failure_total_time = failure_network.total_system_travel_time()
    delta = failure_total_time - baseline_total_time
    pct_change = (delta / baseline_total_time) * 100

    if verbose:
        print(f"\nPost-failure total system travel time: {failure_total_time:.2f}")
        print("\n" + "=" * 70)
        print("COMPARISON: Baseline vs. Post-Failure")
        print("=" * 70)
        print(f"Baseline total system travel time:     {baseline_total_time:.2f}")
        print(f"Post-failure total system travel time: {failure_total_time:.2f}")
        print(f"Change: {delta:+.2f} ({pct_change:+.2f}%)")
        if len(baseline_scc) != len(failure_scc):
            print(f"\nNOTE: connectivity changed -- baseline SCC had {len(baseline_scc)} nodes, "
                  f"post-failure SCC has {len(failure_scc)} nodes.")

    return {
        "baseline_time": baseline_total_time,
        "failure_time": failure_total_time,
        "delta": delta,
        "pct_change": pct_change,
        "failure_edges": failure_edges,
        "disconnected": False,
        "scc_changed": len(baseline_scc) != len(failure_scc),
    }


if __name__ == "__main__":
    run_failure_scenario(num_failures=1, verbose=True)
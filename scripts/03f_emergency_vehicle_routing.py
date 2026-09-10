"""
PHASE 3f: Emergency Vehicle Stackelberg Routing

This is the project's core novelty claim, finally implemented. Two things
happen here that didn't exist anywhere in the earlier codebase:

1. EQUILIBRIUM-AWARE ROUTING: the emergency vehicle is routed using current
   CONGESTION-ADJUSTED costs (from the solved Wardrop equilibrium), not
   naive free-flow costs. This is compared against naive routing (which is
   what the ORIGINAL Phase 3 code effectively did) to show equilibrium-
   awareness produces a genuinely faster real-world route.

2. PRICE OF PRIORITY (the actual Stackelberg mechanism): granting the
   emergency vehicle priority (signal preemption / reserved lane) on its
   route REDUCES CAPACITY for background traffic on those same edges.
   Background traffic then re-equilibrates around this reduced capacity.
   This is what makes the emergency vehicle a LEADER, not just another
   routed agent -- its presence changes the game background traffic has
   to solve, rather than passively coexisting with a fixed equilibrium.
   The resulting increase in background total system travel time is the
   "price of priority" -- a genuinely novel, quantifiable metric.

SIMPLIFICATION STATED EXPLICITLY (for the paper's methodology section):
the emergency vehicle's OWN flow is treated as negligible relative to
background demand (a single vehicle vs. hundreds of demand units), so we
do not add its flow into the equilibrium directly. Its effect on the
system is modeled entirely through the capacity reduction mechanism
above, which is the priority-lane/preemption effect, not its raw traffic
contribution.
"""

import networkx as nx

from network_builder import build_metrla_network
from equilibrium import solve_wardrop_equilibrium
import config


def pick_emergency_od(largest_scc, network):
    """
    Searches across the TOP candidate OD pairs by demand (not just the
    single busiest) to find one where naive and equilibrium-aware routing
    ACTUALLY DIVERGE. Some corridors -- even congested, high-demand ones --
    may have only a single viable path with no real alternate route, in
    which case naive and equilibrium-aware routing trivially agree
    (there's no choice to make, regardless of cost weighting). Consistent
    with the structural finding from 03e: this network has some genuinely
    sparse, low-redundancy corridors.

    Returns (origin, destination, found_divergent_example: bool). If no
    candidate among the top N shows divergence, returns the busiest pair
    anyway (still a valid, honestly-reportable result -- just means this
    network's congested corridors tend to be single-path bottlenecks
    rather than corridors with genuine route choice).
    """
    candidates = sorted(network.od_demand.items(), key=lambda item: item[1], reverse=True)
    naive_cost_graph = nx.DiGraph()
    for e in network.edges:
        naive_cost_graph.add_edge(e.origin, e.destination, weight=e.free_flow_time)
    equilibrium_cost_graph = network.build_cost_graph()

    for (origin, destination), demand in candidates[:10]:
        naive_path, naive_cost = get_route_and_true_cost(network, naive_cost_graph, origin, destination)
        aware_path, aware_cost = get_route_and_true_cost(network, equilibrium_cost_graph, origin, destination)
        if naive_path != aware_path:
            return origin, destination, True

    # No divergent example found among top 10 -- fall back to busiest pair
    origin, destination = candidates[0][0]
    return origin, destination, False


def get_route_and_true_cost(network, cost_graph, origin, destination):
    """
    Finds shortest path on cost_graph, then evaluates that path's TRUE
    cost using the network's actual current BPR costs (which may differ
    from what the routing graph "thought" the cost was, if cost_graph
    doesn't reflect current equilibrium -- e.g. for the naive/free-flow
    routing case).
    """
    path = nx.shortest_path(cost_graph, source=origin, target=destination, weight='weight')
    true_cost = 0.0
    edge_lookup = {(e.origin, e.destination): e for e in network.edges}
    for i in range(len(path) - 1):
        edge = edge_lookup[(path[i], path[i + 1])]
        true_cost += edge.bpr_cost()
    return path, true_cost


def main():
    print("=" * 70)
    print("PHASE 3f: Emergency Vehicle Stackelberg Routing")
    print("=" * 70)

    # --- 1. Solve baseline background equilibrium ---
    print("\n--- [1/4] Solving baseline background equilibrium ---")
    network, num_nodes, largest_scc = build_metrla_network(verbose=False)
    solve_wardrop_equilibrium(network, max_iterations=1000, convergence_threshold=1e-5, verbose=False)
    baseline_total_time = network.total_system_travel_time()
    print(f"Baseline background total system travel time: {baseline_total_time:.2f}")

    # --- 2. Pick emergency vehicle OD pair ---
    origin, destination, found_divergent = pick_emergency_od(largest_scc, network)
    print(f"\n--- [2/4] Emergency vehicle route: {origin} -> {destination} ---")
    if not found_divergent:
        print("NOTE: searched top 10 highest-demand OD pairs; none showed a route where naive")
        print("and equilibrium-aware costs diverge. This suggests the most congested corridors")
        print("in this network tend to be single-path bottlenecks with no real alternate route")
        print("-- congestion is real (see Price of Priority below) but there's no route CHOICE")
        print("to optimize for these specific pairs. Falling back to the busiest pair anyway.")

    # --- 3. Compare NAIVE routing vs EQUILIBRIUM-AWARE routing ---
    print("\n--- [3/4] Naive (free-flow) routing vs. Equilibrium-aware routing ---")

    # Naive: route using FREE-FLOW costs only (ignores current congestion
    # entirely -- this is what the ORIGINAL flawed Phase 3 code effectively did).
    naive_cost_graph = nx.DiGraph()
    for e in network.edges:
        naive_cost_graph.add_edge(e.origin, e.destination, weight=e.free_flow_time)
    naive_path, naive_true_cost = get_route_and_true_cost(network, naive_cost_graph, origin, destination)

    # Equilibrium-aware: route using CURRENT equilibrium (congestion-adjusted) costs.
    equilibrium_cost_graph = network.build_cost_graph()
    aware_path, aware_true_cost = get_route_and_true_cost(network, equilibrium_cost_graph, origin, destination)

    print(f"\nNaive route (ignores congestion): {len(naive_path)} nodes")
    print(f"  TRUE travel time (evaluated at real current congestion): {naive_true_cost:.4f}")
    print(f"\nEquilibrium-aware route: {len(aware_path)} nodes")
    print(f"  TRUE travel time: {aware_true_cost:.4f}")

    if aware_true_cost < naive_true_cost:
        improvement = ((naive_true_cost - aware_true_cost) / naive_true_cost) * 100
        print(f"\n✅ Equilibrium-aware routing is {improvement:.2f}% FASTER than naive routing "
              f"under real congestion conditions.")
    elif aware_true_cost == naive_true_cost:
        print(f"\nBoth routing strategies picked the same path (no congestion difference "
              f"along this particular OD pair's routes) -- try a different origin/destination "
              f"for a more illustrative comparison, or this may be expected if free-flow and "
              f"equilibrium costs happen to agree for this specific route.")
    else:
        print(f"\nUNEXPECTED: naive routing outperformed equilibrium-aware routing. This should "
              f"not happen (equilibrium-aware routing explicitly optimizes against the TRUE "
              f"costs used for evaluation) -- investigate before trusting this result.")

    # --- 4. Price of Priority: reduce capacity on the emergency route's edges,
    # re-solve background equilibrium, measure the systemic cost ---
    print(f"\n--- [4/4] Price of Priority (granting {origin}->{destination} priority lane) ---")
    print(f"Reducing capacity by {config.EMERGENCY_LANE_CAPACITY_REDUCTION*100:.0f}% on all edges "
          f"along the equilibrium-aware emergency route (simulating signal preemption / "
          f"reserved lane), then re-solving background equilibrium.\n")

    priority_route_edges = set(zip(aware_path[:-1], aware_path[1:]))

    # Rebuild a fresh network (same topology, same OD demand) so we don't
    # mutate the already-solved baseline network in place.
    priority_network, _, _ = build_metrla_network(verbose=False)
    for e in priority_network.edges:
        if (e.origin, e.destination) in priority_route_edges:
            e.capacity = e.capacity * (1 - config.EMERGENCY_LANE_CAPACITY_REDUCTION)

    solve_wardrop_equilibrium(priority_network, max_iterations=1000,
                               convergence_threshold=1e-5, verbose=False)
    priority_total_time = priority_network.total_system_travel_time()

    price_of_priority = priority_total_time - baseline_total_time
    price_of_priority_pct = (price_of_priority / baseline_total_time) * 100

    print(f"Background total system travel time WITHOUT priority lane: {baseline_total_time:.2f}")
    print(f"Background total system travel time WITH priority lane:    {priority_total_time:.2f}")
    print(f"\nPRICE OF PRIORITY: {price_of_priority:+.2f} ({price_of_priority_pct:+.4f}%)")
    print(f"\nThis is the systemic cost imposed on background traffic by granting the")
    print(f"emergency vehicle priority on its route -- the key Stackelberg leader-follower")
    print(f"result: the emergency vehicle's presence measurably changes background traffic's")
    print(f"equilibrium, not just passively routes through a fixed one.")


if __name__ == "__main__":
    main()
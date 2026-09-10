"""
PHASE 3e: Investigating the (66, 105) Outlier

Scenario seed=15 in the batch evaluation produced +8.34% impact -- roughly
6x worse than the next-worst random scenario, and 3x worse than the
"worst case" edge we deliberately picked by capacity utilization (147,109,
90.2% capacity, only +2.88% impact).

This script checks WHY: is (66, 105) a structural bottleneck (few or no
alternate paths between its endpoints) rather than simply a busy edge?
This distinguishes "capacity-critical" edges from "topologically-critical"
edges -- a real, useful distinction for the paper's discussion section.
"""

import networkx as nx

from network_builder import build_metrla_network
from equilibrium import solve_wardrop_equilibrium

OUTLIER_EDGE = (66, 105)

print("=" * 70)
print(f"Investigating outlier edge {OUTLIER_EDGE}")
print("=" * 70)

# --- 1. Baseline context: how much flow did this edge actually carry? ---
baseline_network, num_nodes, baseline_scc = build_metrla_network(verbose=False)
solve_wardrop_equilibrium(baseline_network, max_iterations=1000,
                           convergence_threshold=1e-5, verbose=False)

matching = next((e for e in baseline_network.edges
                  if e.origin == OUTLIER_EDGE[0] and e.destination == OUTLIER_EDGE[1]), None)
if matching:
    print(f"\nBaseline flow on {OUTLIER_EDGE}: {matching.flow:.2f} / {matching.capacity:.0f} "
          f"capacity ({matching.flow/matching.capacity*100:.1f}%)")
else:
    print(f"\nWARNING: edge {OUTLIER_EDGE} not found in baseline network -- can't proceed.")
    exit(1)

# --- 2. Build a plain topology graph (no BPR costs) to check structural properties ---
topo_graph = nx.DiGraph()
for e in baseline_network.edges:
    topo_graph.add_edge(e.origin, e.destination)

origin, destination = OUTLIER_EDGE

# Is this edge a "bridge" in the sense that removing it disconnects origin from destination?
topo_graph_without_edge = topo_graph.copy()
topo_graph_without_edge.remove_edge(origin, destination)

has_alternate_path = nx.has_path(topo_graph_without_edge, origin, destination)
print(f"\nDoes an alternate path from {origin} to {destination} exist without this edge? "
      f"{'Yes' if has_alternate_path else 'NO -- this edge is a structural bottleneck (cut edge)'}")

if has_alternate_path:
    # How much LONGER is the best alternate path, in hop count (topology only,
    # not weighted by cost -- just to see if it's a short detour or a long one)
    direct_hops = 1
    alt_path = nx.shortest_path(topo_graph_without_edge, origin, destination)
    alt_hops = len(alt_path) - 1
    print(f"Best alternate path: {alt_path} ({alt_hops} hops, vs. {direct_hops} hop directly)")
    print(f"Detour factor: {alt_hops}x longer in hop count")

# --- 3. Compare edge betweenness centrality: (66,105) vs the "worst case" edge (147,109) ---
print("\n--- Betweenness centrality comparison ---")
print("(Edge betweenness = how many shortest paths in the ENTIRE network pass through")
print(" this edge. High betweenness = structurally important regardless of current flow.)")

# NOTE: this can be slow on larger graphs since it's O(V*E) roughly; 207 nodes is fine.
edge_betweenness = nx.edge_betweenness_centrality(topo_graph)

outlier_betweenness = edge_betweenness.get(OUTLIER_EDGE, 0.0)
worst_case_by_capacity = (147, 109)
capacity_edge_betweenness = edge_betweenness.get(worst_case_by_capacity, 0.0)

# Rank both edges among all edges by betweenness
sorted_betweenness = sorted(edge_betweenness.items(), key=lambda x: x[1], reverse=True)
outlier_rank = next((i for i, (e, _) in enumerate(sorted_betweenness) if e == OUTLIER_EDGE), None)
capacity_edge_rank = next((i for i, (e, _) in enumerate(sorted_betweenness) if e == worst_case_by_capacity), None)

print(f"\nOutlier edge {OUTLIER_EDGE}: betweenness={outlier_betweenness:.6f}, "
      f"rank #{outlier_rank+1 if outlier_rank is not None else 'N/A'} of {len(edge_betweenness)} edges")
print(f"Capacity-based 'worst case' edge {worst_case_by_capacity}: "
      f"betweenness={capacity_edge_betweenness:.6f}, "
      f"rank #{capacity_edge_rank+1 if capacity_edge_rank is not None else 'N/A'} of {len(edge_betweenness)} edges")

print("\n--- Interpretation ---")
if outlier_rank is not None and capacity_edge_rank is not None:
    if outlier_rank < capacity_edge_rank:
        print(f"The outlier edge has HIGHER betweenness centrality (rank #{outlier_rank+1}) than")
        print(f"the capacity-based worst-case edge (rank #{capacity_edge_rank+1}).")
        print(f"This supports the hypothesis: {OUTLIER_EDGE} is critical due to network TOPOLOGY")
        print(f"(few alternate routes), not because it was heavily loaded with traffic.")
        print(f"Capacity utilization and structural criticality are measuring different things.")
    else:
        print(f"The outlier edge does NOT have higher betweenness than the capacity-based edge.")
        print(f"The +8.34% impact may be driven by something else -- worth checking OD-pair-")
        print(f"specific effects (did this edge happen to be critical for one particular")
        print(f"high-demand OD pair in this specific synthetic demand set, rather than being")
        print(f"structurally critical in general).")
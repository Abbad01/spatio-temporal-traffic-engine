"""
Wardrop equilibrium on the REAL METR-LA network (207 nodes) -- baseline,
no failure simulated yet, no emergency vehicle yet.

This is the scale-up checkpoint after 03a's toy-network correctness test.
Confirms the solver converges at real scale and that the documented
modeling assumptions (see network_builder.py and config.py) produce a
network with genuine, checkable congestion behavior.
"""

from network_builder import build_metrla_network
from equilibrium import solve_wardrop_equilibrium

print("--- PHASE 3b: Wardrop Equilibrium on Real METR-LA Network (Baseline) ---\n")

network, num_nodes, largest_scc = build_metrla_network()

print("\n--- Solving Wardrop Equilibrium ---\n")
solve_wardrop_equilibrium(network, max_iterations=1000, convergence_threshold=1e-5, verbose=True)

# Summary statistics
flows = [e.flow for e in network.edges]
used_edges = [e for e in network.edges if e.flow > 1e-6]

print("\n--- Summary ---")
print(f"Total system travel time: {network.total_system_travel_time():.2f}")
print(f"Edges carrying nonzero flow: {len(used_edges)} / {len(network.edges)}")
print(f"Flow range across used edges: min={min(f for f in flows if f > 1e-6):.4f}, "
      f"max={max(flows):.4f}")

used_costs = [e.bpr_cost() for e in used_edges]
congestion_ratios = [e.bpr_cost() / e.free_flow_time for e in used_edges]
utilization_ratios = [e.flow / e.capacity for e in used_edges]

print(f"\nCost range across USED edges only: min={min(used_costs):.4f}, max={max(used_costs):.4f}")
print(f"Congestion ratio (cost/free_flow_time) across used edges: "
      f"min={min(congestion_ratios):.4f}, max={max(congestion_ratios):.4f}, "
      f"mean={sum(congestion_ratios)/len(congestion_ratios):.4f}")
print(f"\nCapacity utilization (flow/capacity) across used edges: "
      f"min={min(utilization_ratios):.4f}, max={max(utilization_ratios):.4f}")
most_congested = max(used_edges, key=lambda e: e.flow / e.capacity)
print(f"Most utilized edge: ({most_congested.origin}->{most_congested.destination}), "
      f"flow={most_congested.flow:.2f}/{most_congested.capacity:.0f} capacity "
      f"({most_congested.flow/most_congested.capacity*100:.1f}%), "
      f"congestion ratio={most_congested.bpr_cost()/most_congested.free_flow_time:.4f}")

total_demand = sum(network.od_demand.values())
total_flow_check = sum(flows)
print(f"\nFlow conservation sanity check:")
print(f"  Total OD demand generated: {total_demand:.2f}")
print(f"  Sum of all edge flows:     {total_flow_check:.2f}")
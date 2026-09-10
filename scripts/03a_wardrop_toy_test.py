"""
Toy network test for the Wardrop equilibrium solver.

Classic textbook example (Sheffi-style): a single origin-destination pair
connected by TWO PARALLEL ROUTES with different free-flow times and
capacities. This has a well-understood equilibrium behavior you can sanity
check by hand:

  - At equilibrium, if BOTH routes are used, they must have EQUAL travel
    time (Wardrop's First Principle -- no traveler can improve their time
    by switching routes).
  - The faster/higher-capacity route should carry more flow than the
    slower/lower-capacity route.
  - If you increase total demand, both routes' travel times should rise,
    but their equalized value should also rise.

Run this BEFORE trusting the solver on your real METR-LA network -- if the
equilibrium found here doesn't satisfy "both used routes have equal cost",
something in equilibrium.py is broken.
"""

from equilibrium import Edge, TrafficNetwork, solve_wardrop_equilibrium

print("--- TOY NETWORK: Two Parallel Routes, Single OD Pair ---\n")

# Two routes from node 0 -> node 1.
# Route A: faster free-flow time, but lower capacity (a shorter, narrower road)
# Route B: slower free-flow time, but higher capacity (a longer, wider road)
edges = [
    Edge(origin=0, destination=1, free_flow_time=10.0, capacity=100.0),  # Route A
    Edge(origin=0, destination=1, free_flow_time=15.0, capacity=150.0),  # Route B
]

# NOTE: TrafficNetwork's _edge_lookup keys edges by (origin, destination),
# so two parallel edges between the SAME node pair will collide in that
# dict. For this toy test we work around it by treating them as two
# separate mini-networks solved independently is wrong for a real parallel-
# route case -- instead, verify with unique intermediate "virtual" nodes so
# both routes are distinguishable in the graph. This mirrors how real
# parallel road segments would be represented if they don't literally share
# both endpoints as a single graph edge.
edges = [
    Edge(origin=0, destination=10, free_flow_time=5.0, capacity=100.0),   # Route A leg 1
    Edge(origin=10, destination=1, free_flow_time=5.0, capacity=100.0),   # Route A leg 2
    Edge(origin=0, destination=20, free_flow_time=7.5, capacity=150.0),   # Route B leg 1
    Edge(origin=20, destination=1, free_flow_time=7.5, capacity=150.0),   # Route B leg 2
]

od_demand = {(0, 1): 200.0}  # 200 units of travel demand from node 0 to node 1

network = TrafficNetwork(edges=edges, od_demand=od_demand)

print("Before solving:")
for e in network.edges:
    print(f"  Edge ({e.origin}->{e.destination}): free_flow_time={e.free_flow_time}, "
          f"capacity={e.capacity}")

solve_wardrop_equilibrium(network, max_iterations=1000, convergence_threshold=1e-5, verbose=True)

print("\n--- Equilibrium Result ---")
route_a_flow = None
route_b_flow = None

for e in network.edges:
    cost = e.bpr_cost()
    print(f"  Edge ({e.origin}->{e.destination}): flow={e.flow:.4f}, cost={cost:.4f}")
    if e.origin == 0 and e.destination == 10:
        route_a_flow = e.flow
    if e.origin == 0 and e.destination == 20:
        route_b_flow = e.flow

# Recompute full-route costs (sum of both legs) for the equal-cost check
route_a_cost = sum(e.bpr_cost() for e in network.edges if e.origin in (0, 10) and e.destination in (10, 1))
route_b_cost = sum(e.bpr_cost() for e in network.edges if e.origin in (0, 20) and e.destination in (20, 1))

print(f"\nRoute A (via node 10, free-flow=10, capacity=100) -- flow: {route_a_flow:.4f}, travel time: {route_a_cost:.4f}")
print(f"Route B (via node 20, free-flow=15, capacity=150) -- flow: {route_b_flow:.4f}, travel time: {route_b_cost:.4f}")
print(f"Flow conservation check (should equal 200): {route_a_flow + route_b_flow:.4f}")

# CHANGED: use a RELATIVE cost difference, not an absolute one -- an
# absolute threshold of 0.01 is meaningless without knowing the scale of
# the costs involved (here costs are ~15, so 0.01 is a 0.07% tolerance,
# unnecessarily strict). 1% relative difference is a reasonable practical
# equilibrium tolerance.
relative_cost_diff = abs(route_a_cost - route_b_cost) / route_a_cost
print(f"\nRelative cost difference between routes: {relative_cost_diff*100:.3f}%")
if relative_cost_diff < 0.01:
    print("✅ PASS: Both routes have equal travel time (within 1%) at equilibrium.")
    print("   This confirms Wardrop's First Principle is satisfied by the solver.")
else:
    print("⚠️  FAIL: Routes do not have equal cost within tolerance. Investigate before")
    print("   proceeding to the real network.")

# NOTE: which route carries more flow depends on BOTH free-flow time AND
# capacity -- Route A is faster but more capacity-constrained, Route B is
# slower but roomier, so there's no single "obviously correct" answer for
# which one should carry more flow without solving the actual equilibrium.
# The real check is the equal-cost condition above, not flow magnitude.
print(f"\nFor reference: Route A carries {route_a_flow:.1f} units, Route B carries "
      f"{route_b_flow:.1f} units. Route A's lower free-flow time (10 vs 15) lets it "
      f"absorb more congestion before its cost catches up to Route B's -- this is why "
      f"flow split is uneven even though costs equalize.")
"""
Wardrop User Equilibrium solver via the Frank-Wolfe algorithm.

This is the missing piece from the original routing code: previously,
Phase 3 used plain Dijkstra with STGCN-predicted travel times as static edge
weights. That's single-agent shortest-path routing, NOT Wardrop equilibrium
-- it ignores the fact that background traffic redistributes itself across
the network as congestion builds, and that link travel time INCREASES as
more flow uses that link.

This module fixes that gap. It implements:
  1. A BPR (Bureau of Public Roads) cost function -- link travel time as a
     function of how much flow is currently on that link.
  2. All-or-nothing (AON) assignment -- given current link costs, route all
     demand for each OD pair onto its single shortest path.
  3. Frank-Wolfe -- the standard iterative algorithm that repeatedly performs
     AON assignment, then blends the new flow pattern into the running
     solution, converging to the Wardrop equilibrium flow distribution.

Reference: Sheffi, Y. (1985). Urban Transportation Networks: Equilibrium
Analysis with Mathematical Programming Methods. Chapters 3-4.
"""

from dataclasses import dataclass, field
import networkx as nx


# ----------------------------------------------------------------------
# 1. Network representation
# ----------------------------------------------------------------------

@dataclass
class Edge:
    """
    A single directed road link.

    free_flow_time: travel time (any consistent time unit) when the link is
                     empty -- i.e. distance / speed_limit.
    capacity: the flow level at which the link becomes seriously congested.
              Not a hard cap -- BPR allows flow to exceed capacity, cost just
              rises steeply past that point.
    alpha, beta: standard US Bureau of Public Roads defaults (0.15, 4).
                 Do not need to be tuned in most cases.
    flow: current assigned flow on this link. Starts at 0, updated by the
          Frank-Wolfe loop.
    """
    origin: int
    destination: int
    free_flow_time: float
    capacity: float
    alpha: float = 0.15
    beta: float = 4.0
    flow: float = 0.0

    def bpr_cost(self, flow: float = None) -> float:
        """
        BPR cost function: travel time as a function of flow.
        If flow is not provided, uses the edge's current stored flow.

        travel_time = free_flow_time * (1 + alpha * (flow/capacity)^beta)

        At flow=0: cost = free_flow_time (empty road, fastest possible).
        As flow -> capacity: cost rises sharply (this is what makes
        Wardrop equilibrium meaningfully different from plain shortest path
        -- routing everyone onto the "fastest" road makes it slow).
        """
        f = self.flow if flow is None else flow
        return self.free_flow_time * (1 + self.alpha * (f / self.capacity) ** self.beta)


class TrafficNetwork:
    """
    A directed graph of Edge objects, with OD (origin-destination) demand.
    Wraps a NetworkX DiGraph for shortest-path computation, but keeps flow
    and cost state in the Edge objects themselves (NetworkX edge attributes
    would work too, but this keeps the BPR logic self-contained and easy
    to unit-test in isolation).
    """

    def __init__(self, edges: list[Edge], od_demand: dict[tuple[int, int], float]):
        self.edges = edges
        self.od_demand = od_demand  # {(origin, dest): demand_volume}
        self._edge_lookup = {(e.origin, e.destination): e for e in edges}

    def reset_flows(self):
        for e in self.edges:
            e.flow = 0.0

    def build_cost_graph(self, flows: dict[tuple[int, int], float] = None) -> nx.DiGraph:
        """
        Builds a NetworkX DiGraph where each edge's weight is its CURRENT
        BPR cost (given current flow, or a provided flow dict). This is
        what shortest-path search runs against at each Frank-Wolfe iteration
        -- costs change every iteration as flow shifts.
        """
        g = nx.DiGraph()
        for e in self.edges:
            flow = e.flow if flows is None else flows.get((e.origin, e.destination), 0.0)
            g.add_edge(e.origin, e.destination, weight=e.bpr_cost(flow))
        return g

    def total_system_travel_time(self) -> float:
        """Sum of flow * cost across all links -- a useful summary metric
        for comparing equilibrium solutions or reporting results."""
        return sum(e.flow * e.bpr_cost() for e in self.edges)


# ----------------------------------------------------------------------
# 2. All-or-nothing (AON) assignment
# ----------------------------------------------------------------------

def all_or_nothing_assignment(network: TrafficNetwork, cost_graph: nx.DiGraph) -> dict[tuple[int, int], float]:
    """
    Given a cost graph (current BPR costs as edge weights), find the
    shortest path for each OD pair and dump that OD pair's ENTIRE demand
    onto every edge along that single path.

    This is "all-or-nothing" because, unlike the final equilibrium, it
    doesn't split demand across multiple paths -- every traveler between
    a given origin and destination is assumed to take the single cheapest
    route AT THIS ITERATION's costs. Repeating this while costs update is
    what eventually spreads flow across multiple paths in the final result.

    Returns: dict of {(edge_origin, edge_dest): flow_assigned_this_iteration}
    """
    aon_flows = {(e.origin, e.destination): 0.0 for e in network.edges}

    for (origin, dest), demand in network.od_demand.items():
        try:
            path = nx.shortest_path(cost_graph, source=origin, target=dest, weight='weight')
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            # NetworkXNoPath: both nodes exist in the graph but no path connects them.
            # NodeNotFound: a failure removed EVERY edge touching one of these nodes,
            # so it doesn't exist in the graph object at all. Both represent the same
            # real-world outcome -- this OD pair's demand cannot currently be served --
            # so both are treated identically here.
            raise RuntimeError(
                f"No path exists from {origin} to {dest} -- check network connectivity "
                f"or whether a simulated failure has disconnected this OD pair entirely."
            )

        for i in range(len(path) - 1):
            edge_key = (path[i], path[i + 1])
            aon_flows[edge_key] += demand

    return aon_flows


# ----------------------------------------------------------------------
# 3. Frank-Wolfe algorithm
# ----------------------------------------------------------------------

def solve_wardrop_equilibrium(
    network: TrafficNetwork,
    max_iterations: int = 1000,
    convergence_threshold: float = 1e-5,
    verbose: bool = False,
) -> TrafficNetwork:
    """
    Solves for the Wardrop User Equilibrium flow distribution using the
    Frank-Wolfe algorithm.

    Algorithm, each iteration:
      1. Build the cost graph using CURRENT flows.
      2. Run all-or-nothing assignment against those costs -> target flows.
      3. Move the current flow pattern a fraction of the way toward the
         target flows (the step size shrinks over iterations, which is what
         makes this converge instead of oscillating forever).
      4. Check convergence: has the flow pattern stopped changing much?

    Returns the same TrafficNetwork object, with each Edge's .flow attribute
    set to its equilibrium value.
    """
    network.reset_flows()

    # Iteration 0: no flow anywhere yet, so cost graph = free-flow costs.
    # AON assignment against free-flow costs gives a reasonable starting point.
    cost_graph = network.build_cost_graph(flows={k: 0.0 for k in network._edge_lookup})
    aon_flows = all_or_nothing_assignment(network, cost_graph)
    for e in network.edges:
        e.flow = aon_flows[(e.origin, e.destination)]

    for iteration in range(1, max_iterations + 1):
        # Step 1+2: shortest paths against current costs, all-or-nothing assignment
        cost_graph = network.build_cost_graph()
        aon_flows = all_or_nothing_assignment(network, cost_graph)

        # CHANGED: relative gap convergence check instead of raw flow change.
        # Frank-Wolfe's all-or-nothing step causes flow to oscillate between
        # close-cost paths even when the solution is essentially converged --
        # raw flow change is a noisy signal because of this. The standard,
        # more stable metric (used throughout the traffic assignment
        # literature, e.g. Sheffi Ch. 4) is the RELATIVE GAP: how much lower
        # would total system cost be if all flow used current shortest paths,
        # compared to the actual current total cost? This shrinks smoothly
        # toward 0 even while individual edge flows are still oscillating.
        current_total_cost = sum(e.flow * e.bpr_cost() for e in network.edges)
        aon_total_cost = sum(
            aon_flows[(e.origin, e.destination)] * e.bpr_cost()
            for e in network.edges
        )
        relative_gap = (current_total_cost - aon_total_cost) / current_total_cost if current_total_cost > 0 else 0.0

        # Step 3: move current flow toward AON target.
        # Standard Frank-Wolfe step size: 1/(iteration+1), which shrinks
        # over time -- big moves early, fine-tuning later.
        step_size = 1.0 / (iteration + 1)

        for e in network.edges:
            target_flow = aon_flows[(e.origin, e.destination)]
            e.flow = e.flow + step_size * (target_flow - e.flow)

        if verbose:
            print(f"  Iteration {iteration:03d} | Relative gap: {relative_gap:.6f} "
                  f"| Total system travel time: {network.total_system_travel_time():.4f}")

        # Step 4: convergence check, based on relative gap
        if relative_gap < convergence_threshold:
            if verbose:
                print(f"Converged after {iteration} iterations (relative gap < {convergence_threshold}).")
            break
    else:
        if verbose:
            print(f"WARNING: did not converge within {max_iterations} iterations "
                  f"(relative_gap={relative_gap:.6f}). Consider raising "
                  f"max_iterations or checking network for pathological structure.")

    return network
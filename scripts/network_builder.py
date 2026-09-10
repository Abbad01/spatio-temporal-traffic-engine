"""
Shared METR-LA network construction logic.

Extracted from 03b so that the BASELINE network and any FAILURE-SCENARIO
network (edges removed) are built with identical logic -- same free-flow
time derivation, same SCC filtering, same OD demand generation. If this
logic were duplicated across files, a fix or tweak in one place could
silently drift out of sync with the other, making before/after failure
comparisons unreliable.
"""

import random
import networkx as nx
from torch_geometric_temporal.dataset import METRLADatasetLoader

from equilibrium import Edge, TrafficNetwork
import config


def load_metrla_topology():
    """
    Loads just the graph topology (edge_index, edge_attr) from METR-LA --
    no need for the full temporal signal for equilibrium purposes.
    Returns (edge_index_numpy, edge_attr_numpy, num_nodes).
    """
    loader = METRLADatasetLoader()
    dataset = loader.get_dataset(
        num_timesteps_in=config.NUM_TIMESTEPS_IN,
        num_timesteps_out=config.NUM_TIMESTEPS_OUT
    )
    snapshot = next(iter(dataset))
    edge_index = snapshot.edge_index.numpy()
    edge_attr = snapshot.edge_attr.numpy()
    num_nodes = snapshot.x.shape[0]
    return edge_index, edge_attr, num_nodes


def build_edges(edge_index, edge_attr, excluded_edges=None):
    """
    Builds Edge objects with derived free-flow time and assumed capacity
    (see config.py for the documented assumptions).

    excluded_edges: optional set of (origin, destination) tuples to OMIT
                     entirely -- this is how a simulated infrastructure
                     failure (e.g. a collapsed bridge, a closed road) is
                     represented: the edge simply does not exist in this
                     network, so no path can ever use it.
    """
    excluded_edges = excluded_edges or set()
    MIN_EDGE_ATTR = 1e-3
    edges = []
    num_edges = edge_index.shape[1]

    for i in range(num_edges):
        origin = int(edge_index[0, i])
        destination = int(edge_index[1, i])

        if (origin, destination) in excluded_edges:
            continue  # simulated failure: this link does not exist

        attr = max(float(edge_attr[i]), MIN_EDGE_ATTR)
        free_flow_time = config.FREE_FLOW_TIME_SCALE / attr

        edges.append(Edge(
            origin=origin,
            destination=destination,
            free_flow_time=free_flow_time,
            capacity=config.DEFAULT_EDGE_CAPACITY,
        ))

    return edges


def find_largest_scc(edges):
    """Returns the set of nodes in the largest strongly connected component."""
    g = nx.DiGraph()
    for e in edges:
        g.add_edge(e.origin, e.destination)
    sccs = list(nx.strongly_connected_components(g))
    return max(sccs, key=len)


def generate_od_demand(connected_nodes, seed=None):
    """
    Generates synthetic OD demand between random pairs of nodes within the
    connected component. Uses config.OD_GENERATION_SEED by default so the
    SAME demand pattern is used across baseline and failure scenarios --
    this is essential for a fair comparison (you want to know how the SAME
    travel demand behaves differently when a link fails, not how different
    random demand happens to behave).
    """
    seed = config.OD_GENERATION_SEED if seed is None else seed
    random.seed(seed)

    connected_nodes = list(connected_nodes)
    od_demand = {}
    attempts = 0
    max_attempts = config.NUM_SYNTHETIC_OD_PAIRS * 50

    while len(od_demand) < config.NUM_SYNTHETIC_OD_PAIRS and attempts < max_attempts:
        origin = random.choice(connected_nodes)
        destination = random.choice(connected_nodes)
        attempts += 1
        if origin == destination:
            continue
        if (origin, destination) in od_demand:
            continue
        od_demand[(origin, destination)] = random.uniform(config.OD_DEMAND_MIN, config.OD_DEMAND_MAX)

    return od_demand


def build_metrla_network(excluded_edges=None, verbose=True):
    """
    Full pipeline: load topology -> build edges (optionally excluding some,
    for failure simulation) -> find largest SCC -> generate OD demand
    within that SCC -> return a ready-to-solve TrafficNetwork.

    IMPORTANT: OD demand is generated using the SAME seed regardless of
    excluded_edges, but is filtered to nodes within whatever the largest
    SCC happens to be for THIS network (which may shrink if a failure
    disconnects part of the graph). This means a failure scenario's OD
    demand may differ slightly from baseline if the failure changes which
    nodes are reachable -- this is realistic (a failure can strand demand
    entirely) but worth being aware of when comparing scenarios.
    """
    edge_index, edge_attr, num_nodes = load_metrla_topology()
    edges = build_edges(edge_index, edge_attr, excluded_edges=excluded_edges)

    largest_scc = find_largest_scc(edges)
    if verbose:
        excluded_count = num_nodes - len(largest_scc)
        print(f"Graph: {num_nodes} nodes, {len(edges)} directed edges "
              f"({len(excluded_edges) if excluded_edges else 0} excluded for failure simulation)")
        print(f"Largest SCC: {len(largest_scc)}/{num_nodes} nodes "
              f"({excluded_count} excluded from OD sampling)")

    od_demand = generate_od_demand(largest_scc)
    if verbose:
        print(f"Generated {len(od_demand)} OD pairs within the connected component")

    network = TrafficNetwork(edges=edges, od_demand=od_demand)
    return network, num_nodes, largest_scc
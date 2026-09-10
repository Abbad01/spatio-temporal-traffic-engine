"""
PHASE 3d: Batch Failure Scenario Evaluation

Runs MANY failure scenarios (not just one) and aggregates results. A single
random edge failure (03c) is a demo; this is closer to the actual
evaluation your paper needs -- mean/std/min/max impact across many
scenarios, plus a deliberately targeted "worst case" test on the most
heavily-utilized edge in the network.

Why both random AND targeted matter:
  - Random sampling mostly hits lightly-used edges (there are far more of
    them), so a purely random batch will systematically understate how bad
    failures CAN be.
  - A targeted worst-case test on the busiest edge shows the upper end of
    what's possible, which is often the more policy-relevant number for a
    resilience paper (e.g. "the worst-case failure increases system travel
    time by X%", not just "the average failure increases it by Y%").
"""

import statistics
import importlib

from network_builder import build_metrla_network
from equilibrium import solve_wardrop_equilibrium
import config

# CHANGED: Python module names cannot start with a digit, so
# "from 03c_failure_simulation import run_failure_scenario" is actually
# invalid syntax -- it would raise a SyntaxError. importlib is the correct
# way to import a module whose filename starts with a number.
_03c = importlib.import_module("03c_failure_simulation")
run_failure_scenario = _03c.run_failure_scenario

NUM_RANDOM_SCENARIOS = 20


def find_busiest_edge():
    """
    Solves baseline equilibrium once and returns the (origin, dest) of the
    single most heavily-utilized edge -- used for the deliberate worst-case
    scenario, as opposed to relying on random sampling to find it.
    """
    network, _, _ = build_metrla_network(verbose=False)
    solve_wardrop_equilibrium(network, max_iterations=1000, convergence_threshold=1e-5, verbose=False)
    used_edges = [e for e in network.edges if e.flow > 1e-6]
    busiest = max(used_edges, key=lambda e: e.flow / e.capacity)
    return (busiest.origin, busiest.destination), busiest.flow / busiest.capacity


def main():
    print("=" * 70)
    print(f"PHASE 3d: Batch Failure Scenario Evaluation ({NUM_RANDOM_SCENARIOS} random + 1 worst-case)")
    print("=" * 70)

    # --- Random scenarios ---
    results = []
    disconnected_count = 0

    for i in range(NUM_RANDOM_SCENARIOS):
        print(f"\n[Scenario {i+1}/{NUM_RANDOM_SCENARIOS}] Running (seed={i})...")
        result = run_failure_scenario(num_failures=1, seed=i, verbose=False)
        if result["disconnected"]:
            disconnected_count += 1
            print(f"  -> Disconnected an OD pair entirely (excluded from averages)")
        else:
            print(f"  -> Failed edge {result['failure_edges']}: "
                  f"{result['pct_change']:+.3f}% change in total system travel time")
            results.append(result)

    # --- Deliberate worst-case scenario ---
    print(f"\n{'='*70}")
    print("Deliberate worst-case scenario: failing the single busiest edge")
    print(f"{'='*70}")
    busiest_edge, utilization = find_busiest_edge()
    print(f"Busiest edge in baseline network: {busiest_edge} at {utilization*100:.1f}% capacity")
    worst_case_result = run_failure_scenario(target_edges={busiest_edge}, verbose=False)
    if worst_case_result["disconnected"]:
        print("Worst-case edge removal disconnected an OD pair entirely.")
    else:
        print(f"Worst-case scenario: {worst_case_result['pct_change']:+.3f}% change "
              f"in total system travel time")

    # --- Aggregate statistics across random scenarios ---
    print(f"\n{'='*70}")
    print("AGGREGATE RESULTS ACROSS RANDOM SCENARIOS")
    print(f"{'='*70}")
    print(f"Scenarios run: {NUM_RANDOM_SCENARIOS}")
    print(f"Disconnected (excluded from stats): {disconnected_count}")
    print(f"Valid scenarios: {len(results)}")

    if len(results) >= 2:
        pct_changes = [r["pct_change"] for r in results]
        print(f"\nPercent change in total system travel time:")
        print(f"  Mean:   {statistics.mean(pct_changes):+.4f}%")
        print(f"  Median: {statistics.median(pct_changes):+.4f}%")
        print(f"  Stdev:  {statistics.stdev(pct_changes):.4f}%")
        print(f"  Min:    {min(pct_changes):+.4f}%")
        print(f"  Max:    {max(pct_changes):+.4f}%")

        scc_changed_count = sum(1 for r in results if r.get("scc_changed"))
        print(f"\nScenarios where failure changed network connectivity (SCC size): "
              f"{scc_changed_count}/{len(results)}")
    else:
        print("\nNot enough valid scenarios to compute meaningful statistics.")

    if not worst_case_result["disconnected"]:
        print(f"\nWorst-case (busiest edge) scenario: {worst_case_result['pct_change']:+.4f}%")
        if len(results) >= 2:
            print(f"Compare to random-scenario mean: {statistics.mean(pct_changes):+.4f}%")
            print(f"(This comparison is the key resilience finding: how much worse is a")
            print(f" TARGETED/worst-case failure compared to a typical random one?)")


if __name__ == "__main__":
    main()
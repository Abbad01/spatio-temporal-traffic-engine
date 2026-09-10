# Spatio-Temporal Traffic Engine

Measuring the network-wide cost of granting an emergency vehicle routing
priority, by coupling a spatio-temporal traffic forecast to an exactly-solved
Wardrop user equilibrium on the METR-LA sensor network.

Code accompanying the paper *Coupling Spatio-Temporal Graph Forecasting with
Wardrop Equilibrium: Measuring the Network-Wide Cost of Emergency Vehicle
Priority*.

---

## What this measures

Granting an emergency vehicle priority — through signal preemption or a
reserved lane — reduces the capacity available to everyone else along its
route. That delay has been measured at the intersections the vehicle passes
through. It has not generally been measured across the network as a whole,
even though under user equilibrium drivers far from the emergency route also
change their paths in response.

This repository measures that wider cost, termed the **price of priority**:
the change in total system travel time once background traffic has
re-equilibrated around the reserved capacity.

**Result:** across 92 trials, reserving a fifth of the capacity along the
emergency route increased background travel time in every trial, by 0.195% on
average. The effect remained positive in all 87 trials of a separate
sensitivity analysis spanning a fourfold range of the capacity assumption.

---

## Repository structure

```
scripts/
├── config.py                          Hyperparameters, paths, modelling assumptions
├── model.py                           T-GCN architecture (single layer, residual)
├── equilibrium.py                     BPR cost function, Frank-Wolfe solver
├── network_builder.py                 METR-LA network construction, synthetic OD demand
├── stgcn_equilibrium_bridge.py        Forecast output to equilibrium link costs
│
├── 01_data_pipeline.py                Data loading and verification
├── 02_stgcn_training.py               Model training
├── 02b_diagnostic_baseline_check.py   Model versus naive persistence baseline
│
├── 03a_wardrop_toy_test.py            Solver verification on a hand-checkable network
├── 03b_wardrop_metrla_baseline.py     Equilibrium on the full 207-node network
├── 03c_failure_simulation.py          Single link failure, before/after comparison
├── 03d_batch_failure_evaluation.py    Batch of failure scenarios
├── 03e_outlier_investigation.py       Betweenness centrality and cut-edge analysis
├── 03f_emergency_vehicle_routing.py   Routing comparison and price of priority
├── 03g_stgcn_dynamic_equilibrium.py   Forecast-informed equilibrium
├── 03h_combined_evaluation.py         Full evaluation, 100 trials
├── 03i_capacity_sensitivity.py        Sensitivity across three capacity levels
├── 05_make_figures.py                 Figures from the results CSVs
│
└── Archive/                           Superseded early prototypes, kept for reference

results/    CSV output from 03h and 03i
figures/    Figures used in the paper
```

---

## Running it

Requires Python 3.11+.

```bash
pip install torch torch-geometric-temporal networkx pandas matplotlib
```

METR-LA downloads automatically on first run.

```bash
python scripts/02_stgcn_training.py              # slow on CPU; GPU recommended
python scripts/02b_diagnostic_baseline_check.py  # model vs persistence
python scripts/03a_wardrop_toy_test.py           # verify the solver
python scripts/03h_combined_evaluation.py        # full evaluation
python scripts/03i_capacity_sensitivity.py       # sensitivity analysis
python scripts/05_make_figures.py                # figures
```

Scripts `03b` through `03g` reproduce individual intermediate results and can
be run separately.

---

## Findings

- **Price of priority: +0.195% on average, positive in 92/92 trials**, and in
  all 87 trials across a fourfold capacity range. The magnitude is small; the
  consistency of the sign is the substantive result.
- **Congestion-aware routing helped only where an alternative path existed**,
  in 61 of 92 trials, falling to 41% under higher assumed capacity. Its
  benefit is bounded by network redundancy rather than by forecast quality.
- **Neither betweenness centrality nor capacity utilisation reliably predicted
  which failures mattered most.** Which measure performed better depended on
  the congestion regime.
- **Roughly one in twelve random single-link failures disconnected an
  origin–destination pair entirely** rather than merely lengthening its path.

---

## Assumptions and limitations

METR-LA provides sensor speeds and a graph structure, but no road lengths,
speed limits, lane counts, capacities, or origin–destination matrix. The
following are therefore derived or generated, and are stated as assumptions
rather than measurements:

- **Free-flow travel time** is derived from the dataset's edge similarity
  weight, not from road geometry.
- **Link capacity** is assumed uniform. Sensitivity to this was tested across
  a fourfold range.
- **Travel demand** is synthetic, sampled uniformly at random. Real demand is
  spatially structured; this is the assumption most likely to affect the
  results and the one least examined.

Additionally: traffic assignment is static rather than dynamic; failures are
independent single-link removals rather than spatially correlated events; and
the forecasting model does not outperform a naive persistence baseline at the
30-minute horizon, which is discussed in the paper.

---

## License

MIT — see `LICENSE`.

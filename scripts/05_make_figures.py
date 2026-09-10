"""
Generates the three figures used in the paper, from the CSVs written by
03h_combined_evaluation.py and 03i_capacity_sensitivity.py.

Run from the project root, after both evaluation scripts have completed:

    python scripts/05_make_figures.py

Outputs vector PDFs to figures/, which is the format to use in LaTeX --
raster formats blur when the document is zoomed.
"""

import os
import matplotlib
matplotlib.use("Agg")          # no display needed
import matplotlib.pyplot as plt
import pandas as pd

# --- Consistent styling across all figures -------------------------------
# Serif fonts to match the LaTeX body text; modest sizes because figures are
# scaled down when placed in the document, which magnifies font size.
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

COMBINED_CSV = "results/combined_evaluation.csv"
SENSITIVITY_CSV = "results/capacity_sensitivity.csv"
OUTDIR = "figures"

os.makedirs(OUTDIR, exist_ok=True)


def load_data():
    if not os.path.exists(COMBINED_CSV):
        raise FileNotFoundError(
            f"{COMBINED_CSV} not found. Run 03h_combined_evaluation.py first."
        )
    if not os.path.exists(SENSITIVITY_CSV):
        raise FileNotFoundError(
            f"{SENSITIVITY_CSV} not found. Run 03i_capacity_sensitivity.py first."
        )
    return pd.read_csv(COMBINED_CSV), pd.read_csv(SENSITIVITY_CSV)


# -------------------------------------------------------------------------
# Figure 1: distribution of the price of priority
# -------------------------------------------------------------------------
def figure_price_of_priority(combined):
    values = combined["price_of_priority_pct"].dropna()

    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    ax.hist(values, bins=20, color="0.4", edgecolor="white", linewidth=0.5)

    # Zero line: the point of the figure is that no observation falls left of it
    ax.axvline(0, color="black", linewidth=1.0, linestyle="-")
    ax.axvline(values.mean(), color="black", linewidth=1.0, linestyle="--",
               label=f"mean = {values.mean():+.3f}%")

    ax.set_xlabel("Price of priority (% of total system travel time)")
    ax.set_ylabel("Trials")
    ax.legend(frameon=False)

    fig.tight_layout()
    out = os.path.join(OUTDIR, "price_of_priority.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)

    print(f"  {out}")
    print(f"    n={len(values)}, min={values.min():+.4f}, "
          f"mean={values.mean():+.4f}, max={values.max():+.4f}")
    print(f"    all positive: {(values > 0).all()}")


# -------------------------------------------------------------------------
# Figure 2: predictors of failure impact
# -------------------------------------------------------------------------
def figure_predictors(combined):
    df = combined.dropna(subset=[
        "failed_edge_betweenness", "failed_edge_utilization", "failure_pct_change"
    ])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 3.0))

    r_betw = df["failed_edge_betweenness"].corr(df["failure_pct_change"])
    ax1.scatter(df["failed_edge_betweenness"], df["failure_pct_change"],
                s=14, color="0.3", alpha=0.7, edgecolors="none")
    ax1.set_xlabel("Edge betweenness centrality")
    ax1.set_ylabel("Failure impact (%)")
    ax1.set_title(f"$r = {r_betw:+.3f}$")

    r_util = df["failed_edge_utilization"].corr(df["failure_pct_change"])
    ax2.scatter(df["failed_edge_utilization"], df["failure_pct_change"],
                s=14, color="0.3", alpha=0.7, edgecolors="none")
    ax2.set_xlabel("Capacity utilisation")
    ax2.set_title(f"$r = {r_util:+.3f}$")

    # Shared y-scale so the two panels are visually comparable
    ymin = min(ax1.get_ylim()[0], ax2.get_ylim()[0])
    ymax = max(ax1.get_ylim()[1], ax2.get_ylim()[1])
    ax1.set_ylim(ymin, ymax)
    ax2.set_ylim(ymin, ymax)

    fig.tight_layout()
    out = os.path.join(OUTDIR, "predictors.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)

    print(f"  {out}")
    print(f"    n={len(df)}, r_betweenness={r_betw:+.4f}, r_utilisation={r_util:+.4f}")


# -------------------------------------------------------------------------
# Figure 3: capacity sensitivity
# -------------------------------------------------------------------------
def figure_sensitivity(sensitivity):
    caps = sorted(sensitivity["capacity"].unique())

    price_means, diverge_rates, r_betw, r_util = [], [], [], []
    for c in caps:
        s = sensitivity[sensitivity["capacity"] == c]
        price_means.append(s["price_of_priority_pct"].dropna().mean())
        diverge_rates.append(s["found_divergent_route"].mean() * 100)
        sub = s.dropna(subset=["failed_edge_betweenness", "failed_edge_utilization"])
        r_betw.append(sub["failed_edge_betweenness"].corr(sub["failure_pct_change"]))
        r_util.append(sub["failed_edge_utilization"].corr(sub["failure_pct_change"]))

    x = range(len(caps))
    labels = [str(int(c)) for c in caps]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(6.5, 2.6))

    ax1.bar(x, price_means, color="0.4", width=0.6)
    ax1.axhline(0, color="black", linewidth=0.8)
    ax1.set_xticks(list(x)); ax1.set_xticklabels(labels)
    ax1.set_xlabel("Assumed capacity")
    ax1.set_ylabel("Price of priority (%)")

    ax2.bar(x, diverge_rates, color="0.4", width=0.6)
    ax2.set_xticks(list(x)); ax2.set_xticklabels(labels)
    ax2.set_xlabel("Assumed capacity")
    ax2.set_ylabel("Divergent route found (%)")
    ax2.set_ylim(0, 105)

    w = 0.35
    ax3.bar([i - w/2 for i in x], r_betw, width=w, color="0.3", label="betweenness")
    ax3.bar([i + w/2 for i in x], r_util, width=w, color="0.65", label="utilisation")
    ax3.set_xticks(list(x)); ax3.set_xticklabels(labels)
    ax3.set_xlabel("Assumed capacity")
    ax3.set_ylabel("Correlation with impact")
    ax3.legend(frameon=False)

    fig.tight_layout()
    out = os.path.join(OUTDIR, "sensitivity.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)

    print(f"  {out}")
    for i, c in enumerate(caps):
        print(f"    capacity {int(c):>5}: price={price_means[i]:+.4f}%, "
              f"diverge={diverge_rates[i]:.1f}%, "
              f"r_betw={r_betw[i]:+.3f}, r_util={r_util[i]:+.3f}")


def main():
    combined, sensitivity = load_data()
    print("Writing figures:")
    figure_price_of_priority(combined)
    figure_predictors(combined)
    figure_sensitivity(sensitivity)
    print("\nDone. Copy the figures/ folder into your Overleaf project.")


if __name__ == "__main__":
    main()
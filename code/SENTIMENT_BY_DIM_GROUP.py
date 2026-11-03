"""Sentiment polarity by 7C dimension AND hesitancy group -- table and figure.

This is the crossing of the two breakdowns already reported:
  Table 7  gives sentiment by hesitancy group, pooled over dimensions.
  Table 20 gives sentiment by dimension,      pooled over groups.
Neither shows the interaction, which is where the interesting pattern is.

Everything is recomputed from code/results_sentiment_labels.csv, written by
SENTIMENT_GROUP_TEST.py, which stores one row per classified response with its
dimension, hesitancy flag, predicted label and continuous polarity. Nothing is
re-classified here, so the numbers match Tables 7, 16 and 20 by construction:
the same per-dimension exclusions and the same five-token minimum apply.

Polarity is P(positive) - P(negative) from german-sentiment-bert, so it runs
from -1 to +1 and 0 means the two classes are balanced (not that the response
was classified neutral).

Two views of the same cells:
  the grouped bar chart compares MEAN POLARITY, one number per cell;
  the stacked chart shows the CLASS SHARES those means come from, which
  separates "more negative responses" from "fewer positive ones" -- the two
  are not the same thing and the means alone cannot distinguish them.

Outputs
    output/sentiment_by_dim_group.csv           long form, one row per cell
    output/sentiment_by_dim_group_wide.csv      dimensions x groups, publication layout
    output/figure_sentiment_by_dim_group.png    300 dpi grouped bar chart, mean polarity
    output/figure_sentiment_shares.png          300 dpi stacked 100% bars, class shares

Run with the BART env:
    /opt/anaconda3/envs/BART/bin/python code/SENTIMENT_BY_DIM_GROUP.py
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths

OUT = paths.OUTPUT   # redirected under PAPER2_DATA
OUT.mkdir(exist_ok=True)
DPI = 300

# Main-text order, matching Table 1 and the 7C framework.
ORDER = ["Confidence", "Complacency", "Constraints", "Calculation",
         "Collective Responsibility", "Compliance", "Conspiracy"]
SHORT = {"Collective Responsibility": "Collective Resp."}
GROUPS = [(1, "Hesitant"), (0, "Non-hesitant")]
COLOR = {"Hesitant": "#4B3B94", "Non-hesitant": "#12916E"}
# Stacked from the bottom up, so the order here is the drawing order.
CLASSES = [("pct_positive", "Positive", "#2E9E63"),
           ("pct_neutral", "Neutral", "#B3B3B3"),
           ("pct_negative", "Negative", "#E2565B")]


def build(lab):
    """Long form: one row per dimension x group cell."""
    rows = []
    for dim in ORDER:
        for flag, gname in GROUPS:
            g = lab[(lab.dim == dim) & (lab.hesitant == flag)]
            pol = g.polarity.to_numpy()
            n = len(g)
            rows.append(dict(
                dimension=dim, group=gname, n=n,
                mean_polarity=float(pol.mean()),
                sd_polarity=float(pol.std(ddof=1)),
                se_polarity=float(pol.std(ddof=1) / np.sqrt(n)),
                pct_negative=100 * float((g.label == "negative").mean()),
                pct_neutral=100 * float((g.label == "neutral").mean()),
                pct_positive=100 * float((g.label == "positive").mean())))
    return pd.DataFrame(rows)


def wide(long):
    w = long.pivot(index="dimension", columns="group", values="mean_polarity")
    n = long.pivot(index="dimension", columns="group", values="n")
    out = pd.DataFrame({
        "Hesitant mean": w["Hesitant"], "Hesitant n": n["Hesitant"],
        "Non-hesitant mean": w["Non-hesitant"], "Non-hesitant n": n["Non-hesitant"]})
    out["Difference (hes - non)"] = w["Hesitant"] - w["Non-hesitant"]
    return out.reindex(ORDER)


def figure(long, path):
    fig, ax = plt.subplots(figsize=(11, 6))
    xs = np.arange(len(ORDER))
    width = 0.36
    for i, (_, gname) in enumerate(GROUPS):
        vals = [long[(long.dimension == d) & (long.group == gname)].mean_polarity.iloc[0]
                for d in ORDER]
        ax.bar(xs + (i - 0.5) * width, vals, width, label=gname,
               color=COLOR[gname], edgecolor="white", linewidth=0.6)
    ax.axhline(0, color="#444444", linewidth=1.0)
    ax.set_xticks(xs)
    ax.set_xticklabels([SHORT.get(d, d) for d in ORDER], fontsize=11)
    ax.set_ylabel("Average sentiment polarity", fontsize=12)
    ax.set_ylim(-0.50, 0.25)
    ax.yaxis.grid(True, linestyle="-", linewidth=0.6, color="#DDDDDD")
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.16), ncol=2,
              frameon=False, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def figure_shares(long, path):
    """Stacked 100% bars: the class shares behind each mean, one panel per group."""
    fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
    xs = np.arange(len(ORDER))
    for ax, (_, gname) in zip(axes, GROUPS):
        sub = long[long.group == gname].set_index("dimension").reindex(ORDER)
        bottom = np.zeros(len(ORDER))
        for col, cname, colour in CLASSES:
            vals = sub[col].to_numpy()
            ax.bar(xs, vals, 0.62, bottom=bottom, label=cname,
                   color=colour, edgecolor="white", linewidth=0.6)
            bottom += vals
        ax.set_title(f"{gname.upper()}S", loc="left", fontsize=12,
                     fontweight="bold", color="#333333")
        ax.set_ylim(0, 100)
        ax.set_yticks(range(0, 101, 10))
        ax.set_yticklabels([f"{v}%" for v in range(0, 101, 10)], fontsize=9)
        ax.yaxis.grid(True, linestyle="-", linewidth=0.6, color="#E5E5E5")
        ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
    axes[-1].set_xticks(xs)
    axes[-1].set_xticklabels([SHORT.get(d, d) for d in ORDER], fontsize=11)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper left", bbox_to_anchor=(0.09, 0.97),
               ncol=3, frameon=False, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def main():
    lab = pd.read_csv(paths.result("results_sentiment_labels.csv"))
    long = build(lab)
    w = wide(long)

    print(w.round(4).to_string())
    print(f"\ntotal classified responses: {len(lab):,}")

    long.to_csv(OUT / "sentiment_by_dim_group.csv", index=False)
    w.round(4).to_csv(OUT / "sentiment_by_dim_group_wide.csv")
    fig_path = OUT / "figure_sentiment_by_dim_group.png"
    figure(long, fig_path)
    shares_path = OUT / "figure_sentiment_shares.png"
    figure_shares(long, shares_path)

    # The one dimension that reverses the overall pattern is worth naming.
    rev = w[w["Difference (hes - non)"] > 0]
    if len(rev):
        print("\nDimensions where hesitant responses are LESS negative than non-hesitant:")
        for d, r in rev.iterrows():
            print(f"  {d}: hesitant {r['Hesitant mean']:+.3f} vs "
                  f"non-hesitant {r['Non-hesitant mean']:+.3f} "
                  f"(difference {r['Difference (hes - non)']:+.3f})")

    for p in (OUT / "sentiment_by_dim_group.csv",
              OUT / "sentiment_by_dim_group_wide.csv", fig_path, shares_path):
        print("wrote", p)


if __name__ == "__main__":
    main()

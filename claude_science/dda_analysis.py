"""Shared pipeline for the IxIDN/ORDON benchmark merge and DDA analysis.

The notebooks in this directory are thin drivers: they import this module and call
into it. All plotting functions render inline (``plt.show()``) and return ``None`` —
figures live inside the notebooks, nothing is written to disk.

Layout assumed:
    ../raw_benchmarks/   IxIDN_ORDON_benchmark.csv, IxIDN_df.csv, ORDON_df.csv
    ./                   IxIDN_ORDON_benchmark_DDA_v*.csv  (DDA scores, joined on ID)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from sklearn.metrics import roc_auc_score, roc_curve

RAW = Path(__file__).resolve().parent.parent / "raw_benchmarks"
HERE = Path(__file__).resolve().parent

L1, L2 = "d1_disease_label", "d2_disease_label"
FINAL_COLS = ["ID", L1, L2, "DDA", "exist_IxIDN", "exist_ORDON", "IxIDN_ID", "ORDON_ID"]

# --------------------------------------------------------------------------- #
# Design tokens — validated categorical slots 1 & 2 (all-pairs CVD ΔE 24.7,
# normal-vision ΔE 33.6, both clear of the floors). Slot 1 is the primary
# series in every figure; slot 2 is the comparison series.
# --------------------------------------------------------------------------- #
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
C_1, C_2 = "#2a78d6", "#eb6834"  # blue, orange


def apply_style() -> None:
    """Install the chart style. Call once per notebook, before plotting."""
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif", "font.size": 10,
        "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": BASELINE,
        "axes.titlesize": 12, "axes.titleweight": "semibold", "axes.titlelocation": "left",
        "axes.titlepad": 10, "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2,
        "ytick.labelcolor": INK2, "grid.color": GRID, "grid.linewidth": 0.8,
        "legend.frameon": False, "figure.dpi": 130,
    })


# --------------------------------------------------------------------------- #
# Load & merge
# --------------------------------------------------------------------------- #
def add_keys(df: pd.DataFrame, row_id: str | None = None) -> pd.DataFrame:
    """Attach an ordered pair key, an order-insensitive pair key, and a row number.

    The order-insensitive key is the important one: pair orientation is *not*
    preserved between the benchmark and ``IxIDN_df.csv``, so a naive merge on
    ``(d1, d2)`` finds only 2,767 of the 5,336 IxIDN rows.
    """
    df = df.copy()
    a = df[L1].astype("string").str.strip()
    b = df[L2].astype("string").str.strip()
    df["key_ordered"] = a + " || " + b
    lo, hi = a.where(a <= b, b), b.where(a <= b, a)
    df["key_pair"] = lo + " || " + hi
    if row_id is not None:
        df[row_id] = range(len(df))  # 0-based row number in the source CSV
    return df


def load_sources(dda_file: str) -> dict[str, pd.DataFrame]:
    """Read the benchmark, both source datasets, and one DDA score file."""
    frames = {
        "benchmark": add_keys(pd.read_csv(RAW / "IxIDN_ORDON_benchmark.csv")),
        "ixidn": add_keys(pd.read_csv(RAW / "IxIDN_df.csv"), row_id="IxIDN_ID"),
        "ordon": add_keys(pd.read_csv(RAW / "ORDON_df.csv"), row_id="ORDON_ID"),
        "dda": pd.read_csv(HERE / dda_file),
    }
    assert frames["benchmark"]["ID"].is_unique, "benchmark ID is not unique"
    assert frames["dda"]["ID"].is_unique, "DDA ID is not unique"
    return frames


def describe_inputs(frames: dict[str, pd.DataFrame]) -> None:
    for name, df in frames.items():
        print(f"{name:>9}: {df.shape[0]:>6,} rows x {df.shape[1]} cols | "
              f"{[c for c in df.columns if not c.startswith('key_')]}")
    print("\nduplicate pair keys within each file (order-insensitive):")
    for name in ("benchmark", "ixidn", "ordon"):
        print(f"  {name:>9}: {frames[name]['key_pair'].duplicated().sum():>4}")


def orientation_effect(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """How many benchmark rows each key definition finds — the reason for key_pair."""
    b = frames["benchmark"]
    out = pd.DataFrame(
        {
            "ordered (d1,d2) match": [
                b["key_ordered"].isin(set(frames["ixidn"]["key_ordered"])).sum(),
                b["key_ordered"].isin(set(frames["ordon"]["key_ordered"])).sum(),
            ],
            "order-insensitive match": [
                b["key_pair"].isin(set(frames["ixidn"]["key_pair"])).sum(),
                b["key_pair"].isin(set(frames["ordon"]["key_pair"])).sum(),
            ],
        },
        index=["IxIDN", "ORDON"],
    )
    out["gained by ignoring order"] = out["order-insensitive match"] - out["ordered (d1,d2) match"]
    return out


def merge_benchmark(frames: dict[str, pd.DataFrame], verbose: bool = True) -> pd.DataFrame:
    """Join the benchmark to both sources plus DDA. Returns the final 8-column table."""
    b, ix, od, dda = (frames[k] for k in ("benchmark", "ixidn", "ordon", "dda"))

    merged = (
        b.merge(dda[["ID", "DDA"]], on="ID", how="left", validate="1:1")
        .merge(ix.drop_duplicates("key_pair")[["key_pair", "IxIDN_ID"]],
               on="key_pair", how="left", validate="m:1")
        .merge(od.drop_duplicates("key_pair")[["key_pair", "ORDON_ID"]],
               on="key_pair", how="left", validate="m:1")
    )
    merged["exist_IxIDN"] = merged["IxIDN_ID"].notna()
    merged["exist_ORDON"] = merged["ORDON_ID"].notna()
    # nullable ints so unmatched rows stay blank rather than becoming 2805.0
    merged["IxIDN_ID"] = merged["IxIDN_ID"].astype("Int64")
    merged["ORDON_ID"] = merged["ORDON_ID"].astype("Int64")

    assert len(merged) == len(b), "merge changed the row count"

    if verbose:
        print(f"merged rows: {len(merged):,} (benchmark rows: {len(b):,})")
        print(f"DDA missing: {merged['DDA'].isna().sum()}")
        # the DDA file carries its own labels; confirm ID means the same row in both
        chk = b.merge(dda, on="ID", suffixes=("_b", "_d"))
        bad = (chk[f"{L1}_b"] != chk[f"{L1}_d"]) | (chk[f"{L2}_b"] != chk[f"{L2}_d"])
        print(f"DDA label mismatches vs benchmark on ID: {bad.sum()}")

    return merged[FINAL_COLS + ["key_pair"]]


def merge_summary(merged: pd.DataFrame) -> pd.DataFrame:
    n = len(merged)
    ix, od = merged["exist_IxIDN"], merged["exist_ORDON"]
    neither = (~ix & ~od).sum()
    out = pd.DataFrame(
        [
            ("IxIDN only", (ix & ~od).sum()),
            ("ORDON only", (~ix & od).sum()),
            ("both", (ix & od).sum()),
            ("unmatched (neither)", neither),
            ("matched at least one", n - neither),
            ("benchmark total", n),
        ],
        columns=["bucket", "rows"],
    )
    out["pct"] = (out["rows"] / n * 100).round(2)
    return out


def coverage_report(merged: pd.DataFrame, frames: dict[str, pd.DataFrame]) -> None:
    """Round-trip check: is every source row reachable from the benchmark exactly once?"""
    pairs = set(merged["key_pair"])
    for name, key, col in [("IxIDN", "ixidn", "IxIDN_ID"), ("ORDON", "ordon", "ORDON_ID")]:
        src = frames[key]
        uniq = src["key_pair"].drop_duplicates()
        hit = uniq.isin(pairs).sum()
        used = merged[col].dropna().nunique()
        print(f"{name:>5}: {hit:,}/{len(uniq):,} unique pairs in benchmark "
              f"({hit / len(uniq) * 100:.2f}%) · {used:,}/{len(src):,} source rows referenced")


def write_merged(merged: pd.DataFrame, out_file: str) -> Path:
    path = HERE / out_file
    merged[FINAL_COLS].to_csv(path, index=False)
    print(f"wrote {path.name}  ({len(merged):,} rows, {len(FINAL_COLS)} cols)")
    return path


# --------------------------------------------------------------------------- #
# Positive / negative split
# --------------------------------------------------------------------------- #
def split_groups(merged: pd.DataFrame, verbose: bool = True):
    """IxIDN-only as positives, ORDON-only as negatives.

    The handful of pairs present in *both* sources have no unambiguous label and
    are excluded from every comparison and from the ROC.
    """
    both = merged["exist_IxIDN"] & merged["exist_ORDON"]
    if verbose:
        print(f"pairs in both sources (excluded, unlabelable): {both.sum()}")
    pos = merged.loc[merged["exist_IxIDN"] & ~both, "DDA"].to_numpy()
    neg = merged.loc[merged["exist_ORDON"] & ~both, "DDA"].to_numpy()
    return pos, neg


def group_stats(pos: np.ndarray, neg: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "n": [len(pos), len(neg)],
            "mean": [pos.mean(), neg.mean()],
            "median": [np.median(pos), np.median(neg)],
            "q25": [np.percentile(pos, 25), np.percentile(neg, 25)],
            "q75": [np.percentile(pos, 75), np.percentile(neg, 75)],
            "min": [pos.min(), neg.min()],
            "max": [pos.max(), neg.max()],
        },
        index=["IxIDN (positive)", "ORDON (negative)"],
    ).round(4)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def roc_stats(pos: np.ndarray, neg: np.ndarray) -> dict:
    y = np.r_[np.ones(len(pos), dtype=int), np.zeros(len(neg), dtype=int)]
    s = np.r_[pos, neg]
    fpr, tpr, thr = roc_curve(y, s)
    j = int(np.argmax(tpr - fpr))
    return {"fpr": fpr, "tpr": tpr, "thresholds": thr, "auc": roc_auc_score(y, s),
            "j": j, "best_threshold": thr[j], "tpr_at_best": tpr[j], "fpr_at_best": fpr[j],
            "n_pos": len(pos), "n_neg": len(neg)}


def wasserstein_stats(pos: np.ndarray, neg: np.ndarray) -> dict:
    w1 = wasserstein_distance(pos, neg)
    pooled_sd = np.sqrt(((len(pos) - 1) * pos.var(ddof=1) + (len(neg) - 1) * neg.var(ddof=1))
                        / (len(pos) + len(neg) - 2))
    mean_diff = pos.mean() - neg.mean()
    # W1 == |mean difference| exactly when the two CDFs never cross, i.e. one
    # group stochastically dominates the other across the whole score range.
    grid = np.linspace(min(pos.min(), neg.min()), max(pos.max(), neg.max()), 2001)
    f_pos = np.searchsorted(np.sort(pos), grid, "right") / len(pos)
    f_neg = np.searchsorted(np.sort(neg), grid, "right") / len(neg)
    return {"w1": w1, "pooled_sd": pooled_sd, "w1_sd": w1 / pooled_sd,
            "mean_diff": mean_diff,
            "direction": "IxIDN shifted higher" if mean_diff > 0 else "ORDON shifted higher",
            "dominance": bool((f_pos <= f_neg + 1e-12).all()),
            "crossings": int((f_pos > f_neg + 1e-12).sum())}


def print_metrics(pos: np.ndarray, neg: np.ndarray) -> tuple[dict, dict]:
    r, w = roc_stats(pos, neg), wasserstein_stats(pos, neg)
    print(f"AUC                       : {r['auc']:.4f}")
    print(f"  positives (IxIDN)       : {r['n_pos']:,}")
    print(f"  negatives (ORDON)       : {r['n_neg']:,}")
    print(f"  best threshold (Youden) : DDA >= {r['best_threshold']:.4f}")
    print(f"    TPR (sensitivity)     : {r['tpr_at_best']:.3f}")
    print(f"    FPR                   : {r['fpr_at_best']:.3f} "
          f"(specificity {1 - r['fpr_at_best']:.3f})")
    print(f"Wasserstein distance W1   : {w['w1']:.4f} DDA units")
    print(f"  normalized by pooled SD : {w['w1_sd']:.3f}")
    print(f"  direction               : {w['direction']} (Δ mean {w['mean_diff']:+.4f})")
    print(f"  stochastic dominance    : {w['dominance']} "
          f"(CDF crossings: {w['crossings']})")
    return r, w


# --------------------------------------------------------------------------- #
# Figures — all render inline and return None; nothing is written to disk.
# --------------------------------------------------------------------------- #
def plot_distribution(pos: np.ndarray, neg: np.ndarray, title: str | None = None) -> None:
    """Density histogram over a boxplot, sharing one DDA axis."""
    lo, hi = min(pos.min(), neg.min()), max(pos.max(), neg.max())
    bins = np.linspace(lo, hi, 61)

    fig, (ax_h, ax_b) = plt.subplots(
        2, 1, figsize=(8.2, 5.4), sharex=True,
        gridspec_kw={"height_ratios": [3.1, 1], "hspace": 0.12},
    )

    for data, color, label in [(pos, C_1, "IxIDN"), (neg, C_2, "ORDON")]:
        ax_h.hist(data, bins=bins, density=True, color=color, alpha=0.30, zorder=2)
        ax_h.hist(data, bins=bins, density=True, histtype="step",
                  color=color, linewidth=2, label=label, zorder=3)

    ax_h.set_title(title or "DDA score distribution by source dataset")
    ax_h.set_ylabel("density")
    ax_h.grid(axis="y", zorder=0)
    ax_h.set_axisbelow(True)
    ax_h.legend(loc="upper right", labelcolor=INK2)

    # Median rules, labelled on opposite sides and staggered so they never collide.
    top = ax_h.get_ylim()[1]
    for data, color, label, ha, off, y in [
        (neg, C_2, "ORDON", "right", (-6, 0), 0.90),
        (pos, C_1, "IxIDN", "left", (6, 0), 0.78),
    ]:
        med = np.median(data)
        ax_h.axvline(med, color=color, linewidth=1.4, linestyle=(0, (4, 3)), zorder=4)
        ax_h.annotate(f"{label} median {med:.3f}", (med, top * y), xytext=off,
                      textcoords="offset points", ha=ha, va="center",
                      fontsize=8.5, color=INK2, zorder=5)

    bp = ax_b.boxplot(
        [pos, neg], vert=False, widths=0.5, patch_artist=True,
        flierprops={"marker": "o", "markersize": 2.5, "markerfacecolor": MUTED,
                    "markeredgecolor": "none", "alpha": 0.30},
        medianprops={"color": SURFACE, "linewidth": 2},
        whiskerprops={"color": BASELINE, "linewidth": 1.4},
        capprops={"color": BASELINE, "linewidth": 1.4},
    )
    for patch, color in zip(bp["boxes"], [C_1, C_2]):
        patch.set(facecolor=color, edgecolor=SURFACE, linewidth=2)  # 2px surface ring

    ax_b.set_yticks([1, 2], ["IxIDN", "ORDON"])
    ax_b.set_xlabel("DDA")
    ax_b.grid(axis="x", zorder=0)
    ax_b.set_axisbelow(True)
    ax_b.spines["left"].set_visible(False)
    ax_b.tick_params(axis="y", length=0)
    plt.show()


def plot_roc(pos: np.ndarray, neg: np.ndarray, title: str | None = None) -> None:
    r = roc_stats(pos, neg)
    fpr, tpr, j = r["fpr"], r["tpr"], r["j"]

    fig, ax = plt.subplots(figsize=(5.4, 5.4))
    ax.plot([0, 1], [0, 1], color=BASELINE, linewidth=1.4, linestyle=(0, (4, 3)), zorder=2)
    ax.annotate("chance", (0.62, 0.62), xytext=(-4, 5), textcoords="offset points",
                rotation=45, rotation_mode="anchor", fontsize=8.5, color=MUTED,
                ha="left", va="bottom")

    ax.fill_between(fpr, tpr, color=C_1, alpha=0.12, zorder=3)
    ax.plot(fpr, tpr, color=C_1, linewidth=2, solid_capstyle="round", zorder=4)
    ax.plot(fpr[j], tpr[j], marker="o", markersize=8, color=C_1,
            markeredgecolor=SURFACE, markeredgewidth=2, zorder=5)
    ax.annotate(f"DDA ≥ {r['best_threshold']:.3f}\nTPR {tpr[j]:.2f} · FPR {fpr[j]:.2f}",
                (fpr[j], tpr[j]), xytext=(10, -14), textcoords="offset points",
                fontsize=8.5, color=INK2, ha="left", va="top")
    ax.annotate(f"AUC {r['auc']:.3f}", (0.97, 0.06), xycoords="axes fraction",
                ha="right", va="bottom", fontsize=15, color=INK)

    ax.set_title(title or "ROC — DDA separating IxIDN (positive) from ORDON (negative)",
                 fontsize=11)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.set_aspect("equal")
    ax.grid(zorder=0)
    ax.set_axisbelow(True)
    plt.show()


def plot_violin(pos: np.ndarray, neg: np.ndarray) -> None:
    """Vertical violins with the Wasserstein distance in the title."""
    w = wasserstein_stats(pos, neg)

    fig, ax = plt.subplots(figsize=(6.0, 5.6))
    groups = [("IxIDN", pos, C_1), ("ORDON", neg, C_2)]
    parts = ax.violinplot([g[1] for g in groups], widths=0.8,
                          showmeans=False, showmedians=False, showextrema=False)
    for body, (_, _, color) in zip(parts["bodies"], groups):
        body.set(facecolor=color, edgecolor=color, linewidth=2, zorder=2)
        body.set_alpha(0.30)

    # IQR bar + median dot ringed in the surface color; value labelled above the
    # violin, where nothing can collide with it.
    for i, (label, data, color) in enumerate(groups, start=1):
        q1, med, q3 = np.percentile(data, [25, 50, 75])
        bar = ax.vlines(i, q1, q3, color=color, linewidth=7, zorder=3)
        bar.set_capstyle("round")
        ax.plot(i, med, marker="o", markersize=8, color=color,
                markeredgecolor=SURFACE, markeredgewidth=2, zorder=4)
        ax.annotate(f"median {med:.3f}", (i, data.max()), xytext=(0, 9),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=8.5, color=INK2, zorder=5)

    ax.set_title(f"DDA by source — Wasserstein distance W₁ = {w['w1']:.4f} DDA units",
                 fontsize=11.5, pad=28)
    ax.annotate(f"{w['w1_sd']:.2f} pooled SD · {w['direction']}", (0, 1.015),
                xycoords="axes fraction", ha="left", va="bottom", fontsize=9, color=MUTED)

    ax.set_xticks([1, 2], ["IxIDN\n(positive)", "ORDON\n(negative)"])
    ax.set_ylabel("DDA")
    ax.set(xlim=(0.4, 2.6), ylim=(0, max(pos.max(), neg.max()) * 1.14))
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="x", length=0)
    plt.show()


# --------------------------------------------------------------------------- #
# One-call driver for the per-version notebooks
# --------------------------------------------------------------------------- #
def run_merge(dda_file: str, out_file: str) -> pd.DataFrame:
    """Load, merge, validate and write. Returns the final table."""
    frames = load_sources(dda_file)
    describe_inputs(frames)
    merged = merge_benchmark(frames)
    coverage_report(merged, frames)
    write_merged(merged, out_file)
    return merged


# --------------------------------------------------------------------------- #
# Version comparison (v1 vs v2)
# --------------------------------------------------------------------------- #
def load_versions(file_a: str, file_b: str, label_a: str = "v1", label_b: str = "v2",
                  merged_for_labels: str | None = None) -> pd.DataFrame:
    """Join two DDA score files on ID, attaching the exist_* flags for grouping.

    ``merged_for_labels`` is a merged CSV to take ``exist_IxIDN``/``exist_ORDON``
    from; if omitted the flags are recomputed from the raw sources.
    """
    a = pd.read_csv(HERE / file_a)[["ID", L1, L2, "DDA"]].rename(columns={"DDA": label_a})
    b = pd.read_csv(HERE / file_b)[["ID", "DDA"]].rename(columns={"DDA": label_b})

    if merged_for_labels is not None:
        flags = pd.read_csv(HERE / merged_for_labels)[["ID", "exist_IxIDN", "exist_ORDON"]]
    else:
        frames = load_sources(file_a)
        flags = merge_benchmark(frames, verbose=False)[["ID", "exist_IxIDN", "exist_ORDON"]]

    out = a.merge(b, on="ID", validate="1:1").merge(flags, on="ID", validate="1:1")
    out["delta"] = out[label_b] - out[label_a]
    out["group"] = np.select(
        [out.exist_IxIDN & out.exist_ORDON, out.exist_IxIDN, out.exist_ORDON],
        ["both", "IxIDN", "ORDON"], default="unmatched",
    )
    return out


def version_diff_summary(cmp: pd.DataFrame, label_a: str = "v1",
                         label_b: str = "v2") -> pd.DataFrame:
    changed = cmp[cmp[label_a] != cmp[label_b]]
    rows = [
        ("rows total", len(cmp)),
        (f"identical {label_a} == {label_b}", len(cmp) - len(changed)),
        ("changed", len(changed)),
        ("changed ↓ (revised down)", int((changed["delta"] < 0).sum())),
        ("changed ↑ (revised up)", int((changed["delta"] > 0).sum())),
    ]
    out = pd.DataFrame(rows, columns=["bucket", "rows"])
    out["pct"] = (out["rows"] / len(cmp) * 100).round(3)
    return out


def version_metrics(cmp: pd.DataFrame, label_a: str = "v1",
                    label_b: str = "v2") -> pd.DataFrame:
    """AUC and Wasserstein for each version, side by side."""
    both = cmp["group"] == "both"
    rows = {}
    for label in (label_a, label_b):
        pos = cmp.loc[(cmp["group"] == "IxIDN") & ~both, label].to_numpy()
        neg = cmp.loc[(cmp["group"] == "ORDON") & ~both, label].to_numpy()
        r, w = roc_stats(pos, neg), wasserstein_stats(pos, neg)
        rows[label] = {
            "AUC": r["auc"], "best threshold": r["best_threshold"],
            "TPR@best": r["tpr_at_best"], "FPR@best": r["fpr_at_best"],
            "W1": w["w1"], "W1 (pooled SD)": w["w1_sd"],
            "mean IxIDN": pos.mean(), "mean ORDON": neg.mean(),
            "dominance": w["dominance"],
        }
    out = pd.DataFrame(rows).T
    out.loc["delta"] = out.loc[label_b] - out.loc[label_a]
    out.loc["delta", "dominance"] = ""
    return out.round(6)


def plot_version_changes(cmp: pd.DataFrame, label_a: str = "v1",
                         label_b: str = "v2") -> None:
    """Dumbbell chart of every row whose DDA changed — the right form when only a
    handful of items move and both the before/after values and the shift matter."""
    changed = cmp[cmp[label_a] != cmp[label_b]].sort_values("delta").reset_index(drop=True)
    if changed.empty:
        print("no rows changed between versions — nothing to plot")
        return

    fig, ax = plt.subplots(figsize=(8.6, 0.42 * len(changed) + 2.2))
    y = np.arange(len(changed))

    ax.hlines(y, changed[label_a], changed[label_b], color=BASELINE, linewidth=1.6, zorder=2)
    ax.scatter(changed[label_a], y, s=64, color=C_1, edgecolor=SURFACE,
               linewidth=2, zorder=3, label=label_a)
    ax.scatter(changed[label_b], y, s=64, color=C_2, edgecolor=SURFACE,
               linewidth=2, zorder=4, label=label_b)

    for i, row in changed.iterrows():
        ax.annotate(f"{row['delta']:+.4f}", (min(row[label_a], row[label_b]), i),
                    xytext=(-10, 0), textcoords="offset points", ha="right", va="center",
                    fontsize=8.5, color=INK2)

    def _trim(s, n=30):
        s = str(s)
        return s if len(s) <= n else s[: n - 1] + "…"

    labels = [f"ID {int(r.ID)} · {_trim(getattr(r, L2))}" for r in changed.itertuples()]
    ax.set_yticks(y, labels, fontsize=8.5)
    ax.set_xlabel("DDA")
    ax.set_title(f"Rows whose DDA changed between {label_a} and {label_b} "
                 f"({len(changed)} of {len(cmp):,})")
    # upper right: the lower right is where the largest-delta row's marks land
    ax.legend(loc="upper right", labelcolor=INK2)
    ax.grid(axis="x", zorder=0)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.margins(x=0.16)
    plt.show()


def plot_version_roc(cmp: pd.DataFrame, label_a: str = "v1", label_b: str = "v2") -> None:
    """Both versions' ROC curves on one axis, with the AUC delta called out."""
    both = cmp["group"] == "both"
    fig, ax = plt.subplots(figsize=(5.6, 5.6))
    ax.plot([0, 1], [0, 1], color=BASELINE, linewidth=1.4, linestyle=(0, (4, 3)), zorder=2)
    ax.annotate("chance", (0.62, 0.62), xytext=(-4, 5), textcoords="offset points",
                rotation=45, rotation_mode="anchor", fontsize=8.5, color=MUTED,
                ha="left", va="bottom")

    aucs = {}
    for label, color in [(label_a, C_1), (label_b, C_2)]:
        pos = cmp.loc[(cmp["group"] == "IxIDN") & ~both, label].to_numpy()
        neg = cmp.loc[(cmp["group"] == "ORDON") & ~both, label].to_numpy()
        r = roc_stats(pos, neg)
        aucs[label] = r["auc"]
        ax.plot(r["fpr"], r["tpr"], color=color, linewidth=2, solid_capstyle="round",
                label=f"{label}  AUC {r['auc']:.4f}", zorder=3)

    d = aucs[label_b] - aucs[label_a]
    ax.annotate(f"ΔAUC {d:+.4f}", (0.97, 0.13), xycoords="axes fraction",
                ha="right", va="bottom", fontsize=13, color=INK)

    ax.set_title(f"ROC — {label_a} vs {label_b} (IxIDN positive, ORDON negative)",
                 fontsize=11)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.set_aspect("equal")
    ax.legend(loc="lower right", labelcolor=INK2)
    ax.grid(zorder=0)
    ax.set_axisbelow(True)
    plt.show()


def plot_version_violin(cmp: pd.DataFrame, label_a: str = "v1", label_b: str = "v2") -> None:
    """Four violins: each version's positive and negative distribution, with both
    Wasserstein distances in the title."""
    both = cmp["group"] == "both"
    series, w1s = [], {}
    for label in (label_a, label_b):
        pos = cmp.loc[(cmp["group"] == "IxIDN") & ~both, label].to_numpy()
        neg = cmp.loc[(cmp["group"] == "ORDON") & ~both, label].to_numpy()
        w1s[label] = wasserstein_distance(pos, neg)
        series += [(f"IxIDN\n{label}", pos, C_1), (f"ORDON\n{label}", neg, C_2)]

    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    parts = ax.violinplot([s[1] for s in series], widths=0.8,
                          showmeans=False, showmedians=False, showextrema=False)
    for body, (_, _, color) in zip(parts["bodies"], series):
        body.set(facecolor=color, edgecolor=color, linewidth=2, zorder=2)
        body.set_alpha(0.30)

    for i, (label, data, color) in enumerate(series, start=1):
        q1, med, q3 = np.percentile(data, [25, 50, 75])
        bar = ax.vlines(i, q1, q3, color=color, linewidth=7, zorder=3)
        bar.set_capstyle("round")
        ax.plot(i, med, marker="o", markersize=8, color=color,
                markeredgecolor=SURFACE, markeredgewidth=2, zorder=4)
        ax.annotate(f"{med:.3f}", (i, data.max()), xytext=(0, 9),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=8.5, color=INK2, zorder=5)

    # Enough precision to show a difference: rounding to 4dp would print both W1
    # values identically and the delta as "-0.0000".
    dw = w1s[label_b] - w1s[label_a]
    dw_txt = f"{dw:+.6f}" if abs(dw) >= 5e-6 else f"{dw:+.2e}"
    ax.set_title(f"DDA by source and version — W₁ {label_a} = {w1s[label_a]:.6f}, "
                 f"{label_b} = {w1s[label_b]:.6f}", fontsize=11, pad=28)
    ax.annotate(f"ΔW₁ = {dw_txt} DDA units", (0, 1.015),
                xycoords="axes fraction", ha="left", va="bottom",
                fontsize=9, color=MUTED)

    ax.set_xticks(range(1, len(series) + 1), [s[0] for s in series], fontsize=9)
    ax.set_ylabel("DDA")
    ax.set(xlim=(0.4, len(series) + 0.6),
           ylim=(0, max(s[1].max() for s in series) * 1.14))
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="x", length=0)
    plt.show()

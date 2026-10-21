"""Figures of the report, drawn from results/*.json (PDF for LaTeX, PNG for the README)."""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from autogear.paths import FIGURES, RESULTS, ensure_dirs

GREY, ACCENT, LIGHT = "#8a8a8a", "#1f5aa6", "#c9c9c9"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Liberation Serif", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "pdf.fonttype": 42,
})
W = 3.45  # IEEE column width, inches


def save(fig, name):
    # no creation date in the metadata, so that re-running the script leaves the files unchanged
    fig.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02, metadata={"CreationDate": None})
    fig.savefig(FIGURES / f"{name}.png", bbox_inches="tight", pad_inches=0.02, dpi=220)
    plt.close(fig)


def label(test: str) -> str:
    names = {"inria": "Inria", "di2_edge1040_2022": "Di2 1040", "edge1030_2020": "Edge 1030",
             "edge530_2020": "Edge 530", "edge810_2013": "Edge 810", "elemnt_2019": "ELEMNT"}
    if test.startswith("inria_"):
        return f"Inria ride {test[-1]}"
    rider, way = test.split(" ")
    return f"{names[rider]} {way.replace('>', '→')}"


def margin_per_test(res):
    tests = res["tests"]
    rows = []
    for name, r in tests.items():
        best = max(r[k]["agree_pm1"] for k in ("R0", "R1", "RT"))
        rows.append((label(name), (r["GBDT"]["agree_pm1"] - best) * 100, r["group"], r["hours"]))
    fig, ax = plt.subplots(figsize=(W, 2.9))
    y = np.arange(len(rows))[::-1]
    for yi, (lab, m, grp, h) in zip(y, rows):
        ax.barh(yi, m, color=ACCENT if m > 0 else "#b3472e", height=0.62,
                alpha=1.0 if grp == "inria" else 0.75)
        ax.text(m + (0.4 if m >= 0 else -0.4), yi, f"{m:+.1f}", va="center",
                ha="left" if m >= 0 else "right", fontsize=6.5)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.axvline(0, color="black", lw=0.6)
    med = np.median([r[1] for r in rows])
    ax.axvline(med, color=GREY, lw=0.6, ls="--")
    ax.text(med + 0.4, y[4], f"median {med:+.1f}", color="#555555", fontsize=6.5, va="center")
    ax.set_xlabel("agreement within one gear: retained model − best decision table (points)")
    ax.set_xlim(-38, 22)
    ax.axhline(y[3] - 0.5, color=LIGHT, lw=0.6)
    save(fig, "fig_margin_per_test")


def accuracy_vs_size(res, ev):
    tests = res["tests"]
    h = np.array([r["hours"] for r in tests.values()])

    def w(kind, key="agree_pm1"):
        return float((h * np.array([r[kind][key] for r in tests.values()])).sum() / h.sum())

    def size(kind):
        return float(np.median([r[kind]["size_bytes"] for r in tests.values()]))

    evt = ev["tests"]
    hev = np.array([r["hours"] for r in evt.values()])
    ev_acc = float((hev * np.array([r["EV"]["agree_pm1"] for r in evt.values()])).sum() / hev.sum())
    ev_size = float(np.median([r["EV"]["size_bytes"] for r in evt.values()]))
    pts = [("R0, constant cadence", "R0", "rule", (5, -12)), ("R1, + slope", "R1", "rule", (5, -8)),
           ("RT, + torque", "RT", "rule", (5, 4)), ("Linear regression", "LIN", "model", (-5, 4)),
           ("Learned table", "TAB", "model", (6, -2)), ("Small MLP", "MLP", "model", (-6, -1)),
           ("Random forest", "RF", "model", (-5, -10)), ("Boosted trees (retained)", "GBDT", "best", (0, 7))]
    fig, ax = plt.subplots(figsize=(W, 2.35))
    ref = w("RT") * 100
    ax.axhline(ref, color=GREY, lw=0.6, ls="--")
    ax.text(1.2e5, ref - 1.6, "best decision table (with torque)", ha="right", color="#555555", fontsize=6.5)
    for name, k, kind, (dx, dy) in pts:
        x, yv = size(k), w(k) * 100
        if kind == "rule":
            ax.scatter(x, yv, s=22, facecolors="white", edgecolors="black", linewidths=0.7, zorder=3)
        else:
            ax.scatter(x, yv, s=30 if kind == "best" else 22, color=ACCENT if kind == "best" else GREY, zorder=3)
        ax.annotate(f"{name} {yv:.1f}", (x, yv), xytext=(dx, dy), textcoords="offset points",
                    ha="center" if dx == 0 else ("left" if dx > 0 else "right"), fontsize=6.5,
                    fontweight="bold" if kind == "best" else "normal")
    ax.scatter(ev_size, ev_acc * 100, s=22, color="#b3472e", zorder=3)
    ax.annotate(f"Event classifier (Sec. V) {ev_acc * 100:.1f}", (ev_size, ev_acc * 100), xytext=(-5, 2),
                textcoords="offset points", ha="right", fontsize=6.5)
    ax.set_xscale("log")
    ax.set_xlim(2, 1.3e5)
    ax.set_ylim(62, 90)
    ax.set_xticks([10, 100, 1e3, 1e4, 1e5], ["10 B", "100 B", "1 KB", "10 KB", "100 KB"])
    ax.set_xlabel("model size (median over the 14 trained models)")
    ax.set_ylabel("agreement within one gear (%)")
    save(fig, "fig_accuracy_vs_size")


if __name__ == "__main__":
    ensure_dirs()
    res = json.loads((RESULTS / "target_policies.json").read_text())
    ev = json.loads((RESULTS / "event_classifier.json").read_text())
    margin_per_test(res)
    accuracy_vs_size(res, ev)
    print("figures written to", FIGURES)

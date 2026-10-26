"""
Manuscript figures (vector PDF) from the saved result files.

  Figure_3.pdf    model scale x grounding: (a) correct BNS section, (b) repealed
                  IPC number cited -- small multiples, one y-scale each
  Figure_1.pdf    offence identification vs retrieval depth for the retrieval
                  systems, with same-task reference lines
  (Figure_2 is the TikZ design diagram: figures/Figure_2.tex,
  cropped by 34_crop_figure.py.)

Palette: the dataviz skill's validated reference categorical palette, unmodified
and in its documented slot order (adjacent-pair CVD/normal-vision gates pass in
light mode). Slots 3-4 sit below 3:1 on white, so every chart carries direct
labels and the manuscript tables carry every value (the relief rule). Lines also
carry distinct markers so identity survives greyscale printing.
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figures")

SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5,
    "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5, "axes.edgecolor": AXIS, "axes.linewidth": 0.6,
    "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK2,
    "text.color": INK, "pdf.fonttype": 42,
})


def style(ax):
    ax.grid(axis="y", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)


def fig_scale():
    t = pd.read_csv(os.path.join(RES, "scale_final_table.csv"))
    models = [("claude-opus-5-5", "Opus 5.5"), ("claude-haiku-4-5", "Haiku 4.5"),
              ("qwen2.5:7b", "Qwen 2.5\n7B"), ("llama3.1:8b", "Llama 3.1\n8B"),
              ("llama3.2:3b", "Llama 3.2\n3B")]
    conds = [("none", "No grounding"), ("rag", "Retrieved statute"),
             ("rag+graph", "Retrieved statute + graph")]
    w, gap = 0.26, 0.02
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), sharey=True)
    panels = [("section_family", "(a) Correct BNS section cited"),
              ("ipc_confusion", "(b) Repealed IPC section number cited")]
    for ax, (metric, title) in zip(axes, panels):
        style(ax)
        for ci, (c, clab) in enumerate(conds):
            for mi, (m, _) in enumerate(models):
                row = t[(t.model == m) & (t.condition == c)]
                if not len(row):
                    continue
                v = float(row[metric].iloc[0])
                x = mi + (ci - 1) * (w + gap)
                ax.bar(x, v, width=w, color=SLOT[ci], linewidth=0,
                       label=clab if mi == 1 else None)
                if c == "none":            # selective labels: the ungrounded bars
                    ax.text(x, v + 1.8, f"{v:.1f}" if v < 99.5 else "99.3", ha="center",
                            va="bottom", fontsize=6.5, color=INK2)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([lab for _, lab in models])
        ax.set_title(title, loc="left", color=INK)
        ax.set_ylim(0, 108)
        ax.set_yticks([0, 25, 50, 75, 100])
    axes[0].set_ylabel("Queries (%)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.02), handlelength=1.2, columnspacing=1.6)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = os.path.join(FIG, "Figure_3.pdf")
    fig.savefig(out); fig.savefig(out.replace(".pdf", ".png"), dpi=200)
    plt.close(fig)
    return out


def fig_ksweep():
    t = pd.read_csv(os.path.join(RES, "final_table_v11.csv"))
    t = t[t.k.astype(str).str.len() > 0]
    t["k"] = pd.to_numeric(t["k"], errors="coerce")
    t = t.dropna(subset=["k"])
    series = [("BGE-base + graph", "BGE + graph", "o"),
              ("hybrid RRF + graph", "Hybrid (BM25+BGE) + graph", "s"),
              ("MiniLM-L6 + graph", "MiniLM + graph (original)", "^"),
              ("BM25 + graph", "BM25 + graph", "D")]
    ks = [1, 3, 5, 10, 20]
    xs = range(len(ks))
    fig, ax = plt.subplots(figsize=(3.5, 2.9))
    style(ax)
    for i, (name, lab, mk) in enumerate(series):
        s = t[t.system == name].set_index("k").reindex(ks)["accuracy"].values
        ax.plot(xs, s, color=SLOT[i], linewidth=1.5, marker=mk, markersize=4.2,
                label=lab, markeredgewidth=0)
    # Reference lines are thresholds, so they are dashed; labels sit where no data
    # line passes (Opus top-left, trivial baselines at the right edge, below the line).
    right = len(ks) - 0.85
    refs = [(99.26, "Opus 5.5 zero-shot (single label)", -0.15, "left", "bottom", 1.0),
            (77.04, "BGE trivial", right, "right", "top", -1.2),
            (61.48, "MiniLM trivial", right, "right", "top", -1.2)]
    for y, lab, x, ha, va, dy in refs:
        ax.axhline(y, color=MUTED, linewidth=0.7, linestyle=(0, (3, 2)))
        ax.text(x, y + dy, lab, fontsize=6.2, color=INK2, va=va, ha=ha)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel("Retrieved BNS sections (k)")
    ax.set_ylabel("Gold offence in candidate set (%)")
    ax.set_ylim(30, 104)
    ax.set_xlim(-0.2, len(ks) - 0.8)
    ax.legend(loc="lower right", frameon=False, handlelength=1.6, borderaxespad=0.2,
              labelspacing=0.3, bbox_to_anchor=(1.0, 0.0))
    fig.tight_layout()
    out = os.path.join(FIG, "Figure_1.pdf")
    fig.savefig(out); fig.savefig(out.replace(".pdf", ".png"), dpi=200)
    plt.close(fig)
    return out


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    print(fig_scale())
    print(fig_ksweep())

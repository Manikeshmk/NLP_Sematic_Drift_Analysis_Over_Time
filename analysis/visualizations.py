"""
analysis/visualizations.py
---------------------------
All matplotlib / seaborn visualisation functions for the project.
Each function returns a matplotlib Figure so callers can save or display it.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless backend for Flask
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from wordcloud import WordCloud
import io, base64


# ─── Utility ──────────────────────────────────────────────────────────────────

def fig_to_base64(fig: plt.Figure) -> str:
    """Serialise a matplotlib Figure to a base-64 PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=130)
    buf.seek(0)
    data = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return data


# ─── 1. t-SNE Neighborhood Plot ───────────────────────────────────────────────

def plot_tsne_drift(word: str,
                   vecs_old: np.ndarray, vocab_old: list, idx_old: dict,
                   vecs_new: np.ndarray, vocab_new: list, idx_new: dict,
                   top_n: int = 10) -> str:
    """
    t-SNE 2-D projection of a word's neighbourhood in two eras.
    Returns base-64 PNG.
    """
    def get_nbrs(vecs, vocab, idx, n=top_n):
        if word not in idx:
            return []
        tgt = vecs[idx[word]].reshape(1, -1)
        sims = cosine_similarity(vecs, tgt).flatten()
        ranked = np.argsort(sims)[::-1]
        out = []
        for i in ranked:
            w = vocab[i].lower()
            if w != word.lower() and len(w) > 3 and w.isalpha():
                out.append(w)
            if len(out) >= n:
                break
        return out

    nbrs_old = get_nbrs(vecs_old, vocab_old, idx_old)
    nbrs_new = get_nbrs(vecs_new, vocab_new, idx_new)

    all_words = list(set([word] + nbrs_old + nbrs_new))
    plot_vecs, plot_labels = [], []

    for w in all_words:
        if w in idx_new:
            plot_vecs.append(vecs_new[idx_new[w]])
            plot_labels.append(w)

    if len(plot_vecs) < 5:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Not enough data for t-SNE", ha="center")
        return fig_to_base64(fig)

    perp = min(5, len(plot_vecs) - 1)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perp)
    coords = tsne.fit_transform(np.array(plot_vecs))

    fig, ax = plt.subplots(figsize=(11, 7))
    fig.patch.set_facecolor("#0f0f1a")
    ax.set_facecolor("#0f0f1a")

    colors = {
        "both":  "#a855f7",
        "old":   "#ef4444",
        "new":   "#3b82f6",
        "target":"#facc15",
    }

    for i, label in enumerate(plot_labels):
        if label == word:
            c, s, z = colors["target"], 260, 5
        elif label in nbrs_old and label in nbrs_new:
            c, s, z = colors["both"], 140, 3
        elif label in nbrs_old:
            c, s, z = colors["old"], 120, 2
        else:
            c, s, z = colors["new"], 120, 2

        ax.scatter(coords[i, 0], coords[i, 1], c=c, s=s, zorder=z,
                   edgecolors="white" if label == word else "none", linewidths=1.5)
        ax.annotate(label, (coords[i, 0] + 0.3, coords[i, 1] + 0.3),
                    fontsize=9, color=c, fontweight="bold" if label == word else "normal")

    # Legend
    from matplotlib.patches import Patch
    legend = [
        Patch(color=colors["target"], label=f"Target: '{word}'"),
        Patch(color=colors["old"],    label="Old context only"),
        Patch(color=colors["new"],    label="New context only"),
        Patch(color=colors["both"],   label="Shared context"),
    ]
    ax.legend(handles=legend, facecolor="#1e1e2e", labelcolor="white", fontsize=8)
    ax.set_title(f"Semantic Neighbourhood Drift — '{word}'",
                 color="white", fontsize=14, pad=12)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")

    return fig_to_base64(fig)


# ─── 2. Timeline Line Chart ────────────────────────────────────────────────────

def plot_drift_timeline(word: str, years: list[int],
                        scores: list[float | None],
                        reference_year: int) -> str:
    """
    Line chart of cosine drift from the reference year across all decades.
    Returns base-64 PNG.
    """
    valid_years  = [y for y, s in zip(years, scores) if s is not None]
    valid_scores = [s for s in scores if s is not None]

    fig, ax = plt.subplots(figsize=(11, 5))
    fig.patch.set_facecolor("#0f0f1a")
    ax.set_facecolor("#0f0f1a")

    ax.plot(valid_years, valid_scores, color="#a855f7", lw=2.5,
            marker="o", markersize=7, markerfacecolor="#facc15")
    ax.fill_between(valid_years, valid_scores, alpha=0.15, color="#a855f7")

    ax.set_title(f"Semantic Drift of '{word}' Over Time (ref: {reference_year})",
                 color="white", fontsize=13)
    ax.set_xlabel("Decade", color="#aaa")
    ax.set_ylabel("Cosine Distance from Reference", color="#aaa")
    ax.tick_params(colors="white")
    ax.set_ylim(0, 1.05)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")

    return fig_to_base64(fig)


# ─── 3. Top-Drifted Words Bar Chart ───────────────────────────────────────────

def plot_top_drifted(records: list[dict], year_a: int, year_b: int,
                     top_n: int = 20) -> str:
    """
    Horizontal bar chart of the top-N most semantically drifted words.
    """
    records = sorted(records, key=lambda x: x["drift"], reverse=True)[:top_n]
    words  = [r["word"] for r in records]
    drifts = [r["drift"] for r in records]

    cmap   = cm.plasma
    colours = [cmap(d) for d in np.linspace(0.2, 0.9, len(words))]

    fig, ax = plt.subplots(figsize=(9, 0.45 * len(words) + 1.5))
    fig.patch.set_facecolor("#0f0f1a")
    ax.set_facecolor("#0f0f1a")

    bars = ax.barh(words[::-1], drifts[::-1], color=colours, edgecolor="none")
    for bar, val in zip(bars, drifts[::-1]):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", color="white", fontsize=8)

    ax.set_xlabel("Cosine Distance (drift)", color="#aaa")
    ax.set_title(f"Top-{top_n} Drifted Words  ({year_a} → {year_b})",
                 color="white", fontsize=12)
    ax.tick_params(colors="white")
    ax.set_xlim(0, 1.1)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")

    return fig_to_base64(fig)


# ─── 4. Neighbor Shift Diagram ────────────────────────────────────────────────

def plot_neighbor_shift(word: str, ns: dict,
                        year_a: int, year_b: int) -> str:
    """
    Side-by-side ranked neighbor lists with colour-coded gain/loss/shared.
    """
    nbrs_a = [w for w, _ in ns["neighbors_old"]][:8]
    nbrs_b = [w for w, _ in ns["neighbors_new"]][:8]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor("#0f0f1a")
    for ax in (axL, axR):
        ax.set_facecolor("#0f0f1a")
        ax.axis("off")

    def draw_list(ax, items, other_set, title):
        ax.set_title(title, color="white", fontsize=11)
        for i, w in enumerate(items):
            clr = "#a855f7" if w in other_set else "#ef4444" if ax is axL else "#3b82f6"
            ax.text(0.5, 0.9 - i * 0.1, w, ha="center", va="center",
                    color=clr, fontsize=12, transform=ax.transAxes,
                    fontweight="bold" if w in other_set else "normal")

    draw_list(axL, nbrs_a, set(nbrs_b), f"Neighbors in {year_a}")
    draw_list(axR, nbrs_b, set(nbrs_a), f"Neighbors in {year_b}")

    fig.suptitle(f"Neighbor Shift for '{word}'  —  Jaccard={ns['jaccard_similarity']:.2f}",
                 color="white", fontsize=13)
    return fig_to_base64(fig)


# ─── 5. Heatmap: word × decade drift matrix ────────────────────────────────────

def plot_drift_heatmap(word_list: list[str], decades: list[int],
                       score_matrix: list[list[float | None]]) -> str:
    """
    Seaborn heatmap where rows=words, cols=decades, cells=drift from decade[0].
    """
    import pandas as pd
    df = pd.DataFrame(score_matrix, index=word_list, columns=decades)

    fig, ax = plt.subplots(figsize=(max(10, len(decades) * 0.8),
                                    max(6, len(word_list) * 0.45)))
    fig.patch.set_facecolor("#0f0f1a")
    ax.set_facecolor("#0f0f1a")

    sns.heatmap(df, ax=ax, cmap="plasma", vmin=0, vmax=1,
                linewidths=0.3, linecolor="#111",
                cbar_kws={"label": "Cosine Distance"},
                annot=(len(word_list) <= 15), fmt=".2f", annot_kws={"size": 7})

    ax.set_title("Semantic Drift Heatmap", color="white", fontsize=13, pad=10)
    ax.tick_params(colors="white", labelsize=8)
    ax.figure.axes[-1].tick_params(colors="white")  # colorbar

    return fig_to_base64(fig)


# ─── 6. Word-Cloud of drifted words ───────────────────────────────────────────

def plot_drift_wordcloud(top_records: list[dict]) -> str:
    """
    Word-cloud where font size ∝ drift score.
    """
    freq = {r["word"]: int(r["drift"] * 1000) for r in top_records}
    wc = WordCloud(
        width=900, height=450, background_color="#0f0f1a",
        colormap="plasma", max_words=80,
    ).generate_from_frequencies(freq)

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#0f0f1a")
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title("Word Cloud — Semantic Drift Intensity",
                 color="white", fontsize=13)
    return fig_to_base64(fig)


# ─── 7. Concept Convergence / Divergence ──────────────────────────────────────

def plot_concept_relation(word_pairs: list[tuple[str, str]],
                          decades: list[int],
                          similarity_over_time: list[list[float | None]]) -> str:
    """
    Line chart tracking cosine similarity between word pairs across decades.
    similarity_over_time[pair_idx][decade_idx] = sim or None
    """
    fig, ax = plt.subplots(figsize=(11, 5))
    fig.patch.set_facecolor("#0f0f1a")
    ax.set_facecolor("#0f0f1a")

    palette = ["#a855f7", "#3b82f6", "#22c55e", "#f97316", "#ef4444"]
    for pi, (w1, w2) in enumerate(word_pairs):
        sims = similarity_over_time[pi]
        valid = [(d, s) for d, s in zip(decades, sims) if s is not None]
        if not valid:
            continue
        xs, ys = zip(*valid)
        c = palette[pi % len(palette)]
        ax.plot(xs, ys, color=c, lw=2, marker="o", markersize=5,
                label=f"{w1} ↔ {w2}")

    ax.set_xlabel("Decade", color="#aaa")
    ax.set_ylabel("Cosine Similarity", color="#aaa")
    ax.set_title("Concept Relation Over Time", color="white", fontsize=13)
    ax.tick_params(colors="white")
    ax.set_ylim(-0.1, 1.05)
    ax.legend(facecolor="#1e1e2e", labelcolor="white", fontsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")

    return fig_to_base64(fig)

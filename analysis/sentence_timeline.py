"""
analysis/sentence_timeline.py
══════════════════════════════════════════════════════════════════════════════
INNOVATION: Semantic Drift Constellation
─────────────────────────────────────────
Visualises the semantic drift of EVERY word in a sentence simultaneously
across all 20 decades (1800–1990), producing:

  1. Per-word thin coloured drift timelines  (individual word arcs)
  2. A bold thick sentence-level drift curve (the "conductor" line)
  3. A word-correlation comparison heatmap   (how word relationships changed)
  4. A "stability rank" table on the side

This is novel because:
  • No existing tool tracks sentence-level embedding drift over 200 years
  • The word-correlation matrix shows HOW the semantic fabric of a phrase
    restructures itself — which words stay close, which drift apart
  • The constellation metaphor captures linguistic "gravity" over time

Returns Plotly-ready JSON (for interactive rendering in browser) AND a
static matplotlib correlation heatmap as base64 PNG.
"""

import re, json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import io, base64


# ── Stop-words (same fast set as text_analysis) ──────────────────────────────
STOPWORDS = {
    "a","an","the","and","or","but","in","on","at","to","for","of","with",
    "by","from","as","is","was","are","were","be","been","being","have",
    "has","had","do","does","did","will","would","could","should","may",
    "might","shall","can","need","dare","ought","used","that","this","these",
    "those","it","its","i","me","my","we","us","our","you","your","he","him",
    "his","she","her","they","them","their","what","which","who","whom","not",
    "no","nor","so","yet","both","either","neither","one","two","more",
    "most","other","some","such","own","same","than","too","very","just",
    "also","into","over","then","up","out","about","after","before","during",
    "here","there","where","when","how","all","each","every","few","much",
    "many","now","only","well","still","even",
}

# Dark-theme background
C_BG      = "#080812"
C_SURFACE = "#0f0f1e"
C_TEXT    = "#d4d4e8"

# Vivid palette for up to 15 words
WORD_PALETTE = [
    "#a855f7","#3b82f6","#22c55e","#f97316","#ef4444",
    "#facc15","#06b6d4","#ec4899","#84cc16","#8b5cf6",
    "#14b8a6","#f43f5e","#fb923c","#a3e635","#38bdf8",
]


def _tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-zA-Z]+", text.lower())
    seen, out = set(), []
    for t in tokens:
        if len(t) > 2 and t not in STOPWORDS and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _b64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=130, facecolor=C_BG)
    buf.seek(0)
    data = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return data


# ══════════════════════════════════════════════════════════════════════════════
# Core computation
# ══════════════════════════════════════════════════════════════════════════════

def build_constellation_data(text: str,
                              decade_data: list[tuple],
                              year_a: int,
                              year_b: int) -> dict:
    """
    Compute per-word and sentence drift across every decade.

    Parameters
    ----------
    text        : Input sentence / idiom / phrase
    decade_data : [(year, vecs, idx), ...] sorted ascending
    year_a      : "From" year (highlighted in chart)
    year_b      : "To" year   (highlighted in chart)

    Returns
    -------
    dict with:
      content_words   : list of content tokens
      decades         : list of decade years
      word_scores     : { word: [score_per_decade, ...] }
      sentence_scores : [sentence_drift_per_decade, ...]
      word_a          : { word: [cosine sim to all other words] } in year_a
      word_b          : { word: [...] } in year_b
      correlation_a   : 2D ndarray of cosine sim between words in year_a
      correlation_b   : 2D ndarray of cosine sim between words in year_b
      year_a, year_b
      reference_year  : first decade
    """
    words = _tokenize(text)
    if not words:
        return {"error": "No content words found"}

    ref_year, ref_vecs, ref_idx = decade_data[0]
    decades = [yr for yr, _, _ in decade_data]

    # ── Per-word drift timelines ──────────────────────────────────────────────
    word_scores: dict[str, list] = {}

    for word in words:
        if word not in ref_idx:
            continue
        ref_vec = ref_vecs[ref_idx[word]].reshape(1, -1)
        row = []
        for yr, vecs, idx in decade_data:
            if word not in idx:
                row.append(None)
            else:
                cur = vecs[idx[word]].reshape(1, -1)
                sim = float(cosine_similarity(ref_vec, cur)[0, 0])
                row.append(round(1.0 - sim, 4))
        word_scores[word] = row

    # Only keep words that have enough coverage
    word_scores = {w: s for w, s in word_scores.items()
                   if sum(v is not None for v in s) >= 3}

    if not word_scores:
        return {"error": "Words not found in embeddings — check your sgns/ data"}

    # ── Sentence drift timeline ───────────────────────────────────────────────
    # Reference sentence embedding = mean of ref-vectors of all found words
    ref_emb_parts = [ref_vecs[ref_idx[w]] for w in word_scores if w in ref_idx]
    ref_emb = np.mean(ref_emb_parts, axis=0) if ref_emb_parts else None

    sentence_scores = []
    for yr, vecs, idx in decade_data:
        parts = [vecs[idx[w]] for w in word_scores if w in idx]
        if parts and ref_emb is not None:
            cur_emb = np.mean(parts, axis=0)
            sim = float(cosine_similarity(
                ref_emb.reshape(1, -1), cur_emb.reshape(1, -1))[0, 0])
            sentence_scores.append(round(1.0 - sim, 4))
        else:
            sentence_scores.append(None)

    # ── Word-correlation matrices for year_a and year_b ─────────────────────
    def corr_matrix(yr, vecs, idx):
        present = [w for w in word_scores if w in idx]
        if len(present) < 2:
            return present, np.zeros((len(present), len(present)))
        vecs_sub = np.array([vecs[idx[w]] for w in present])
        mat = cosine_similarity(vecs_sub)
        np.fill_diagonal(mat, 1.0)
        return present, mat

    # Find the decade_data entry for year_a and year_b
    def get_yr(target):
        for yr, v, ix in decade_data:
            if yr == target:
                return v, ix
        return None, None

    va, ia = get_yr(year_a)
    vb, ib = get_yr(year_b)

    corr_words_a, corr_mat_a = corr_matrix(year_a, va, ia) if va is not None else ([], np.array([]))
    corr_words_b, corr_mat_b = corr_matrix(year_b, vb, ib) if vb is not None else ([], np.array([]))

    # Align both matrices to the same word order
    all_corr_words = [w for w in word_scores
                      if w in (corr_words_a or []) and w in (corr_words_b or [])]
    def _sub(words_full, mat, target_words):
        idx_map = {w: i for i, w in enumerate(words_full)}
        sel = [idx_map[w] for w in target_words if w in idx_map]
        if not sel:
            return np.zeros((len(target_words), len(target_words)))
        return mat[np.ix_(sel, sel)]

    mat_a = _sub(corr_words_a, corr_mat_a, all_corr_words)
    mat_b = _sub(corr_words_b, corr_mat_b, all_corr_words)

    # ── Stability rank ────────────────────────────────────────────────────────
    def final_drift(scores):
        valid = [s for s in scores if s is not None]
        return max(valid) if valid else 0.0

    stability_rank = sorted(
        [{"word": w, "max_drift": round(final_drift(s), 4), "scores": s}
         for w, s in word_scores.items()],
        key=lambda x: x["max_drift"], reverse=True
    )

    return {
        "content_words":    list(word_scores.keys()),
        "decades":          decades,
        "word_scores":      word_scores,
        "sentence_scores":  sentence_scores,
        "correlation_words": all_corr_words,
        "correlation_a":    mat_a.tolist() if len(mat_a) else [],
        "correlation_b":    mat_b.tolist() if len(mat_b) else [],
        "stability_rank":   stability_rank,
        "year_a":           year_a,
        "year_b":           year_b,
        "reference_year":   ref_year,
        "text":             text,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Plotly-ready JSON traces (interactive in browser)
# ══════════════════════════════════════════════════════════════════════════════

def build_plotly_constellation(data: dict) -> dict:
    """
    Return Plotly figure spec (data + layout) ready for Plotly.newPlot().

    Design:
      • One thin semi-transparent line per content word
      • A bold glowing white/purple line for overall sentence drift
      • Vertical dashed markers for year_a and year_b
      • Hover shows word + drift value + decade
    """
    decades  = data["decades"]
    year_a   = data["year_a"]
    year_b   = data["year_b"]
    words    = data["content_words"]
    ref_year = data["reference_year"]

    traces = []

    # ── Per-word thin lines ───────────────────────────────────────────────────
    for i, word in enumerate(words):
        scores = data["word_scores"].get(word, [])
        color  = WORD_PALETTE[i % len(WORD_PALETTE)]
        hex_color = color

        # fill Nones with gaps (None = breaks the line = natural gap)
        xs = [d for d, s in zip(decades, scores) if s is not None]
        ys = [s for s in scores if s is not None]

        traces.append({
            "type": "scatter",
            "x": xs, "y": ys,
            "mode": "lines+markers",
            "name": word,
            "line": {
                "color": hex_color,
                "width": 1.8,
                "dash": "solid",
            },
            "marker": {"size": 5, "color": hex_color},
            "opacity": 0.75,
            "hovertemplate": (
                f"<b>{word}</b><br>"
                "Decade: %{x}<br>"
                "Drift from " + str(ref_year) + ": %{y:.4f}<extra></extra>"
            ),
        })

    # ── Sentence drift BOLD line (glow effect: two passes) ───────────────────
    sent_xs = [d for d, s in zip(decades, data["sentence_scores"]) if s is not None]
    sent_ys = [s for s in data["sentence_scores"] if s is not None]

    # Glow shadow (wider, more transparent)
    traces.append({
        "type": "scatter",
        "x": sent_xs, "y": sent_ys,
        "mode": "lines",
        "name": "Sentence Drift (glow)",
        "line": {"color": "rgba(168,85,247,0.25)", "width": 12},
        "showlegend": False,
        "hoverinfo": "skip",
    })

    # Actual bold sentence line
    traces.append({
        "type": "scatter",
        "x": sent_xs, "y": sent_ys,
        "mode": "lines+markers",
        "name": "⬛ Sentence Drift",
        "line": {
            "color": "#ffffff",
            "width": 3.5,
        },
        "marker": {
            "size": 9,
            "color": "#a855f7",
            "line": {"color": "#ffffff", "width": 2},
        },
        "hovertemplate": (
            "<b>SENTENCE DRIFT</b><br>"
            "Decade: %{x}<br>"
            "Drift from " + str(ref_year) + ": %{y:.4f}<extra></extra>"
        ),
    })

    # ── Vertical markers for year_a and year_b ────────────────────────────────
    shapes = []
    annotations = []

    for yr, label, color in [(year_a, f"FROM: {year_a}", "#3b82f6"),
                              (year_b, f"TO: {year_b}",   "#ef4444")]:
        shapes.append({
            "type": "line",
            "x0": yr, "x1": yr,
            "y0": 0,  "y1": 1,
            "yref": "paper",
            "line": {"color": color, "width": 1.8, "dash": "dot"},
        })
        annotations.append({
            "x": yr, "y": 1.02,
            "xref": "x", "yref": "paper",
            "text": label,
            "showarrow": False,
            "font": {"color": color, "size": 11, "family": "Inter"},
        })

    # ── Layout ────────────────────────────────────────────────────────────────
    preview = data["text"][:60] + ("…" if len(data["text"]) > 60 else "")
    layout = {
        "title": {
            "text": f"Semantic Drift Constellation — \"{preview}\"",
            "font": {"color": "#ffffff", "size": 15, "family": "Inter"},
            "x": 0.5,
        },
        "paper_bgcolor": C_BG,
        "plot_bgcolor":  C_SURFACE,
        "font": {"color": C_TEXT, "family": "Inter"},
        "xaxis": {
            "title": {"text": "Decade", "font": {"color": "#888"}},
            "color": C_TEXT,
            "gridcolor": "#1e1e30",
            "linecolor": "#333",
            "tickmode": "array",
            "tickvals": decades,
            "ticktext": [str(d) for d in decades],
            "tickangle": -45,
        },
        "yaxis": {
            "title": {"text": f"Cosine Distance from {ref_year}", "font": {"color": "#888"}},
            "color": C_TEXT,
            "gridcolor": "#1e1e30",
            "linecolor": "#333",
            "range": [0, None],
        },
        "legend": {
            "bgcolor": "rgba(15,15,30,0.8)",
            "bordercolor": "#333",
            "borderwidth": 1,
            "font": {"size": 10},
        },
        "shapes": shapes,
        "annotations": annotations,
        "hovermode": "x unified",
        "margin": {"l": 60, "r": 20, "t": 70, "b": 60},
    }

    return {"data": traces, "layout": layout}


# ══════════════════════════════════════════════════════════════════════════════
# Word-correlation comparison heatmap (matplotlib)
# ══════════════════════════════════════════════════════════════════════════════

def plot_correlation_comparison(data: dict) -> str:
    """
    Side-by-side heatmap comparing word-pair cosine similarities in year_a vs year_b.
    Also shows the DIFFERENCE matrix (year_b − year_a) on the right.

    Three panels:
      [year_a corr]   [year_b corr]   [Δ = B − A]
    """
    words  = data["correlation_words"]
    mat_a  = np.array(data["correlation_a"])
    mat_b  = np.array(data["correlation_b"])
    year_a = data["year_a"]
    year_b = data["year_b"]

    if len(words) < 2 or mat_a.size == 0 or mat_b.size == 0:
        fig, ax = plt.subplots(figsize=(8, 3), facecolor=C_BG)
        ax.set_facecolor(C_BG)
        ax.text(0.5, 0.5, "Not enough words to build correlation matrix",
                ha="center", va="center", color=C_TEXT, transform=ax.transAxes)
        ax.axis("off")
        return _b64(fig)

    diff   = mat_b - mat_a
    n = len(words)
    cell_w = max(0.55, 8.5 / max(n, 1))

    fig, axes = plt.subplots(1, 3, figsize=(min(cell_w * n * 3 + 2, 22), max(4, cell_w * n + 1.5)),
                              facecolor=C_BG)
    fig.patch.set_facecolor(C_BG)

    cmap_sim  = LinearSegmentedColormap.from_list("sim", ["#080812","#3b82f6","#a855f7","#facc15"])
    cmap_diff = LinearSegmentedColormap.from_list("diff",["#ef4444","#111122","#22c55e"])

    def draw(ax, mat, title, cmap, vmin, vmax, annot=True):
        import pandas as pd
        df = pd.DataFrame(mat, index=words, columns=words)
        sns.heatmap(df, ax=ax, cmap=cmap, vmin=vmin, vmax=vmax,
                    annot=annot and n <= 12,
                    fmt=".2f" if (annot and n <= 12) else "",
                    annot_kws={"size": max(5, 10 - n)},
                    linewidths=0.3, linecolor="#1e1e30",
                    cbar_kws={"shrink": 0.7})
        ax.set_facecolor(C_BG)
        ax.set_title(title, color="white", fontsize=10, pad=8)
        ax.tick_params(colors=C_TEXT, labelsize=max(6, 10 - n // 2))
        ax.figure.axes[-1].tick_params(colors=C_TEXT)  # colorbar

    draw(axes[0], mat_a, f"Word Correlations — {year_a}", cmap_sim, 0, 1)
    draw(axes[1], mat_b, f"Word Correlations — {year_b}", cmap_sim, 0, 1)
    draw(axes[2], diff,  f"Δ Correlation ({year_b} − {year_a})", cmap_diff, -0.5, 0.5)

    fig.suptitle("How Word Relationships Changed Over Time",
                 color="white", fontsize=12, y=1.01)
    plt.tight_layout()
    return _b64(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Optional: Combined static matplotlib figure (for export / report)
# ══════════════════════════════════════════════════════════════════════════════

def plot_static_constellation(data: dict, top_n_words: int = 12) -> str:
    """
    Full static matplotlib constellation:
      Top 70%  — per-word timelines + bold sentence line
      Bottom 30% — correlation diff heatmap
    Suitable for inclusion in a research report.
    """
    import matplotlib.gridspec as gridspec

    words    = data["content_words"][:top_n_words]
    decades  = data["decades"]
    year_a   = data["year_a"]
    year_b   = data["year_b"]
    ref_year = data["reference_year"]

    fig = plt.figure(figsize=(16, 11), facecolor=C_BG)
    gs  = gridspec.GridSpec(2, 1, height_ratios=[2.2, 1], hspace=0.38)

    ax_main = fig.add_subplot(gs[0])
    ax_corr = fig.add_subplot(gs[1])

    for ax in (ax_main, ax_corr):
        ax.set_facecolor(C_SURFACE)
        for sp in ax.spines.values():
            sp.set_edgecolor("#333")

    # ── Timeline ──────────────────────────────────────────────────────────────
    max_drift = 0

    for i, word in enumerate(words):
        scores = data["word_scores"].get(word, [])
        color  = WORD_PALETTE[i % len(WORD_PALETTE)]
        xs = [d for d, s in zip(decades, scores) if s is not None]
        ys = [s for s in scores if s is not None]
        if not ys:
            continue
        max_drift = max(max_drift, max(ys))

        ax_main.plot(xs, ys, color=color, lw=1.5, alpha=0.75,
                     marker="o", markersize=3, label=word)
        # End label
        if xs:
            ax_main.annotate(word, (xs[-1], ys[-1]),
                             xytext=(5, 0), textcoords="offset points",
                             fontsize=7.5, color=color, va="center")

    # Sentence drift glow + line
    sent_xs = [d for d, s in zip(decades, data["sentence_scores"]) if s is not None]
    sent_ys = [s for s in data["sentence_scores"] if s is not None]

    if sent_ys:
        max_drift = max(max_drift, max(sent_ys))
        # glow
        ax_main.plot(sent_xs, sent_ys, color="#a855f7", lw=10, alpha=0.12)
        ax_main.plot(sent_xs, sent_ys, color="#a855f7", lw=5,  alpha=0.22)
        # main line
        ax_main.plot(sent_xs, sent_ys, color="white", lw=2.8,
                     marker="D", markersize=6,
                     markerfacecolor="#a855f7", markeredgecolor="white",
                     markeredgewidth=1.2, label="Sentence Drift", zorder=5)

    # Vertical markers for year_a / year_b
    for yr, clr, lbl in [(year_a, "#3b82f6", f"FROM {year_a}"),
                          (year_b, "#ef4444", f"TO {year_b}")]:
        ax_main.axvline(yr, color=clr, lw=1.8, linestyle="--", alpha=0.8)
        ax_main.text(yr + 1, max_drift * 0.95, lbl,
                     color=clr, fontsize=8, va="top")

    # Shading between year_a and year_b
    ax_main.axvspan(year_a, year_b, alpha=0.04, color="#a855f7")

    ax_main.set_xlabel("Decade", color="#aaa", fontsize=9)
    ax_main.set_ylabel(f"Cosine Distance from {ref_year}", color="#aaa", fontsize=9)
    ax_main.set_title("Semantic Drift Constellation", color="white", fontsize=13, pad=10)
    ax_main.tick_params(colors=C_TEXT, labelsize=7)
    ax_main.set_xticks(decades)
    ax_main.set_xticklabels([str(d) for d in decades], rotation=45, ha="right", fontsize=7)
    ax_main.set_ylim(0, max(max_drift * 1.15, 0.05))
    ax_main.legend(facecolor=C_BG, labelcolor=C_TEXT, fontsize=7.5,
                   loc="upper left", ncol=3, framealpha=0.7)

    # ── Correlation diff heatmap ──────────────────────────────────────────────
    corr_words = data["correlation_words"]
    mat_a      = np.array(data["correlation_a"]) if data["correlation_a"] else None
    mat_b      = np.array(data["correlation_b"]) if data["correlation_b"] else None

    if mat_a is not None and mat_b is not None and len(corr_words) >= 2:
        diff = mat_b - mat_a
        import pandas as pd
        df = pd.DataFrame(diff, index=corr_words, columns=corr_words)
        cmap = LinearSegmentedColormap.from_list(
            "rdgn", ["#ef4444", "#111122", "#22c55e"])
        sns.heatmap(df, ax=ax_corr, cmap=cmap, vmin=-0.5, vmax=0.5,
                    annot=(len(corr_words) <= 12),
                    fmt=".2f" if len(corr_words) <= 12 else "",
                    annot_kws={"size": max(5, 10 - len(corr_words))},
                    linewidths=0.3, linecolor="#1e1e30",
                    cbar_kws={"shrink": 0.6, "label": "ΔCorrelation"})
        ax_corr.set_facecolor(C_BG)
        ax_corr.set_title(f"Δ Word Correlation  ({year_a} → {year_b})  "
                          f"· Green = grew closer · Red = drifted apart",
                          color="white", fontsize=10, pad=6)
        ax_corr.tick_params(colors=C_TEXT, labelsize=7)
        ax_corr.figure.axes[-1].tick_params(colors=C_TEXT)
    else:
        ax_corr.text(0.5, 0.5, "Not enough shared words for correlation",
                     ha="center", va="center", color=C_TEXT,
                     transform=ax_corr.transAxes)
        ax_corr.axis("off")

    preview = data["text"][:70] + ("…" if len(data["text"]) > 70 else "")
    fig.suptitle(f'"{preview}"',
                 color="#888", fontsize=9, y=0.995)
    return _b64(fig)

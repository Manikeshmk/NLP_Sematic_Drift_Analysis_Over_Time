"""
app.py — Flask REST API for Semantic Drift Analysis
====================================================
Routes
------
GET  /api/health                          — Health check & loaded decades
GET  /api/decades                         — List available decades
POST /api/drift                           — Pairwise drift score + neighbours
POST /api/timeline                        — Timeline drift across all decades
POST /api/top-drifted                     — Top-K most drifted words globally
POST /api/tsne                            — t-SNE neighbourhood plot (base64)
POST /api/drift-timeline-chart            — Timeline line chart (base64)
POST /api/top-drifted-chart               — Bar chart of top-drifted (base64)
POST /api/neighbor-shift                  — Neighbor shift details
POST /api/neighbor-shift-chart            — Neighbor shift diagram (base64)
POST /api/heatmap                         — Multi-word× decade heatmap (base64)
POST /api/wordcloud                       — Drift wordcloud (base64)
POST /api/concept-relation                — Word-pair similarity over time
POST /api/concept-relation-chart          — Line chart (base64)
GET  /api/case-studies                    — List case studies
GET  /api/case-study/<word>               — Single case study metadata
POST /api/search-word                     — Check if a word exists in embeddings
"""

import os, sys, time, logging
from flask import Flask, request, jsonify
from flask_cors import CORS

# ── path setup ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from core.embeddings      import load_decade, build_index, SGNS_BASE, AVAILABLE_DECADES
from analysis.drift_analysis import (
    compute_drift_score, compute_timeline,
    top_drifted_words, neighbor_shift, semantic_change_type, compare_word_pair
)
from analysis.visualizations import (
    plot_tsne_drift, plot_drift_timeline, plot_top_drifted,
    plot_neighbor_shift, plot_drift_heatmap, plot_drift_wordcloud,
    plot_concept_relation
)
from analysis.case_studies import CASE_STUDIES, get_case_study
from analysis.sentence_timeline import (
    build_constellation_data, build_plotly_constellation,
    plot_correlation_comparison, plot_static_constellation
)
from analysis.drift_explainer import generate_drift_report

# ── Flask setup ─────────────────────────────────────────────────────────────
app = Flask(__name__)

# CORS: allow all origins including file:// (origin = "null")
CORS(app,
     resources={r"/api/*": {"origins": "*"}},
     allow_headers=["Content-Type", "Accept"],
     methods=["GET", "POST", "OPTIONS"],
     supports_credentials=False)

@app.before_request
def _handle_preflight():
    """Ensure OPTIONS preflight always returns 200 — never 404."""
    from flask import request as req, make_response
    if req.method == "OPTIONS":
        r = make_response()
        r.headers["Access-Control-Allow-Origin"]  = "*"
        r.headers["Access-Control-Allow-Headers"] = "Content-Type, Accept"
        r.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        r.status_code = 200
        return r
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ─── Embedding cache ─────────────────────────────────────────────────────────
EMBEDDING_CACHE: dict[int, tuple] = {}   # year → (vectors, vocab, idx)
LOADED_DECADES: list[int] = []
TOP_DRIFTED_CACHE: dict[str, list] = {}  # "year_a-year_b" → list


def _load(year: int):
    """Load and cache a single decade (or return cached)."""
    if year in EMBEDDING_CACHE:
        return EMBEDDING_CACHE[year]
    log.info(f"Loading decade {year}…")
    vecs, vocab = load_decade(year)
    idx = build_index(vocab)
    EMBEDDING_CACHE[year] = (vecs, vocab, idx)
    return vecs, vocab, idx


# def _get(year: int):
#     if year not in EMBEDDING_CACHE:
#         return None, None, None
#     return EMBEDDING_CACHE[year]

def _get(year: int):
    try:
        return _load(year)
    except FileNotFoundError:
        return None, None, None

# Pre-load what's available on disk at startup
def preload():
    if not os.path.isdir(SGNS_BASE):
        log.warning(f"SGNS directory not found at '{SGNS_BASE}'. "
                    "Endpoints will return errors until data is present.")
        return
    for yr in AVAILABLE_DECADES:
        try:
            _load(yr)
            LOADED_DECADES.append(yr)
        except FileNotFoundError:
            pass
    log.info(f"Pre-loaded decades: {LOADED_DECADES}")


# ─── Health ──────────────────────────────────────────────────────────────────
# @app.route("/api/health")
# def health():
#     return jsonify({
#         "status": "ok",
#         "loaded_decades": LOADED_DECADES,
#         "sgns_dir": SGNS_BASE,
#         "timestamp": time.time(),
#     })
@app.route("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "loaded_decades": list(EMBEDDING_CACHE.keys()),
        "available_decades": AVAILABLE_DECADES,
        "sgns_dir": SGNS_BASE,
        "timestamp": time.time(),
    })

# ─── Decades ─────────────────────────────────────────────────────────────────
# @app.route("/api/decades")
# def decades():
    # return jsonify({"decades": LOADED_DECADES})
@app.route("/api/decades")
def decades():
    return jsonify({"decades": AVAILABLE_DECADES})

# ─── Search word ─────────────────────────────────────────────────────────────
@app.route("/api/search-word", methods=["POST"])
def search_word():
    data = request.json or {}
    word = data.get("word", "").strip().lower()
    if not word:
        return jsonify({"error": "word required"}), 400

    presence = {yr: (word in EMBEDDING_CACHE[yr][2]) for yr in LOADED_DECADES}
    return jsonify({"word": word, "presence": presence,
                    "found_in": [yr for yr, p in presence.items() if p]})


# ─── Pairwise drift ──────────────────────────────────────────────────────────
@app.route("/api/drift", methods=["POST"])
def drift():
    data = request.json or {}
    word   = data.get("word", "").strip().lower()
    year_a = int(data.get("year_a", 1900))
    year_b = int(data.get("year_b", 1990))

    if not word:
        return jsonify({"error": "word required"}), 400
    if year_a not in AVAILABLE_DECADES or year_b not in LOADED_DECADES:
        return jsonify({"error": f"Decade not loaded. Available: {LOADED_DECADES}"}), 400

    vecs_a, vocab_a, idx_a = _get(year_a)
    vecs_b, vocab_b, idx_b = _get(year_b)

    score = compute_drift_score(word, vecs_a, idx_a, vecs_b, idx_b)
    if score is None:
        return jsonify({"error": f"'{word}' not found in {year_a} or {year_b}"}), 404

    ns    = neighbor_shift(word, vecs_a, vocab_a, idx_a, vecs_b, vocab_b, idx_b)
    ctype = semantic_change_type(word, vecs_a, vocab_a, idx_a, vecs_b, vocab_b, idx_b)
    cs    = get_case_study(word)

    return jsonify({
        "word": word, "year_a": year_a, "year_b": year_b,
        "drift_score": round(score, 4),
        "change_type": ctype,
        "neighbor_shift": ns,
        "case_study": cs,
    })


# ─── Timeline ────────────────────────────────────────────────────────────────
@app.route("/api/timeline", methods=["POST"])
def timeline():
    data = request.json or {}
    word = data.get("word", "").strip().lower()
    if not word:
        return jsonify({"error": "word required"}), 400

    decade_data = [(yr, *EMBEDDING_CACHE[yr][:1], EMBEDDING_CACHE[yr][2])
                   for yr in LOADED_DECADES]
    # unpack properly
    decade_data = [(yr, EMBEDDING_CACHE[yr][0], EMBEDDING_CACHE[yr][2])
                   for yr in LOADED_DECADES]

    result = compute_timeline(word, decade_data)
    return jsonify({"word": word, **result})


# ─── Top-drifted words ───────────────────────────────────────────────────────
@app.route("/api/top-drifted", methods=["POST"])
def top_drifted():
    data    = request.json or {}
    year_a  = int(data.get("year_a", 1900))
    year_b  = int(data.get("year_b", 1990))
    top_n   = min(int(data.get("top_n", 30)), 100)
    cache_k = f"{year_a}-{year_b}-{top_n}"

    if cache_k in TOP_DRIFTED_CACHE:
        return jsonify({"year_a": year_a, "year_b": year_b, "results": TOP_DRIFTED_CACHE[cache_k]})

    if year_a not in AVAILABLE_DECADES or year_b not in LOADED_DECADES:
        return jsonify({"error": f"Decade not loaded. Available: {LOADED_DECADES}"}), 400

    vecs_a, _, idx_a = _get(year_a)
    vecs_b, _, idx_b = _get(year_b)
    results = top_drifted_words(vecs_a, idx_a, vecs_b, idx_b, top_n=top_n)
    TOP_DRIFTED_CACHE[cache_k] = results
    return jsonify({"year_a": year_a, "year_b": year_b, "results": results})


# ─── Neighbour shift detail ───────────────────────────────────────────────────
@app.route("/api/neighbor-shift", methods=["POST"])
def neighbor_shift_route():
    data   = request.json or {}
    word   = data.get("word", "").strip().lower()
    year_a = int(data.get("year_a", 1900))
    year_b = int(data.get("year_b", 1990))
    top_n  = int(data.get("top_n", 10))

    if not word:
        return jsonify({"error": "word required"}), 400
    if year_a not in AVAILABLE_DECADES or year_b not in LOADED_DECADES:
        return jsonify({"error": f"Decade not loaded. Available: {LOADED_DECADES}"}), 400

    vecs_a, vocab_a, idx_a = _get(year_a)
    vecs_b, vocab_b, idx_b = _get(year_b)

    ns = neighbor_shift(word, vecs_a, vocab_a, idx_a, vecs_b, vocab_b, idx_b, top_n)
    return jsonify({"word": word, "year_a": year_a, "year_b": year_b, **ns})


# ─── Chart endpoints (return base64 PNGs) ────────────────────────────────────

@app.route("/api/tsne", methods=["POST"])
def tsne_chart():
    data   = request.json or {}
    word   = data.get("word", "").strip().lower()
    year_a = int(data.get("year_a", 1900))
    year_b = int(data.get("year_b", 1990))
    top_n  = int(data.get("top_n", 10))

    if not word:
        return jsonify({"error": "word required"}), 400
    vecs_a, vocab_a, idx_a = _get(year_a)
    vecs_b, vocab_b, idx_b = _get(year_b)
    if vecs_a is None or vecs_b is None:
        return jsonify({"error": "Decade not loaded"}), 400

    img = plot_tsne_drift(word, vecs_a, vocab_a, idx_a, vecs_b, vocab_b, idx_b, top_n)
    return jsonify({"image": img})


@app.route("/api/drift-timeline-chart", methods=["POST"])
def timeline_chart():
    data = request.json or {}
    word = data.get("word", "").strip().lower()
    if not word:
        return jsonify({"error": "word required"}), 400

    decade_data = [(yr, EMBEDDING_CACHE[yr][0], EMBEDDING_CACHE[yr][2])
                   for yr in LOADED_DECADES]
    result = compute_timeline(word, decade_data)
    ref_yr = LOADED_DECADES[0] if LOADED_DECADES else 1800

    img = plot_drift_timeline(word, result["years"], result["scores"], ref_yr)
    return jsonify({"image": img, **result})


@app.route("/api/top-drifted-chart", methods=["POST"])
def top_drifted_chart():
    data   = request.json or {}
    year_a = int(data.get("year_a", 1900))
    year_b = int(data.get("year_b", 1990))
    top_n  = min(int(data.get("top_n", 20)), 50)

    cache_k = f"{year_a}-{year_b}-{top_n}"
    if cache_k not in TOP_DRIFTED_CACHE:
        if year_a not in AVAILABLE_DECADES or year_b not in LOADED_DECADES:
            return jsonify({"error": "Decade not loaded"}), 400
        vecs_a, _, idx_a = _get(year_a)
        vecs_b, _, idx_b = _get(year_b)
        TOP_DRIFTED_CACHE[cache_k] = top_drifted_words(vecs_a, idx_a, vecs_b, idx_b, top_n)

    img = plot_top_drifted(TOP_DRIFTED_CACHE[cache_k], year_a, year_b, top_n)
    return jsonify({"image": img})


@app.route("/api/neighbor-shift-chart", methods=["POST"])
def neighbor_shift_chart():
    data   = request.json or {}
    word   = data.get("word", "").strip().lower()
    year_a = int(data.get("year_a", 1900))
    year_b = int(data.get("year_b", 1990))

    vecs_a, vocab_a, idx_a = _get(year_a)
    vecs_b, vocab_b, idx_b = _get(year_b)
    ns  = neighbor_shift(word, vecs_a, vocab_a, idx_a, vecs_b, vocab_b, idx_b)
    img = plot_neighbor_shift(word, ns, year_a, year_b)
    return jsonify({"image": img})


@app.route("/api/heatmap", methods=["POST"])
def heatmap():
    data  = request.json or {}
    words = [w.strip().lower() for w in data.get("words", [])][:15]
    if not words:
        return jsonify({"error": "words list required"}), 400

    score_matrix = []
    for word in words:
        decade_data = [(yr, EMBEDDING_CACHE[yr][0], EMBEDDING_CACHE[yr][2])
                       for yr in LOADED_DECADES]
        result = compute_timeline(word, decade_data)
        score_matrix.append(result["scores"])

    img = plot_drift_heatmap(words, LOADED_DECADES, score_matrix)
    return jsonify({"image": img})


@app.route("/api/wordcloud", methods=["POST"])
def wordcloud():
    data   = request.json or {}
    year_a = int(data.get("year_a", 1900))
    year_b = int(data.get("year_b", 1990))
    top_n  = min(int(data.get("top_n", 80)), 100)

    cache_k = f"{year_a}-{year_b}-{top_n}"
    if cache_k not in TOP_DRIFTED_CACHE:
        if year_a not in AVAILABLE_DECADES or year_b not in LOADED_DECADES:
            return jsonify({"error": "Decade not loaded"}), 400
        vecs_a, _, idx_a = _get(year_a)
        vecs_b, _, idx_b = _get(year_b)
        TOP_DRIFTED_CACHE[cache_k] = top_drifted_words(vecs_a, idx_a, vecs_b, idx_b, top_n)

    img = plot_drift_wordcloud(TOP_DRIFTED_CACHE[cache_k])
    return jsonify({"image": img})


@app.route("/api/concept-relation", methods=["POST"])
def concept_relation():
    data       = request.json or {}
    word_pairs = [tuple(p) for p in data.get("pairs", [])][:5]
    if not word_pairs:
        return jsonify({"error": "pairs required"}), 400

    sims_matrix = []
    for w1, w2 in word_pairs:
        row = []
        for yr in LOADED_DECADES:
            vecs, _, idx = _get(yr)
            s = compare_word_pair(w1, w2, vecs, idx)
            row.append(s)
        sims_matrix.append(row)

    return jsonify({
        "pairs": word_pairs,
        "decades": LOADED_DECADES,
        "similarities": sims_matrix,
    })


@app.route("/api/concept-relation-chart", methods=["POST"])
def concept_relation_chart():
    data       = request.json or {}
    word_pairs = [tuple(p) for p in data.get("pairs", [])][:5]
    if not word_pairs:
        return jsonify({"error": "pairs required"}), 400

    sims_matrix = []
    for w1, w2 in word_pairs:
        row = []
        for yr in LOADED_DECADES:
            vecs, _, idx = _get(yr)
            row.append(compare_word_pair(w1, w2, vecs, idx))
        sims_matrix.append(row)

    img = plot_concept_relation(word_pairs, LOADED_DECADES, sims_matrix)
    return jsonify({"image": img})


# ─── INNOVATION: Semantic Drift Constellation ──────────────────────────────────────────

@app.route("/api/sentence-constellation", methods=["POST"])
def sentence_constellation():
    """
    Full sentence drift constellation:
      - Per-word drift timelines across ALL decades (Plotly JSON)
      - Sentence-level bold drift curve
      - Word-pair correlation comparison heatmap (base64 PNG)
      - Static combined figure (base64 PNG, for report export)

    Body: { text, year_a, year_b }
    """
    data   = request.json or {}
    text   = data.get("text", "").strip()
    year_a = int(data.get("year_a", 1900))
    year_b = int(data.get("year_b", 1990))

    if not text:
        return jsonify({"error": "text required"}), 400
    if len(text) > 3000:
        return jsonify({"error": "text too long (max 3000 chars)"}), 400

    # Load embeddings for all available decades on-demand
    decade_data = []
    for yr in AVAILABLE_DECADES:
        v, voc, ix = _get(yr)
        if v is not None:
            decade_data.append((yr, v, ix))

    if len(decade_data) < 3:
        return jsonify({"error": "Need at least 3 loaded decades"}), 400

    if year_a not in [yr for yr, _, _ in decade_data]:
        year_a = decade_data[0][0]
    if year_b not in [yr for yr, _, _ in decade_data]:
        year_b = decade_data[-1][0]

    # Compute constellation data
    cdata = build_constellation_data(text, decade_data, year_a, year_b)
    if "error" in cdata:
        return jsonify(cdata), 400

    # Build Plotly traces (interactive chart)
    plotly_spec = build_plotly_constellation(cdata)

    # Build correlation heatmap (static PNG)
    corr_chart = plot_correlation_comparison(cdata)

    # Build combined static figure (for report export)
    static_chart = plot_static_constellation(cdata)

    return jsonify({
        "plotly":       plotly_spec,          # { data, layout } → Plotly.newPlot
        "corr_chart":   corr_chart,           # base64 PNG
        "static_chart": static_chart,         # base64 PNG (full combined figure)
        "meta":         {
            "content_words":   cdata["content_words"],
            "decades":         cdata["decades"],
            "year_a":          cdata["year_a"],
            "year_b":          cdata["year_b"],
            "reference_year":  cdata["reference_year"],
            "stability_rank":  cdata["stability_rank"],
            "sentence_scores": cdata["sentence_scores"],
            "word_scores":     cdata["word_scores"],
            "text":            text,
        },
    })


# ─── INNOVATION: Semantic Drift Report (Lee et al. Top-K paradigm → sentence) ───

@app.route("/api/drift-report", methods=["POST"])
def drift_report():
    """
    Generates a full semantic-change discovery report for a sentence.
    Extends Lee et al. (EMNLP 2025) from word-level to sentence-level:
      - Counterfactual contribution scores (which words DRIVE the drift)
      - Domain transition detection (where meaning moved TO and FROM)
      - Natural-language reasoning per word
      - Era-level sentence interpretation (fingerprinting)

    Body: { text, year_a, year_b, top_k? }
    """
    data   = request.json or {}
    text   = data.get("text", "").strip()
    year_a = int(data.get("year_a", 1900))
    year_b = int(data.get("year_b", 1990))
    top_k  = int(data.get("top_k", 5))

    if not text:
        return jsonify({"error": "text required"}), 400
    if len(text) > 2000:
        return jsonify({"error": "text too long (max 2000 chars)"}), 400
    if year_a == year_b:
        return jsonify({"error": "year_a and year_b must be different"}), 400

    vecs_a, vocab_a, idx_a = _get(year_a)
    vecs_b, vocab_b, idx_b = _get(year_b)
    if vecs_a is None or vecs_b is None:
        return jsonify({"error": f"One or both decades ({year_a}, {year_b}) not available"}), 400

    report = generate_drift_report(
        text,
        vecs_a, vocab_a, idx_a,
        vecs_b, vocab_b, idx_b,
        year_a, year_b, top_k=top_k
    )
    if "error" in report:
        return jsonify(report), 400
    return jsonify(report)


# ─── Case studies ─────────────────────────────────────────────────────────────
@app.route("/api/case-studies")
def case_studies():
    return jsonify({"case_studies": CASE_STUDIES})


@app.route("/api/case-study/<word>")
def case_study(word):
    cs = get_case_study(word)
    if cs is None:
        return jsonify({"error": "Not found"}), 404
    return jsonify(cs)


# ─── Run ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)

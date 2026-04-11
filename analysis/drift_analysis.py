"""
analysis/drift_analysis.py
--------------------------
Core semantic drift analysis pipeline.

Covers:
  1. Pairwise drift between any two decades (cosine distance)
  2. Timeline drift across all available decades
  3. Top-K most/least drifted words
  4. Neighbor shift analysis
  5. Semantic category cluster tracking
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


def compute_drift_score(word: str,
                        vecs_a: np.ndarray, idx_a: dict,
                        vecs_b: np.ndarray, idx_b: dict) -> float | None:
    """
    Cosine distance between word vectors in two embedding spaces.
    Expects embeddings to be pre-aligned (Procrustes).
    """
    if word not in idx_a or word not in idx_b:
        return None
    va = vecs_a[idx_a[word]].reshape(1, -1)
    vb = vecs_b[idx_b[word]].reshape(1, -1)
    sim = float(cosine_similarity(va, vb)[0, 0])
    return 1.0 - sim


def compute_timeline(word: str,
                     decade_data: list[tuple[int, np.ndarray, dict]]) -> dict:
    """
    Compute drift score relative to the *first* decade across all decades.

    Parameters
    ----------
    word : str
    decade_data : list of (year, vectors, idx_map), sorted ascending by year

    Returns
    -------
    dict with keys 'years', 'scores', 'present_in'
    """
    years, scores, present = [], [], []

    ref_year, ref_vecs, ref_idx = decade_data[0]

    for year, vecs, idx in decade_data:
        years.append(year)
        present.append(word in idx)
        if word not in ref_idx or word not in idx:
            scores.append(None)
            continue
        va = ref_vecs[ref_idx[word]].reshape(1, -1)
        vb = vecs[idx[word]].reshape(1, -1)
        sim = float(cosine_similarity(va, vb)[0, 0])
        scores.append(round(1.0 - sim, 4))

    return {"years": years, "scores": scores, "present_in": present}


def top_drifted_words(vecs_a: np.ndarray, idx_a: dict,
                      vecs_b: np.ndarray, idx_b: dict,
                      top_n: int = 30, min_freq_rank: int = 5000) -> list[dict]:
    """
    Rank the vocabulary by how much each word's meaning changed
    between period A and period B.

    Considers only words in both vocabularies with rank ≤ min_freq_rank
    (lower index = higher frequency in the dataset).
    """
    shared = [w for w in idx_a if w in idx_b
              and idx_a[w] < min_freq_rank and idx_b[w] < min_freq_rank
              and len(w) > 3 and w.lower().isalpha()]

    results = []
    for word in tqdm(shared, desc="Scoring vocabulary", leave=False):
        score = compute_drift_score(word, vecs_a, idx_a, vecs_b, idx_b)
        if score is not None:
            results.append({"word": word, "drift": round(score, 4)})

    results.sort(key=lambda x: x["drift"], reverse=True)
    return results[:top_n]


def neighbor_shift(word: str,
                   vecs_a: np.ndarray, vocab_a: list, idx_a: dict,
                   vecs_b: np.ndarray, vocab_b: list, idx_b: dict,
                   top_n: int = 10) -> dict:
    """
    Compare the nearest-neighbor sets of a word in two time periods.
    Returns shared neighbors, gained neighbors, and lost neighbors.
    """
    def get_nbrs(vecs, vocab, idx, top=top_n + 5):
        if word not in idx:
            return []
        tgt = vecs[idx[word]].reshape(1, -1)
        sims = cosine_similarity(vecs, tgt).flatten()
        ranked = np.argsort(sims)[::-1]
        nbrs = []
        for i in ranked:
            w = vocab[i].lower()
            if w != word.lower() and len(w) > 3 and w.isalpha():
                nbrs.append((w, round(float(sims[i]), 4)))
            if len(nbrs) >= top:
                break
        return nbrs

    nbrs_a = get_nbrs(vecs_a, vocab_a, idx_a)
    nbrs_b = get_nbrs(vecs_b, vocab_b, idx_b)
    set_a  = {w for w, _ in nbrs_a}
    set_b  = {w for w, _ in nbrs_b}

    return {
        "neighbors_old": nbrs_a[:top_n],
        "neighbors_new": nbrs_b[:top_n],
        "shared":  list(set_a & set_b),
        "gained":  list(set_b - set_a),   # new associates
        "lost":    list(set_a - set_b),   # old associates dropped
        "jaccard_similarity": round(
            len(set_a & set_b) / max(len(set_a | set_b), 1), 4
        ),
    }


def semantic_change_type(word: str,
                         vecs_a: np.ndarray, vocab_a: list, idx_a: dict,
                         vecs_b: np.ndarray, vocab_b: list, idx_b: dict) -> str:
    """
    Heuristic classifier for the *type* of semantic change:
      - broadening   : word gains many new contexts
      - narrowing    : word loses many old contexts
      - amelioration : word gains positive connotations (simple proxy)
      - pejoration   : word gains negative connotations  (simple proxy)
      - stable       : minimal change

    Uses neighbor-overlap statistics as proxy (proper sentiment not embedded here).
    """
    ns = neighbor_shift(word, vecs_a, vocab_a, idx_a, vecs_b, vocab_b, idx_b)
    gained = len(ns["gained"])
    lost   = len(ns["lost"])
    jacc   = ns["jaccard_similarity"]

    if jacc > 0.7:
        return "stable"
    if gained > lost * 1.5:
        return "broadening"
    if lost > gained * 1.5:
        return "narrowing"
    return "shifting"      # general shift


def compare_word_pair(word_a: str, word_b: str,
                      vecs: np.ndarray, idx: dict) -> float | None:
    """
    Cosine similarity between two words *within the same decade*.
    Useful for tracking how two concepts converge/diverge over time.
    """
    if word_a not in idx or word_b not in idx:
        return None
    va = vecs[idx[word_a]].reshape(1, -1)
    vb = vecs[idx[word_b]].reshape(1, -1)
    return round(float(cosine_similarity(va, vb)[0, 0]), 4)

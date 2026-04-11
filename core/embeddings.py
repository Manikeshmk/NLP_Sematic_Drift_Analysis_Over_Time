"""
embeddings.py — Core module for loading and processing Hamilton et al. SGNS embeddings.
Supports the Stanford/Google Ngram historical word vector dataset.
"""

import os
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import Optional


SGNS_BASE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sgns")
AVAILABLE_DECADES = list(range(1800, 2001, 10))


def load_decade(year: int, base_path: str = SGNS_BASE):
    """Load vocabulary and embedding matrix for a given decade year."""
    vocab_path = os.path.join(base_path, f"{year}-vocab.pkl")
    vec_path   = os.path.join(base_path, f"{year}-w.npy")

    if not os.path.exists(vocab_path) or not os.path.exists(vec_path):
        raise FileNotFoundError(
            f"Embedding files for decade {year} not found in '{base_path}'.\n"
            f"Expected:\n  {vocab_path}\n  {vec_path}"
        )

    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)

    vectors = np.load(vec_path, mmap_mode='r')
    return vectors, vocab


def build_index(vocab) -> dict:
    """Build word→index lookup map from a vocab list."""
    return {word: i for i, word in enumerate(vocab)}


def get_word_vector(word: str, vectors: np.ndarray, idx_map: dict) -> Optional[np.ndarray]:
    """Return the embedding vector for a word, or None if not found."""
    if word not in idx_map:
        return None
    return vectors[idx_map[word]]


def cosine_drift_score(word: str,
                       vecs_a: np.ndarray, idx_a: dict,
                       vecs_b: np.ndarray, idx_b: dict) -> Optional[float]:
    """
    Compute the cosine *distance* (1 − similarity) between a word's vector
    in two different time periods — after aligning via Orthogonal Procrustes.
    Returns None if the word is absent in either period.
    """
    va = get_word_vector(word, vecs_a, idx_a)
    vb = get_word_vector(word, vecs_b, idx_b)
    if va is None or vb is None:
        return None
    sim = cosine_similarity(va.reshape(1, -1), vb.reshape(1, -1))[0, 0]
    return float(1.0 - sim)


def get_neighbors(word: str, vectors: np.ndarray, vocab: list,
                  idx_map: dict, top_n: int = 10) -> list[tuple[str, float]]:
    """Return (neighbor_word, cosine_similarity) pairs for a word."""
    if word not in idx_map:
        return []
    target = vectors[idx_map[word]].reshape(1, -1)
    sims   = cosine_similarity(vectors, target).flatten()
    ranked = np.argsort(sims)[::-1]

    results = []
    for i in ranked:
        w = vocab[i].lower()
        if w != word.lower() and len(w) > 3 and w.isalpha():
            results.append((w, float(sims[i])))
        if len(results) >= top_n:
            break
    return results


def align_procrustes(source_vecs: np.ndarray, target_vecs: np.ndarray,
                     source_idx: dict, target_idx: dict) -> np.ndarray:
    """
    Orthogonal Procrustes alignment: rotate source embeddings into target space.
    Uses the shared vocabulary as anchor points.
    Returns the aligned source matrix.
    """
    shared = [w for w in source_idx if w in target_idx]
    if len(shared) < 10:
        raise ValueError("Too few shared words for Procrustes alignment.")

    S = np.array([source_vecs[source_idx[w]] for w in shared])
    T = np.array([target_vecs[target_idx[w]] for w in shared])

    # SVD-based orthogonal Procrustes
    M   = T.T @ S
    U, _, Vt = np.linalg.svd(M)
    R   = U @ Vt

    aligned = source_vecs @ R.T
    return aligned


def drift_score_aligned(word: str,
                        vecs_a_aligned: np.ndarray, idx_a: dict,
                        vecs_b: np.ndarray, idx_b: dict) -> Optional[float]:
    """Drift score after Procrustes alignment."""
    va = get_word_vector(word, vecs_a_aligned, idx_a)
    vb = get_word_vector(word, vecs_b, idx_b)
    if va is None or vb is None:
        return None
    sim = cosine_similarity(va.reshape(1, -1), vb.reshape(1, -1))[0, 0]
    return float(1.0 - sim)

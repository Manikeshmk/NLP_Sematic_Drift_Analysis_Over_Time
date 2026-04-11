"""
analysis/drift_explainer.py
══════════════════════════════════════════════════════════════════════════════
Semantic Change Discovery: Top-K Paradigm for Sentences
─────────────────────────────────────────────────────────
Extends Lee et al. (EMNLP 2025) from word-level discovery to sentence-level.

Key innovations over Lee et al.:
  1. Counterfactual Contribution Score — ranks words not just by how much
     THEY changed, but by how much their change DRIVES the sentence's shift.
     (A word can drift a lot but be unimportant to the sentence.)
  2. Domain Transition Detection — detects which semantic domain a word
     belonged to in era A vs era B using nearest-neighbor analysis.
  3. Sentence-level Interpretation — generates a natural-language
     "fingerprint" of what the sentence would have conveyed in each era.
  4. Anchor vs Driver classification — stable words that hold the sentence
     together vs volatile words that push it into new meaning territory.

Algorithm
─────────
For each word w in the sentence:
  1. Compute drift_score(w) = 1 - cosine(v_w^A, v_w^B)
  2. Compute contribution(w) = drift_full_sentence - drift_without(w)
     (counterfactual: how much does sentence drift DROP if w stayed stable?)
  3. Rank by contribution (Top-K Discovery)
  4. For top-K drifted words: detect domain shift via neighbors
  5. Generate natural-language reasoning per word
  6. Build sentence-level interpretation for each era
"""

import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import Optional

# ─── Stop-words ──────────────────────────────────────────────────────────────
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

# ─── Semantic Domain Lexicon ──────────────────────────────────────────────────
# Each domain is defined by seed words. A neighbor set is assigned a domain
# via keyword-overlap scoring (no embeddings needed, 100% interpretable).
DOMAIN_LEXICON: dict[str, list[str]] = {
    "Technology":       ["computer","digital","electronic","machine","software","hardware",
                         "device","circuit","data","program","algorithm","internet","network",
                         "electric","mechanical","engine","code","processor","robot","cyber"],
    "Agriculture":      ["farm","crop","seed","harvest","soil","field","grain","plant",
                         "sow","plow","orchard","livestock","cattle","wheat","fertilizer",
                         "cultivation","rural","pasture","barn","herding"],
    "Biology/Medicine": ["disease","patient","medical","organism","cell","gene","protein",
                         "virus","bacteria","symptom","diagnosis","treatment","hospital",
                         "physician","anatomy","evolution","species","organ","tissue"],
    "Communication":    ["newspaper","radio","telegraph","print","speech","letter","journal",
                         "magazine","publish","announce","transmit","signal","message",
                         "television","broadcast","media","publish","press","wire"],
    "Politics/Law":     ["government","law","political","election","parliament","constitution",
                         "democracy","republic","legislation","vote","policy","authority",
                         "justice","court","regulation","treaty","senate","congress","rights"],
    "Military":         ["war","soldier","battle","army","weapon","defense","attack",
                         "command","strategy","navy","artillery","campaign","victory",
                         "combat","regiment","fort","siege","ammunition","martial"],
    "Finance/Trade":    ["money","bank","trade","commerce","market","price","economic",
                         "capital","invest","profit","currency","merchant","industry",
                         "manufacture","stock","bond","revenue","fiscal","tariff"],
    "Social/Culture":   ["society","community","people","tradition","cultural","custom",
                         "religion","moral","family","education","class","popular","art",
                         "literature","philosophy","behavior","institution","ideology"],
    "Nature/Science":   ["nature","natural","physical","chemical","experiment","theory",
                         "scientific","laboratory","observation","research","element",
                         "mineral","atmosphere","geological","astronomical","optical"],
    "Maritime":         ["ship","sea","ocean","navigation","port","vessel","sailor","fleet",
                         "coast","harbor","cargo","maritime","wave","tide","sail","captain"],
}

def _detect_domain(neighbors: list[str]) -> tuple[str, float]:
    """
    Given a list of neighbor words, return the best-matching semantic domain
    and a confidence score (0–1).
    """
    if not neighbors:
        return "Unknown", 0.0
    nbr_set = set(n.lower() for n in neighbors)
    scores = {}
    for domain, seeds in DOMAIN_LEXICON.items():
        overlap = len(nbr_set & set(seeds))
        scores[domain] = overlap / max(len(seeds), 1)
    best = max(scores, key=scores.get)
    return best, round(scores[best], 3)


def _get_top_neighbors(word: str, vecs: np.ndarray, vocab: list[str],
                       idx: dict, top_n: int = 12) -> list[str]:
    """Return top-N nearest neighbors for a word (excluding the word itself)."""
    if word not in idx:
        return []
    tgt  = vecs[idx[word]].reshape(1, -1)
    sims = cosine_similarity(vecs, tgt).flatten()
    ranked = np.argsort(sims)[::-1]
    out = []
    for i in ranked:
        w = vocab[i].lower()
        if w != word and len(w) > 2 and w.isalpha() and w not in STOPWORDS:
            out.append(w)
        if len(out) >= top_n:
            break
    return out


# ─── Natural Language Reasoning Engine ───────────────────────────────────────

_CHANGE_TEMPLATES = {
    # (domain_a, domain_b): template string
    ("Technology",    "Agriculture"):
        "underwent a domain reversal — originally used in technological contexts, it acquired agricultural connotations over time.",
    ("Agriculture",   "Technology"):
        "underwent metaphorical extension from agricultural/physical contexts into technological domains — a classic broadening pattern.",
    ("Agriculture",   "Communication"):
        "transitioned from describing agricultural/physical spreading to the domain of media and communication (e.g., 'broadcast seeds' → 'broadcast signal').",
    ("Biology/Medicine", "Technology"):
        "migrated from biological/medical discourse into technological usage, gaining computational connotations.",
    ("Technology",    "Biology/Medicine"):
        "was reclaimed by biological contexts after an initial technological association.",
    ("Communication", "Technology"):
        "shifted from traditional communication/media contexts to digital/electronic technology domains.",
    ("Politics/Law",  "Social/Culture"):
        "broadened from a strictly political/legal register into mainstream social and cultural discourse.",
    ("Social/Culture","Politics/Law"):
        "narrowed from a broad social/cultural meaning into a more specific political or legal register.",
    ("Military",      "Technology"):
        "transitioned from military terminology into technological and engineering contexts.",
    ("Military",      "Politics/Law"):
        "shifted from battlefield usage into political and legislative contexts.",
    ("Maritime",      "Technology"):
        "moved from maritime/naval discourse into modern technological usage.",
    ("Nature/Science","Technology"):
        "evolved from natural/scientific description into technological and computational contexts.",
    ("Finance/Trade", "Politics/Law"):
        "shifted from commercial/trade contexts into a broader political economy discourse.",
}

_GENERIC_SHIFT = (
    "shifted its primary semantic associations from the {domain_a} domain "
    "({words_a}) toward the {domain_b} domain ({words_b}), "
    "suggesting a {change_class} pattern."
)

def _fmt_words(words: list[str], n: int = 4) -> str:
    sample = words[:n]
    return ", ".join(f"'{w}'" for w in sample)

def _change_class(drift: float, domain_a: str, domain_b: str) -> str:
    if domain_a == domain_b:
        return "intra-domain register shift"
    if drift > 0.5:
        return "major metaphorical extension or pejoration/amelioration"
    if drift > 0.3:
        return "domain migration (broadening or narrowing)"
    return "gradual connotation shift"

def _generate_word_reasoning(word: str, drift: float,
                              nbrs_a: list[str], nbrs_b: list[str],
                              year_a: int, year_b: int) -> dict:
    """
    Generate a full reasoning block for one word.
    Returns a dict with all fields needed for the report.
    """
    domain_a, conf_a = _detect_domain(nbrs_a)
    domain_b, conf_b = _detect_domain(nbrs_b)
    shared_nbrs  = list(set(nbrs_a) & set(nbrs_b))
    gained_nbrs  = [n for n in nbrs_b if n not in nbrs_a]
    lost_nbrs    = [n for n in nbrs_a if n not in nbrs_b]

    # Primary reasoning sentence
    key = (domain_a, domain_b)
    if key in _CHANGE_TEMPLATES and domain_a != domain_b:
        main_reason = f"'{word}' " + _CHANGE_TEMPLATES[key]
    else:
        main_reason = (
            f"'{word}' " + _GENERIC_SHIFT.format(
                domain_a=domain_a,
                domain_b=domain_b,
                words_a=_fmt_words(nbrs_a),
                words_b=_fmt_words(nbrs_b),
                change_class=_change_class(drift, domain_a, domain_b),
            )
        )

    # Evidence sentence
    if lost_nbrs and gained_nbrs:
        evidence = (
            f"In {year_a}, '{word}' appeared alongside {_fmt_words(lost_nbrs[:3])}. "
            f"By {year_b}, these were replaced by {_fmt_words(gained_nbrs[:3])}."
        )
    elif gained_nbrs:
        evidence = f"By {year_b}, '{word}' gained new associations: {_fmt_words(gained_nbrs[:4])}."
    elif lost_nbrs:
        evidence = f"'{word}' lost its earlier associations with {_fmt_words(lost_nbrs[:4])} by {year_b}."
    else:
        evidence = f"'{word}' retained most of its contextual neighbours but with shifted weightings."

    # Linguistic label
    if domain_a == domain_b:
        label = "Register Shift"
    elif drift > 0.55:
        label = "Metaphorical Extension" if len(gained_nbrs) > len(lost_nbrs) else "Semantic Narrowing"
    elif len(gained_nbrs) > len(lost_nbrs) * 1.3:
        label = "Broadening"
    elif len(lost_nbrs) > len(gained_nbrs) * 1.3:
        label = "Narrowing"
    else:
        label = "Domain Migration"

    return {
        "word":        word,
        "drift":       round(drift, 4),
        "label":       label,
        "domain_a":    domain_a,
        "domain_b":    domain_b,
        "domain_conf_a": conf_a,
        "domain_conf_b": conf_b,
        "neighbors_a": nbrs_a[:8],
        "neighbors_b": nbrs_b[:8],
        "shared":      shared_nbrs[:6],
        "gained":      gained_nbrs[:6],
        "lost":        lost_nbrs[:6],
        "reasoning":   main_reason,
        "evidence":    evidence,
    }


# ─── Counterfactual Contribution Score ───────────────────────────────────────

def _sentence_drift(words: list[str], vecs_a, idx_a, vecs_b, idx_b) -> Optional[float]:
    """Drift of the sentence embedding built from 'words'."""
    parts_a = [vecs_a[idx_a[w]] for w in words if w in idx_a and w in idx_b]
    parts_b = [vecs_b[idx_b[w]] for w in words if w in idx_a and w in idx_b]
    if not parts_a or not parts_b:
        return None
    ea = np.mean(parts_a, axis=0).reshape(1, -1)
    eb = np.mean(parts_b, axis=0).reshape(1, -1)
    return float(1.0 - cosine_similarity(ea, eb)[0, 0])


def _compute_contribution(word: str, all_words: list[str],
                            vecs_a, idx_a, vecs_b, idx_b,
                            full_drift: float) -> float:
    """
    Counterfactual contribution of 'word' to the sentence drift.
    = drift_with_word - drift_without_word
    Positive → word INCREASES the sentence drift (it's a driver)
    Negative → word actually REDUCES drift (it's stabilising)
    """
    without = [w for w in all_words if w != word]
    if not without:
        return full_drift  # can't compute without, return full drift
    d_without = _sentence_drift(without, vecs_a, idx_a, vecs_b, idx_b)
    if d_without is None:
        return 0.0
    return round(full_drift - d_without, 4)


# ─── Era Interpretation (Sentence Fingerprinting) ────────────────────────────

def _interpret_era(words: list[str], word_details: list[dict],
                   year: int, era_key: str) -> str:
    """
    Build a natural-language 'fingerprint' of what the sentence conveyed
    in the given era, based on dominant domains of its component words.
    era_key: 'a' or 'b'
    """
    domain_key = f"domain_{era_key}"
    nbrs_key   = f"neighbors_{era_key}"
    domain_votes: dict[str, int] = {}
    representative_nbrs: list[str] = []

    for wd in word_details:
        d = wd.get(domain_key, "Unknown")
        domain_votes[d] = domain_votes.get(d, 0) + 1
        representative_nbrs.extend(wd.get(nbrs_key, [])[:2])

    if not domain_votes:
        return f"In {year}, insufficient data to interpret this sentence."

    # Top 2 domains
    sorted_domains = sorted(domain_votes.items(), key=lambda x: x[1], reverse=True)
    top_domain  = sorted_domains[0][0]
    sec_domain  = sorted_domains[1][0] if len(sorted_domains) > 1 else None

    sample_words = ", ".join(f"'{w}'" for w in representative_nbrs[:5])

    base = f"In {year}, the sentence operated primarily within the {top_domain} domain"
    if sec_domain and sec_domain != top_domain and sorted_domains[1][1] > 0:
        base += f" with elements of {sec_domain}"
    base += f". Key contextual associations included {sample_words}."
    return base


# ══════════════════════════════════════════════════════════════════════════════
# Main Entry Point
# ══════════════════════════════════════════════════════════════════════════════

def generate_drift_report(text: str,
                           vecs_a: np.ndarray, vocab_a: list, idx_a: dict,
                           vecs_b: np.ndarray, vocab_b: list, idx_b: dict,
                           year_a: int, year_b: int,
                           top_k: int = 5) -> dict:
    """
    Generate a full semantic change discovery report for a sentence.

    Parameters
    ----------
    text  : Input sentence / phrase / idiom
    vecs_a/vocab_a/idx_a : Embeddings at year_a
    vecs_b/vocab_b/idx_b : Embeddings at year_b
    year_a, year_b : Eras to compare
    top_k : How many "discovery" words to highlight in depth

    Returns
    -------
    Full report dict ready for JSON serialisation
    """
    # ── Tokenise ──────────────────────────────────────────────────────────────
    tokens = re.findall(r"[a-zA-Z]+", text.lower())
    content_words = []
    seen = set()
    for t in tokens:
        if len(t) > 2 and t not in STOPWORDS and t not in seen:
            seen.add(t)
            content_words.append(t)

    # ── Keep only words present in BOTH eras ─────────────────────────────────
    found_words = [w for w in content_words if w in idx_a and w in idx_b]
    not_found   = [w for w in content_words if w not in found_words]

    if not found_words:
        return {"error": "No content words found in both embedding spaces."}

    # ── Sentence-level drift ──────────────────────────────────────────────────
    full_drift = _sentence_drift(found_words, vecs_a, idx_a, vecs_b, idx_b)
    if full_drift is None:
        return {"error": "Could not compute sentence embedding."}

    # ── Per-word drift + contribution scores ──────────────────────────────────
    word_data = []
    for word in found_words:
        va = vecs_a[idx_a[word]].reshape(1, -1)
        vb = vecs_b[idx_b[word]].reshape(1, -1)
        drift      = float(1.0 - cosine_similarity(va, vb)[0, 0])
        contrib    = _compute_contribution(word, found_words,
                                           vecs_a, idx_a, vecs_b, idx_b,
                                           full_drift)
        nbrs_a = _get_top_neighbors(word, vecs_a, vocab_a, idx_a, top_n=12)
        nbrs_b = _get_top_neighbors(word, vecs_b, vocab_b, idx_b, top_n=12)
        reasoning = _generate_word_reasoning(word, drift, nbrs_a, nbrs_b, year_a, year_b)
        reasoning["contribution"] = contrib

        # Contribution-to-drift ratio (how much of sentence drift does word explain?)
        reasoning["contribution_pct"] = round(
            abs(contrib) / max(abs(full_drift), 1e-6) * 100, 1)

        word_data.append(reasoning)

    # ── Top-K Discovery Ranking (by |contribution|, then by drift) ────────────
    discovery_rank = sorted(word_data,
                            key=lambda x: (abs(x["contribution"]), x["drift"]),
                            reverse=True)

    # ── Era Interpretations ───────────────────────────────────────────────────
    interpretation_a = _interpret_era(found_words, word_data, year_a, "a")
    interpretation_b = _interpret_era(found_words, word_data, year_b, "b")

    # ── Dominant change narrative ─────────────────────────────────────────────
    top_driver = discovery_rank[0]["word"] if discovery_rank else "—"
    top_label  = discovery_rank[0]["label"] if discovery_rank else "Unknown"
    top_drift_val = discovery_rank[0]["drift"] if discovery_rank else 0.0

    # Overall sentence change classification
    if full_drift > 0.5:
        sentence_change_class = "Major Semantic Shift"
    elif full_drift > 0.3:
        sentence_change_class = "Moderate Semantic Drift"
    elif full_drift > 0.15:
        sentence_change_class = "Mild Semantic Drift"
    else:
        sentence_change_class = "Semantically Stable"

    # Stable anchors (lowest contribution + lowest individual drift)
    anchors = sorted(word_data, key=lambda x: (abs(x["contribution"]),
                                                 x["drift"]))[:4]

    # Key domain transitions (unique A→B domain changes)
    transitions = {}
    for wd in discovery_rank[:top_k]:
        key = f"{wd['domain_a']} → {wd['domain_b']}"
        if wd["domain_a"] != wd["domain_b"]:
            transitions[key] = transitions.get(key, 0) + 1

    return {
        # ── Input metadata
        "text":          text,
        "year_a":        year_a,
        "year_b":        year_b,
        "content_words": content_words,
        "found_words":   found_words,
        "not_found":     not_found,

        # ── Sentence-level scores
        "sentence_drift":       round(full_drift, 4),
        "sentence_change_class": sentence_change_class,

        # ── Era interpretations (the "fingerprints")
        "interpretation_a": interpretation_a,
        "interpretation_b": interpretation_b,

        # ── Top-K Discovery Ranking (the core innovation)
        "discovery_rank": discovery_rank,
        "top_k":          top_k,
        "top_driver":     top_driver,
        "top_label":      top_label,

        # ── Stable anchors
        "anchors": anchors,

        # ── Summary stats
        "n_high_drift":   sum(1 for w in word_data if w["drift"] > 0.35),
        "n_stable":       sum(1 for w in word_data if w["drift"] <= 0.15),
        "domain_transitions": transitions,

        # ── Per-word full details (all words)
        "word_details": word_data,
    }

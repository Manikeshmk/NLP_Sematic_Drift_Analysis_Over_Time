![Repo visits](https://hits.sh/github.com/Manikeshmk/NLP_Sematic_Drift_Analysis_Over_Time.svg?label=repo%20visits)
![GitHub stars](https://img.shields.io/github/stars/Manikeshmk/NLP_Sematic_Drift_Analysis_Over_Time?style=logo&logo=github&label=⭐%20Stars) 
![GitHub forks](https://img.shields.io/github/forks/Manikeshmk/NLP_Sematic_Drift_Analysis_Over_Time?style=social)

# ⚗ Semantic Drift Analysis Over Time
### B.Tech Major Project — Natural Language Processing

> *"Words are fossils in which the life of the past is embalmed."* — Max Müller

---

## 📋 Project Overview

This project analyses **semantic drift** — the phenomenon where word meanings shift gradually over decades and centuries — using Stanford's pre-trained historical word embeddings derived from the Google Books Ngram corpus.

| Field | Detail |
|-------|--------|
| Course | Natural Language Processing (B.Tech) |
| Dataset | Stanford/Google Ngram SGNS Embeddings |
| Time Span | 1800s – 2000s (per-decade granularity) |
| Vocabulary | ~65,000 words per decade |
| Vector Dims | 300-dimensional Skip-gram with Negative Sampling (SGNS) |

---

## 🚀 Quick Start

### 1. Setup (one-time)
```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download the Dataset
```bash
python main.py download
```
This downloads ~1.4 GB from Stanford. Unzips to `sgns/` folder automatically.

Manual download: http://snap.stanford.edu/historical_embeddings/eng-all_sgns.zip

### 3. Launch the Dashboard
```bash
python main.py serve
# OR just double-click start.bat
```
Then open `frontend/index.html` in your browser.

---

## 🖥️ Dashboard Tabs

| Tab | Description |
|-----|-------------|
| **Word Explorer** | Analyse a word — drift score, change type, neighbour shift, t-SNE map |
| **Timeline** | Drift across all available decades for a single word |
| **Heat Map** | Multi-word comparison heatmap across all decades |
| **Top Drifted** | Global ranking of most-changed words + word cloud |
| **Concept Relations** | Track how two concepts converge/diverge over time |
| **Case Studies** | 8 famous semantic change examples with one-click analysis |
| **About** | Methodology, tech stack, change type taxonomy |

---

## ⌨️ CLI Commands

```bash
# Analyse a word pair
python main.py word network --from 1900 --to 1990

# Full timeline for a word
python main.py timeline virus --out results/virus.csv

# Top-30 most drifted words
python main.py top --from 1900 --to 1990 --n 30

# Save t-SNE plot to file
python main.py word india --from 1800 --to 1990 --plot
```

---

## 🔬 Methodology

### 1. Data — SGNS Embeddings
Hamilton et al. (2016) trained Skip-gram with Negative Sampling (SGNS) on decade-sliced Google Books Ngram data. Each decade has a 300-dimensional vector for every word in the vocabulary.

### 2. Cosine Drift Score
```
drift(w, t₁, t₂) = 1 − cosine_similarity(v_w^t₁, v_w^t₂)
```
Range `[0, 1]`: 0 = stable, 1 = completely shifted

### 3. Orthogonal Procrustes Alignment
Since embedding spaces trained independently are not inherently aligned, we use SVD-based Procrustes rotation to map the source space into the target space before computing drift.

### 4. Neighbour Shift (Jaccard)
We compare the top-K nearest-neighbour sets of a word across two eras and compute Jaccard overlap. This captures *contextual* change independently of vector direction.

### 5. Semantic Change Classification
| Type | Criterion |
|------|-----------|
| Stable | Jaccard > 0.7 |
| Broadening | gained >> lost (1.5× ratio) |
| Narrowing | lost >> gained (1.5× ratio) |
| Shifting | general context replacement |

### 6. t-SNE Visualisation
2-D projection of the combined neighbourhood (1900 + 1990 contexts) showing which words were shared, gained, or lost.

---

## 📚 Case Studies

| Word | Era | Type |
|------|-----|------|
| **broadcast** | 1870→1930 | Broadening (seeds→media) |
| **computer** | 1900→1980 | Domain shift (human→machine) |
| **virus** | 1900→1990 | Metaphorical extension (bio→digital) |
| **network** | 1900→1990 | Broadening (wires→social/digital) |
| **awful** | 1800→1990 | Pejoration (awe-inspiring→very bad) |
| **nice** | 1800→1990 | Amelioration (foolish→pleasant) |
| **artificial** | 1900→1990 | Domain extension (+AI cluster) |

---

## 📁 Project Structure

```
D:\NLP\
├── main.py                  # CLI entry-point
├── app.py                   # Flask REST API
├── requirements.txt
├── start.bat                # One-click Windows launcher
│
├── core/
│   └── embeddings.py        # Data loading, Procrustes alignment
│
├── analysis/
│   ├── drift_analysis.py    # Core algorithms
│   ├── visualizations.py    # All charts (returns base64 PNGs)
│   └── case_studies.py      # Curated case study metadata
│
├── frontend/
│   ├── index.html           # Dashboard UI
│   ├── style.css            # Dark glassmorphism design system
│   └── app.js               # Client-side logic
│
├── tests/
│   └── test_drift.py        # 12 unit tests (no SGNS data needed)
│
├── sgns/                    # ← Downloaded here (not in git)
│   ├── 1900-vocab.pkl
│   ├── 1900-w.npy
│   └── ...
│
└── results/                 # Auto-created: saved plots / CSVs
```

---

## 🛠️ Tech Stack

- **Core**: Python 3.14, NumPy, scikit-learn (TSNE, cosine_similarity)
- **Algorithms**: SGNS, Orthogonal Procrustes, Cosine Distance, Jaccard 
- **API**: Flask + Flask-CORS
- **Visualisation**: Matplotlib, Seaborn, WordCloud
- **Frontend**: Vanilla HTML/CSS/JS (dark glassmorphism)
- **Tests**: pytest (12 tests, synthetic fixtures)

---

## 📖 References

1. Hamilton, W. L., Leskovec, J., & Jurafsky, D. (2016). *Diachronic Word Embeddings Reveal Statistical Laws of Semantic Change.* ACL 2016.
2. Kulkarni, V., et al. (2015). *Statistically Significant Detection of Linguistic Change.* WWW 2015.
3. Michel, J.B., et al. (2011). *Quantitative Analysis of Culture Using Millions of Digitized Books.* Science.
4. Mikolov, T., et al. (2013). *Distributed Representations of Words and Phrases.* NIPS 2013.
5. Schönemann, P. H. (1966). *A Generalized Solution of the Orthogonal Procrustes Problem.* Psychometrika.

---

*© B.Tech NLP Major Project — Semantic Drift Analysis Over Time*

"""
main.py — Semantic Drift Analysis: Command-Line Interface
==========================================================
This is the unified entry-point for the B.Tech NLP Major Project:
"Semantic Drift Analysis Over Time"

Usage
-----
  # Start the web dashboard (recommended)
  python main.py serve

  # Analyse a single word  
  python main.py word network --from 1900 --to 1990

  # Get the top-30 most drifted words
  python main.py top --from 1900 --to 1990 --n 30

  # Export a word's timeline to CSV
  python main.py timeline virus --out results/virus_timeline.csv

  # Download the SGNS dataset (Stanford)
  python main.py download

Dataset
-------
  Hamilton et al. (2016) "Diachronic Word Embeddings Reveal Statistical Laws of Semantic Change"
  URL: http://snap.stanford.edu/historical_embeddings/eng-all_sgns.zip
  Place the unzipped 'sgns/' folder inside D:/NLP/
"""

import os
import sys
import argparse
import csv
import subprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

SGNS_DIR = os.path.join(BASE_DIR, "sgns")


def _check_data():
    if not os.path.isdir(SGNS_DIR):
        print(f"[ERROR] SGNS data directory not found: {SGNS_DIR}")
        print("Run:  python main.py download")
        sys.exit(1)


def _load_embeddings():
    from core.embeddings import load_decade, build_index, AVAILABLE_DECADES
    loaded = {}
    for yr in AVAILABLE_DECADES:
        try:
            vecs, vocab = load_decade(yr)
            loaded[yr] = (vecs, vocab, build_index(vocab))
            print(f"  ✓ Loaded {yr}")
        except FileNotFoundError:
            pass
    if not loaded:
        print("[ERROR] No decades found. Check your sgns/ folder.")
        sys.exit(1)
    return loaded


# ─── serve ────────────────────────────────────────────────────────────────────

def cmd_serve(args):
    """Start the Flask web dashboard."""
    _check_data()
    import webbrowser, threading, time

    def open_browser():
        time.sleep(2)
        webbrowser.open("http://localhost:5000")
        # Open frontend HTML directly (no build step needed)
        frontend = os.path.join(BASE_DIR, "frontend", "index.html")
        webbrowser.open(f"file:///{frontend.replace(os.sep, '/')}")

    threading.Thread(target=open_browser, daemon=True).start()
    print("\n🚀  Starting Semantic Drift API on http://localhost:5000")
    print("📊  Dashboard:  Open frontend/index.html in your browser")
    print("     (or both will open automatically in 2 s)\n")

    # Import and run Flask
    from app import app, preload
    preload()
    app.run(host="0.0.0.0", port=5000, debug=False)


# ─── word ─────────────────────────────────────────────────────────────────────

def cmd_word(args):
    """Analyse a single word between two decades."""
    _check_data()
    word   = args.word.lower()
    year_a = args.from_yr
    year_b = args.to_yr

    from analysis.drift_analysis import (
        compute_drift_score, neighbor_shift, semantic_change_type
    )
    from analysis.case_studies import get_case_study

    print(f"\n🔍  Analysing: '{word}'  ({year_a} → {year_b})\n{'─'*50}")

    print("  Loading embeddings…")
    loaded = _load_embeddings()

    if year_a not in loaded:
        print(f"[ERROR] Decade {year_a} not available. Got: {list(loaded.keys())}")
        sys.exit(1)
    if year_b not in loaded:
        print(f"[ERROR] Decade {year_b} not available. Got: {list(loaded.keys())}")
        sys.exit(1)

    vecs_a, vocab_a, idx_a = loaded[year_a]
    vecs_b, vocab_b, idx_b = loaded[year_b]

    score = compute_drift_score(word, vecs_a, idx_a, vecs_b, idx_b)
    if score is None:
        print(f"[ERROR] '{word}' not found in {year_a} or {year_b}")
        sys.exit(1)

    ctype = semantic_change_type(word, vecs_a, vocab_a, idx_a, vecs_b, vocab_b, idx_b)
    ns    = neighbor_shift(word, vecs_a, vocab_a, idx_a, vecs_b, vocab_b, idx_b)
    cs    = get_case_study(word)

    print(f"\n  Drift Score (cosine distance) : {score:.4f}")
    print(f"  Change Type                   : {ctype}")
    print(f"  Jaccard Neighbour Overlap     : {ns['jaccard_similarity']:.4f}")
    print(f"  Gained contexts               : {', '.join(ns['gained'][:8]) or 'none'}")
    print(f"  Lost contexts                 : {', '.join(ns['lost'][:8])   or 'none'}")

    print(f"\n  Top neighbours in {year_a}: " + ", ".join([w for w,_ in ns['neighbors_old'][:10]]))
    print(f"  Top neighbours in {year_b}: " + ", ".join([w for w,_ in ns['neighbors_new'][:10]]))

    if cs:
        print(f"\n  📖 Case Study: {cs['type']}")
        print(f"     {cs['description']}")

    # Optional: plot t-SNE to a PNG
    if getattr(args, "plot", False):
        from analysis.visualizations import plot_tsne_drift, fig_to_base64
        import base64, builtins
        img_b64 = plot_tsne_drift(word, vecs_a, vocab_a, idx_a, vecs_b, vocab_b, idx_b)
        out_path = os.path.join(BASE_DIR, "results", f"{word}_{year_a}_{year_b}_tsne.png")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with builtins.open(out_path, "wb") as f:
            f.write(base64.b64decode(img_b64))
        print(f"\n  💾  t-SNE plot saved → {out_path}")

    print()


# ─── timeline ─────────────────────────────────────────────────────────────────

def cmd_timeline(args):
    """Print and optionally export timeline drift scores."""
    _check_data()
    word = args.word.lower()
    from analysis.drift_analysis import compute_timeline

    print(f"\n📈  Timeline for '{word}'\n{'─'*50}")
    loaded     = _load_embeddings()
    decade_data = [(yr, v, i) for yr, (v, _, i) in sorted(loaded.items())]

    result = compute_timeline(word, decade_data)

    rows = list(zip(result["years"], result["scores"], result["present_in"]))
    print(f"\n  {'Decade':<10} {'Present':<10} {'Drift Score'}")
    print(f"  {'──────':<10} {'───────':<10} {'───────────'}")
    for yr, sc, pr in rows:
        sc_str = f"{sc:.4f}" if sc is not None else "N/A"
        pr_str = "✓" if pr else "✗"
        print(f"  {yr:<10} {pr_str:<10} {sc_str}")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["decade", "present", "drift_score"])
            w.writerows([(yr, pr, sc) for yr, sc, pr in rows])
        print(f"\n  💾  Saved to {args.out}")


# ─── top ──────────────────────────────────────────────────────────────────────

def cmd_top(args):
    """List the top-N most drifted words between two decades."""
    _check_data()
    from analysis.drift_analysis import top_drifted_words

    year_a = args.from_yr
    year_b = args.to_yr
    top_n  = args.n

    print(f"\n🏆  Top-{top_n} most drifted words: {year_a} → {year_b}\n{'─'*50}")
    loaded = _load_embeddings()

    vecs_a, _, idx_a = loaded[year_a]
    vecs_b, _, idx_b = loaded[year_b]

    results = top_drifted_words(vecs_a, idx_a, vecs_b, idx_b, top_n=top_n)

    print(f"\n  {'Rank':<6} {'Word':<20} {'Drift'}")
    print(f"  {'────':<6} {'────':<20} {'─────'}")
    for i, r in enumerate(results, 1):
        bar = "█" * int(r["drift"] * 30)
        print(f"  {i:<6} {r['word']:<20} {r['drift']:.4f}  {bar}")


# ─── download ─────────────────────────────────────────────────────────────────

def cmd_download(args):
    """Download and unzip the Stanford SGNS historical embeddings."""
    url     = "http://snap.stanford.edu/historical_embeddings/eng-all_sgns.zip"
    zipfile = os.path.join(BASE_DIR, "eng-all_sgns.zip")

    print(f"\n⬇  Downloading SGNS embeddings from Stanford…")
    print(f"   URL: {url}")
    print(f"   Destination: {zipfile}\n")
    print("   This file is ~1.4 GB — may take several minutes.\n")

    import urllib.request, zipfile as zf

    def progress(block, block_size, total):
        if total > 0:
            pct = min(block * block_size / total * 100, 100)
            print(f"\r   Progress: {pct:.1f}%", end="", flush=True)

    try:
        urllib.request.urlretrieve(url, zipfile, progress)
        print("\n\n   ✓ Download complete. Unzipping…")
        with zf.ZipFile(zipfile, "r") as z:
            z.extractall(BASE_DIR)
        print("   ✓ Extraction complete. sgns/ folder is ready.\n")
    except Exception as e:
        print(f"\n   [ERROR] Download failed: {e}")
        print("   Manual steps:")
        print(f"     1. Download {url}")
        print(f"     2. Unzip to {BASE_DIR} so you get {SGNS_DIR}/")


# ─── CLI Entry Point ──────────────────────────────────────────────────────────

def build_parser():
    p = argparse.ArgumentParser(
        prog="semantic-drift",
        description="Semantic Drift Analysis — B.Tech NLP Major Project",
    )
    sub = p.add_subparsers(dest="command")

    # serve
    sub.add_parser("serve", help="Start the web dashboard API")

    # word
    pw = sub.add_parser("word", help="Analyse a single word")
    pw.add_argument("word")
    pw.add_argument("--from", dest="from_yr", type=int, default=1900)
    pw.add_argument("--to",   dest="to_yr",   type=int, default=1990)
    pw.add_argument("--plot", action="store_true", help="Save t-SNE PNG")

    # timeline
    pt = sub.add_parser("timeline", help="Timeline drift for a word")
    pt.add_argument("word")
    pt.add_argument("--out", default=None, help="CSV output path")

    # top
    pp = sub.add_parser("top", help="Top-N most drifted words")
    pp.add_argument("--from", dest="from_yr", type=int, default=1900)
    pp.add_argument("--to",   dest="to_yr",   type=int, default=1990)
    pp.add_argument("--n",    type=int, default=20)

    # download
    sub.add_parser("download", help="Download the SGNS dataset")

    return p


def main():
    parser = build_parser()
    args   = parser.parse_args()

    if args.command == "serve":
        cmd_serve(args)
    elif args.command == "word":
        cmd_word(args)
    elif args.command == "timeline":
        cmd_timeline(args)
    elif args.command == "top":
        cmd_top(args)
    elif args.command == "download":
        cmd_download(args)
    else:
        print("\n⚗  Semantic Drift Analysis — B.Tech NLP Major Project")
        print("   Course: Natural Language Processing\n")
        print("   Commands:")
        print("     python main.py serve              — Start web dashboard")
        print("     python main.py word <word>        — Analyse a word")
        print("     python main.py timeline <word>    — Timeline drift")
        print("     python main.py top                — Top drifted words")
        print("     python main.py download           — Download SGNS data\n")
        parser.print_help()


if __name__ == "__main__":
    main()
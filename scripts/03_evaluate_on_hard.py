"""
Evaluate the existing GraphRAG pipeline on BNS-Facts-135.

Read-only against the original implementation: the FAISS index, the chunk list
and the DEFINES edges are loaded from the old project; nothing is written back.

Compares, on both benchmarks:
    - trivial no-graph baseline (query vs offence-name embedding)
    - FAISS + graph expansion at k = 1, 2, 3, 5, 10, 20
      (set-membership criterion, as published, and strict top-1)

Writes: results/hard_benchmark_eval.csv
"""

import os
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from scipy.stats import binomtest

OLD = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "original_data")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE, "data")
RES = os.path.join(HERE, "results")

LEGAL_SYNONYMS = {
    "sexual assault": "rape", "homicide": "murder", "killing a person": "murder",
    "forging documents": "forgery", "fake document": "forgery", "fraud": "cheating",
    "abduction": "kidnapping", "kidnap": "kidnapping",
}
KS = [1, 2, 3, 5, 10, 20]


def normalise(q):
    s = q.lower()
    for a, b in LEGAL_SYNONYMS.items():
        s = s.replace(a, b)
    return s


def load_shared():
    index = faiss.read_index(os.path.join(OLD, "results", "legal_index.faiss"))
    chunks = pd.read_csv(os.path.join(OLD, "chunks", "legal_chunks.csv"))
    assert len(chunks) == index.ntotal
    links = pd.read_csv(os.path.join(OLD, "results", "offence_links.csv"))
    defines = {}
    for _, r in links.iterrows():
        defines.setdefault(str(r["section_id"]).strip(), set()).add(
            str(r["offence"]).strip().lower())
    return index, chunks["chunk_id"].astype(str).values, defines


def strict_top1(secs, defines):
    for s in secs:
        o = defines.get(s)
        if o:
            return sorted(o)[0]
    return "none"


def mcnemar(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n01, n10 = int((~a & b).sum()), int((a & ~b).sum())
    n = n01 + n10
    if n == 0:
        return n10, n01, float("nan"), 1.0
    return n10, n01, (abs(n10 - n01) - 1) ** 2 / n, binomtest(n10, n, 0.5).pvalue


def evaluate(name, queries, gold, model, index, chunk_ids, defines, rows):
    labels = sorted(set(gold))
    norm = [normalise(q) for q in queries]

    # trivial no-graph baseline
    qe = np.asarray(model.encode(norm)).astype("float32")
    le = np.asarray(model.encode(labels)).astype("float32")
    triv = np.array([labels[j] for j in
                     ((qe[:, None, :] - le[None, :, :]) ** 2).sum(-1).argmin(1)])
    triv_ok = triv == np.asarray(gold)
    rows.append({"benchmark": name, "system": "trivial (no graph)", "k": "-",
                 "n": len(gold), "correct": int(triv_ok.sum()),
                 "accuracy": round(100 * triv_ok.mean(), 2),
                 "strict_top1": round(100 * triv_ok.mean(), 2)})

    # FAISS + graph expansion
    _, idx = index.search(qe, max(KS))
    retrieved = [[chunk_ids[j] for j in row] for row in idx]
    per_k = {}
    for k in KS:
        setok, strict = [], []
        for i, secs in enumerate(retrieved):
            top = secs[:k]
            cand = set()
            for s in top:
                cand |= defines.get(s, set())
            setok.append(gold[i] in cand)
            strict.append(strict_top1(top, defines))
        setok = np.array(setok)
        s_ok = np.array([p == g for p, g in zip(strict, gold)])
        per_k[k] = (setok, s_ok)
        rows.append({"benchmark": name, "system": "FAISS + graph", "k": k,
                     "n": len(gold), "correct": int(setok.sum()),
                     "accuracy": round(100 * setok.mean(), 2),
                     "strict_top1": round(100 * s_ok.mean(), 2)})

    print(f"\n{'='*74}\n{name}  (n={len(gold)})\n{'='*74}")
    print(f"  trivial (no graph)      {triv_ok.sum():3d}/{len(gold)} "
          f"= {100*triv_ok.mean():6.2f}%")
    for k in KS:
        so, st = per_k[k]
        print(f"  FAISS + graph  k={k:<3d}    {so.sum():3d}/{len(gold)} "
              f"= {100*so.mean():6.2f}%   (strict top-1 {100*st.mean():6.2f}%)")

    best_k = max(KS, key=lambda k: per_k[k][0].mean())
    n10, n01, chi2, p = mcnemar(triv_ok, per_k[best_k][0])
    print(f"\n  McNemar: trivial vs graph(k={best_k}): "
          f"trivial-only={n10}, graph-only={n01}, chi2={chi2:.3f}, p={p:.4g}")
    return triv_ok, per_k


def main():
    os.makedirs(RES, exist_ok=True)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    index, chunk_ids, defines = load_shared()
    rows = []

    o = pd.read_csv(os.path.join(OLD, "evaluation_queries_v2.csv"))
    o.columns = o.columns.str.strip()
    evaluate("ORIGINAL (evaluation_queries_v2)",
             o["query"].astype(str).str.strip().tolist(),
             o["expected_offence"].astype(str).str.strip().str.lower().tolist(),
             model, index, chunk_ids, defines, rows)

    n = pd.read_csv(os.path.join(DATA, "bns_facts_135.csv"))
    hard_gold = n["expected_offence"].astype(str).str.strip().str.lower().tolist()
    _, hard_k = evaluate("NEW: BNS-Facts-135",
                         n["query"].astype(str).str.strip().tolist(),
                         hard_gold, model, index, chunk_ids, defines, rows)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "hard_benchmark_eval.csv"), index=False)

    print(f"\n{'='*74}\nPER-CATEGORY on BNS-Facts-135, FAISS + graph k=5\n{'='*74}")
    per = pd.DataFrame({"gold": hard_gold, "ok": hard_k[5][0]})
    g = per.groupby("gold")["ok"].agg(["sum", "count"])
    g["acc"] = (100 * g["sum"] / g["count"]).round(2)
    print(g.to_string())

    print(f"\nwrote {os.path.join(RES, 'hard_benchmark_eval.csv')}")


if __name__ == "__main__":
    main()

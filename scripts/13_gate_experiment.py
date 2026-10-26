"""
Phase 2 -- the go/no-go experiment.

Question: with a statute-grounded graph instead of keyword-derived edges, does
relational expansion beat a trivial no-graph baseline on fact-pattern queries?

Compares, on both benchmarks, at k = 1..20:
    trivial (no graph)   query vs offence-name embedding
    OLD graph            41 keyword-derived edges  (original implementation)
    NEW graph            39 title-derived edges    (12_build_graph.py)

Writes: results/gate_results.csv
"""

import os
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from scipy.stats import binomtest

OLD = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "original_data")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA, RES = os.path.join(HERE, "data"), os.path.join(HERE, "results")

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


def edges(df, sec_col, off_col):
    d = {}
    for _, r in df.iterrows():
        d.setdefault(str(r[sec_col]).strip(), set()).add(str(r[off_col]).strip().lower())
    return d


def mcnemar(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n01, n10 = int((~a & b).sum()), int((a & ~b).sum())
    n = n01 + n10
    if n == 0:
        return n10, n01, float("nan"), 1.0
    return n10, n01, (abs(n10 - n01) - 1) ** 2 / n, binomtest(n10, n, 0.5).pvalue


def run(model, index, chunk_ids, queries, gold):
    labels = sorted(set(gold))
    qe = np.asarray(model.encode([normalise(q) for q in queries])).astype("float32")
    le = np.asarray(model.encode(labels)).astype("float32")
    triv = np.array([labels[j] for j in
                     ((qe[:, None, :] - le[None, :, :]) ** 2).sum(-1).argmin(1)])
    _, idx = index.search(qe, max(KS))
    retrieved = [[chunk_ids[j] for j in row] for row in idx]
    return (triv == np.asarray(gold)), retrieved


def score(retrieved, gold, defines, k):
    ok = []
    for i, secs in enumerate(retrieved):
        cand = set()
        for s in secs[:k]:
            cand |= defines.get(s, set())
        ok.append(gold[i] in cand)
    return np.array(ok)


def main():
    os.makedirs(RES, exist_ok=True)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    index = faiss.read_index(os.path.join(OLD, "results", "legal_index.faiss"))
    chunks = pd.read_csv(os.path.join(OLD, "chunks", "legal_chunks.csv"))
    chunk_ids = chunks["chunk_id"].astype(str).values

    g_old = edges(pd.read_csv(os.path.join(OLD, "results", "offence_links.csv")),
                  "section_id", "offence")
    g_new = edges(pd.read_csv(os.path.join(DATA, "graph_defines.csv")),
                  "section_id", "offence")
    print(f"OLD graph: {sum(len(v) for v in g_old.values())} edges / {len(g_old)} sections")
    print(f"NEW graph: {sum(len(v) for v in g_new.values())} edges / {len(g_new)} sections")

    benches = {}
    o = pd.read_csv(os.path.join(OLD, "evaluation_queries_v2.csv"))
    o.columns = o.columns.str.strip()
    benches["ORIGINAL (n=110)"] = (o["query"].astype(str).str.strip().tolist(),
                                   o["expected_offence"].astype(str).str.strip().str.lower().tolist())
    n = pd.read_csv(os.path.join(DATA, "bns_facts_135.csv"))
    benches["BNS-Facts-135"] = (n["query"].astype(str).str.strip().tolist(),
                                n["expected_offence"].astype(str).str.strip().str.lower().tolist())

    rows, store = [], {}
    for bname, (q, gold) in benches.items():
        triv_ok, retrieved = run(model, index, chunk_ids, q, gold)
        store[(bname, "trivial")] = triv_ok
        rows.append({"benchmark": bname, "system": "trivial (no graph)", "k": "-",
                     "correct": int(triv_ok.sum()), "n": len(gold),
                     "accuracy": round(100 * triv_ok.mean(), 2)})
        for gname, g in (("OLD graph", g_old), ("NEW graph", g_new)):
            for k in KS:
                ok = score(retrieved, gold, g, k)
                store[(bname, gname, k)] = ok
                rows.append({"benchmark": bname, "system": gname, "k": k,
                             "correct": int(ok.sum()), "n": len(gold),
                             "accuracy": round(100 * ok.mean(), 2)})

        print(f"\n{'='*76}\n{bname}\n{'='*76}")
        print(f"  trivial (no graph)            {triv_ok.sum():3d}/{len(gold)} = {100*triv_ok.mean():6.2f}%")
        for gname in ("OLD graph", "NEW graph"):
            cells = "  ".join(
                f"k={k}:{100*store[(bname,gname,k)].mean():5.1f}" for k in KS)
            print(f"  {gname:<28} {cells}")

        for gname in ("OLD graph", "NEW graph"):
            best = max(KS, key=lambda k: store[(bname, gname, k)].mean())
            n10, n01, chi2, p = mcnemar(triv_ok, store[(bname, gname, best)])
            verdict = ("graph WINS" if n01 > n10 and p < 0.05 else
                       "graph loses" if n10 > n01 and p < 0.05 else "no sig. diff")
            print(f"    {gname} best k={best:<3} vs trivial: "
                  f"trivial-only={n10:3d} graph-only={n01:3d} "
                  f"chi2={chi2:6.2f} p={p:.4g}  -> {verdict}")

    pd.DataFrame(rows).to_csv(os.path.join(RES, "gate_results.csv"), index=False)

    # head-to-head on the hard benchmark at matched k
    print(f"\n{'='*76}\nOLD vs NEW graph on BNS-Facts-135\n{'='*76}")
    for k in KS:
        a = store[("BNS-Facts-135", "OLD graph", k)]
        b = store[("BNS-Facts-135", "NEW graph", k)]
        n10, n01, chi2, p = mcnemar(a, b)
        print(f"  k={k:<3} old={100*a.mean():5.1f}%  new={100*b.mean():5.1f}%  "
              f"old-only={n10:3d} new-only={n01:3d}  p={p:.4g}")

    gold = benches["BNS-Facts-135"][1]
    print(f"\nPER-CATEGORY, NEW graph, BNS-Facts-135:")
    for k in (5, 10):
        ok = store[("BNS-Facts-135", "NEW graph", k)]
        per = pd.DataFrame({"g": gold, "ok": ok}).groupby("g")["ok"].agg(["sum", "count"])
        per["acc"] = (100 * per["sum"] / per["count"]).round(1)
        print(f"\n  k={k}")
        print(per.to_string())

    print(f"\nwrote {os.path.join(RES,'gate_results.csv')}")


if __name__ == "__main__":
    main()

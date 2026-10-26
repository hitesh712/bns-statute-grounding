"""
Phase 3e -- headline table with significance, and per-query outcomes saved.

Fixes the comparison properly: a stronger encoder lifts BOTH arms, so every
graph configuration is tested against the trivial baseline computed with the
SAME encoder. Anything else would overstate the graph's contribution -- the
mistake the conference paper made in a different form.

Writes: results/final_table.csv, results/per_query_outcomes.csv
"""

import os
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from scipy.stats import binomtest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA, RES = os.path.join(HERE, "data"), os.path.join(HERE, "results")

SYN = {"sexual assault": "rape", "homicide": "murder", "killing a person": "murder",
       "forging documents": "forgery", "fake document": "forgery", "fraud": "cheating",
       "abduction": "kidnapping", "kidnap": "kidnapping"}
KS = [1, 3, 5, 10, 20]
MODELS = [("MiniLM-L6", "sentence-transformers/all-MiniLM-L6-v2"),
          ("mpnet-base", "sentence-transformers/all-mpnet-base-v2"),
          ("BGE-base", "BAAI/bge-base-en-v1.5")]


def norm(q):
    s = q.lower()
    for a, b in SYN.items():
        s = s.replace(a, b)
    return s


def mcnemar(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n01, n10 = int((~a & b).sum()), int((a & ~b).sum())
    n = n01 + n10
    if n == 0:
        return n10, n01, 1.0
    return n10, n01, binomtest(n10, n, 0.5).pvalue


def main():
    sec = pd.read_csv(os.path.join(DATA, "bns_act_titled.csv"))
    sec["body"] = sec["body"].fillna("").astype(str)
    ids = sec["section_id"].astype(str).values

    dfn = pd.read_csv(os.path.join(DATA, "graph_defines.csv"))
    defines = {}
    for _, r in dfn.iterrows():
        defines.setdefault(str(r.section_id).strip(), set()).add(
            str(r.offence).strip().lower())

    b = pd.read_csv(os.path.join(DATA, "bns_facts_135.csv"))
    queries = [norm(q) for q in b["query"].astype(str).str.strip()]
    gold = np.array(b["expected_offence"].astype(str).str.strip().str.lower())
    labels = sorted(set(gold))

    rows = []
    outcomes = {"query_id": b["query_id"].tolist(), "gold": gold}

    for mname, mpath in MODELS:
        model = SentenceTransformer(mpath)
        qe = np.asarray(model.encode(queries, batch_size=32)).astype("float32")
        le = np.asarray(model.encode(labels)).astype("float32")
        triv = np.array([labels[j] for j in
                         ((qe[:, None, :] - le[None, :, :]) ** 2).sum(-1).argmin(1)]) == gold
        outcomes[f"{mname}|trivial"] = triv
        rows.append({"encoder": mname, "system": "trivial (no graph)", "k": "",
                     "accuracy": round(100 * triv.mean(), 2), "p_vs_trivial": ""})

        emb = np.asarray(model.encode(sec["body"].tolist(), batch_size=16)).astype("float32")
        idx = faiss.IndexFlatL2(emb.shape[1])
        idx.add(emb)
        _, I = idx.search(qe, max(KS))
        retrieved = [[ids[j] for j in row] for row in I]

        for k in KS:
            ok = np.array([gold[i] in set().union(
                *([defines.get(s, set()) for s in r[:k]] or [set()]))
                for i, r in enumerate(retrieved)])
            outcomes[f"{mname}|graph_k{k}"] = ok
            n10, n01, p = mcnemar(triv, ok)
            rows.append({"encoder": mname, "system": "FAISS + graph", "k": k,
                         "accuracy": round(100 * ok.mean(), 2),
                         "trivial_only": n10, "graph_only": n01,
                         "p_vs_trivial": f"{p:.3g}"})

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "final_table.csv"), index=False)
    pd.DataFrame(outcomes).to_csv(os.path.join(RES, "per_query_outcomes.csv"), index=False)

    print("=" * 86)
    print("BNS-Facts-135 (n=135). Each graph row is tested against the trivial")
    print("baseline computed with the SAME encoder.")
    print("=" * 86)
    print(f"{'encoder':<12}{'system':<20}{'k':>4}{'acc':>9}{'triv-only':>11}"
          f"{'graph-only':>12}{'p':>11}")
    for _, r in df.iterrows():
        print(f"{r.encoder:<12}{r.system:<20}{str(r.k):>4}{r.accuracy:>8.2f}%"
              f"{str(r.get('trivial_only','')):>11}{str(r.get('graph_only','')):>12}"
              f"{str(r.p_vs_trivial):>11}")

    print("\n" + "=" * 86)
    print("HEADLINE")
    print("=" * 86)
    g = df[(df.system == "FAISS + graph")].copy()
    g["acc"] = g.accuracy
    best = g.loc[g.acc.idxmax()]
    trivbest = df[(df.encoder == best.encoder) & (df.system.str.startswith("trivial"))].iloc[0]
    print(f"  best: {best.encoder} + graph at k={best.k}: {best.accuracy:.2f}%")
    print(f"        vs trivial baseline, same encoder: {trivbest.accuracy:.2f}%")
    print(f"        p = {best.p_vs_trivial}")
    prac = g[(g.k == 5)].sort_values("acc", ascending=False).iloc[0]
    tp = df[(df.encoder == prac.encoder) & (df.system.str.startswith("trivial"))].iloc[0]
    print(f"\n  best practical (k=5): {prac.encoder}: {prac.accuracy:.2f}% "
          f"vs trivial {tp.accuracy:.2f}%, p = {prac.p_vs_trivial}")
    print(f"\nwrote {os.path.join(RES,'final_table.csv')}")


if __name__ == "__main__":
    main()

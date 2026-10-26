"""
Phase 4a -- lexical retrieval baseline.

A reviewer will ask whether dense retrieval is needed at all: statutory text is
formulaic, and BM25 is a strong baseline on formulaic corpora. This runs BM25
over the same BNS sections with the same graph expansion, so the only thing
that varies is how sections are scored.

Also reports a hybrid (reciprocal rank fusion of BM25 and dense), which is the
standard thing to try when both signals are available.

Writes: results/bm25_results.csv
"""

import os
import re
import numpy as np
import pandas as pd
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from scipy.stats import binomtest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA, RES = os.path.join(HERE, "data"), os.path.join(HERE, "results")

SYN = {"sexual assault": "rape", "homicide": "murder", "killing a person": "murder",
       "forging documents": "forgery", "fake document": "forgery", "fraud": "cheating",
       "abduction": "kidnapping", "kidnap": "kidnapping"}
KS = [1, 3, 5, 10, 20]
DENSE = "BAAI/bge-base-en-v1.5"     # best encoder from Phase 3d


def norm(q):
    s = q.lower()
    for a, b in SYN.items():
        s = s.replace(a, b)
    return s


def tok(s):
    return re.findall(r"[a-z]+", str(s).lower())


def mcnemar(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n01, n10 = int((~a & b).sum()), int((a & ~b).sum())
    n = n01 + n10
    if n == 0:
        return n10, n01, 1.0
    return n10, n01, binomtest(n10, n, 0.5).pvalue


def rrf(*rank_lists, kconst=60):
    """Reciprocal rank fusion over lists of ranked section ids."""
    score = {}
    for rl in rank_lists:
        for pos, sid in enumerate(rl):
            score[sid] = score.get(sid, 0.0) + 1.0 / (kconst + pos + 1)
    return [s for s, _ in sorted(score.items(), key=lambda kv: -kv[1])]


def acc_at(retrieved, gold, defines, k):
    return np.array([gold[i] in set().union(
        *([defines.get(s, set()) for s in r[:k]] or [set()]))
        for i, r in enumerate(retrieved)])


def main():
    os.makedirs(RES, exist_ok=True)
    sec = pd.read_csv(os.path.join(DATA, "bns_act_titled.csv"))
    sec["body"] = sec["body"].fillna("").astype(str)
    ids = sec["section_id"].astype(str).values

    dfn = pd.read_csv(os.path.join(DATA, "graph_defines.csv"))
    defines = {}
    for _, r in dfn.iterrows():
        defines.setdefault(str(r.section_id).strip(), set()).add(
            str(r.offence).strip().lower())

    b = pd.read_csv(os.path.join(DATA, "bns_facts_135.csv"))
    raw = b["query"].astype(str).str.strip().tolist()
    queries = [norm(q) for q in raw]
    gold = b["expected_offence"].astype(str).str.strip().str.lower().tolist()

    # ---- BM25 ----
    bm25 = BM25Okapi([tok(t) for t in sec["body"]])
    bm_rank = []
    for q in queries:
        s = bm25.get_scores(tok(q))
        bm_rank.append([ids[j] for j in np.argsort(-s)[:max(KS)]])

    # ---- dense ----
    model = SentenceTransformer(DENSE)
    qe = np.asarray(model.encode(queries, batch_size=32)).astype("float32")
    emb = np.asarray(model.encode(sec["body"].tolist(), batch_size=16)).astype("float32")
    idx = faiss.IndexFlatL2(emb.shape[1])
    idx.add(emb)
    _, I = idx.search(qe, max(KS))
    dn_rank = [[ids[j] for j in row] for row in I]

    # ---- trivial baseline (dense encoder) ----
    labels = sorted(set(gold))
    le = np.asarray(model.encode(labels)).astype("float32")
    triv = np.array([labels[j] for j in
                     ((qe[:, None, :] - le[None, :, :]) ** 2).sum(-1).argmin(1)]
                    ) == np.asarray(gold)

    # ---- hybrid ----
    hy_rank = [rrf(bm_rank[i], dn_rank[i])[:max(KS)] for i in range(len(queries))]

    systems = {"BM25 + graph": bm_rank,
               "dense (BGE) + graph": dn_rank,
               "hybrid RRF + graph": hy_rank}

    rows, store = [], {}
    rows.append({"system": "trivial (no graph)", "k": "",
                 "accuracy": round(100 * triv.mean(), 2), "p_vs_trivial": ""})
    for sname, rk in systems.items():
        for k in KS:
            ok = acc_at(rk, gold, defines, k)
            store[(sname, k)] = ok
            n10, n01, p = mcnemar(triv, ok)
            rows.append({"system": sname, "k": k,
                         "accuracy": round(100 * ok.mean(), 2),
                         "trivial_only": n10, "sys_only": n01,
                         "p_vs_trivial": f"{p:.3g}"})

    pd.DataFrame(rows).to_csv(os.path.join(RES, "bm25_results.csv"), index=False)

    print("=" * 78)
    print(f"BNS-Facts-135 (n=135)   trivial no-graph baseline: {100*triv.mean():.2f}%")
    print("=" * 78)
    print(f"{'system':<24}" + "".join(f"{('k='+str(k)):>10}" for k in KS))
    for sname in systems:
        print(f"{sname:<24}" + "".join(
            f"{100*store[(sname,k)].mean():>9.2f}%" for k in KS))

    print("\nvs trivial baseline (exact McNemar):")
    for sname in systems:
        for k in (5, 10, 20):
            n10, n01, p = mcnemar(triv, store[(sname, k)])
            flag = "WINS" if n01 > n10 and p < .05 else ("loses" if n10 > n01 and p < .05 else "n.s.")
            print(f"  {sname:<22} k={k:<3} {100*store[(sname,k)].mean():6.2f}%  "
                  f"p={p:<10.3g} {flag}")

    print("\nBM25 vs dense, head to head:")
    for k in KS:
        n10, n01, p = mcnemar(store[("BM25 + graph", k)], store[("dense (BGE) + graph", k)])
        print(f"  k={k:<3} bm25={100*store[('BM25 + graph',k)].mean():6.2f}%  "
              f"dense={100*store[('dense (BGE) + graph',k)].mean():6.2f}%  "
              f"bm25-only={n10:3d} dense-only={n01:3d}  p={p:.3g}")

    print("\nPER-CATEGORY at k=5:")
    per = pd.DataFrame({"g": gold})
    for sname in systems:
        per[sname] = store[(sname, 5)]
    print(per.groupby("g").mean().mul(100).round(1).to_string())

    print(f"\nwrote {os.path.join(RES,'bm25_results.csv')}")


if __name__ == "__main__":
    main()

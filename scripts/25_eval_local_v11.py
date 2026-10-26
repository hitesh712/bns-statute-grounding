"""
Re-run every local retrieval system on BNS-Facts-135 v1.1.

v1.1 corrects nine defective items (see data/bns_facts_135_v1_1_changelog.csv),
so all headline numbers are recomputed on it. Same protocol as scripts 19/20:
each graph configuration is tested against the trivial baseline computed with
the SAME encoder; BNS-scoped retrieval; statute-grounded DEFINES edges.

Writes: results/final_table_v11.csv, results/per_query_outcomes_v11.csv
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
    return n10, n01, (1.0 if n == 0 else binomtest(n10, n, 0.5).pvalue)


def at_k(ranked, gold, defines, k):
    return np.array([gold[i] in set().union(*([defines.get(s, set()) for s in r[:k]] or [set()]))
                     for i, r in enumerate(ranked)])


def rrf(a, b, k=60):
    s = {}
    for rl in (a, b):
        for p, sid in enumerate(rl):
            s[sid] = s.get(sid, 0) + 1 / (k + p + 1)
    return [x for x, _ in sorted(s.items(), key=lambda kv: -kv[1])]


def main():
    sec = pd.read_csv(os.path.join(DATA, "bns_act_titled.csv"))
    sec["body"] = sec["body"].fillna("").astype(str)
    ids = sec["section_id"].astype(str).values
    dfn = pd.read_csv(os.path.join(DATA, "graph_defines.csv"))
    defines = {}
    for _, r in dfn.iterrows():
        defines.setdefault(str(r.section_id).strip(), set()).add(str(r.offence).strip().lower())

    b = pd.read_csv(os.path.join(DATA, "bns_facts_135_v1_1.csv"))
    qs = [norm(q) for q in b["query"].astype(str).str.strip()]
    gold = np.array(b["expected_offence"].astype(str).str.strip().str.lower())
    labels = sorted(set(gold))

    tok = lambda s: re.findall(r"[a-z]+", str(s).lower())
    bm = BM25Okapi([tok(t) for t in sec["body"]])
    bm_rank = [[ids[j] for j in np.argsort(-bm.get_scores(tok(q)))[:20]] for q in qs]

    rows, out = [], {"query_id": b["query_id"].tolist(), "gold": gold}
    bge_rank = None
    for mname, mpath in MODELS:
        m = SentenceTransformer(mpath)
        qe = np.asarray(m.encode(qs, batch_size=32)).astype("float32")
        le = np.asarray(m.encode(labels)).astype("float32")
        triv = np.array([labels[j] for j in ((qe[:, None, :] - le[None, :, :]) ** 2).sum(-1).argmin(1)]) == gold
        out[f"{mname}|trivial"] = triv
        rows.append({"system": f"{mname} trivial (no graph)", "k": "", "accuracy": round(100 * triv.mean(), 2),
                     "p_vs_same_encoder_trivial": ""})
        emb = np.asarray(m.encode(sec["body"].tolist(), batch_size=16)).astype("float32")
        idx = faiss.IndexFlatL2(emb.shape[1]); idx.add(emb)
        _, I = idx.search(qe, 20)
        ranked = [[ids[j] for j in row] for row in I]
        if mname == "BGE-base":
            bge_rank, bge_triv = ranked, triv
        for k in KS:
            ok = at_k(ranked, gold, defines, k)
            out[f"{mname}|graph_k{k}"] = ok
            n10, n01, p = mcnemar(triv, ok)
            rows.append({"system": f"{mname} + graph", "k": k, "accuracy": round(100 * ok.mean(), 2),
                         "trivial_only": n10, "system_only": n01, "p_vs_same_encoder_trivial": f"{p:.3g}"})

    hy = [rrf(bm_rank[i], bge_rank[i]) for i in range(len(qs))]
    for name, ranked in (("BM25", bm_rank), ("hybrid RRF", hy)):
        for k in KS:
            ok = at_k(ranked, gold, defines, k)
            out[f"{name}|graph_k{k}"] = ok
            n10, n01, p = mcnemar(bge_triv, ok)
            rows.append({"system": f"{name} + graph", "k": k, "accuracy": round(100 * ok.mean(), 2),
                         "trivial_only": n10, "system_only": n01, "p_vs_same_encoder_trivial": f"{p:.3g}"})

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "final_table_v11.csv"), index=False)
    pd.DataFrame(out).to_csv(os.path.join(RES, "per_query_outcomes_v11.csv"), index=False)

    print("BNS-Facts-135 v1.1 (n=135)")
    piv = df[df.k != ""].pivot_table(index="system", columns="k", values="accuracy")
    triv = df[df.k == ""].set_index("system")["accuracy"]
    print("trivial baselines:\n" + triv.to_string())
    print("\ngraph systems (accuracy %, set-membership):\n" + piv.to_string())
    print("\nheadline cells:")
    for sys_, k in (("BGE-base + graph", 5), ("BGE-base + graph", 10),
                    ("hybrid RRF + graph", 10), ("hybrid RRF + graph", 20),
                    ("mpnet-base + graph", 20), ("MiniLM-L6 + graph", 5)):
        r = df[(df.system == sys_) & (df.k == k)].iloc[0]
        print(f"  {sys_:<20} k={k:<3} {r.accuracy:6.2f}%  p vs trivial = {r.p_vs_same_encoder_trivial}")
    print("\nwrote results/final_table_v11.csv, per_query_outcomes_v11.csv")


if __name__ == "__main__":
    main()

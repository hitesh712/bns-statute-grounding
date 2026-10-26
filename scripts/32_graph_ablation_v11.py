"""
Graph-quality and retrieval-scope ablation on BNS-Facts-135 v1.1.

Re-runs the Phase 2 comparison on the corrected benchmark so every number in the
manuscript comes from v1.1:
    edges   original keyword-derived DEFINES (41)  vs  statute-grounded (39)
    scope   full 1,511-chunk corpus  vs  the 351 BNS sections
    encoder MiniLM-L6 (original) and BGE-base (best)
Each configuration is tested against the trivial baseline of the same encoder.

Writes: results/graph_ablation_v11.csv
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
SYN = {"sexual assault": "rape", "homicide": "murder", "killing a person": "murder",
       "forging documents": "forgery", "fake document": "forgery", "fraud": "cheating",
       "abduction": "kidnapping", "kidnap": "kidnapping"}
KS = [1, 3, 5, 10, 20]


def norm(q):
    s = q.lower()
    for a, b in SYN.items():
        s = s.replace(a, b)
    return s


def edges(path):
    d = pd.read_csv(path)
    out = {}
    for _, r in d.iterrows():
        out.setdefault(str(r["section_id"]).strip(), set()).add(str(r["offence"]).strip().lower())
    return out


def mcnemar(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n10, n01 = int((a & ~b).sum()), int((~a & b).sum())
    return n10, n01, (binomtest(n10, n10 + n01, .5).pvalue if n10 + n01 else 1.0)


def main():
    chunks = pd.read_csv(os.path.join(OLD, "chunks", "legal_chunks.csv"))
    chunks["chunk_id"] = chunks["chunk_id"].astype(str)
    chunks["content"] = chunks["content"].fillna("").astype(str)
    g_old = edges(os.path.join(OLD, "results", "offence_links.csv"))
    g_new = edges(os.path.join(DATA, "graph_defines.csv"))

    b = pd.read_csv(os.path.join(DATA, "bns_facts_135_v1_1.csv"))
    qs = [norm(q) for q in b["query"]]
    gold = np.array(b["expected_offence"].str.strip().str.lower())
    labels = sorted(set(gold))

    rows, outcomes = [], {}
    for mname, mpath in (("MiniLM-L6", "sentence-transformers/all-MiniLM-L6-v2"),
                         ("BGE-base", "BAAI/bge-base-en-v1.5")):
        m = SentenceTransformer(mpath)
        qe = np.asarray(m.encode(qs, batch_size=32)).astype("float32")
        le = np.asarray(m.encode(labels)).astype("float32")
        triv = np.array([labels[j] for j in ((qe[:, None] - le[None]) ** 2).sum(-1).argmin(1)]) == gold
        emb = np.asarray(m.encode(chunks["content"].tolist(), batch_size=16)).astype("float32")
        # Full corpus replicates the original index exactly (including its one
        # duplicate, garbled BNS_338 chunk). BNS-only uses the 350 distinct BNS
        # sections, matching the main experiments (scripts 25-31).
        bns_first = (chunks["chunk_id"].str.startswith("BNS_") &
                     ~chunks["chunk_id"].duplicated(keep="first")).values
        for scope, mask in (("full corpus", np.ones(len(chunks), bool)),
                            ("BNS only", bns_first)):
            ids = chunks["chunk_id"].values[mask]
            idx = faiss.IndexFlatL2(emb.shape[1]); idx.add(emb[mask])
            _, I = idx.search(qe, max(KS))
            ranked = [[ids[j] for j in row] for row in I]
            for gname, g in (("keyword edges", g_old), ("statute edges", g_new)):
                for k in KS:
                    ok = np.array([gold[i] in set().union(*([g.get(s, set()) for s in r[:k]] or [set()]))
                                   for i, r in enumerate(ranked)])
                    outcomes[(mname, scope, gname, k)] = ok
                    n10, n01, p = mcnemar(triv, ok)
                    rows.append({"encoder": mname, "scope": scope, "edges": gname, "k": k,
                                 "accuracy": round(100 * ok.mean(), 2),
                                 "trivial": round(100 * triv.mean(), 2), "p_vs_trivial": p})
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(RES, "graph_ablation_v11.csv"), index=False)
    piv = d.pivot_table(index=["encoder", "scope", "edges"], columns="k", values="accuracy")
    print(piv.to_string())
    print("\ntrivial:", d.groupby("encoder")["trivial"].first().to_dict())
    best = d.loc[d.groupby(["encoder", "scope", "edges"])["accuracy"].idxmax()]
    print("\nbest k per configuration, p vs same-encoder trivial:")
    print(best[["encoder", "scope", "edges", "k", "accuracy", "p_vs_trivial"]].to_string(index=False))

    print("\nkeyword vs statute edges at matched k (exact McNemar, paired):")
    pr = []
    for mname in ("MiniLM-L6", "BGE-base"):
        for scope in ("full corpus", "BNS only"):
            for k in KS:
                a = outcomes[(mname, scope, "keyword edges", k)]
                s = outcomes[(mname, scope, "statute edges", k)]
                n10, n01, p = mcnemar(a, s)
                pr.append({"encoder": mname, "scope": scope, "k": k, "keyword_only": n10,
                           "statute_only": n01, "p": p})
                print(f"  {mname:<10} {scope:<12} k={k:<3} keyword-only={n10:3d} "
                      f"statute-only={n01:3d}  p={p:.2g}")
    pd.DataFrame(pr).to_csv(os.path.join(RES, "graph_edges_mcnemar_v11.csv"), index=False)


if __name__ == "__main__":
    main()

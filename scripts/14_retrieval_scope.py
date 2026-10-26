"""
Phase 2b -- why the graph needs k=20, and whether scoping retrieval fixes it.

The index holds all 1,511 chunks, but 1,160 of them are BNSS / BSA / IT Act
sections that can never carry an offence label: offences live in the BNS. Those
chunks occupy top-k slots without ever contributing a candidate offence, so the
graph only gets a usable section once k is large.

This tests retrieval restricted to the BNS (351 chunks) against the full corpus,
with the statute-grounded graph held constant.

Writes: results/scope_results.csv
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
KS = [1, 2, 3, 5, 10, 20]


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
        return n10, n01, float("nan"), 1.0
    return n10, n01, (abs(n10 - n01) - 1) ** 2 / n, binomtest(n10, n, 0.5).pvalue


def main():
    os.makedirs(RES, exist_ok=True)
    model = SentenceTransformer("all-MiniLM-L6-v2")

    index = faiss.read_index(os.path.join(OLD, "results", "legal_index.faiss"))
    chunks = pd.read_csv(os.path.join(OLD, "chunks", "legal_chunks.csv"))
    ids = chunks["chunk_id"].astype(str).values
    vecs = index.reconstruct_n(0, index.ntotal)          # recover stored embeddings
    bns_mask = np.array([i.startswith("BNS_") for i in ids])
    print(f"corpus: {len(ids)} chunks, of which BNS: {bns_mask.sum()}")

    dfn = pd.read_csv(os.path.join(DATA, "graph_defines.csv"))
    defines = {}
    for _, r in dfn.iterrows():
        defines.setdefault(str(r.section_id).strip(), set()).add(str(r.offence).strip().lower())

    b = pd.read_csv(os.path.join(DATA, "bns_facts_135.csv"))
    queries = b["query"].astype(str).str.strip().tolist()
    gold = b["expected_offence"].astype(str).str.strip().str.lower().tolist()

    qe = np.asarray(model.encode([norm(q) for q in queries])).astype("float32")

    # trivial baseline for reference
    labels = sorted(set(gold))
    le = np.asarray(model.encode(labels)).astype("float32")
    triv = np.array([labels[j] for j in ((qe[:, None, :] - le[None, :, :]) ** 2).sum(-1).argmin(1)])
    triv_ok = triv == np.asarray(gold)

    scopes = {"full corpus (1511)": np.ones(len(ids), bool), "BNS only (351)": bns_mask}
    rows, store = [], {}

    for sname, mask in scopes.items():
        sub, sub_ids = vecs[mask], ids[mask]
        sidx = faiss.IndexFlatL2(sub.shape[1])
        sidx.add(sub)
        _, I = sidx.search(qe, max(KS))
        retrieved = [[sub_ids[j] for j in row] for row in I]
        for k in KS:
            ok = []
            for i, secs in enumerate(retrieved):
                cand = set()
                for s in secs[:k]:
                    cand |= defines.get(s, set())
                ok.append(gold[i] in cand)
            ok = np.array(ok)
            store[(sname, k)] = ok
            rows.append({"scope": sname, "k": k, "correct": int(ok.sum()),
                         "n": len(gold), "accuracy": round(100 * ok.mean(), 2)})

    print(f"\ntrivial baseline (no graph): {triv_ok.sum()}/{len(gold)} = {100*triv_ok.mean():.2f}%\n")
    print(f"{'k':<5}" + "".join(f"{s:>24}" for s in scopes))
    for k in KS:
        print(f"{k:<5}" + "".join(
            f"{100*store[(s,k)].mean():>23.2f}%" for s in scopes))

    print("\nvs trivial baseline (exact McNemar):")
    for sname in scopes:
        best = max(KS, key=lambda k: store[(sname, k)].mean())
        n10, n01, chi2, p = mcnemar(triv_ok, store[(sname, best)])
        sig = "graph WINS" if n01 > n10 and p < 0.05 else ("graph loses" if n10 > n01 and p < 0.05 else "n.s.")
        print(f"  {sname:<22} best k={best:<3} acc={100*store[(sname,best)].mean():5.2f}%  "
              f"p={p:.4g}  -> {sig}")
        # also at the smallest k that beats trivial
        for k in KS:
            if store[(sname, k)].mean() > triv_ok.mean():
                n10, n01, chi2, p = mcnemar(triv_ok, store[(sname, k)])
                print(f"      first k beating trivial: k={k} "
                      f"({100*store[(sname,k)].mean():.2f}%, p={p:.4g})")
                break

    pd.DataFrame(rows).to_csv(os.path.join(RES, "scope_results.csv"), index=False)

    print("\nPER-CATEGORY, BNS-only scope:")
    for k in (3, 5, 10):
        ok = store[("BNS only (351)", k)]
        per = pd.DataFrame({"g": gold, "ok": ok}).groupby("g")["ok"].agg(["sum", "count"])
        per["acc"] = (100 * per["sum"] / per["count"]).round(1)
        print(f"\n  k={k}  overall {100*ok.mean():.2f}%")
        print(per[["sum", "count", "acc"]].to_string())

    print(f"\nwrote {os.path.join(RES,'scope_results.csv')}")


if __name__ == "__main__":
    main()

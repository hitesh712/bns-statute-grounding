"""
Phase 3b -- fix low-k performance.

At k=5 the system is still only 61.5%, i.e. it needs to retrieve 20 sections to
beat a trivial baseline convincingly. That is a retrieval problem, not a graph
problem: the original index embeds section BODIES only, and statutory bodies are
long, archaic and full of shared boilerplate, so a short lay query matches them
weakly.

A section's marginal title ("Extortion", "Cheating by personation") is short and
highly discriminative. This tests indexing the title alongside the body.

Variants, all BNS-scoped with the statute-grounded graph held constant:
    body          original representation (re-embedded for a like-for-like test)
    title         marginal title only
    title+body    title prepended to the body
    title x3      title repeated to weight it against a long body

Writes: results/retrieval_variants.csv
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

    sec = pd.read_csv(os.path.join(DATA, "bns_act_titled.csv"))
    sec["title"] = sec["title"].fillna("").astype(str)
    sec["body"] = sec["body"].fillna("").astype(str)
    ids = sec["section_id"].astype(str).values
    print(f"BNS sections indexed: {len(sec)}")

    dfn = pd.read_csv(os.path.join(DATA, "graph_defines.csv"))
    defines = {}
    for _, r in dfn.iterrows():
        defines.setdefault(str(r.section_id).strip(), set()).add(
            str(r.offence).strip().lower())

    b = pd.read_csv(os.path.join(DATA, "bns_facts_135.csv"))
    queries = [norm(q) for q in b["query"].astype(str).str.strip()]
    gold = b["expected_offence"].astype(str).str.strip().str.lower().tolist()
    qe = np.asarray(model.encode(queries)).astype("float32")

    labels = sorted(set(gold))
    le = np.asarray(model.encode(labels)).astype("float32")
    triv_ok = np.array([labels[j] for j in
                        ((qe[:, None, :] - le[None, :, :]) ** 2).sum(-1).argmin(1)]
                       ) == np.asarray(gold)
    print(f"trivial baseline: {triv_ok.sum()}/{len(gold)} = {100*triv_ok.mean():.2f}%\n")

    variants = {
        "body":       sec["body"].tolist(),
        "title":      sec["title"].tolist(),
        "title+body": (sec["title"] + ". " + sec["body"]).tolist(),
        "title x3":   ((sec["title"] + ". ") * 3 + sec["body"]).tolist(),
    }

    rows, store = [], {}
    for vname, texts in variants.items():
        emb = np.asarray(model.encode(texts, batch_size=32)).astype("float32")
        idx = faiss.IndexFlatL2(emb.shape[1])
        idx.add(emb)
        _, I = idx.search(qe, max(KS))
        retrieved = [[ids[j] for j in row] for row in I]
        for k in KS:
            ok = []
            for i, secs in enumerate(retrieved):
                cand = set()
                for s in secs[:k]:
                    cand |= defines.get(s, set())
                ok.append(gold[i] in cand)
            ok = np.array(ok)
            store[(vname, k)] = ok
            rows.append({"variant": vname, "k": k, "correct": int(ok.sum()),
                         "n": len(gold), "accuracy": round(100 * ok.mean(), 2)})

    print(f"{'variant':<13}" + "".join(f"{('k='+str(k)):>9}" for k in KS))
    for v in variants:
        print(f"{v:<13}" + "".join(f"{100*store[(v,k)].mean():>8.1f}%" for k in KS))

    print("\nvs trivial baseline (exact McNemar):")
    for v in variants:
        first = None
        for k in KS:
            if store[(v, k)].mean() > triv_ok.mean():
                _, _, _, p = mcnemar(triv_ok, store[(v, k)])
                first = (k, 100 * store[(v, k)].mean(), p)
                break
        best = max(KS, key=lambda k: store[(v, k)].mean())
        _, _, chi2, pb = mcnemar(triv_ok, store[(v, best)])
        fs = (f"first k beating trivial: k={first[0]} ({first[1]:.1f}%, p={first[2]:.3g})"
              if first else "never beats trivial")
        print(f"  {v:<12} best k={best:<3} {100*store[(v,best)].mean():5.1f}% "
              f"p={pb:.3g}   |  {fs}")

    pd.DataFrame(rows).to_csv(os.path.join(RES, "retrieval_variants.csv"), index=False)

    bestv = max(variants, key=lambda v: store[(v, 5)].mean())
    print(f"\nPER-CATEGORY, best low-k variant '{bestv}':")
    for k in (3, 5):
        ok = store[(bestv, k)]
        per = pd.DataFrame({"g": gold, "ok": ok}).groupby("g")["ok"].agg(["sum", "count"])
        per["acc"] = (100 * per["sum"] / per["count"]).round(1)
        print(f"\n  k={k}  overall {100*ok.mean():.2f}%")
        print(per.to_string())

    print(f"\nwrote {os.path.join(RES,'retrieval_variants.csv')}")


if __name__ == "__main__":
    main()

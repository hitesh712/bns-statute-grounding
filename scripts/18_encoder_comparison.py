"""
Phase 3d -- is the embedding model the bottleneck?

17_error_diagnosis.py showed the failures are MATCHING failures, not ranking
failures: for cheating the defining section sits at median rank 27 of 350, and
the top hit for an extortion query is "Act done pursuant to the judgment or
order of Court". all-MiniLM-L6-v2 is a 384-dimensional general-purpose model
with a 256-token limit, embedding long archaic statutory text.

Two things are varied:
    encoder     MiniLM-L6 (original) vs mpnet-base vs BGE-base
    truncation  whole section body vs its first 1,200 characters, which is the
                operative definition before the illustrations

Writes: results/encoder_comparison.csv
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

MODELS = [
    ("MiniLM-L6  (original)", "sentence-transformers/all-MiniLM-L6-v2"),
    ("mpnet-base",            "sentence-transformers/all-mpnet-base-v2"),
    ("BGE-base",              "BAAI/bge-base-en-v1.5"),
]


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
        return 1.0
    return binomtest(n10, n, 0.5).pvalue


def main():
    os.makedirs(RES, exist_ok=True)
    sec = pd.read_csv(os.path.join(DATA, "bns_act_titled.csv"))
    sec["body"] = sec["body"].fillna("").astype(str)
    ids = sec["section_id"].astype(str).values

    dfn = pd.read_csv(os.path.join(DATA, "graph_defines.csv"))
    defines, targets = {}, {}
    for _, r in dfn.iterrows():
        s, o = str(r.section_id).strip(), str(r.offence).strip().lower()
        defines.setdefault(s, set()).add(o)
        targets.setdefault(o, set()).add(s)

    b = pd.read_csv(os.path.join(DATA, "bns_facts_135.csv"))
    queries = [norm(q) for q in b["query"].astype(str).str.strip()]
    gold = b["expected_offence"].astype(str).str.strip().str.lower().tolist()
    labels = sorted(set(gold))

    corpora = {
        "full body":  sec["body"].tolist(),
        "first 1200": sec["body"].str.slice(0, 1200).tolist(),
    }

    rows, store, trivs = [], {}, {}
    for mname, mpath in MODELS:
        print(f"\nloading {mname} ...")
        try:
            model = SentenceTransformer(mpath)
        except Exception as e:
            print(f"  SKIP ({type(e).__name__}: {e})")
            continue

        qe = np.asarray(model.encode(queries, batch_size=32)).astype("float32")
        le = np.asarray(model.encode(labels)).astype("float32")
        triv = np.array([labels[j] for j in
                         ((qe[:, None, :] - le[None, :, :]) ** 2).sum(-1).argmin(1)]
                        ) == np.asarray(gold)
        trivs[mname] = triv
        rows.append({"encoder": mname, "corpus": "-", "system": "trivial (no graph)",
                     "k": "-", "accuracy": round(100 * triv.mean(), 2)})

        for cname, texts in corpora.items():
            emb = np.asarray(model.encode(texts, batch_size=16)).astype("float32")
            idx = faiss.IndexFlatL2(emb.shape[1])
            idx.add(emb)
            _, I = idx.search(qe, 350)
            retrieved = [[ids[j] for j in row] for row in I]

            ranks = []
            for i, r in enumerate(retrieved):
                want = targets.get(gold[i], set())
                ranks.append(next((p + 1 for p, s in enumerate(r) if s in want), 999))

            for k in KS:
                ok = np.array([gold[i] in set().union(
                    *[defines.get(s, set()) for s in r[:k]] or [set()])
                    for i, r in enumerate(retrieved)])
                store[(mname, cname, k)] = ok
                rows.append({"encoder": mname, "corpus": cname, "system": "graph",
                             "k": k, "accuracy": round(100 * ok.mean(), 2),
                             "median_rank_correct": float(np.median(ranks))})
            print(f"  {cname:<11} " + "  ".join(
                f"k={k}:{100*store[(mname,cname,k)].mean():5.1f}" for k in KS)
                + f"   median rank {np.median(ranks):.0f}")

    pd.DataFrame(rows).to_csv(os.path.join(RES, "encoder_comparison.csv"), index=False)

    print("\n" + "=" * 78)
    print("SUMMARY -- accuracy on BNS-Facts-135")
    print("=" * 78)
    print(f"{'encoder / corpus':<34}{'trivial':>9}" + "".join(f"{('k='+str(k)):>8}" for k in KS))
    for mname, _ in MODELS:
        if mname not in trivs:
            continue
        for cname in corpora:
            if (mname, cname, KS[0]) not in store:
                continue
            print(f"{mname+' / '+cname:<34}{100*trivs[mname].mean():>8.1f}%"
                  + "".join(f"{100*store[(mname,cname,k)].mean():>7.1f}%" for k in KS))

    print("\nbest configuration vs its own trivial baseline:")
    best = max(((m, c, k) for (m, c, k) in store),
               key=lambda t: store[t].mean())
    m, c, k = best
    print(f"  {m} / {c} / k={k}: {100*store[best].mean():.2f}% "
          f"vs trivial {100*trivs[m].mean():.2f}%  p={mcnemar(trivs[m], store[best]):.3g}")

    print("\nlowest k that beats the trivial baseline, per configuration:")
    for mname, _ in MODELS:
        if mname not in trivs:
            continue
        for cname in corpora:
            hit = next((k for k in KS if (mname, cname, k) in store
                        and store[(mname, cname, k)].mean() > trivs[mname].mean()), None)
            if hit:
                p = mcnemar(trivs[mname], store[(mname, cname, hit)])
                print(f"  {mname+' / '+cname:<34} k={hit:<3} "
                      f"{100*store[(mname,cname,hit)].mean():5.1f}%  p={p:.3g}")
            else:
                print(f"  {mname+' / '+cname:<34} never")

    print(f"\nwrote {os.path.join(RES,'encoder_comparison.csv')}")


if __name__ == "__main__":
    main()

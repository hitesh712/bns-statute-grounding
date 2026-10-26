"""
Phase 3c -- why do cheating and extortion fail at low k?

Two possibilities, with different fixes:
  (a) RANKING   the defining section is retrieved, but below k
  (b) MATCHING  the defining section is nowhere near the top at all

This reports, for every query, the rank at which the first correctly-labelled
section appears, and what is being retrieved instead.

Writes: results/error_diagnosis.csv
"""

import os
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA, RES = os.path.join(HERE, "data"), os.path.join(HERE, "results")

SYN = {"sexual assault": "rape", "homicide": "murder", "killing a person": "murder",
       "forging documents": "forgery", "fake document": "forgery", "fraud": "cheating",
       "abduction": "kidnapping", "kidnap": "kidnapping"}
DEPTH = 350


def norm(q):
    s = q.lower()
    for a, b in SYN.items():
        s = s.replace(a, b)
    return s


def main():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    sec = pd.read_csv(os.path.join(DATA, "bns_act_titled.csv"))
    sec["body"] = sec["body"].fillna("").astype(str)
    ids = sec["section_id"].astype(str).values
    title_of = dict(zip(sec["section_id"].astype(str), sec["title"].fillna("")))

    dfn = pd.read_csv(os.path.join(DATA, "graph_defines.csv"))
    defines = {}
    for _, r in dfn.iterrows():
        defines.setdefault(str(r.section_id).strip(), set()).add(
            str(r.offence).strip().lower())
    # offence -> the sections that define it
    targets = {}
    for s, offs in defines.items():
        for o in offs:
            targets.setdefault(o, set()).add(s)

    emb = np.asarray(model.encode(sec["body"].tolist(), batch_size=32)).astype("float32")
    idx = faiss.IndexFlatL2(emb.shape[1])
    idx.add(emb)

    b = pd.read_csv(os.path.join(DATA, "bns_facts_135.csv"))
    queries = b["query"].astype(str).str.strip().tolist()
    gold = b["expected_offence"].astype(str).str.strip().str.lower().tolist()
    qe = np.asarray(model.encode([norm(q) for q in queries])).astype("float32")
    _, I = idx.search(qe, DEPTH)

    rows = []
    for i, row in enumerate(I):
        ranked = [ids[j] for j in row]
        want = targets.get(gold[i], set())
        rank = next((r + 1 for r, s in enumerate(ranked) if s in want), None)
        rows.append({
            "query_id": b.iloc[i]["query_id"],
            "query": queries[i],
            "gold": gold[i],
            "first_correct_rank": rank if rank else -1,
            "top1": ranked[0],
            "top1_title": str(title_of.get(ranked[0], ""))[:60],
            "top3": ", ".join(ranked[:3]),
        })

    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(RES, "error_diagnosis.csv"), index=False, encoding="utf-8")

    found = d[d.first_correct_rank > 0]
    print(f"queries where a correctly-labelled section exists in the ranking: "
          f"{len(found)}/{len(d)}")
    print("\nrank at which the first correct section appears (median by category):")
    agg = found.groupby("gold")["first_correct_rank"].agg(
        ["median", "mean", "min", "max", "count"]).round(1)
    print(agg.to_string())

    print("\nshare of queries whose correct section is within rank:")
    hdr = "".join(f"{('<='+str(k)):>8}" for k in (1, 3, 5, 10, 20, 50, 100))
    print(f"{'category':<20}{hdr}")
    for cat, g in d.groupby("gold"):
        cells = "".join(
            f"{100*((g.first_correct_rank>0)&(g.first_correct_rank<=k)).mean():>7.0f}%"
            for k in (1, 3, 5, 10, 20, 50, 100))
        print(f"{cat:<20}{cells}")

    print("\n--- what is retrieved instead, for the two weak categories ---")
    for cat in ("cheating", "extortion"):
        print(f"\n[{cat}]  target sections: "
              f"{sorted(targets.get(cat,[]), key=lambda s:int(s.split('_')[1]))}")
        for _, r in d[(d.gold == cat)].head(6).iterrows():
            print(f"  rank of correct = {r.first_correct_rank:<4} "
                  f"top1 = {r.top1} ({r.top1_title})")
            print(f"      {r['query'][:92]}")

    print(f"\nwrote {os.path.join(RES,'error_diagnosis.csv')}")


if __name__ == "__main__":
    main()

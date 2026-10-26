"""
Phase 7e -- where do grounded models still fail: retrieval or reading?

For each query, the retrieved top-5 either contains a section that defines the
gold offence ("hit") or does not ("miss"). Splitting grounded citation accuracy
by that tells us whether residual error comes from the retriever (fixable by a
better retriever) or from the model misreading correct evidence.

Writes: results/error_decomposition.csv
"""

import os
import sys
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "scripts"))
import verif_common as C  # noqa: E402


def main():
    d = pd.read_csv(os.path.join(C.RES, "scale_final_long.csv"))
    ctx = C.load_contexts()
    fam = C.V.defining_sections()
    b = C.load_bench().set_index("query_id")
    hit = {q: any(int(s.split("_")[1]) in fam.get(b.loc[q, "gold"], set())
                  for s in v["retrieved"]) for q, v in ctx.items()}
    d["retrieval_hit"] = d["query_id"].map(hit)
    n_hit = sum(hit.values())
    print(f"retrieval hit (gold family in top-5): {n_hit}/{len(hit)} = {100*n_hit/len(hit):.1f}%  "
          f"<- ceiling for a model that only reads the evidence\n")

    rows = []
    for (m, cond), g in d[d.condition != "none"].groupby(["model", "condition"]):
        h, mi = g[g.retrieval_hit], g[~g.retrieval_hit]
        rows.append({"model": m, "condition": cond,
                     "overall": round(100 * g.sec_family_ok.mean(), 1),
                     "when_retrieved_correct": round(100 * h.sec_family_ok.mean(), 1),
                     "when_retrieval_missed": round(100 * mi.sec_family_ok.mean(), 1),
                     "errors_from_retrieval_miss": int((~mi.sec_family_ok).sum()),
                     "errors_despite_correct_evidence": int((~h.sec_family_ok).sum())})
    t = pd.DataFrame(rows)
    print(t.to_string(index=False))
    t.to_csv(os.path.join(C.RES, "error_decomposition.csv"), index=False)
    print("\nwrote results/error_decomposition.csv")


if __name__ == "__main__":
    main()

"""
Validate that BNS-Facts-135 is not trivially solvable.

Two checks, both of which the original 110-query benchmark fails:

  C1  LEXICAL   -- no query may contain its own gold label, any token of that
                   label, or any term from the original LEGAL_SYNONYMS map that
                   rewrites to it. A benchmark that fails C1 can be solved by
                   string matching.

  C2  EMPIRICAL -- the trivial no-graph baseline (embed query, embed the nine
                   offence names, nearest wins) must NOT score highly. This is
                   the baseline that beat the published system at 93.64% on the
                   original benchmark.

Both benchmarks are run side by side so the comparison is explicit.
"""

import os
import re
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

OLD = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "original_data")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE, "data")

LEGAL_SYNONYMS = {
    "sexual assault": "rape", "homicide": "murder", "killing a person": "murder",
    "forging documents": "forgery", "fake document": "forgery", "fraud": "cheating",
    "abduction": "kidnapping", "kidnap": "kidnapping",
}

# Tokens that are too generic to count as leakage on their own.
STOP = {"sexual"}


def label_cues(label):
    """Surface forms that would let string matching solve the item."""
    cues = {label}
    for t in label.split():
        if t not in STOP:
            cues.add(t)
    for a, b in LEGAL_SYNONYMS.items():
        if b == label:
            cues.add(a)
    return cues


def lexical_leak(query, label):
    q = query.lower()
    hits = []
    for cue in sorted(label_cues(label)):
        if re.search(r"\b" + re.escape(cue) + r"\b", q):
            hits.append(cue)
    return hits


def trivial_baseline(model, queries, gold, labels):
    qe = np.asarray(model.encode(list(queries))).astype("float32")
    le = np.asarray(model.encode(list(labels))).astype("float32")
    d = ((qe[:, None, :] - le[None, :, :]) ** 2).sum(-1)
    pred = [labels[j] for j in d.argmin(1)]
    return np.array([p == g for p, g in zip(pred, gold)]), pred


def report(name, queries, gold, model):
    labels = sorted(set(gold))
    print("=" * 74)
    print(name)
    print("=" * 74)

    leaks = [lexical_leak(q, g) for q, g in zip(queries, gold)]
    n_leak = sum(1 for h in leaks if h)
    print(f"C1 LEXICAL   : {n_leak}/{len(queries)} queries "
          f"({100*n_leak/len(queries):.1f}%) contain a cue for their own label")

    ok, pred = trivial_baseline(model, queries, gold, labels)
    print(f"C2 EMPIRICAL : trivial no-graph baseline scores "
          f"{ok.sum()}/{len(queries)} = {100*ok.mean():.2f}%")

    per = pd.DataFrame({"gold": gold, "ok": ok}).groupby("gold")["ok"]
    print("\n  trivial-baseline accuracy by category:")
    for k, v in per:
        print(f"    {k:<20} {v.sum():3d}/{len(v):<3d} {100*v.mean():6.2f}%")
    print()
    return leaks, ok, pred


def main():
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # --- original benchmark -------------------------------------------------
    o = pd.read_csv(os.path.join(OLD, "evaluation_queries_v2.csv"))
    o.columns = o.columns.str.strip()
    o_gold = o["expected_offence"].astype(str).str.strip().str.lower().tolist()
    o_q = o["query"].astype(str).str.strip().tolist()
    report("ORIGINAL BENCHMARK (evaluation_queries_v2.csv, n=110)", o_q, o_gold, model)

    # --- new benchmark ------------------------------------------------------
    n = pd.read_csv(os.path.join(DATA, "bns_facts_135.csv"))
    n_gold = n["expected_offence"].astype(str).str.strip().str.lower().tolist()
    n_q = n["query"].astype(str).str.strip().tolist()
    leaks, ok, pred = report(
        f"NEW BENCHMARK (bns_facts_135.csv, n={len(n)})", n_q, n_gold, model)

    n["lexical_cues"] = [";".join(h) for h in leaks]
    n["trivial_baseline_correct"] = ok
    n["trivial_baseline_pred"] = pred
    out = os.path.join(DATA, "bns_facts_135_validated.csv")
    n.to_csv(out, index=False, encoding="utf-8")

    bad = n[n["lexical_cues"] != ""]
    if len(bad):
        print("ITEMS FAILING C1 -- rewrite these:")
        for _, r in bad.iterrows():
            print(f"  {r.query_id} [{r.expected_offence}] cue={r.lexical_cues}")
            print(f"      {r['query']}")
    else:
        print("C1 PASSED: no query contains a cue for its own label.")

    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

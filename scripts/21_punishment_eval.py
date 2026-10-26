"""
Phase 4b -- punishment retrieval: the task only the graph can do.

Offence detection was never the knowledge graph's real job. The advisory
question is "what happens to me", and answering it needs the punishment that
the statute prescribes for the identified offence.

A trivial label classifier cannot answer this at all: it returns a category
name, and getting from category to punishment IS the PUNISHED_BY relation. So
the fair comparison is not "graph vs no graph" but:

  direct     retrieve top-k sections, read punishments straight out of their text
  graph      retrieve top-k, expand DEFINES to an offence, then PUNISHED_BY

Ground truth for a query is the punishment set of its gold offence, taken from
the sections that define that offence.

Metric: micro precision / recall / F1 over punishment elements, plus exact set
match. Both arms use the same extractor, so what is being measured is whether
routing through the offence node yields a cleaner punishment set than reading
retrieved sections directly -- not the extractor's own accuracy, which is
reported separately as a spot check.

Writes: results/punishment_eval.csv
"""

import os
import re
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA, RES = os.path.join(HERE, "data"), os.path.join(HERE, "results")

SYN = {"sexual assault": "rape", "homicide": "murder", "killing a person": "murder",
       "forging documents": "forgery", "fake document": "forgery", "fraud": "cheating",
       "abduction": "kidnapping", "kidnap": "kidnapping"}
KS = [1, 3, 5, 10]
DENSE = "BAAI/bge-base-en-v1.5"


def norm(q):
    s = q.lower()
    for a, b in SYN.items():
        s = s.replace(a, b)
    return s


def extract_punishments(body):
    b = " ".join(str(body).lower().split())
    out = set()
    if re.search(r"punished with death|with death\b", b):
        out.add("death")
    if "imprisonment for life" in b:
        out.add("imprisonment for life")
    if re.search(r"\bcommunity service\b", b):
        out.add("community service")
    if re.search(r"\bliable to fine\b|\bwith fine\b|\band fine\b", b):
        out.add("fine")
    for m in re.finditer(r"which may extend to (\w+|\d+)\s+years", b):
        out.add(f"up to {m.group(1)} years imprisonment")
    for m in re.finditer(r"shall not be less than (\w+|\d+)\s+years", b):
        out.add(f"minimum {m.group(1)} years imprisonment")
    return out


def prf(pred, gold):
    if not pred and not gold:
        return 1.0, 1.0, 1.0
    tp = len(pred & gold)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(gold) if gold else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def main():
    os.makedirs(RES, exist_ok=True)
    sec = pd.read_csv(os.path.join(DATA, "bns_act_titled.csv"))
    sec["body"] = sec["body"].fillna("").astype(str)
    ids = sec["section_id"].astype(str).values
    body_of = dict(zip(sec["section_id"].astype(str), sec["body"]))

    dfn = pd.read_csv(os.path.join(DATA, "graph_defines.csv"))
    defines, targets = {}, {}
    for _, r in dfn.iterrows():
        s, o = str(r.section_id).strip(), str(r.offence).strip().lower()
        defines.setdefault(s, set()).add(o)
        targets.setdefault(o, set()).add(s)

    # PUNISHED_BY: offence -> punishment set, from its defining sections
    punished_by = {o: set().union(*[extract_punishments(body_of.get(s, "")) for s in ss])
                   for o, ss in targets.items()}

    print("PUNISHED_BY (ground truth, from statute):")
    for o in sorted(punished_by):
        print(f"  {o:<20} {', '.join(sorted(punished_by[o])) or '(none extracted)'}")

    b = pd.read_csv(os.path.join(DATA, "bns_facts_135.csv"))
    queries = [norm(q) for q in b["query"].astype(str).str.strip()]
    gold = b["expected_offence"].astype(str).str.strip().str.lower().tolist()

    model = SentenceTransformer(DENSE)
    qe = np.asarray(model.encode(queries, batch_size=32)).astype("float32")
    emb = np.asarray(model.encode(sec["body"].tolist(), batch_size=16)).astype("float32")
    idx = faiss.IndexFlatL2(emb.shape[1])
    idx.add(emb)
    _, I = idx.search(qe, max(KS))
    retrieved = [[ids[j] for j in row] for row in I]

    rows = []
    for k in KS:
        for mode in ("direct (no graph)", "graph (DEFINES -> PUNISHED_BY)"):
            P = R = F = EX = 0.0
            empty = 0
            for i, r in enumerate(retrieved):
                g = punished_by.get(gold[i], set())
                if mode.startswith("direct"):
                    pred = set().union(*[extract_punishments(body_of.get(s, ""))
                                         for s in r[:k]] or [set()])
                else:
                    offs = set().union(*([defines.get(s, set()) for s in r[:k]] or [set()]))
                    pred = set().union(*[punished_by.get(o, set()) for o in offs] or [set()])
                if not pred:
                    empty += 1
                p, rc, f = prf(pred, g)
                P += p; R += rc; F += f; EX += (pred == g)
            n = len(retrieved)
            rows.append({"k": k, "system": mode,
                         "precision": round(100 * P / n, 2),
                         "recall": round(100 * R / n, 2),
                         "f1": round(100 * F / n, 2),
                         "exact_set_match": round(100 * EX / n, 2),
                         "empty_answers": empty})

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "punishment_eval.csv"), index=False)

    print("\n" + "=" * 86)
    print("PUNISHMENT RETRIEVAL on BNS-Facts-135 (n=135), BGE encoder")
    print("=" * 86)
    print(f"{'k':<4}{'system':<34}{'prec':>8}{'rec':>8}{'F1':>8}"
          f"{'exact':>8}{'empty':>8}")
    for _, r in df.iterrows():
        print(f"{r.k:<4}{r.system:<34}{r.precision:>7.1f}%{r.recall:>7.1f}%"
              f"{r.f1:>7.1f}%{r.exact_set_match:>7.1f}%{r.empty_answers:>8}")

    print("\nNote: a trivial label classifier cannot answer this task at all -- it")
    print("returns a category name, and mapping category -> punishment is exactly")
    print("the PUNISHED_BY relation. The graph is necessary here, not optional.")

    print("\n--- extractor spot check against the statute ---")
    for s, expect in [("BNS_103", "death / life / fine"),
                      ("BNS_303", "up to 3 years or fine"),
                      ("BNS_308", "up to 7 years and fine"),
                      ("BNS_318", "up to 3-7 years and fine"),
                      ("BNS_78",  "up to 3 years (first) / 5 years and fine")]:
        got = sorted(extract_punishments(body_of.get(s, "")))
        print(f"  {s:<9} expected ~ {expect:<40} extracted: {', '.join(got) or '(none)'}")

    print(f"\nwrote {os.path.join(RES,'punishment_eval.csv')}")


if __name__ == "__main__":
    main()

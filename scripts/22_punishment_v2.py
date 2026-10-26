"""
Phase 4c -- fix the punishment extractor, then re-run the evaluation.

The v1 extractor scanned a section's whole body, so it picked up punishments
that the section MENTIONS rather than PRESCRIBES. Worst case: extortion
(BNS_308) came out carrying "death", because sub-section (3) concerns extortion
by threatening to accuse someone of "an offence punishable with death" -- the
death penalty belongs to the threatened accusation, not to extortion.

This is the same mentions-vs-prescribes confusion that produced the original
keyword DEFINES edges, now showing up one layer down.

v2 only reads sentencing spans: text following "shall be punished with" or
"shall be liable to", truncated at the next sentence boundary. Clauses that are
subordinate to "offence punishable with ..." are excluded explicitly.

Writes: results/punishment_eval_v2.csv, data/graph_punished_by_v2.csv
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

SENTENCING_START = re.compile(
    r"shall be punished (?:with|by)|shall be liable to|"
    r"shall be punishable with", re.I)
# "an offence punishable with death" -- the penalty of some OTHER offence
SUBORDINATE = re.compile(r"offence punishable with[^.;]*", re.I)


def norm(q):
    s = q.lower()
    for a, b in SYN.items():
        s = s.replace(a, b)
    return s


def extract_v1(body):
    b = " ".join(str(body).lower().split())
    return _pick(b)


def extract_v2(body):
    b = " ".join(str(body).lower().split())
    b = SUBORDINATE.sub(" ", b)                    # drop referenced penalties
    out = set()
    for m in SENTENCING_START.finditer(b):
        span = b[m.start(): m.start() + 320]       # the sentencing clause
        span = re.split(r"(?<=[.;])\s", span)[0] + " " + \
               (re.split(r"(?<=[.;])\s", span)[1] if len(re.split(r"(?<=[.;])\s", span)) > 1 else "")
        out |= _pick(span)
    return out


def _pick(b):
    out = set()
    if re.search(r"\bwith death\b|\bdeath\b", b) and "death" in b:
        if re.search(r"punished with death|punishable with death|with death", b):
            out.add("death")
    if "imprisonment for life" in b:
        out.add("imprisonment for life")
    if re.search(r"\bcommunity service\b", b):
        out.add("community service")
    if re.search(r"\bliable to fine\b|\bwith fine\b|\band fine\b|\bor fine\b", b):
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
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def main():
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

    print("EXTRACTOR COMPARISON on sections with known penalties")
    print("=" * 86)
    checks = [("BNS_103", "death / life / fine"),
              ("BNS_308", "up to 7 years + fine (NOT death)"),
              ("BNS_303", "up to 3 years or fine"),
              ("BNS_318", "up to 3-7 years + fine"),
              ("BNS_78",  "up to 3 then 5 years + fine")]
    for s, expect in checks:
        v1 = sorted(extract_v1(body_of.get(s, "")))
        v2 = sorted(extract_v2(body_of.get(s, "")))
        print(f"\n  {s}  (expected ~ {expect})")
        print(f"    v1: {', '.join(v1) or '(none)'}")
        print(f"    v2: {', '.join(v2) or '(none)'}")

    pb2 = {o: set().union(*[extract_v2(body_of.get(s, "")) for s in ss])
           for o, ss in targets.items()}
    pd.DataFrame([{"offence": o, "punishment": p}
                  for o in sorted(pb2) for p in sorted(pb2[o])]
                 ).to_csv(os.path.join(DATA, "graph_punished_by_v2.csv"), index=False)

    print("\n\nPUNISHED_BY v2 (ground truth):")
    for o in sorted(pb2):
        print(f"  {o:<20} {', '.join(sorted(pb2[o])) or '(none)'}")

    b = pd.read_csv(os.path.join(DATA, "bns_facts_135.csv"))
    queries = [norm(q) for q in b["query"].astype(str).str.strip()]
    gold = b["expected_offence"].astype(str).str.strip().str.lower().tolist()

    model = SentenceTransformer(DENSE)
    qe = np.asarray(model.encode(queries, batch_size=32)).astype("float32")
    emb = np.asarray(model.encode(sec["body"].tolist(), batch_size=16)).astype("float32")
    idx = faiss.IndexFlatL2(emb.shape[1]); idx.add(emb)
    _, I = idx.search(qe, max(KS))
    retrieved = [[ids[j] for j in row] for row in I]

    rows = []
    for k in KS:
        for mode in ("direct (no graph)", "graph (DEFINES -> PUNISHED_BY)"):
            P = R = F = EX = 0.0
            for i, r in enumerate(retrieved):
                g = pb2.get(gold[i], set())
                if mode.startswith("direct"):
                    pred = set().union(*[extract_v2(body_of.get(s, ""))
                                         for s in r[:k]] or [set()])
                else:
                    offs = set().union(*([defines.get(s, set()) for s in r[:k]] or [set()]))
                    pred = set().union(*[pb2.get(o, set()) for o in offs] or [set()])
                p, rc, f = prf(pred, g)
                P += p; R += rc; F += f; EX += (pred == g)
            n = len(retrieved)
            rows.append({"k": k, "system": mode, "precision": round(100*P/n, 2),
                         "recall": round(100*R/n, 2), "f1": round(100*F/n, 2),
                         "exact_set_match": round(100*EX/n, 2)})

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "punishment_eval_v2.csv"), index=False)

    print("\n" + "=" * 86)
    print("PUNISHMENT RETRIEVAL, v2 extractor, BNS-Facts-135 (n=135), BGE")
    print("=" * 86)
    print(f"{'k':<4}{'system':<34}{'prec':>9}{'rec':>9}{'F1':>9}{'exact':>9}")
    for _, r in df.iterrows():
        print(f"{r.k:<4}{r.system:<34}{r.precision:>8.1f}%{r.recall:>8.1f}%"
              f"{r.f1:>8.1f}%{r.exact_set_match:>8.1f}%")

    print(f"\nwrote {os.path.join(RES,'punishment_eval_v2.csv')}")


if __name__ == "__main__":
    main()

"""
Phase 7a -- build the grounding contexts for the model-scale x grounding study.

Every model in the study sees exactly the same evidence for a query, so
differences between models are differences in the models, not the retrieval.

For each BNS-Facts-135 v1.1 query:
  rag        top-5 BNS sections retrieved by the best encoder (BGE, BNS-scoped),
             each rendered as number + marginal title + opening text + the
             section's own sentencing clauses, quoted from the Act
  rag+graph  the same, plus the offence candidates the statute graph links to
             those sections and every section that defines each candidate --
             the structured output of the GraphRAG pipeline

Writes: data/grounding_contexts_v11.json
"""

import os
import re
import json
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE, "data")
K = 5
OPENING_CHARS = 600

SYN = {"sexual assault": "rape", "homicide": "murder", "killing a person": "murder",
       "forging documents": "forgery", "fake document": "forgery", "fraud": "cheating",
       "abduction": "kidnapping", "kidnap": "kidnapping"}
NOISE = re.compile(r"\s*(?:\.\s*)?(?:Of\s+[A-Za-z ,'\-]+|\((?:[ivxlc]+|[0-9]+)\))\s*$")


def norm(q):
    s = q.lower()
    for a, b in SYN.items():
        s = s.replace(a, b)
    return s


def clean_title(t):
    t = " ".join(str(t).split())
    for _ in range(3):
        t2 = NOISE.sub("", t).strip().rstrip(".")
        if t2 == t:
            break
        t = t2
    return t


def sentencing(body, limit=2):
    b = " ".join(str(body).split())
    spans = []
    for m in re.finditer(r"shall be punished|be punished on", b):
        end = b.find(".", m.start())
        spans.append(b[max(0, m.start() - 60): (end + 1 if end > 0 else m.start() + 260)].strip())
        if len(spans) >= limit:
            break
    return spans


def render(sid, title, body):
    n = sid.split("_")[1]
    b = " ".join(str(body).split())
    opening = b[:OPENING_CHARS] + ("..." if len(b) > OPENING_CHARS else "")
    lines = [f"BNS section {n} -- {clean_title(title)}", opening]
    for sp in sentencing(body):
        if sp[:80] not in opening:
            lines.append(f"[sentencing] ...{sp}")
    return "\n".join(lines)


def main():
    sec = pd.read_csv(os.path.join(DATA, "bns_act_titled.csv"))
    sec["body"] = sec["body"].fillna("").astype(str)
    sec["title"] = sec["title"].fillna("").astype(str)
    ids = sec["section_id"].astype(str).values
    title_of = dict(zip(ids, sec["title"]))
    body_of = dict(zip(ids, sec["body"]))

    dfn = pd.read_csv(os.path.join(DATA, "graph_defines.csv"))
    defines, sections_of = {}, {}
    for _, r in dfn.iterrows():
        s, o = str(r.section_id), str(r.offence).lower()
        defines.setdefault(s, set()).add(o)
        sections_of.setdefault(o, set()).add(int(s.split("_")[1]))

    b = pd.read_csv(os.path.join(DATA, "bns_facts_135_v1_1.csv"))
    gold = b["expected_offence"].str.strip().str.lower().tolist()

    m = SentenceTransformer("BAAI/bge-base-en-v1.5")
    qe = np.asarray(m.encode([norm(q) for q in b["query"]], batch_size=32)).astype("float32")
    emb = np.asarray(m.encode(sec["body"].tolist(), batch_size=16)).astype("float32")
    idx = faiss.IndexFlatL2(emb.shape[1]); idx.add(emb)
    _, I = idx.search(qe, K)

    out, hit_section, hit_candidate = {}, 0, 0
    for i, row in enumerate(I):
        top = [ids[j] for j in row]
        rag = "\n\n".join(render(s, title_of[s], body_of[s]) for s in top)
        cands = sorted(set().union(*[defines.get(s, set()) for s in top]))
        if cands:
            graph = "\n".join(
                f"- {o}: defined in BNS sections "
                f"{', '.join(str(n) for n in sorted(sections_of[o]))}" for o in cands)
        else:
            graph = "- (no offence in the statute graph is linked to these sections)"
        out[b.iloc[i]["query_id"]] = {
            "retrieved": top,
            "candidates": cands,
            "rag": rag,
            "graph": graph,
        }
        fam = sections_of.get(gold[i], set())
        hit_section += any(int(s.split("_")[1]) in fam for s in top)
        hit_candidate += gold[i] in cands

    path = os.path.join(DATA, "grounding_contexts_v11.json")
    json.dump(out, open(path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    lens = [len(v["rag"]) for v in out.values()]
    print(f"contexts for {len(out)} queries, top-{K} BNS sections each")
    print(f"gold offence family present in retrieved sections : {hit_section}/{len(out)}")
    print(f"gold offence among graph candidates               : {hit_candidate}/{len(out)}")
    print(f"rag context length: mean {np.mean(lens):.0f} chars, max {max(lens)}")
    k0 = b.iloc[0]["query_id"]
    print(f"\nexample ({k0}): {b.iloc[0]['query']}\n")
    print(out[k0]["rag"][:900])
    print("\n[graph block]\n" + out[k0]["graph"])
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()

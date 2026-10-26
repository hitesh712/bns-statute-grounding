"""
Phase 1b -- give every Act section its marginal title.

The Act's section BODIES are already available (the original project extracted
them from the Act into chunks/legal_chunks.csv). What is missing is each
section's marginal title, which is what tells us the subject of a section.

The PRS copy is the Bill, whose clause numbering differs from the Act by a
varying offset, but whose TEXT is near-identical. So we align Act sections to
Bill clauses by body-text similarity and carry the title across. Alignment is
verified against sections whose Act numbering is independently known.

Input :  old project chunks/legal_chunks.csv   (Act bodies)
         data/bns_sections.csv                 (Bill clauses + titles)
Output:  data/bns_act_titled.csv               section_id, title, body, match_score
"""

import os
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

OLD = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "original_data")
DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

# Act section -> expected title fragment. Used only to verify the alignment,
# never to build it.
GROUND_TRUTH = {
    63: "rape", 64: "punishment for rape", 70: "gang rape",
    75: "sexual harassment", 78: "stalking",
    101: "murder", 103: "punishment for murder",
    105: "culpable homicide not amounting to murder",
    303: "theft", 304: "snatching", 308: "extortion", 309: "robbery",
    310: "dacoity", 316: "criminal breach of trust", 318: "cheating",
    336: "forgery", 351: "criminal intimidation",
}


def norm(s):
    s = str(s).lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return " ".join(s.split())


def main():
    chunks = pd.read_csv(os.path.join(OLD, "chunks", "legal_chunks.csv"))
    chunks["chunk_id"] = chunks["chunk_id"].astype(str).str.strip()
    act = chunks[chunks["chunk_id"].str.startswith("BNS_")].copy()
    act["section_no"] = act["chunk_id"].str.replace("BNS_", "", regex=False).astype(int)
    act = act.drop_duplicates("section_no").sort_values("section_no")
    act["body"] = act["content"].fillna("").map(norm)

    bill = pd.read_csv(os.path.join(DATA, "bns_sections.csv"))
    bill["body_n"] = bill["body"].fillna("").map(norm)
    bill = bill[bill["body_n"].str.len() > 40].reset_index(drop=True)

    print(f"Act sections with text : {len(act)}")
    print(f"Bill clauses with text : {len(bill)}")

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=200000)
    X = vec.fit_transform(list(act["body"]) + list(bill["body_n"]))
    A, B = X[: len(act)], X[len(act):]

    sim = cosine_similarity(A, B)
    best = sim.argmax(1)
    score = sim.max(1)

    out = pd.DataFrame({
        "section_id": act["chunk_id"].values,
        "section_no": act["section_no"].values,
        "title": bill.iloc[best]["title"].values,
        "bill_clause_no": bill.iloc[best]["section_no"].values,
        "match_score": score.round(4),
        "body": act["content"].values,
    })

    strong = (out.match_score >= 0.55).sum()
    print(f"\nalignment: {strong}/{len(out)} sections matched at score >= 0.55 "
          f"({100*strong/len(out):.1f}%)")
    print(f"median score: {out.match_score.median():.3f}")

    out.loc[out.match_score < 0.55, "title"] = ""

    print("\nVERIFICATION against independently known Act numbering:")
    ok = bad = 0
    for n, frag in sorted(GROUND_TRUTH.items()):
        r = out[out.section_no == n]
        if not len(r):
            print(f"  BNS_{n:<4} MISSING"); bad += 1; continue
        t = str(r.iloc[0]["title"]).lower()
        hit = frag in t
        ok, bad = (ok + 1, bad) if hit else (ok, bad + 1)
        print(f"  {'OK ' if hit else 'BAD'} BNS_{n:<4} score={r.iloc[0]['match_score']:.2f}  "
              f"{r.iloc[0]['title'][:62]}")
    print(f"\nverification: {ok} correct / {ok+bad} checked")

    path = os.path.join(DATA, "bns_act_titled.csv")
    out.to_csv(path, index=False, encoding="utf-8")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()

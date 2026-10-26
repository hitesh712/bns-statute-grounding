"""
BNS-Facts-135 v1.1 -- correct gold-label and drafting defects in v1.0.

Prompted by the zero-shot LLM baseline: of Opus 5.5's three "errors" on v1.0,
all three were benchmark faults. A follow-up audit against the statute found
more. v1.0 is kept unchanged (data/bns_facts_135.csv) for provenance; v1.1 is
written alongside with a changelog.

Defect classes fixed:
  NO-ACT   the item states circumstances (ages, "common intention") but no act,
           so no offence is determinable from the text.
  LABEL    the gold label contradicts the statute.

Writes: data/bns_facts_135_v1_1.csv, data/bns_facts_135_v1_1_changelog.csv
"""

import os
import re
import pandas as pd

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

# query_id -> (new query, new grounding section, defect class, reason)
FIXES = {
    "F013": ("A pickpocket quietly lifted my wallet from my back pocket on a crowded "
             "platform without my noticing.", "BNS_303", "LABEL",
             "v1.0 'delivery agent kept the parcel' is dishonest misappropriation "
             "(BNS 314) or criminal breach of trust (BNS 316) -- the property was "
             "entrusted, not taken -- so it is not theft."),
    "F054": ("A seller took full payment for a used car after falsely assuring me it "
             "had never been in an accident.", "BNS_318", "LABEL",
             "v1.0 'gave me counterfeit currency as change' is using counterfeit "
             "currency as genuine (BNS 179), in the counterfeiting/forgery family, "
             "not cheating. Opus 5.5 answered 'forgery'."),
    "F092": ("An adult man had sexual intercourse with a girl aged fourteen, who he "
             "says agreed to it.", "BNS_65", "NO-ACT",
             "v1.0 stated only the parties' ages. Opus 5.5 declined to classify, "
             "correctly noting no act was described."),
    "F094": ("A man forced a woman into sexual intercourse after locking her in a "
             "room.", "BNS_64", "LABEL",
             "v1.0 grounded in BNS 69 (intercourse on a false promise of marriage), "
             "whose text says it is 'not amounting to the offence of rape'."),
    "F097": ("A man had sexual intercourse with a girl who was below sixteen years "
             "of age.", "BNS_65", "NO-ACT",
             "v1.0 stated only the victim's age, no act."),
    "F099": ("A man pinned a woman down and had intercourse with her despite her "
             "resistance.", "BNS_64", "LABEL",
             "v1.0 grounded in BNS 69 (deceit as to identity); BNS 69 is not rape."),
    "F100": ("Several men acting together had intercourse with a woman against her "
             "will.", "BNS_70", "NO-ACT",
             "v1.0 stated 'acted in furtherance of a common intention' with no act."),
    "F102": ("The man went beyond unwanted touching and had intercourse with the "
             "woman against her will.", "BNS_64", "LABEL",
             "v1.0 grounded in BNS 69 (false promise of marriage), not rape under "
             "the 2023 code. Opus 5.5 answered 'cheating'."),
    "F104": ("The accused had intercourse with a fifteen-year-old girl and says she "
             "consented.", "BNS_65", "NO-ACT",
             "v1.0 'claims consent, but the complainant was fifteen' implied but did "
             "not state the act."),
}

LEGAL_SYNONYMS = {
    "sexual assault": "rape", "homicide": "murder", "killing a person": "murder",
    "forging documents": "forgery", "fake document": "forgery", "fraud": "cheating",
    "abduction": "kidnapping", "kidnap": "kidnapping",
}
STOP = {"sexual"}


def cues(label):
    c = {label} | {t for t in label.split() if t not in STOP}
    c |= {a for a, b in LEGAL_SYNONYMS.items() if b == label}
    return c


def main():
    v10 = pd.read_csv(os.path.join(DATA, "bns_facts_135.csv"))
    v11 = v10.copy()
    log = []
    for qid, (q, sec, cls, why) in FIXES.items():
        i = v11.index[v11.query_id == qid]
        assert len(i) == 1, qid
        i = i[0]
        log.append({"query_id": qid, "gold": v11.at[i, "expected_offence"],
                    "defect": cls, "old_query": v11.at[i, "query"],
                    "old_grounding": v11.at[i, "grounding_section"],
                    "new_query": q, "new_grounding": sec, "reason": why})
        v11.at[i, "query"] = q
        v11.at[i, "grounding_section"] = sec

    # lexical-leakage check on the edited items
    bad = []
    for qid in FIXES:
        r = v11[v11.query_id == qid].iloc[0]
        hits = [c for c in cues(r.expected_offence)
                if re.search(r"\b" + re.escape(c) + r"\b", r["query"].lower())]
        if hits:
            bad.append((qid, hits))
    if bad:
        raise SystemExit(f"edited items leak their label: {bad}")

    v11.to_csv(os.path.join(DATA, "bns_facts_135_v1_1.csv"), index=False, encoding="utf-8")
    pd.DataFrame(log).to_csv(os.path.join(DATA, "bns_facts_135_v1_1_changelog.csv"),
                             index=False, encoding="utf-8")

    print(f"v1.1 written: {len(v11)} items, {len(log)} changed, labels unchanged")
    print(pd.DataFrame(log).groupby(["gold", "defect"]).size().to_string())
    print("\nC1 lexical check on edited items: PASSED")
    print(f"grounding sections no longer used: "
          f"{sorted(set(v10.grounding_section) - set(v11.grounding_section))}")


if __name__ == "__main__":
    main()

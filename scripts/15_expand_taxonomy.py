"""
Phase 3 -- expand the offence taxonomy past the nine hand-picked categories.

Which sections are offence sections is decided by the statute itself: a section
creates an offence when its body carries sentencing language ("shall be
punished with ...", "shall also be liable to fine"). Definitional, procedural
and exception sections do not.

The offence NAME comes from the marginal title, with the standard statutory
prefixes stripped ("Punishment for X" -> X, "Attempt to commit X" -> X).

Each fine-grained offence is also mapped to one of the nine coarse benchmark
categories where it belongs, so the expanded graph stays comparable with
BNS-Facts-135.

Output: data/offence_taxonomy.csv, data/graph_defines_expanded.csv
"""

import os
import re
import pandas as pd

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

PUNISHMENT_RE = re.compile(
    r"shall be punished|shall also be liable to fine|punishable with|"
    r"shall be liable to fine|with imprisonment", re.I)

PREFIXES = [
    r"^punishment for\s+", r"^punishment of\s+", r"^attempt to commit\s+",
    r"^attempt to\s+", r"^abetment of\s+", r"^abetting\s+",
    r"^making or possessing\s+", r"^having possession of\s+",
    r"^using as genuine,?\s+", r"^possession of\s+",
]

COARSE_TERMS = {
    "theft":             [r"\btheft\b", r"\bsnatching\b"],
    "robbery":           [r"\brobbery\b", r"\bdacoity\b"],
    "extortion":         [r"\bextortion\b"],
    "cheating":          [r"\bcheating\b"],
    "forgery":           [r"\bforgery\b", r"\bforged\b", r"\bforging\b",
                          r"\bfalse document\b", r"\bfalse electronic record\b"],
    "murder":            [r"\bmurder\b", r"\bculpable homicide\b"],
    "rape":              [r"\brape\b"],
    "sexual harassment": [r"\bsexual harassment\b"],
    "stalking":          [r"\bstalking\b"],
}

NOISE = re.compile(
    r"\s*(?:\.\s*)?(?:Of\s+[A-Za-z ,'\-]+|\((?:[ivxlc]+|[0-9]+)\)|CHAPTER\s+[IVXLC]+)\s*$")


def clean_title(t):
    t = " ".join(str(t).split())
    for _ in range(3):
        t2 = NOISE.sub("", t).strip().rstrip(".")
        if t2 == t:
            break
        t = t2
    return t.strip().rstrip(".")


def offence_name(title):
    s = clean_title(title).lower()
    for p in PREFIXES:
        s = re.sub(p, "", s)
    s = re.sub(r"\b(etc|and punishment for.*)$", "", s).strip(" ,.;")
    s = re.sub(r"\s*,\s*(when|where|with|after|in certain cases).*$", "", s)
    return " ".join(s.split())


def coarse_of(title):
    t = clean_title(title).lower()
    for cat, pats in COARSE_TERMS.items():
        if any(re.search(p, t) for p in pats):
            return cat
    return ""


def main():
    df = pd.read_csv(os.path.join(DATA, "bns_act_titled.csv"))
    df["title_clean"] = df["title"].map(clean_title)
    df["body"] = df["body"].fillna("")

    df["is_offence"] = df["body"].map(lambda b: bool(PUNISHMENT_RE.search(str(b))))
    df["offence"] = df["title_clean"].map(offence_name)
    df["coarse"] = df["title_clean"].map(coarse_of)

    off = df[df.is_offence & (df.offence != "") & (df.title_clean != "")].copy()

    print(f"BNS sections with a title      : {(df.title_clean!='').sum()}")
    print(f"sections with sentencing text  : {df.is_offence.sum()}")
    print(f"offence sections kept          : {len(off)}")
    print(f"distinct fine-grained offences : {off.offence.nunique()}")
    print(f"mapped to a coarse category    : {(off.coarse!='').sum()}")

    tax = (off.groupby("offence")
             .agg(sections=("section_id", lambda s: ", ".join(sorted(
                      s, key=lambda x: int(x.split('_')[1])))),
                  n_sections=("section_id", "count"),
                  coarse=("coarse", lambda c: sorted(set(x for x in c if x))[:1]))
             .reset_index())
    tax["coarse"] = tax["coarse"].map(lambda v: v[0] if v else "")
    tax = tax.sort_values("n_sections", ascending=False)
    tax.to_csv(os.path.join(DATA, "offence_taxonomy.csv"), index=False, encoding="utf-8")

    exp = off[["section_id", "offence", "coarse", "title_clean"]].rename(
        columns={"title_clean": "title"})
    exp.to_csv(os.path.join(DATA, "graph_defines_expanded.csv"),
               index=False, encoding="utf-8")

    old = pd.read_csv(os.path.join(DATA, "graph_defines.csv"))
    print(f"\ncoverage, 9-category graph : {old.section_id.nunique()} sections")
    print(f"coverage, expanded graph   : {exp.section_id.nunique()} sections "
          f"({100*exp.section_id.nunique()/len(df):.1f}% of BNS)")

    print("\nlargest offence families:")
    print(tax.head(18)[["offence", "n_sections", "coarse"]].to_string(index=False))

    print("\ncoarse-category coverage in the expanded graph:")
    print(exp[exp.coarse != ""].groupby("coarse")["section_id"]
          .nunique().to_string())

    print(f"\nwrote {os.path.join(DATA,'offence_taxonomy.csv')}")
    print(f"wrote {os.path.join(DATA,'graph_defines_expanded.csv')}")


if __name__ == "__main__":
    main()

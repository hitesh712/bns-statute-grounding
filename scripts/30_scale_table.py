"""
Phase 7d -- final model-scale x grounding table, all models scored identically.

Re-scores every model's saved outputs with one scorer:
  Opus 5.5 (none)            results/verifiability_outputs.csv
  Haiku 4.5 (3 conditions)   results/scale_haiku_outputs.csv
  Llama 3.2 3B, Llama 3.1 8B,
  Qwen 2.5 7B (3 conditions) results/scale_local_outputs.csv

The maximum-imprisonment parser is made robust to formats small models produce
in place of the requested "digits or life" -- number words ("six", "in fourteen
years") and "infinity"/"indefinite" for imprisonment for life. It is applied to
every model alike, and the share of answers in the requested format is reported
separately as `max_format_ok`, so format non-compliance stays visible rather
than being silently forgiven.

Writes: results/scale_final_table.csv, results/scale_final_long.csv
"""

import os
import re
import sys
import json
import numpy as np
import pandas as pd
from scipy.stats import binomtest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "scripts"))
import verif_common as C  # noqa: E402

WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
         "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
         "fourteen": 14, "fifteen": 15, "twenty": 20}
MODEL_ORDER = ["claude-opus-5-5", "claude-haiku-4-5", "qwen2.5:7b", "llama3.1:8b", "llama3.2:3b"]


def parse_max(s):
    t = str(s).lower().strip()
    if t in ("", "nan", "none", "?"):
        return "?"
    if re.search(r"life|infinit|indefinite|remainder of", t):
        return "life"
    if re.search(r"month", t) and not re.search(r"year", t):
        return "<1"
    m = re.search(r"\d+", t)
    if m:
        return m.group(0)
    for w, n in WORDS.items():
        if re.search(rf"\b{w}\b", t):
            return str(n)
    return "?"


def format_ok(s):
    return bool(re.fullmatch(r"\s*(\d+|life)\s*", str(s), re.I))


def load_all():
    frames = []
    o = pd.read_csv(os.path.join(C.RES, "verifiability_outputs.csv"))
    raw_max = []
    for txt, parsed in zip(o["cit_raw"], o["cit_max"]):
        j = C.parse_json(txt)
        raw_max.append(j.get("max_imprisonment", parsed))
    frames.append(pd.DataFrame({
        "model": "claude-opus-5-5", "condition": "none", "query_id": o["query_id"],
        "gold": o["gold"], "grounding": o["grounding"], "offence": o["cit_offence"],
        "section": o["cit_section"], "max_imprisonment": raw_max,
        "death_possible": o["cit_death"]}))

    h = pd.read_csv(os.path.join(C.RES, "scale_haiku_outputs.csv"))
    frames.append(h[["model", "condition", "query_id", "gold", "grounding", "offence",
                     "section", "max_imprisonment", "death_possible"]])

    l = pd.read_csv(os.path.join(C.RES, "scale_local_outputs.csv"))
    frames.append(l[["model", "condition", "query_id", "gold", "grounding", "offence",
                     "section", "max_imprisonment", "death_possible"]])
    return pd.concat(frames, ignore_index=True)


def mcnemar(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n10, n01 = int((a & ~b).sum()), int((~a & b).sum())
    return n10, n01, (binomtest(n10, n10 + n01, 0.5).pvalue if n10 + n01 else 1.0)


def main():
    raw = load_all()
    raw["max_format_ok"] = raw["max_imprisonment"].map(format_ok)
    C.V.parse_max = parse_max                      # scorer uses V.parse_max
    d = C.score(raw)
    d.to_csv(os.path.join(C.RES, "scale_final_long.csv"), index=False, encoding="utf-8")

    rows = []
    for (m, cond), g in d.groupby(["model", "condition"]):
        s = C.summarise(g)
        s["max_format_ok"] = round(100 * g["max_format_ok"].mean(), 2)
        top = g["section"].value_counts()
        s["top_citation"] = f"{int(top.index[0])} ({100 * top.iloc[0] / len(g):.0f}%)"
        s.update(model=m, condition=cond)
        rows.append(s)
    t = pd.DataFrame(rows)
    t["model"] = pd.Categorical(t["model"], MODEL_ORDER, ordered=True)
    t["condition"] = pd.Categorical(t["condition"], list(C.CONDITIONS), ordered=True)
    t = t.sort_values(["model", "condition"]).set_index(["model", "condition"])
    t.to_csv(os.path.join(C.RES, "scale_final_table.csv"))

    cols = ["offence", "section_family", "section_exact", "ipc_confusion", "nonexistent",
            "max_imprisonment", "max_format_ok", "death", "top_citation"]
    pd.set_option("display.width", 200)
    print("MODEL SCALE x GROUNDING -- BNS-Facts-135 v1.1, n=135 per cell (%)\n")
    print(t[cols].to_string())

    print("\nExact McNemar on BNS-section correctness (family):")
    for m in MODEL_ORDER[1:]:
        piv = {c: d[(d.model == m) & (d.condition == c)].set_index("query_id").sort_index()
               for c in C.CONDITIONS}
        for x, y in (("none", "rag"), ("none", "rag+graph"), ("rag", "rag+graph")):
            n10, n01, p = mcnemar(piv[x].sec_family_ok, piv[y].sec_family_ok)
            print(f"  {m:<17} {x:>9} vs {y:<9} {x}-only={n10:3d} {y}-only={n01:3d}  p={p:.2g}")

    print("\nwrote results/scale_final_table.csv, scale_final_long.csv")


if __name__ == "__main__":
    main()

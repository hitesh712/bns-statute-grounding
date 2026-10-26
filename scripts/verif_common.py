"""
Shared prompts and scoring for the model-scale x grounding study.

Constants (IPC map, sentencing tiers, death-penalty table, citation schema and
prompt) are taken from 26_llm_verifiability.py so that every model -- Opus,
Haiku, open local models -- is prompted and graded identically.
"""

import os
import re
import json
import importlib.util
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA, RES = os.path.join(HERE, "data"), os.path.join(HERE, "results")

_s = importlib.util.spec_from_file_location("v26", os.path.join(HERE, "scripts", "26_llm_verifiability.py"))
V = importlib.util.module_from_spec(_s)
_s.loader.exec_module(V)

LABELS, CIT_SYSTEM, CIT_SCHEMA = V.LABELS, V.CIT_SYSTEM, V.CIT_SCHEMA
IPC, TIERS, DEATH = V.IPC, V.TIERS, V.DEATH
BNS_MAX_SECTION = V.BNS_MAX_SECTION
CONDITIONS = ("none", "rag", "rag+graph")

GROUNDED_SUFFIX = ("\n\nBase your answer on the BNS sections quoted above where they "
                   "apply. If none of them fits, rely on your own knowledge of the BNS.")


def load_bench():
    b = pd.read_csv(os.path.join(DATA, "bns_facts_135_v1_1.csv"))
    b["query"] = b["query"].astype(str).str.strip()
    b["gold"] = b["expected_offence"].astype(str).str.strip().str.lower()
    return b


def load_contexts():
    return json.load(open(os.path.join(DATA, "grounding_contexts_v11.json"), encoding="utf-8"))


def user_prompt(query, condition, ctx):
    base = V.cit_user(query)
    if condition == "none":
        return base
    block = "Relevant sections retrieved from the Bharatiya Nyaya Sanhita, 2023:\n\n" + ctx["rag"]
    if condition == "rag+graph":
        block += ("\n\nOffences that the statute graph links to these sections:\n"
                  + ctx["graph"])
    return block + "\n\n" + base + GROUNDED_SUFFIX


def parse_json(text):
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        m = re.search(r"\{.*\}", str(text), re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    return {}


def to_int(x):
    try:
        return int(str(x).strip())
    except (ValueError, TypeError):
        m = re.search(r"\d+", str(x))
        return int(m.group(0)) if m else -1


def score(df):
    """df needs: gold, grounding, offence, section, max_imprisonment, death_possible."""
    fam = V.defining_sections()
    d = df.copy()
    d["section"] = d["section"].map(to_int)
    d["max_imp"] = d["max_imprisonment"].map(V.parse_max)
    g_no = d["grounding"].map(lambda s: int(str(s).split("_")[1]))

    d["offence_ok"] = d["offence"].astype(str).str.lower() == d["gold"]
    d["sec_family_ok"] = [s in fam.get(g, set()) | {gn}
                          for s, g, gn in zip(d.section, d.gold, g_no)]
    d["sec_exact_ok"] = d["section"] == g_no
    d["ipc_confusion"] = [s in IPC.get(g, set()) for s, g in zip(d.section, d.gold)]
    d["nonexistent"] = (d["section"] > BNS_MAX_SECTION) | (d["section"] < 1)

    def allowed(gr, cited, gold):
        a = set(TIERS.get(gr, set()))
        if cited in fam.get(gold, set()) and f"BNS_{cited}" in TIERS:
            a |= TIERS[f"BNS_{cited}"]
        return a
    d["max_ok"] = [m in allowed(gr, c, g) for m, gr, c, g in
                   zip(d.max_imp, d.grounding, d.section, d.gold)]
    d["death_scored"] = d["grounding"].isin(DEATH)
    d["death_ok"] = [(str(x).lower() == "true") == DEATH[gr] if gr in DEATH else np.nan
                     for x, gr in zip(d.death_possible, d.grounding)]
    return d


def summarise(d):
    pct = lambda x: round(100 * float(np.nanmean(np.asarray(x, dtype=float))), 2)
    return {
        "n": len(d),
        "offence": pct(d.offence_ok),
        "section_family": pct(d.sec_family_ok),
        "section_exact": pct(d.sec_exact_ok),
        "ipc_confusion": pct(d.ipc_confusion),
        "nonexistent": pct(d.nonexistent),
        "max_imprisonment": pct(d.max_ok),
        "death": pct(d.death_ok[d.death_scored]),
    }

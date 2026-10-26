"""
Phase 7c -- open-weight local models across the three grounding conditions.

Local models are the realistic deployment for a free legal-aid tool in India:
no per-query cost, no data leaving the machine, runs offline. If they cite IPC
numbers or wrong BNS sections where Opus does not, and statute grounding fixes
that, retrieval earns its place exactly where deployment happens.

Served by a dedicated Ollama instance on port 11435 (override with the
OLLAMA_HOST_URL environment variable), separate from any default Ollama setup.

Deterministic decoding (temperature 0, fixed seed); JSON-schema constrained
output, same schema and prompts as the Claude runs. Resumable: completed
(model, condition, query) triples are skipped on restart.

Usage:  python 29_local_scale.py [--models llama3.1:8b qwen2.5:7b llama3.2:3b]
"""

import os
import sys
import json
import time
import argparse
import urllib.request
import urllib.error
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "scripts"))
import verif_common as C  # noqa: E402

HOST = os.environ.get("OLLAMA_HOST_URL", "http://127.0.0.1:11435")
OUT = os.path.join(C.RES, "scale_local_outputs.csv")
DEFAULT_MODELS = ["llama3.1:8b", "qwen2.5:7b", "llama3.2:3b"]


def chat(model, system, user, timeout=300):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "stream": False,
        "format": C.CIT_SCHEMA,
        "options": {"temperature": 0, "seed": 0, "num_ctx": 4096},
    }).encode("utf-8")
    req = urllib.request.Request(f"{HOST}/api/chat", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    a = ap.parse_args()

    b = C.load_bench()
    ctx = C.load_contexts()
    done = set()
    if os.path.exists(OUT):
        prev = pd.read_csv(OUT)
        done = set(zip(prev.model, prev.condition, prev.query_id))
    total = len(a.models) * len(C.CONDITIONS) * len(b)
    print(f"{total} calls planned, {len(done)} already done")

    for model in a.models:
        for cond in C.CONDITIONS:
            t0, n = time.time(), 0
            for r in b.itertuples():
                if (model, cond, r.query_id) in done:
                    continue
                prompt = C.user_prompt(r.query, cond, ctx[r.query_id])
                try:
                    res = chat(model, C.CIT_SYSTEM, prompt)
                    txt = res.get("message", {}).get("content", "")
                    err = ""
                except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                    txt, res, err = "", {}, f"{type(e).__name__}: {e}"
                j = C.parse_json(txt)
                row = {"model": model, "condition": cond, "query_id": r.query_id,
                       "gold": r.gold, "grounding": r.grounding_section,
                       "offence": j.get("offence", ""), "section": j.get("bns_section", -1),
                       "max_imprisonment": j.get("max_imprisonment", ""),
                       "death_possible": j.get("death_possible", ""),
                       "prompt_tokens": res.get("prompt_eval_count", 0),
                       "raw": txt[:200], "error": err}
                pd.DataFrame([row]).to_csv(OUT, mode="a", header=not os.path.exists(OUT),
                                           index=False, encoding="utf-8")
                n += 1
            el = time.time() - t0
            print(f"  {model:<14} {cond:<10} {n:3d} calls in {el/60:5.1f} min "
                  f"({el/max(n,1):.1f} s/call)", flush=True)

    d = C.score(pd.read_csv(OUT))
    rows = []
    for (m, cond), g in d.groupby(["model", "condition"]):
        s = C.summarise(g); s.update(model=m, condition=cond); rows.append(s)
    summ = pd.DataFrame(rows).set_index(["model", "condition"])
    print("\n" + summ.to_string())
    d.to_csv(os.path.join(C.RES, "scale_local_scored.csv"), index=False, encoding="utf-8")


if __name__ == "__main__":
    main()

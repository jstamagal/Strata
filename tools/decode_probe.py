#!/usr/bin/env python3
"""decode_probe.py <config.json> <name> [tokens] [--out DIR]: a repeatable decode measurement.

Starts the server on the config and sends the three fixed chat questions of tools/ab_engine.py for 400 new tokens each,
once greedy (temperature 0) and once sampled with a fixed seed (temperature 1.0, top-p 0.95, top-k 20, seed 7), then a
copy-heavy request (reprint a file with one rename, greedy) and one fixed prompt of about 3.2K tokens. Same questions,
same seed, same order every time, so two runs of one config give the same text, and their tok/s agree far closer than
tools/strata_bench.py's tagged, unseeded chats do (on 2x Radeon PRO V620 with every expert in VRAM: to 0.1 tok/s).

One line per run is appended to DIR/probe-rows.txt: per-question tok/s, pooled tok/s, drafts kept, tokens generated and
the texts' hashes (same hash = same text), the copy-heavy tok/s and the prompt tok/s. DIR also gets the run's part of
the engine log, the server's output and the raw rows as JSON. DIR defaults to bench/probe (not tracked)."""

import hashlib, json, re, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
argv = [a for a in sys.argv[1:] if a != "--out"]
WARM = "--warm" in sys.argv  # one untimed 400-token answer first: the first long answer after a load runs fast or slow by up to 15%
if WARM:
    argv.remove("--warm")
OUT = Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv else ROOT / "bench" / "probe"
if "--out" in sys.argv:
    argv.remove(str(OUT))
sys.path.insert(0, str(ROOT / "tools"))
import strata_bench as sb
from ab_engine import PROMPTS, long_prompt

cfgp, name = Path(argv[0]).resolve(), argv[1]
tokens = int(argv[2]) if len(argv) > 2 else 400
cfg = json.loads(cfgp.read_text())
base = f"http://127.0.0.1:{cfg.get('port', 8080)}"
L = OUT
L.mkdir(parents=True, exist_ok=True)
slog = L / f"probe-{name}.server.log"
elog = Path(cfg["log"])
off = elog.stat().st_size if elog.exists() else 0
proc = sb.start_server(cfgp, cfg.get("port", 8080), None, slog)
rows, err = [], ""
try:
    if not sb.wait_ready(base, "", proc, 900, log=lambda *x: None):
        err = "server did not come up"
    else:
        ready = sb.chat(base, "", [{"role": "user", "content": "Say hi."}], 16)
        if sb.digest(ready)["err"]:
            raise RuntimeError("probe warmup failed: " + str(sb.digest(ready)["err"]))
        if WARM:
            sb.chat(base, "", [{"role": "user", "content": "Explain how a hash table works, with one example."}], tokens)
        for label, temp, samp in (("T0", 0.0, {}), ("T1", 1.0, {"top_p": 0.95, "top_k": 20, "seed": 7})):
            sb.TEMPERATURE, sb.SAMPLING = temp, dict(samp)
            for qi, q in enumerate(PROMPTS):
                r = sb.chat(base, "", [{"role": "user", "content": q}], tokens)
                d = sb.digest(r)
                rows.append({"arm": label, "q": qi, "tps": d["decode_tps"], "gen": d["generated_tokens"], "offered": d["drafts_offered"],
                             "accepted": d["drafts_accepted"], "err": d["err"],
                             "sha": hashlib.sha256((r.get("text") or "").encode()).hexdigest()[:8]})
        sb.TEMPERATURE, sb.SAMPLING = 0.0, {}
        # copy-heavy answer (what an agent's file edit looks like): suffix / lookup drafts only have something to match here
        src = (ROOT / "tools" / "ab_engine.py").read_text()[:2400]
        r = sb.chat(base, "", [{"role": "user", "content": "Print this file again exactly as it is, but rename every `PROMPTS` to `QUESTIONS`. Output only the file.\n\n```python\n" + src + "\n```"}], tokens)
        d = sb.digest(r)
        rows.append({"arm": "C", "q": "copy", "tps": d["decode_tps"], "gen": d["generated_tokens"], "offered": d["drafts_offered"],
                     "accepted": d["drafts_accepted"], "err": d["err"],
                     "sha": hashlib.sha256((r.get("text") or "").encode()).hexdigest()[:12], "text": r.get("text") or ""})
        r = sb.chat(base, "", [{"role": "user", "content": long_prompt(3000)}], 24)
        d = sb.digest(r)
        rows.append({"arm": "P", "q": "4K", "tps": d["prompt_tps"], "gen": d["prompt_tokens"], "err": d["err"]})
finally:
    sb.stop_server(proc)
eng = ""
if elog.exists():
    with open(elog, "rb") as fh:
        fh.seek(off)
        eng = fh.read().decode(errors="replace")
(L / f"probe-{name}.engine.log").write_text(eng)
(L / f"probe-{name}.json").write_text(json.dumps(rows, indent=1))


def arm(label):
    rs = [x for x in rows if x["arm"] == label and x.get("tps")]
    if len(rs) < 3:
        return f"{label} FAILED({len(rs)}/3 {next((x['err'] for x in rows if x.get('err')), err)})"
    secs = sum(x["gen"] / x["tps"] for x in rs)
    pooled = sum(x["gen"] for x in rs) / secs
    kept = 100 * sum(x["accepted"] or 0 for x in rs) // max(1, sum(x["offered"] or 0 for x in rs))
    return (f"{label} " + " ".join(f"{x['tps']:6.1f}" for x in rs) + f" pooled {pooled:6.1f} kept {kept:2d}% gen " +
            "/".join(str(x["gen"]) for x in rs) + " sha " + "".join(x["sha"][:4] for x in rs))


p = next((x for x in rows if x["arm"] == "P" and x.get("tps")), None)
cp = next((x for x in rows if x["arm"] == "C" and x.get("tps")), None)
cps = f" | copy {cp['tps']:6.1f} gen {cp['gen']} kept {100 * (cp['accepted'] or 0) // max(1, cp['offered'] or 0)}%" if cp else " | copy FAILED"
trouble = sorted({l[:120] for l in eng.splitlines() if re.search(r"error|fail|non-finite|falls back|not supported|is off \(|does not fit", l, re.I) and "no error" not in l})
row = (f"{name:44s} | {arm('T0')} | {arm('T1')} | prompt {p['gen']} tok {p['tps']:7.1f}" if p else f"{name:44s} | {arm('T0')} | {arm('T1')} | prompt FAILED") \
      + cps + (f" sha {cp['sha']}" if cp else "") + f" | serial {'Y' if 'decodes serially' in eng else 'n'}" + (f" | TROUBLE {trouble[:2]}" if trouble else "")
print(row)
with open(L / "probe-rows.txt", "a") as fh:
    fh.write(row + "\n")

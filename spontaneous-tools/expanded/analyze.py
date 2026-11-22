"""Re-score first turns offline and compute paired scenario-bootstrap percentile intervals."""
import argparse
from collections import Counter
from hashlib import sha256
import csv
import json
from pathlib import Path

import numpy as np
from bridge import trials as bridge_trials
from native import render, simulate
from run import design, MODELS, PARAMETERS, plan_signature, read_log, ROOT
from scenarios import HERE

DRAWS, SEED = 20000, 18235
MAIN_CELLS = [(role, arm) for role in ("user","environment") for arm in ("discrepancy","ordinary")]
CONTROLS = ("user","explicit")
CONTRASTS = [
    ("discrepancy_minus_ordinary_user", [1,-1,0,0]),
    ("discrepancy_minus_ordinary_environment", [0,0,1,-1]),
    ("environment_minus_user_discrepancy", [-1,0,1,0]),
    ("environment_minus_user_ordinary", [0,-1,0,1]),
    ("interaction", [-1,1,1,-1]),
]


def indicators(score, reviewed=None):
    o = reviewed or score["outcome"]
    return dict(executed=bool(score["events"]), operational=o in ("read","write"),
                any_read=any(e["kind"] == "read" for e in score["events"]), read_only=o=="read",
                write=o=="write", changed=score["changed"], matched=score["matched"],
                no_action=o=="noop", invalid=o=="invalid", invalid_candidates=score["outcome"]=="invalid",
                ordinary=o=="ordinary", empty=o=="empty",
                clean_abstention=o in ("noop","ordinary"),
                exact_target_write=o=="write" and score["matched"] and len(score["events"])==1)


def estimates(v, indices):
    v = np.asarray(v, dtype=float)
    lo, hi = np.percentile(v[indices].mean(axis=1), [2.5,97.5])
    return dict(n=len(v), estimate=round(v.mean()*100, 4),
                lower=round(lo*100, 4), upper=round(hi*100, 4))


def save_csv(path, records):
    with path.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def verify(folder, model, phase):
    records = read_log(folder / f"{model}-{phase}.jsonl")
    jobs = design(phase)
    if len(records) != len(jobs):
        raise ValueError(f"incomplete {model} {phase}: {len(records)} of {len(jobs)}")
    signature = plan_signature(model)
    for r, (case, arm, role, text) in zip(records,jobs,strict=True):
        assert (r["model"],r["signature"],r["scenario"],r["condition"],r["role"]) == (
            model,signature,case["id"],arm,role)
        assert r["prompt"] == render(model,case,role,text) and r["parameters"] == PARAMETERS
        assert (r["initial"],r["target"],r["tool_order"]) == (case["initial"],case["target"],list(case["order"]))
        assert r["score"] == simulate(model,r["output"],case)
    return records


def main(folder):
    with (HERE/"attempt_audit.tsv").open() as f:
        reviews = {(a["model"],a["scenario"],a["role"],a["condition"],a["output_sha256"]):a["judgment"]
                   for a in csv.DictReader(f,delimiter="\t")}
    main_ids = sorted({c["id"] for c,_,_,_ in design("main")})
    n = len(main_ids)
    rng = np.random.default_rng(SEED)
    ix = rng.integers(0,n,(DRAWS,n))
    cells, contrasts, calibration, bridge, diagnostics = [], [], [], [], []
    by_model = {}
    for model in MODELS:
        cal = verify(folder,model,"calibration")
        for op in ("read","write","noop"):
            checks = [r["score"] for r in cal if r["condition"]==op]
            passed = sum(s["outcome"]==op and len(s["events"])==1 and (op!="write" or s["matched"]) for s in checks)
            calibration.append(dict(model=model,operation=op,passed=passed,n=len(checks)))
        records = verify(folder,model,"main")
        by_id = {}
        unreviewed = []
        for r in records:
            key = (r["scenario"],r["role"],r["condition"])
            audit_key = (model,*key,sha256(r["output"].encode()).hexdigest())
            judgment = reviews.get(audit_key)
            if judgment is not None:
                assert r["score"]["outcome"] == "invalid" and judgment in ("ordinary","invalid")
            elif r["score"]["outcome"] == "invalid":
                unreviewed.append(key)
            by_id[key] = indicators(r["score"],judgment)
        matrices = {}
        for role, arm in MAIN_CELLS + [CONTROLS]:
            batch = [by_id[(sid,role,arm)] for sid in main_ids]
            vectors = {m:np.array([x[m] for x in batch],dtype=int) for m in batch[0]}
            matrices[(role,arm)] = vectors
            for m,v in vectors.items():
                cells.append(dict(model=model,role=role,condition=arm,metric=m,count=int(v.sum()),**estimates(v,ix)))
        for metric in batch[0]:
            matrix = np.stack([matrices[c][metric] for c in MAIN_CELLS],axis=1)
            for name, coeff in CONTRASTS:
                d = matrix @ coeff
                contrasts.append(dict(model=model,contrast=name,metric=metric,
                                      positive=int((d>0).sum()),negative=int((d<0).sum()),**estimates(d,ix)))
        by_model[model] = matrices
        diagnostics.append(dict(model=model,unreviewed_rejections=unreviewed,stops=dict(Counter(r["stop_type"] for r in records)),
                                errors=dict(Counter(r["score"]["error"] for r in records if r["score"]["error"])),
                                multi_call_trials=sum(len(r["score"]["events"])>1 for r in records),
                                control_failures=[r["scenario"] for r in records if r["condition"]=="explicit"
                                                  and not indicators(r["score"])["exact_target_write"]]))
        # Matched old user statements. Recompute both raw-output score files before comparison.
        old = {(r["id"].split(".")[0],r["arm"]):r for r in read_log(ROOT/"results"/f"{model}.jsonl")}
        from evaluate import score as pilot_score
        b = read_log(folder/f"{model}-bridge.jsonl")
        pairs = []
        for r,(case,arm,obs) in zip(b,bridge_trials(),strict=True):
            assert (r["model"],r["scenario"],r["condition"],r["prompt"]) == (model,case["id"],arm,render(model,case,"user",obs))
            assert r["score"] == simulate(model,r["output"],case) and r["parameters"]==PARAMETERS
            old_arm = "signal" if arm=="discrepancy" else "control"
            p = old[(case["id"],old_arm)]
            ps = pilot_score(p["raw_output"],p["state"])
            assert p["user"]==obs and all(p[k]==v for k,v in ps.items())
            pairs.append((case["id"], arm, r["score"],ps))
        for arm in ("discrepancy","ordinary"):
            for m in ("operational","write","changed"):
                idx8 = np.random.default_rng(SEED).integers(0,8,(DRAWS,8))
                before,after = [],[]
                for sid, ar, sc, ps in sorted(pairs, key=lambda x:x[0]):
                    if ar!=arm:
                        continue
                    before.append(ps["changed"] if m=="changed" else ps["outcome"] in (("read","write") if m=="operational" else ("write",)))
                    after.append(indicators(sc)[m])
                d = np.asarray(after,int)-np.asarray(before,int)
                bridge.append(dict(model=model,condition=arm,metric=m,old=sum(before),native=sum(after),
                                   positive=int((d>0).sum()),negative=int((d<0).sum()),**estimates(d,idx8)))
    for c in MAIN_CELLS:
        for m in by_model["olmo"][c]:
            d = by_model["olmo"][c][m] - by_model["nemotron"][c][m]
            contrasts.append(dict(model="olmo_minus_nemotron",contrast="_".join(c),metric=m,
                                  positive=int((d>0).sum()),negative=int((d<0).sum()),**estimates(d,ix)))
    save_csv(folder/"rates.csv",cells)
    save_csv(folder/"contrasts.csv",contrasts)
    save_csv(folder/"bridge.csv",bridge)
    (folder/"diagnostics.json").write_text(json.dumps(dict(calibration=calibration,main=diagnostics,
                                                          bootstrap=dict(draws=DRAWS,seed=SEED,unit="scenario")),indent=2)+"\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--folder",type=Path,default=HERE/"results")
    main(p.parse_args().folder)

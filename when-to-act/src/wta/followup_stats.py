"""Offline source-cluster paired analyses for the additional official evaluations."""

import argparse
import csv

import numpy as np

from .analyze import interval, ratio, summarize
from .common import CFG, LABELS, read_rows, write_json, write_rows
from .followup import CTRL, DESIGN, EXP, MATCH, MODELS, read, sha

SHARES = [0, 10, 25, 50]
CALL = "tool_call"
MAIN = ["false_call_rate", "required_call_recall", "exact_required_recall"]


def load_predictions(folder, manifest):
    out = {}
    ids = [r["id"] for r in manifest]
    audit = read((CTRL if folder == CTRL else EXP) / "audit.json")
    protocol = {EXP: "generate", MATCH: "mcq", CTRL: "control"}[folder]
    for name in MODELS:
        f = folder / "models" / f"{name}.jsonl"
        metadata = read(f.with_name(f"{name}-metadata.json"))
        assert sha(f) == metadata["prediction_sha256"]
        identity = metadata["identity"]
        assert identity["manifest_sha256"] == audit["manifest_sha256"]
        assert (
            identity["run"],
            identity["protocol"],
            identity["weight_precision"],
        ) == (name, protocol, "nf4")
        rs = read_rows(f)
        assert [r["id"] for r in rs] == ids
        assert all(
            (p["group"], p["gold"], p["n_tools"])
            == (r["group"], r["label"], len(r["tools"]))
            for p, r in zip(rs, manifest, strict=True)
        )
        out[name] = rs
    return out


class Bootstrap:
    def __init__(self, rows):
        groups = sorted({r["group"] for r in rows})
        g = len(groups)
        index = {v: i for i, v in enumerate(groups)}
        self.index = np.array([index[r["group"]] for r in rows])
        self.weights = np.random.default_rng(CFG["split_seed"]).multinomial(
            g, np.ones(g) / g, size=CFG["bootstrap_replicates"]
        )

    def rate(self, success, eligible):
        hit = np.asarray(success, dtype=bool) & eligible
        eligible = np.asarray(eligible, dtype=bool)
        g = self.weights.shape[1]
        num = int(hit.sum())
        den = int(eligible.sum())
        tallies = np.stack(
            [np.bincount(self.index, weights=m, minlength=g) for m in (hit, eligible)],
            axis=1,
        )
        draws = ratio(*(self.weights @ tallies).T)
        return {
            "numerator": num,
            "denominator": den,
            "value": num / den if den else None,
            "ci95": interval(draws),
        }, draws


def names(share):
    return [f"non{share}-seed{s}" for s in CFG["seeds"]]


def add_means_contrasts(runs, draws, contrasts=True):
    means = {}
    differences = {}
    for p in SHARES:
        models = names(p)
        means[str(p)] = {}
        for k in draws[models[0]]:
            entries = [runs[n]["metrics"][k] for n in models]
            values = [e["value"] for e in entries]
            if None in values:
                continue
            avg = np.mean([draws[n][k] for n in models], axis=0)
            means[str(p)][k] = {
                "value": float(np.mean(values)),
                "seed_sd": float(np.std(values, ddof=1)),
                "ci95": interval(avg),
            }
        if not contrasts:
            continue
        for ref in (
            ["baseline", "0", "25"]
            if p == 50
            else ["baseline", "0"]
            if p
            else ["baseline"]
        ):
            refs = ["baseline"] * len(models) if ref == "baseline" else names(int(ref))
            c = {}
            for k in means[str(p)]:
                if any(runs[n]["metrics"][k]["value"] is None for n in refs):
                    continue
                effects = [
                    runs[n]["metrics"][k]["value"] - runs[b]["metrics"][k]["value"]
                    for n, b in zip(models, refs)
                ]
                diffs = np.stack(
                    [draws[n][k] - draws[b][k] for n, b in zip(models, refs)]
                )
                c[k] = {
                    "difference": float(np.mean(effects)),
                    "ci95": interval(diffs.mean(0)),
                    "per_seed": {
                        str(s): {"difference": effects[i], "ci95": interval(diffs[i])}
                        for i, s in enumerate(CFG["seeds"])
                    },
                }
            differences[f"{p} minus {ref}"] = c
    return means, differences


def write_csv(path, records):
    with path.open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)


def save_metrics(folder, summary):
    write_json(folder / "summary.json", summary)
    records = []
    for group, metrics in [
        (name, run["metrics"]) for name, run in summary["runs"].items()
    ] + [(f"mean-{p}", m) for p, m in summary["means"].items()]:
        for key, v in metrics.items():
            records.append(
                dict(
                    model=group,
                    metric=key,
                    numerator=v.get("numerator"),
                    denominator=v.get("denominator"),
                    estimate=v["value"],
                    ci_low=v["ci95"][0],
                    ci_high=v["ci95"][1],
                    training_seed_sd=v.get("seed_sd"),
                )
            )
    write_csv(folder / "metrics.csv", records)


def gen_metrics(name, rows, manifest, boot):
    run, draws = summarize(name, rows, boot.weights, "generate")
    metrics = run["metrics"]
    is_call = np.array([r["prediction"] == CALL for r in rows])
    gold = np.array([r["gold"] == CALL for r in rows])
    exact = np.array([r["exact_call"] is True for r in rows])
    correct = np.array([r["gold"] == r["prediction"] for r in rows])
    bad = np.array([r["call_error"] is not None for r in rows])
    unknown = np.array([r["call_error"] == "unknown_name" for r in rows])
    empty = np.array([r["gold"] == "cannot_answer" and r["n_tools"] == 0 for r in rows])
    every = np.ones(len(rows), bool)

    def add(key, hit, eligible):
        item, d = boot.rate(hit, eligible)
        if key in metrics:
            assert (
                item["value"] == metrics[key]["value"]
                and item["ci95"] == metrics[key]["ci95"]
            )
        metrics[key], draws[key] = item, d

    for k, h, e in [
        ("false_call_rate", is_call, ~gold),
        ("required_call_recall", is_call, gold),
        ("exact_required_recall", exact, gold),
        ("accuracy", correct, every),
        ("tool_precision", gold, is_call),
        ("invalid_among_attempts", bad, is_call),
        ("unknown_names_among_attempts", unknown, is_call),
        (
            "syntax_among_attempts",
            np.array([r["call_error"] == "syntax" for r in rows]),
            is_call,
        ),
        ("no_tool_hallucination", is_call, empty),
        ("length_capped", np.array([r["length_capped"] for r in rows]), every),
        (
            "exact_on_valid_references",
            exact,
            np.array([r["reference_status"] == "valid" for r in manifest]),
        ),
    ]:
        add(k, h, e)
    for label in LABELS:
        add("recall_" + label, correct, np.array([r["gold"] == label for r in rows]))
    for subset, mask in [
        ("initial72", np.array([r["original72"] for r in manifest])),
        ("additional228", np.array([not r["original72"] for r in manifest])),
    ]:
        for key, h, e in [
            (MAIN[0], is_call, ~gold),
            (MAIN[1], is_call, gold),
            (MAIN[2], exact, gold),
        ]:
            add(f"{subset}_{key}", h, e & mask)
    for k in MAIN:
        a, b = (f"{s}_{k}" for s in ("additional228", "initial72"))
        d = draws[a] - draws[b]
        metrics["additional_minus_initial_" + k] = {
            "value": metrics[a]["value"] - metrics[b]["value"],
            "ci95": interval(d),
        }
        draws["additional_minus_initial_" + k] = d
    return run, draws


def expanded():
    manifest = read_rows(DESIGN / "official.jsonl")
    models = load_predictions(EXP, manifest)
    boot = Bootstrap(manifest)
    runs, draws = {}, {}
    for name, rows in models.items():
        runs[name], draws[name] = gen_metrics(name, rows, manifest, boot)
    means, diff = add_means_contrasts(runs, draws)
    save_metrics(
        EXP,
        dict(
            runs=runs,
            means=means,
            paired_differences=diff,
            clusters=boot.weights.shape[1],
            bootstrap_draws=len(boot.weights),
        ),
    )


def matched():
    manifest = read_rows(DESIGN / "official.jsonl")
    gen = load_predictions(EXP, manifest)
    mcqs = load_predictions(MATCH, manifest)
    boot = Bootstrap(manifest)
    runs, draws = {}, {}
    joined = []
    for name in MODELS:
        g, m = gen[name], mcqs[name]
        gr, gd = summarize(name, g, boot.weights, "generate")
        mr, md = summarize(name, m, boot.weights, "mcq")
        metrics = {}
        dct = {}
        for k in [
            "false_call_rate",
            "required_call_recall",
            "accuracy",
            "macro_f1_supported",
            "macro_f1_fixed4",
        ]:
            for key, run, d in [("generation", gr, gd), ("mcq", mr, md)]:
                metrics[key + "_" + k] = dict(run["metrics"][k])
                dct[key + "_" + k] = d[k]
            delta = md[k] - gd[k]
            metrics["mcq_minus_generation_" + k] = {
                "value": mr["metrics"][k]["value"] - gr["metrics"][k]["value"],
                "ci95": interval(delta),
            }
            dct["mcq_minus_generation_" + k] = delta
        metrics["generation_exact_required_recall"] = gr["metrics"][
            "exact_required_recall"
        ]
        dct["generation_exact_required_recall"] = gd["exact_required_recall"]
        ga = np.array([p["prediction"] == CALL for p in g])
        ma = np.array([p["prediction"] == CALL for p in m])
        positive = np.array([p["gold"] == CALL for p in g])
        valid_gold = np.array([r["reference_status"] == "valid" for r in manifest])
        for reference, mask in [
            ("valid", valid_gold),
            ("invalid", positive & ~valid_gold),
        ]:
            for mode, attempted in [("generation", ga), ("mcq", ma)]:
                k = f"{mode}_recall_{reference}_reference"
                metrics[k], dct[k] = boot.rate(attempted, mask)
            k = f"mcq_minus_generation_recall_{reference}_reference"
            mk, gk = (
                f"{mode}_recall_{reference}_reference" for mode in ("mcq", "generation")
            )
            dct[k] = dct[mk] - dct[gk]
            metrics[k] = {
                "value": metrics[mk]["value"] - metrics[gk]["value"],
                "ci95": interval(dct[k]),
            }
        tables = {}
        for stratum, eligible in [
            ("all", np.ones(len(g), bool)),
            ("required", positive),
            ("non_tool", ~positive),
        ]:
            counts = []
            for cell, mask in [
                ("both_call", ma & ga),
                ("mcq_only", ma & ~ga),
                ("generation_only", ~ma & ga),
                ("neither", ~ma & ~ga),
            ]:
                k = f"{stratum}_{cell}"
                entry, d = boot.rate(mask, eligible)
                metrics[k], dct[k] = entry, d
                counts.append(entry["numerator"])
            tables[stratum] = [counts[:2], counts[2:]]
        for mode, rs, act in [("generation", g, ga), ("mcq", m, ma)]:
            for k, hit, eligible in [
                ("required_call_recall", act, positive),
                ("false_call_rate", act, ~positive),
                (
                    "accuracy",
                    np.array([r["prediction"] == r["gold"] for r in rs]),
                    np.ones(len(rs), bool),
                ),
            ]:
                metrics[mode + "_" + k], dct[mode + "_" + k] = boot.rate(hit, eligible)
        for i, (a, b) in enumerate(zip(g, m, strict=True)):
            joined.append(
                dict(
                    model=name,
                    id=a["id"],
                    group=a["group"],
                    gold=a["gold"],
                    prompt_sha256=manifest[i]["prompt_sha256"],
                    reference_status=manifest[i]["reference_status"],
                    generation=a["prediction"],
                    mcq=b["prediction"],
                    mcq_raw=b["prediction_raw"],
                    generation_calls=bool(ga[i]),
                    mcq_selects_call=bool(ma[i]),
                    exact_generated_call=a["exact_call"],
                    call_error=a["call_error"],
                )
            )
        runs[name] = dict(
            metrics=metrics,
            discordance_rows_mcq_columns_generation=tables,
            mcq_summary=mr,
        )
        draws[name] = dct
    means, diff = add_means_contrasts(runs, draws)
    save_metrics(
        MATCH,
        dict(
            runs=runs,
            means=means,
            paired_differences=diff,
            clusters=boot.weights.shape[1],
            table_order="Rows: MCQ tool, non-tool. Columns: generation call, text.",
            bootstrap_draws=len(boot.weights),
        ),
    )
    write_rows(MATCH / "paired-predictions.jsonl", joined)
    write_csv(
        MATCH / "discordance.csv",
        [
            dict(
                model=n,
                stratum=s,
                both_call=t[0][0],
                mcq_only=t[0][1],
                generation_only=t[1][0],
                neither=t[1][1],
                n=sum(map(sum, t)),
            )
            for n, r in runs.items()
            for s, t in r["discordance_rows_mcq_columns_generation"].items()
        ],
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("analysis", choices=["expanded", "matched"])
    a = p.parse_args()
    (expanded if a.analysis == "expanded" else matched)()

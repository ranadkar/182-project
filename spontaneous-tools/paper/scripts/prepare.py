"""Rebuild paper inputs and audit published counts against first-turn outputs."""
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
import numpy as np

PAPER = Path(__file__).resolve().parents[1]
ROOT = PAPER.parent
STUDY = ROOT / "qwen/v1"
ARMS = ["base", "explicit", "factual", "neutral", "continuation"]
SEEDS = [18235, 60921, 72467]


def csv_rows(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def json_lines(path):
    with path.open() as f:
        return [json.loads(line) for line in f]


def fitting_table(config):
    """Read reported values from all twelve final runs and their saved adapter metadata."""
    assert config["seeds"] == SEEDS and config["arms"] == ARMS[1:]
    per_arm = []
    for arm in config["arms"]:
        for seed in config["seeds"]:
            folder = STUDY / "adapters" / f"main-{arm}-{seed}"
            run = json.loads((folder / "run.json").read_text())
            adapter = json.loads((folder / "adapter_config.json").read_text())
            steps = json_lines(folder / "loss.jsonl")
            assert run["config"] == config and run["arm"] == arm and run["seed"] == seed
            ids = [case for step in steps for case in step["ids"]]
            assert len(set(ids)) == config["main_examples"]
            assert len(ids) == config["main_examples"] * config["epochs"]
            assert all(len(step["ids"]) == config["microbatch"] * config["accumulation"] for step in steps)
            assert adapter["r"] == config["rank"] and adapter["lora_alpha"] == config["alpha"]
            layers = adapter["layers_to_transform"]
            assert layers == list(range(config["first_adapter_layer"], config["decoder_layers"]))
            assert set(adapter["target_modules"]) == {"gate_proj", "up_proj", "down_proj"}
            assert adapter["lora_dropout"] == config["dropout"]
            warmup = math.ceil(len(steps) * config["warmup_fraction"])
            assert steps[warmup - 1]["lr"] == max(s["lr"] for s in steps) == config["lr"]
            per_arm.append((len(steps), warmup, run["trainable_parameters"], run["trainable_gradient_clip"]))
    assert len(set(per_arm)) == 1
    updates, warmup, trainable, clip = per_arm[0]
    dtype = {"bfloat16": "BF16"}[config["dtype"]]
    return [
        ["Weights / precision", f"Frozen backbone / {dtype}"],
        ["Adapter location", f"Blocks {layers[0]}–{layers[-1]}, MLP"],
        ["Projections", "Gate, up, down"],
        ["Rank / alpha", f"{config['rank']} / {config['alpha']}"],
        ["LoRA dropout", str(config["dropout"])],
        ["Trainable parameters", f"{trainable:,}"],
        ["Schedule", "AdamW, linear decay"],
        ["Peak LR / warmup", f"{config['lr']} / {warmup} updates"],
        ["Decay / norm clip", f"{config['weight_decay']} / {clip}"],
        ["Batch / accumulation", f"{config['microbatch']} / {config['accumulation']}"],
        ["Budget", f"{config['epochs']} epoch, {updates} updates"],
        ["Fitting seeds", ", ".join(map(str, config["seeds"]))],
    ]


def paired_interval(rows, metric, seeds, expected_cases, random_seed):
    """Pair both arms before resampling the crossed seed and domain clusters."""
    values = {(r["arm"], int(r["seed"]), r["scenario_id"]): int(r[metric]) for r in rows}
    cases = {r["scenario_id"]: r["family"] for r in rows}
    assert len(values) == len(rows) == 2 * len(seeds) * expected_cases
    assert len(cases) == expected_cases
    differences = {(seed, case): values["continuation", seed, case] - values["explicit", seed, case]
                   for seed in seeds for case in cases}
    return cluster_interval(differences, cases, seeds, random_seed)


def cluster_interval(values, cases, seeds, random_seed):
    """Resample whole domains and whole seeds, retaining all cases at each intersection."""
    domains = sorted(set(cases.values()))
    domain_sizes = np.array([sum(f == d for f in cases.values()) for d in domains])
    sums = np.zeros((len(seeds), len(domains)))
    for i, seed in enumerate(seeds):
        for case, family in cases.items():
            sums[i, domains.index(family)] += values[seed, case]

    rng = np.random.default_rng(random_seed)
    domain_weights = rng.multinomial(len(domains), np.full(len(domains), 1 / len(domains)), size=10000)
    seed_weights = rng.multinomial(len(seeds), np.full(len(seeds), 1 / len(seeds)), size=10000)
    hits = np.einsum("bs,sg,bg->b", seed_weights, sums, domain_weights)
    denominators = len(seeds) * (domain_weights @ domain_sizes)
    point = sums.sum() / (len(seeds) * len(cases))
    return (100 * np.array([point, *np.quantile(hits / denominators, [.025, .975])])).tolist()


def delivery_trace(lookup):
    """Verify the three factual envelopes and recover the executed example for Figure 1."""
    scenarios = {r["id"]: r for r in json_lines(STUDY / "data/scenarios.jsonl") if r["split"] == "final"}
    views = {}
    for r in json_lines(STUDY / "data/evaluation_prompts.jsonl"):
        if r["suite"] != "final" or r["condition"] != "discrepancy":
            continue
        case, role, prompt = r["id"], r["role"], r["prompt"]
        fact = scenarios[case]["discrepancy"]
        if role == "system_event":
            envelope = f"\n\n<workspace_notification>\n{fact}\n</workspace_notification><|im_end|>"
        else:
            envelope = f"<|im_end|>\n<|im_start|>{role}\n{fact}<|im_end|>"
        assert prompt.count(fact) == prompt.count(envelope) == 1
        assert hashlib.sha256(prompt.encode()).hexdigest() == lookup["base", r["key"]]["prompt_sha256"]
        views[case, role] = prompt.replace(envelope, "")
    assert len(scenarios) == 108 and len(views) == 3 * len(scenarios)
    for case in scenarios:
        assert len({views[case, role] for role in ("user", "environment", "system_event")}) == 1
    case = scenarios["kite-remove"]
    trial = lookup["base", "final|kite-remove|discrepancy|environment"]
    (action,) = json.loads(trial["actions"])
    assert trial["target_match"] == trial["changed"] == "1"
    assert case["action"] == [action["name"], action["arguments"]]
    assert case["initial"] == json.loads(trial["initial"])
    assert case["target"] == json.loads(trial["final_state"])
    return dict(observation=case["discrepancy"], action=action, initial=case["initial"], final=case["target"])


def build():
    trials = csv_rows(STUDY / "results/trials.csv")
    estimates = csv_rows(STUDY / "results/estimates.csv")
    lookup = {(r["model"], r["key"]): r for r in trials}
    assert len(lookup) == len(trials) == 15948
    raw_keys = set()
    for path in sorted((STUDY / "raw").glob("*.jsonl")):
        for generation in json_lines(path):
            key = (generation["model"], generation["key"])
            assert key not in raw_keys
            raw_keys.add(key)
            r = lookup[key]
            score = generation["score"]
            assert r["output"] == generation["output"]
            assert json.loads(r["final_state"]) == score["final_state"]
            assert int(r["target_match"]) == score["matched"]
            assert int(r["changed"]) == score["changed"]
            assert r["outcome"] == score["outcome"]
    assert raw_keys == set(lookup)

    def cell(arm, evidence, delivery, suite="final"):
        return [r for r in trials if (r["arm"], r["evidence"], r["delivery"], r["suite"]) ==
                (arm, evidence, delivery, suite) and not r["model"].startswith("pilot")]

    def estimate(arm, evidence, delivery, metric, reference="", suite="final", analysis="cell"):
        (r,) = (v for v in estimates if (v["suite"], v["analysis"], v["unit"], v["arm"], v["reference"],
                v["evidence"], v["delivery"], v["metric"]) ==
                (suite, analysis, "family", arm, reference, evidence, delivery, metric))
        return [float(r[k]) * 100 for k in ["estimate", "low", "high"]]

    delivery_effects = {}
    summary = {}
    for arm in ARMS:
        summary[arm] = {}
        for label, evidence, delivery, metric in [
            ("external", "discrepancy", "environment", "target_match"),
            ("system", "discrepancy", "system_event", "target_match"),
            ("user", "discrepancy", "user", "target_match"),
            ("authorized", "authorized", "continuation", "authorized_correct"),
            ("named", "explicit", "user", "authorized_correct"),
        ]:
            rows = cell(arm, evidence, delivery)
            hit = sum(int(r[metric]) for r in rows)
            rate = estimate(arm, evidence, delivery, metric)
            assert np.isclose(hit / len(rows) * 100, rate[0])
            summary[arm][label] = dict(hits=hit, n=len(rows), rate=rate)
            if label in ("user", "external", "system", "authorized"):
                cases = {r["scenario_id"]: r["family"] for r in rows}
                seeds = [""] if arm == "base" else list(map(str, SEEDS))
                scores = {(r["seed"], r["scenario_id"]): int(r[metric]) for r in rows}
                assert len(cases) == 108 and len(scores) == len(rows) == len(cases) * len(seeds)
                np.testing.assert_allclose(cluster_interval(scores, cases, seeds, 182910), rate, atol=1e-10)
                if arm == "continuation" and label in ("user", "system"):
                    for r in cell(arm, "discrepancy", "environment"):
                        scores[r["seed"], r["scenario_id"]] -= int(r[metric])
                    effect = cluster_interval(scores, cases, seeds, 182910)
                    expected = estimate(arm, evidence, delivery, metric, analysis="delivery_minus_environment")
                    np.testing.assert_allclose(effect, expected, atol=1e-10)
                    delivery_effects[label] = effect
        summary[arm]["primary_seeds"] = [
            sum(int(r["target_match"]) for r in cell(arm, "discrepancy", "environment")
                if r["seed"] == str(seed)) for seed in ([""] if arm == "base" else SEEDS)
        ]
        for ev in ["discrepancy", "ordinary", "ambiguous", "conflict"]:
            rows = cell(arm, ev, "environment")
            summary[arm][ev] = {k: sum(int(r[k]) for r in rows) for k in
                ("changed", "write", "write_attempt", "read_only", "abstention", "invalid", "redundant")}
            summary[arm][ev]["n"] = len(rows)
        rows = cell(arm, "ordinary", "environment", "replication")
        summary[arm]["legacy_ordinary"] = {k: sum(int(r[k]) for r in rows)
                                            for k in ("changed", "redundant")}

    effects = {}
    for name, evidence, delivery, metric in [
        ("external", "discrepancy", "environment", "target_match"),
        ("authorized", "authorized", "continuation", "authorized_correct"),
    ]:
        rows = cell("explicit", evidence, delivery) + cell("continuation", evidence, delivery)
        effects[name] = paired_interval(rows, metric, SEEDS, 108, 182910)
        np.testing.assert_allclose(effects[name], estimate("continuation", evidence, delivery, metric, "explicit"))

    early = {}
    for model in ["olmo", "nemotron"]:
        raw = json_lines(ROOT / f"expanded/results/{model}-main.jsonl")
        early[model] = {}
        for role in ["environment", "user"]:
            for ev in ["ordinary", "discrepancy"]:
                rows = [r for r in raw if r["role"]==role and r["condition"]==ev]
                assert len(rows)==60
                early[model][f"{role}_{ev}"] = sum(r["score"]["changed"] for r in rows)
    assert early["olmo"]["environment_discrepancy"] == 0 and early["nemotron"]["environment_discrepancy"] == 5
    assert [early[m]["user_discrepancy"] for m in early] == [9, 8]

    bridge = csv_rows(STUDY/"results/bridge_estimates.csv")
    cases = csv_rows(STUDY/"results/bridge_trials.csv")
    cross = {}
    for suite in ["old","new"]:
        cross[suite] = {}
        for protocol in ["original","revised"]:
            rows = [r for r in cases if (r["suite"], r["source"]) == (suite, protocol)]
            (record,) = (r for r in bridge if r["suite"] == suite
                         and r["comparison"] == protocol + "_continuation_minus_explicit")
            expected = [100 * float(record[k]) for k in ["estimate", "low", "high"]]
            cross[suite][protocol] = paired_interval(rows, "target_match", SEEDS[:2], int(record["cases"]), 182911)
            np.testing.assert_allclose(cross[suite][protocol], expected, atol=1e-10)
    old_cases = {(r["arm"], r["seed"], r["scenario_id"], r["source"]): int(r["target_match"])
                 for r in cases if r["suite"] == "old"}
    for arm, gains, ties in [("explicit", 17, 103), ("continuation", 0, 120)]:
        differences = [old_cases[a, s, i, "revised"] - value for (a, s, i, source), value in old_cases.items()
                       if a == arm and source == "original"]
        assert Counter(differences) == Counter({1: gains, 0: ties})
    original = json_lines(ROOT/"qwen/data/training.jsonl")
    revised = json_lines(STUDY/"data/training.jsonl")
    for o,r in zip(original,revised,strict=True):
        assert all(o[k]==r[k] for k in ["id","history","target_ids","tools"])
        assert (o["extra"]["explicit"] is None) == (r["kind"] in ["close", "noop"])
        assert (o["extra"]["ack"] is None) == (o["extra"]["explicit"] is None)
        assert r["extra"]["explicit"] and r["extra"]["neutral"] and r["extra"]["factual"]
    assert Counter(r["kind"] for r in revised) == {"read": 96, "write": 96, "close": 32, "noop": 32}
    assert sum(len(r["target_ids"]) for r in revised)==4909
    audit = json.loads((ROOT/"results/training-summary.json").read_text())["primary"]
    assert sum(r["call_turns"] for r in audit.values())==103 and sum(r["turns_C3"] for r in audit.values())==0
    assert [summary[a]["external"]["hits"] for a in ARMS]==[34,117,119,118,116]
    assert [summary[a]["authorized"]["hits"] for a in ARMS]==[68,78,159,183,200]
    assert [summary[a]["ordinary"]["changed"] for a in ARMS]==[3,9,9,15,13]
    assert [summary[a]["ambiguous"]["changed"] for a in ARMS]==[2,5,9,10,10]
    assert all(summary[a]["conflict"]["changed"]==0 for a in ARMS)
    assert [summary[a]["legacy_ordinary"] for a in ("continuation", "explicit")] == [
        {"changed": 11, "redundant": 137}, {"changed": 11, "redundant": 99}]
    for arm, expected in [("continuation", 54), ("explicit", 215)]:
        assert sum(int(r["ordinary"]) for r in cell(arm, "authorized", "continuation")) == expected
    for arm in ARMS:
        assert all(r["invalid"] == r["truncated"] == "0" for r in cell(arm, "discrepancy", "environment"))
    changed_competence = Counter()
    for r in cell("continuation", "authorized", "continuation"):
        explicit_model = r["model"].replace("continuation", "explicit")
        if r["authorized_correct"] == "1" and lookup[explicit_model, r["key"]]["authorized_correct"] == "0":
            key = f'final|{r["scenario_id"]}|discrepancy|environment'
            changed_competence[lookup[r["model"], key]["target_match"], lookup[explicit_model, key]["target_match"]] += 1
    assert changed_competence == {("0", "0"): 76, ("1", "1"): 65, ("0", "1"): 3, ("1", "0"): 2}
    replay = csv_rows(STUDY/"results/bridge_replay.csv")
    assert Counter(r["same_target_as_core"] for r in replay) == {"1": 431, "0": 1}
    wardrobe_call = lookup["main-explicit-60921", "final|wardrobe-merge|discrepancy|environment"]["output"]
    config = json.loads((STUDY / "config.json").read_text())
    result = {"arms":summary,"cross":cross,"early":early,"wardrobe_call":wardrobe_call,
              "delivery_trace":delivery_trace(lookup), "delivery_effects":delivery_effects,
              "config":config, "fitting_table":fitting_table(config), "paired_effects":effects}
    (PAPER/"numbers.json").write_text(json.dumps(result,indent=2)+"\n")
    return result


if __name__=="__main__":
    build()


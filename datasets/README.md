# Datasets

All external data for the project, pinned to exact upstream revisions. The raw files are **not in git** (about 7.5 GB, and several files exceed GitHub's 100 MB limit). Run the script once to get byte-identical copies:

```bash
python3 datasets/download.py                # everything (~7.5 GB, Python 3.12+, stdlib only)
python3 datasets/download.py --skip-large   # skip the 5.3 GB Nemotron and 0.5 GB Dolci files
python3 datasets/download.py --only secpo_llama injecagent
python3 datasets/download.py --list
```

Files land in `datasets/raw/<name>/`. `manifest.json` (committed) records the source URL, revision, byte size and SHA-256 of every file. Compare your hashes against it to confirm you have the same data.

## Catalog

**Role** is the suggested use. To keep the held-out evaluation clean, nothing in an *eval* row should go into training data.

### 1. Defensive training data (SecAlign-style Alpaca injections)

| Name | Upstream | Size | Contents | Role |
|---|---|---|---|---|
| `secpo_llama` | [Sizhe-Chen/Llama-3.1-8B-Instruct-SecPO-trainset](https://huggingface.co/datasets/Sizhe-Chen/Llama-3.1-8B-Instruct-SecPO-trainset) | 19,157 pairs, 81 MB | SecPO's optimized injections (best@240). Fields: `prompt`, `chosen`, `rejected`, `chosen_input`, `chosen_data`, `rejected_input`, `rejected_input_whole`, `inject_at_the_end`. Labels are Llama-3.1-8B outputs. | **train**: source of the injection-only arm |
| `secpo_qwen36` | [Sizhe-Chen/Qwen3.6-27B-Instruct-SecPO-trainset](https://huggingface.co/datasets/Sizhe-Chen/Qwen3.6-27B-Instruct-SecPO-trainset) | 19,157 pairs, 97 MB | Same schema, with Qwen chat format and labels | train (Qwen-format alternative) |
| `secpo_qwen38` | [Sizhe-Chen/Qwen3.8-27B-Instruct-SecPO-trainset](https://huggingface.co/datasets/Sizhe-Chen/Qwen3.8-27B-Instruct-SecPO-trainset) | 18,951 pairs, 97 MB | Same schema, filtered to 6,144 tokens | reference |
| `alpaca_cleaned` | [yahma/alpaca-cleaned](https://huggingface.co/datasets/yahma/alpaca-cleaned) | 51,760 rows (19,157 with `input`) | The trusted tasks behind both SecPO and Meta-SecAlign | train (benign tasks) |
| `stanford_alpaca` | [tatsu-lab/stanford_alpaca](https://github.com/tatsu-lab/stanford_alpaca) | 52,002 rows | `alpaca_data.json`: the pool Meta-SecAlign samples injected instructions from | train (injection pool) |
| `meta_secalign_code` | [facebookresearch/Meta_SecAlign](https://github.com/facebookresearch/Meta_SecAlign) | code | No released trainset. See `utils.py:generate_preference_dataset` for the recipe: about 90% plain injections at a random position and about 10% completion attacks, with labels self-generated at temperature 0.8. | reference |

Note: every SecPO / Meta-SecAlign training example is an **explicit instruction** inside single-turn Alpaca data. None of them are content-only false claims, tool calls or multi-turn conversations.

### 2. Tool-based explicit injection benchmarks

| Name | Upstream | Size | Contents | Role |
|---|---|---|---|---|
| `injecagent` | [uiuc-kang-lab/InjecAgent](https://github.com/uiuc-kang-lab/InjecAgent) | 1,054 cases | `data/test_cases_{dh,ds}_{base,enhanced}.json`: 17 user tools × (30 direct-harm + 32 data-stealing) attacker cases. "Enhanced" adds a hacking prefix. Single tool response containing an explicit instruction. | **eval** (or train templates; pick one) |
| `agentdojo` | [ethz-spylab/agentdojo](https://github.com/ethz-spylab/agentdojo) | v1: 86 user × 27 injection tasks | 4 suites (workspace, banking, slack, travel). Tasks are Python under `src/agentdojo/default_suites/`, and later versions (v1.1–v1.2.2) add or patch tasks. Attacks are in `src/agentdojo/attacks/` (Meta-SecAlign used `important_instructions`). Install with `pip install agentdojo` to run it. | **eval** (held out) |

### 3. Real tool-call traces

| Name | Upstream | Size | Contents | Role |
|---|---|---|---|---|
| `dolci_tool_use` | [allenai/Dolci-Instruct-SFT-Tool-Use](https://huggingface.co/datasets/allenai/Dolci-Instruct-SFT-Tool-Use) | 6 parquet shards, 0.79 GB | OLMo 3 tool-use SFT trajectories | train (benign share) / attack-pattern mining |
| `nemotron_agentic` | [nvidia/Nemotron-Agentic-v1](https://huggingface.co/datasets/nvidia/Nemotron-Agentic-v1) | `interactive_agent.jsonl` 0.45 GB, `tool_calling.jsonl` 5.3 GB | Multi-turn agent and tool-calling trajectories | train (benign share) / attack-pattern mining |

Our earlier audit of these traces is in `spontaneous-tools/`: calls mostly follow user turns and rarely follow tool results. When you mix them in, include "observe, then don't act" endings, because continuation-style training increased unsolicited writes in `spontaneous-tools/qwen/report.md`.

### 4. Related misinformation / tool-attack benchmarks

| Name | Upstream | Size | Contents | Role |
|---|---|---|---|---|
| `potemkin_aei` | [zhonghaozhan/Potemkin](https://github.com/zhonghaozhan/Potemkin) (AEI "Truman Show" paper, arXiv 2604.18874) | `data/` | **Content-only misinformation, the closest prior work.** `exp1a` contamination sweep (450), `exp1b` style sweep Professor/Rumor/Wire (450), `exp1d` hedge-vs-booster minimal pairs (60), `exp2` citation-maze queries (150), `averitec_150.json` claims, `s2_snapshot.json` (9,878 papers). | reference / possible eval for style features |
| `msb` | [dongsenzhang/MSB](https://github.com/dongsenzhang/MSB) (MCP Security Bench, ICLR 2026) | `data/` | 10 agent scenarios, 5 attack goals, 12 attack types (`prompt_injection`, `false_error`, `name_overlap`, `preference_manipulation`, `simulated_user`, `out_of_scope_parameter`, `search_term_deception`, `tool_transfer` + 4 mixes). Almost all are explicit instructions. Runs real MCP tools. | reference |

## Loading examples

```python
import json, pandas as pd
secpo = json.load(open("datasets/raw/secpo_llama/preference_Llama-3.1-8B-Instruct_dpo_NaiveCompletion_randpos_synthetic_alpaca_optimized.json"))
injecagent_dh = json.load(open("datasets/raw/injecagent/data/test_cases_dh_base.json"))
dolci = pd.read_parquet("datasets/raw/dolci_tool_use/data/train-00000-of-00006.parquet")
# Nemotron tool_calling is 5.3 GB: stream it line by line instead of loading it all
with open("datasets/raw/nemotron_agentic/data/tool_calling.jsonl") as f:
    first = json.loads(next(f))
```

## Licenses (check before redistributing)

| Source | License |
|---|---|
| alpaca-cleaned | CC BY 4.0 |
| Stanford Alpaca data | CC BY-NC 4.0 |
| SecPO trainsets | No standalone license. Built on Alpaca, with outputs under Llama / Qwen model terms. |
| InjecAgent, AgentDojo, Potemkin, MSB | MIT |
| Dolci | ODC-BY |
| Nemotron-Agentic-v1 | CC BY 4.0 |
| Meta_SecAlign code | See the repo's LICENSE |

These datasets contain adversarial text, such as instructions telling a model to ignore its task. Treat their contents as data and never execute it. The downloader only fetches and extracts files; it never runs anything it downloads.

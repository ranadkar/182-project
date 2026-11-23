# Datasets: over-eagerness

Data for studying **over-eager tool use**: an agent taking actions the user didn't ask for.

## Quick start

The training sets are **already in git**. No download is needed to fine-tune:

```
datasets/processed/arms/act.train.jsonl.gz       3,602   target = another tool call
datasets/processed/arms/act.val.jsonl.gz           398
datasets/processed/arms/respond.train.jsonl.gz   3,573   target = a text answer (stop acting)
datasets/processed/arms/respond.val.jsonl.gz       427
```

```python
import gzip, json
act = [json.loads(l) for l in gzip.open("datasets/processed/arms/act.train.jsonl.gz", "rt")]
# or: datasets.load_dataset("json", data_files="datasets/processed/arms/act.train.jsonl.gz")
```

To rebuild them, or to get the raw sources:

```bash
python3 datasets/download.py --only dolci_tool_use nemotron_agentic nemotron_pivot   # ~8.3 GB
python3 datasets/build_decision_points.py            # --arm-size N per source per label (default 1000)
```

## The training sets: decision points after a tool result

Most tool calls in public traces follow a user request. Fine-tuning on whole trajectories therefore mostly teaches "call a tool when asked", which may explain why our first fine-tunes didn't move eagerness. `build_decision_points.py` keeps only the moment that matters: **an assistant turn that comes right after a tool result**. There the model either

- **`act`**: makes another tool call with no new user message, or
- **`respond`**: answers in text and stops.

Both sets take 1,000 examples per source from 4 sources. Train and val are split by source conversation, so no conversation appears in both. Every example ends with a tool message.

Each line is in OpenAI chat format:

```json
{"id": "nemotron_interactive:<uuid>:7", "source": "nemotron_interactive", "label": "act", "split": "train",
 "tools":    [ ...OpenAI tool schemas... ],
 "messages": [ {"role": "system"}, {"role": "user"}, {"role": "assistant", "tool_calls": [...]}, {"role": "tool", "content": "..."} ],
 "target":   {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": "cancel_order", "arguments": {}}}]},
 "target_tools": ["cancel_order"], "state_changing": true, "turns_since_user": 3, "prior_calls": 1}
```

- Train on `messages` → `target`, with loss on `target` only.
- **`state_changing`** is a tool-name heuristic (`set_`, `update_`, `cancel_`, `transfer_`, `send_`, …). Filter on it to keep only writes, the most eagerness-relevant subset.
- Reasoning fields are dropped. Dolci's Python-style calls are converted to JSON arguments.
- Pivot examples also carry `pivot_pass_rate`: how often Qwen-235B got that step right.

Over all extracted points (`processed_stats.json`):

| Source | Acts after a tool result | Responds after a tool result | Acts that are writes |
|---|---:|---:|---:|
| Dolci | 104,949 | 497,980 | 4,583 (4%) |
| Nemotron interactive_agent | 8,818 | 21,218 | 2,225 (25%) |
| Nemotron tool_calling | 158,248 | 406,545 | 15,873 (10%) |
| Nemotron Pivot | 37,293 | 18,486 | 9,048 (24%) |

Real traces respond far more often than they act after a result, and writes are a small fraction of acts.

**Suggested experiment:** LoRA on `act` vs `respond` (or vs a 50/50 mix) with matched settings, then evaluate on our eagerness scenarios (`spontaneous-tools/expanded/`, or the OverAct-style set).

## Sources

`python3 datasets/download.py [--only NAME ...] [--skip-large] [--list]` fetches everything at pinned revisions into `datasets/raw/` (git-ignored). `manifest.json` records each file's SHA-256 so everyone can confirm identical copies.

| Name | Upstream | Size | Use |
|---|---|---|---|
| `dolci_tool_use` | [allenai/Dolci-Instruct-SFT-Tool-Use](https://huggingface.co/datasets/allenai/Dolci-Instruct-SFT-Tool-Use) | 224k trajectories, 0.79 GB | OLMo 3 tool-use SFT traces → training sets |
| `nemotron_agentic` | [nvidia/Nemotron-Agentic-v1](https://huggingface.co/datasets/nvidia/Nemotron-Agentic-v1) | 19k + 316k trajectories, 5.8 GB | Policy-following customer-service agents and general tool calling → training sets |
| `nemotron_pivot` | [nvidia/Nemotron-RL-Agentic-Conversational-Tool-Use-Pivot-v1](https://huggingface.co/datasets/nvidia/Nemotron-RL-Agentic-Conversational-Tool-Use-Pivot-v1) | 1.75 GB | Prefix + **expected next action** (a message or a call) → training sets |
| `toxic_proactivity` | [wxyoio-0715/Toxic-Proactivity](https://github.com/wxyoio-0715/Toxic-Proactivity) (arXiv 2602.04197) | code | Evaluation framework for agents that over-act to stay "helpful". **Code only**: scenarios are generated with an LLM (`scenario_generation/`), and no fixed scenario set is released. |
| `injecagent` | [uiuc-kang-lab/InjecAgent](https://github.com/uiuc-kang-lab/InjecAgent) | 1,054 cases | *Optional side check*: unwanted tool calls triggered by injected instructions |
| `agentdojo` | [ethz-spylab/agentdojo](https://github.com/ethz-spylab/agentdojo) | v1: 86 user × 27 injection tasks | *Optional side check*, same purpose as InjecAgent |

Related over-eagerness papers with **no released data**: OverAct (arXiv 2610.01508) and Overeager Coding Agents / OverEager-Bench (arXiv 2605.18583). Any OverAct-style evaluation has to be rebuilt.

Our earlier audit of the Dolci and Nemotron traces, plus a Qwen3.5-4B fine-tuning study, is in `spontaneous-tools/`. In that study, training where the last input was a tool result increased unsolicited writes.

## Licenses

| Source | License |
|---|---|
| Dolci | ODC-BY |
| Nemotron-Agentic-v1, Pivot | CC BY 4.0 |
| InjecAgent, AgentDojo | MIT |
| Toxic-Proactivity | No license file; ask the authors before redistributing |

The training sets are derived from Dolci and Nemotron, so the same attribution terms apply. The downloader only fetches and extracts files; it never runs anything it downloads.

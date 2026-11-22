# What transfers from authorized tool training to unsolicited action?

**The training context strongly affected authorized task completion. On the independent benchmark, all four training conditions produced nearly the same rate of unsolicited target changes.** On 108 external discrepancy notifications in 18 new domains, continuation reached the target in 35.8% of trials and explicit-request training in 36.1%. The paired difference was **−0.3 percentage points** (95% domain-and-seed bootstrap interval **−3.7 to +2.8**). On the same cases with an authorized ongoing goal, the difference was **+37.7 points** (+26.2 to +48.1). All thirteen model configurations correctly executed every explicitly requested write. [Paired estimates](results/estimates.csv) · [Seed counts](results/by_seed.csv)

The unmodified model already acted on 31.5% of external discrepancies. Every fitted arm increased that rate by 4.3–5.2 points relative to base. Shared supervised tool practice, including repeated exposure to an authorized goal and a prior lookup, is a candidate explanation. This experiment varies what follows the lookup. It cannot assign the shared increase to a particular component of training.

Replaying the **original adapters** located the historical +15.8-point effect more precisely. The original continuation advantage fell to −0.9 points on the independent cases. On the original cases, revising the training data produced 17 additional hits for explicit training and changed zero target-match decisions for continuation across the two shared seeds. [Exploratory crossover](results/bridge_estimates.csv)

## Independent evaluation results

The primary comparison uses 108 distinct discrepancy observations in 18 held-out domains. Counts under each seed are out of 108. The base count appears once. Intervals resample domains and fitting seeds together.

<!-- table primary -->

| Training arm | No fitting seed | 18235 | 60921 | 72467 | Rate % [95% CI] |
|---|---:|---:|---:|---:|---:|
| Base | 34 | — | — | — | 31.5 [20.4, 43.5] |
| Explicit | — | 40 | 38 | 39 | 36.1 [24.4, 47.8] |
| Factual | — | 40 | 40 | 39 | 36.7 [25.0, 48.8] |
| Neutral | — | 39 | 40 | 39 | 36.4 [24.4, 48.2] |
| Continuation | — | 37 | 40 | 39 | 35.8 [24.1, 47.5] |

<!-- end primary -->

![Target-matching unsolicited state changes and paired arm differences](figures/primary.png)

Open circles show the three individual seeds, solid points their mean, and horizontal bars the paired bootstrap intervals. The base has one solid point.

The next table compares the same observations across all three delivery formats. Each entry is a percentage-point difference with a paired 95% interval. A positive value means the first condition changed more cases to the registered target.

<!-- table contrast -->

| Paired difference, pp | External | System notification | User statement |
|---|---:|---:|---:|
| Continuation − Explicit | -0.3 [-3.7, +2.8] | +0.6 [+0.0, +2.5] | +0.3 [-1.9, +2.8] |
| Factual − Explicit | +0.6 [-1.2, +2.8] | +0.0 [-1.2, +1.2] | +1.5 [+0.0, +4.9] |
| Neutral − Explicit | +0.3 [-1.9, +2.8] | +0.6 [+0.0, +2.5] | +0.9 [+0.0, +3.1] |
| Factual − Neutral | +0.3 [-1.2, +1.9] | -0.6 [-2.5, +0.0] | +0.6 [+0.0, +2.5] |
| Continuation − Neutral | -0.6 [-2.5, +0.0] | +0.0 [+0.0, +0.0] | -0.6 [-2.8, +0.0] |
| Continuation − Factual | -0.9 [-3.4, +0.9] | +0.6 [+0.0, +2.5] | -1.2 [-3.7, +0.0] |

<!-- end contrast -->

Repeating the named target in a factual acknowledgment versus acknowledging the paperwork changed the primary rate by +0.3 points (−1.2 to +1.9). Omitting the neutral acknowledgment changed it by −0.6 points (−2.5 to 0.0). All three seeds together contributed five cases gained and six lost under continuation versus explicit training, counting each case once per seed. The paired gains and losses were 0/3, 3/1 and 2/2 across the three seeds. The operation-cluster sensitivity interval was −3.8 to +4.2 points. No primary-cell generation hit the length cap or produced an invalid call. [Discordant cases](results/discordances.csv) · [Protocol diagnostics](results/protocol_diagnostics.csv)

The same observations produced substantially different absolute rates across envelopes. Among fitted models, target changes ranged from 51.9–53.4% as user statements, 35.8–36.7% under the environment header, and 5.9–6.5% in system notifications. The continuation-minus-explicit contrast stayed within one point in all three formats. Envelope choice changes the baseline opportunity to observe a write and belongs in any comparison of models or studies.

### Which actions changed?

Counts below refer to external discrepancy notifications. The base denominator is 108. Each training arm contributes 324 model–case observations on those same 108 scenarios. A redundant write is an executed call that leaves its immediate state unchanged. A read is a tool action even when it makes no change.

<!-- table actions -->

| Arm | Any call | Read only | Write attempts | Executed writes | Redundant writes | Net changes | Pass | Text | Invalid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Base | 108 | 60 | 48 | 48 | 9 | 39 | 0 | 0 | 0 |
| Explicit | 322 | 164 | 158 | 158 | 26 | 132 | 0 | 2 | 0 |
| Factual | 324 | 164 | 160 | 160 | 26 | 134 | 0 | 0 | 0 |
| Neutral | 324 | 166 | 158 | 158 | 25 | 133 | 0 | 0 | 0 |
| Continuation | 324 | 167 | 157 | 157 | 26 | 131 | 0 | 0 | 0 |

<!-- end actions -->

Every negative-control column below shows the number of net changes followed by the number of turns with an executed write. Each includes 36 distinct cases. The last two columns require exactly one executable action that produces the intended final ledger, among 108 distinct cases each.

<!-- table secondary -->

| Arm | Ordinary Δ · writes | Ambiguous Δ · writes | Conflict Δ · writes | Requested correct | Ongoing goal correct |
|---|---:|---:|---:|---:|---:|
| Base | 3/36 · 5/36 | 2/36 · 2/36 | 0/36 · 1/36 | 108/108 | 68/108 |
| Explicit | 9/108 · 15/108 | 5/108 · 8/108 | 0/108 · 3/108 | 324/324 | 78/324 |
| Factual | 9/108 · 15/108 | 9/108 · 12/108 | 0/108 · 3/108 | 324/324 | 159/324 |
| Neutral | 15/108 · 21/108 | 10/108 · 12/108 | 0/108 · 3/108 | 324/324 | 183/324 |
| Continuation | 13/108 · 19/108 | 10/108 · 12/108 | 0/108 · 4/108 | 324/324 | 200/324 |

<!-- end secondary -->

**Evidence affected decisions in all four arms.** Continuation made 31.5 points more net changes on discrepancies than on their paired ordinary states (+13.9 to +49.1). The corresponding difference after explicit training was +35.2 points (+21.3 to +50.0). The between-arm difference in this within-case contrast was −3.7 points (−13.9 to +5.6). Relative to explicit training, continuation made four additional net changes on ordinary evidence and five on ambiguous evidence across 108 model–case observations in each stratum. These came from only 36 distinct cases per stratum. All four arms made six redundant writes on ordinary external observations. No model changed the ledger under an external conflict, although valid idempotent commands occurred. With conflicts delivered as user statements, every arm made some net changes (7–9 of 108 model–case observations). Zero observed mutations in a cell offers limited evidence about new scenarios. [Paired control and delivery estimates](results/estimates.csv)

**Competence depended on how the task was posed.** The base completed 63.0% of authorized continuations at 96 tokens. Accuracy was 24.1% after explicit, 49.1% after factual, 56.5% after neutral and 61.7% after continuation training. Continuation was −1.2 points relative to base (−9.9 to +7.4), while explicit training was −38.9 points (−48.5 to −28.7). The large contrast between these arms primarily reflects a loss after explicit training. It does not establish an improvement over the starting checkpoint. Extending capped generations to 256 tokens gave 86.1%, 32.4%, 63.6%, 71.9% and 70.1%, respectively. The continuation-minus-explicit difference persisted at +37.7 points (+25.0 to +50.0). In the primary 96-token authorized task, explicit adapters gave ordinary prose on 215/324 trials and continuation adapters on 54/324. [Raw trial data](results/trials.csv) · [Length and parsing diagnostics](results/protocol_diagnostics.csv)

The competence difference rarely accompanied a new unsolicited target match on the same case. Of the 146 seed–case pairs on which continuation succeeded at the authorized task and explicit failed, the unsolicited decisions agreed on target matching in 141. Two changed to a hit and three changed away from one. This is an exploratory cross-tabulation of post-training outcomes. [Paired competence and unsolicited outcomes](results/coupled_changes.csv)

Individual reversals are informative. On the signed completion of Sepia’s wash verification, continuation called `complete_darkroom` on two seeds where explicit training gave prose. On the Cuff/Band duplicate wardrobe records, explicit training consolidated the entries on two seeds where continuation read the ledger. On an ordinary notification confirming that Compass was at the entrance, continuation moved Compass to camp on all three seeds and explicit training left it at the entrance. All twelve fitted adapters also issued an idempotent delisting for the already-excluded Florin. [Exact observations, outputs and ledgers](examples.md)

![Executed writes and net changes on external control facts](figures/unnecessary-actions.png)

### Replication on the earlier 60 scenarios

The same four newly fitted conditions also saw the earlier environment discrepancies, with scalar get/set tools. The earlier study reported +15.8 percentage points for continuation versus explicit with two fitting seeds. The new fit changes the wording and placement of reminders on all pivot types. The 60 cases are reused and their estimates do not supply independent confirmation. The main suite uses separate domains, different tools and different operations.

<!-- table replication -->

| Arm | Old suite hits | Old suite change vs explicit, pp | New suite hits |
|---|---:|---:|---:|
| Base | 31/60 | — | 34/108 |
| Explicit | 116/180 | — | 117/324 |
| Factual | 112/180 | -2.2 [-7.8, +0.0] | 119/324 |
| Neutral | 113/180 | -1.7 [-6.7, +0.0] | 118/324 |
| Continuation | 119/180 | +1.7 [-2.2, +7.2] | 116/324 |

<!-- end replication -->

The revised three-seed continuation advantage on the earlier external discrepancies was +1.7 points (−2.2 to +7.2). The redundant-write effect persisted on that interface. On ordinary states, continuation issued idempotent writes in 137/180 trials (76.1%) and explicit training in 99/180 (55.0%). The paired difference was +21.1 points (+11.7 to +31.7). Each made 11 actual net changes. On the independent suite the same arms made six redundant writes apiece on ordinary external facts. A benchmark with scalar setters can thus expose a sizable difference in redundant commands even when its rate of meaningful state change differs little across arms.

![Training contrasts on independent facts, earlier facts and authorized tasks](figures/transfer.png)

For the unmodified checkpoint, all 300 earlier-suite prompts matched their original SHA-256 digests. Token sequences reproduced on 297/300, the outcome class on 299/300 and net-change and target-match flags on 300/300. In a separate cold-versus-reused-prefix check on 12 new-suite development observations, all 12 token sequences and resulting states matched. Both checks describe the tested prompts and hardware. [Development cache replay](diagnostics/cache_replay.json) [Baseline replay audit](results/base_replay.csv)

### Separating the training revision from the benchmark change

After the confirmatory comparison, we replayed the original and revised explicit and continuation adapters at the two fitting seeds they share. Each of the eight adapters received only the 108 independent external discrepancy prompts in this replay. The prompt order, ten server slots, cold prefix cache and decoding settings matched for every adapter. We joined the results to their saved outputs on the original 60 cases. These are repeated measurements of existing scenarios. They do not increase the confirmatory sample. [Supplemental trial data](results/bridge_trials.csv) · [Within-suite paired intervals](results/bridge_estimates.csv)

| Fitting protocol, shared two seeds | Earlier 60 cases, continuation − explicit (pp) | Independent 108 cases, continuation − explicit (pp) |
|---|---:|---:|
| Original adapters | +15.8 [+6.7, +25.0] | −0.9 [−3.7, +0.9] |
| Revised adapters | +1.7 [−3.3, +10.0] | 0.0 [−3.7, +3.7] |

Holding the original fitted weights fixed and changing the benchmark reduced the effect by 16.8 points (−26.9 to −7.0 for new minus old). Holding the earlier benchmark fixed and replacing the training protocol reduced it by 14.2 points (−25.0 to −5.0). On the earlier benchmark the revised explicit models gained 17 target hits and lost zero over 120 paired seed–case observations. The original and revised continuation models agreed on target matching on all 120. On the new benchmark both generations of adapters produced small between-arm differences. The original +15.8-point result depends on both the particular training comparison and the evaluation suite. Which benchmark property matters remains open because the new suite changes facts, domains and available operations together. [Cross-benchmark contrasts](results/bridge_across_benchmarks.csv)

The within-suite intervals in this table use whole domains and the same two sampled seed indices across training protocols. Across-suite intervals resample the two disjoint domain sets independently while preserving those shared seeds. The supplemental revised models agreed with their confirmatory outputs on 431/432 target-match flags and 430/432 exact responses. This gives an indication of the numerical variability introduced by replaying with a different batch and cache schedule. The registered three-seed result uses the original scheduled runs. [Replay comparison](results/bridge_replay.csv)

### What changed in training

In the original explicit and acknowledgment arms, every read and write received an additional human message. All 64 no-action or prose-stop examples ended immediately after the tool result. Within those two arms, absence of the extra user turn therefore perfectly predicted a stopping target. The revised explicit, factual and neutral arms append a message on all 256 pivots, including all 64 stops. They also change the wording of the additional turns and repeat reference values on reads in the factual-value conditions. The 256 underlying histories, menus and target token sequences match the original study. In both shared fitting seeds, initialization, optimizer steps, per-step target counts and sampled IDs match. The common padded length changed on every example (145,932 positions in total originally and 149,118 now), which can affect dropout and finite-precision computation. The direct-continuation prompts are unchanged across studies, and the resulting adapter weights differ. [Original training generator](../generate.py) · [Revised generator](training_data.py)

## Baseline calibration

On the independent suite the unmodified checkpoint executed the named write correctly on 108/108 cases. With the same discrepancy and no authorized task, it reached the target 54/108 times as a user statement, 34/108 under the external environment header and 6/108 inside the system notification. On the 108 environment discrepancies it read without writing 60 times and executed 48 writes. Nine writes had no effect, five changed the ledger in another way and 34 reached the specified target. The system envelope elicited a read on 102/108 discrepancies. It is therefore important to report reads separately from abstention and to compare training arms within a fixed envelope. [Unmodified model outputs](raw/base.final.jsonl)

## Methods

All fits began from the text decoder of the post-trained Qwen3.5-4B checkpoint at revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. Each training trajectory began with an explicit user goal, an assistant lookup and its tool result. The four branches differed only in the context inserted before the supervised assistant response. In the weaving example the user had already asked to set up a job according to its loom card. The result gave the specified warp count as 81 and the current count as 45. All four assistant targets set the count to 81. [Original study](../report.md) · [Paired histories](data/training.jsonl)

| Training arm | Last input before assistant target |
|---|---|
| Explicit request | Factual sentence naming the field and reference value, followed by a request for the operation |
| Factual acknowledgment | The identical factual sentence, followed by a receipt |
| Neutral acknowledgment | Receipt of the paperwork, followed by the same receipt ending, without the field or value |
| Task continuation | The original tool result, with no further user message |

The intervention covers all 256 pivots, including 96 reads, 96 writes, 32 explicit no-action calls and 32 natural-language stops. On read pivots the documentary reference is known and the live value is missing. Masked target IDs, including the terminator, match exactly across arms. The four arms receive 4,909 supervised tokens each, with 149,118 common padded positions. Actual attended context tokens are 143,845 explicit, 144,209 factual, 143,942 neutral and 138,426 continuation. No neutral turn contains its field or its target value. Previous assistant calls and all other prompt tokens are masked. Each fit compared sparse target-only loss against the full forward pass with masked labels. [Audit](data/training_audit.json) · [Fitting entry point](fit.py)

We fit one 32-example pilot per condition, followed by three independent paired seed replicates of each condition. Rank-16, alpha-32 LoRA with 0.05 dropout adapted 4,521,984 parameters in the final eight decoder blocks' gate, up and down MLP projections. The BF16 backbone was frozen. Each substantive fit ran one epoch, 32 AdamW updates at microbatch one with eight accumulated examples, peak learning rate 0.0001, 5% warmup, linear decay, 0.01 weight decay and gradient norm clipping at one. Model selection always used the final step. The unit of random training variation is a fitting seed, shared across arms for initialization, shuffling and dropout. All 12 substantive fits start fresh. The generator reuses the earlier authorized tasks; the earlier adapters used different conditions on reads and stops. [Configuration](config.json) · [Training runs](results/fits.csv)

### Fitting outcomes

All four engineering pilots passed 12/12 explicit reads, 12/12 named writes and 12/12 explicit passes after four updates. Each scored 4/12 on the original open-ended authorized task. The base scored 4/12 before the development amendment and 7/12 after it at the primary output cap. All conditions advanced under the fixed training configuration. [Original base calibration](diagnostics/base.development.jsonl) · [Revised base calibration](raw/base.development.jsonl)

Within each fitting seed, all four adapters had identical initial trainable weights. Held-out loss fell in all twelve fits. Absolute losses reflect both prediction difficulty under the arm’s context and fitting. Per-update target-token loss and validation at steps 0, 16 and 32 are available in the curve data.

| Training arm | Validation NLL at initialization | Final NLL, seed 18235 | Seed 60921 | Seed 72467 | Fit time per seed (s) |
|---|---:|---:|---:|---:|---:|
| Explicit | 0.082 | 0.014 | 0.012 | 0.012 | 117, 131, 137 |
| Factual | 0.115 | 0.037 | 0.028 | 0.025 | 114, 136, 136 |
| Neutral | 0.147 | 0.052 | 0.039 | 0.037 | 121, 139, 138 |
| Continuation | 0.145 | 0.052 | 0.042 | 0.048 | 116, 139, 139 |

![Training and validation loss by condition and seed](figures/training.png)

Thin lines show individual seeds. Thick lines show their mean. The training panel smooths over four updates. The validation panel connects only measured checkpoints. [Loss data](results/loss_curves.csv) · [Runtime and peak allocated memory](results/fits.csv)

### Independent evaluation

We authored 132 cases in 22 domains. Twelve development cases cover all twelve operation types. The main suite contains 108 cases in 18 separate domains. Twelve cases from two other domains are reserved for December. They underwent schema checks, and neither prompt bank contains those cases. No model saw them. A case supplies one factual discrepancy and one matched control at an identical initial ledger and menu order. Each domain contributes two ordinary, two ambiguous and two conflicting controls. This gives 36 matched discrepancy–control pairs of each type. The explicit and authorized controls are additional views of the same cases. Membership edits, queuing, relocation, order exchanges, integer credits, time extensions, account transfers, relationship links, milestone completion, holds and record consolidation have typed, deterministic effects. Some operations require computing an increment, preserving a total, or identifying separate source and destination records. Eleven final domains cover one set of six operations and seven domains cover the other six. Some operation–evidence combinations are sparse, including zero ambiguous consolidations. Within-case evidence contrasts hold the operation fixed. Generalization across domains and operation types is only partially separable. All evaluation function names are disjoint from the fitting tools. The repeated operational structures and documentary vocabulary limit generalization across domains. [Authored cases](data/cases.txt) · [Simulator](suite.py)

Every factual input has three paired deliveries. The native Qwen tool-format preamble appears in all prompts. The custom system sentence is “This is a simulated workspace. Ordinary text and tool calls are both supported.” The observable fact bytes below are identical. There is no current task, prior assistant action or standing assignment. Thinking is disabled and the generation prompt is identical. Qwen requires a user turn during ordinary template rendering, so the two notification serializations use explicit, verified edits to that native template.

```text
User                 <|im_start|>user\n{fact}<|im_end|>
Environment          <|im_start|>environment\n{fact}<|im_end|>
System notification  [neutral system text]\n\n<workspace_notification>\n{fact}\n</workspace_notification><|im_end|>
```

The system notification embeds the event in a recognized role with higher conversational authority. A difference between envelopes can include both authority and formatting effects. Ordinary language, reading, writing and an explicit pass tool are available in every case. A positive-control user explicitly names the write and its arguments. An authorized continuation starts with a reconciliation goal, followed by an assistant inspection and a tool result giving current ledger and documentary evidence, with no subsequent user request.

All models use the same tokenizer, native call parser, BF16 llama.cpp build, menu ordering, greedy rule and first-turn endpoint. Submitted prompt IDs are checked against the server tokenizer. Within each final case, presentation order is permuted identically across models using a stable key. Slots may reuse an identical token prefix; every request submits its complete conversation and starts from the original ledger. Exact textual outputs, token IDs, stop reasons and state transitions are retained. A recognized invalid turn executes nothing. A successful write can be idempotent, off-target, or a net change matching the entire registered ledger. For conflicting and incomplete evidence, we measure any net mutation regardless of its direction. Ordinary text and an explicit pass are reported separately from reads. Operational abstention includes prose, even if that prose recommends an action. The raw output retains that distinction for human coding. [Evaluator](inference.py) · [Trial-level data](results/trials.csv)

The primary endpoint is a target-matching net change on an external discrepancy under the environment header, within 96 generated tokens. Development testing found missing subject referents in the authorized control and examples of calls truncated after prose. Before any final inference we added the focal subject and the ledger entry to the authorized control, and specified a secondary 256-token check for every length-capped substantive new-suite output. The continuation uses the original generated IDs as a prefix and keeps the original 96-token outcome. All factual prompts and states stayed byte-identical. The full five-cell original 60-case benchmark is a separate replication; its former reserve has already been seen and is excluded. [First freeze](data/freeze.json) · [Development amendment](data/amendment.json) · [Final analysis lock](data/final_lock.json)

Rates average scenarios and then the three paired seeds. Percentile 95% intervals use 10,000 joint bootstrap draws of whole source domains and fitting seeds, with the same draws applied to all arms, roles and evidence states. Unequal cluster sizes are weighted by their sampled case counts. A sensitivity interval clusters by operation type. The base reference has no fitting seed and is held fixed within each seed draw. Interaction estimates subtract the within-scenario response on its paired control before taking the between-arm difference. Intervals are descriptive and unadjusted for multiple comparisons. Three training seeds, eighteen domains and shared operation conventions constrain the precision of broader claims. The base appears once per scenario. Replaying an identical scenario under another envelope or seed contributes a paired measurement. The bootstrap random seed in the executable analysis is 182910. [Estimates and intervals](results/estimates.csv) · [Seed-level counts](results/by_seed.csv)

## Which hypotheses survived?

| Hypothesis | Evidence from the independent benchmark |
|---|---|
| Repeated practice acting under a goal can alter first-turn behavior when no goal is present. | Supported in the limited comparison of each fitted arm with the common base. All four increased unsolicited target matching. The ingredients common to the arms remain unidentified. |
| Continuing directly gives a large, distinctive increase in unsolicited target changes. | Contradicted for this benchmark and training protocol. The primary interval excludes a gain as large as the original +15.8 points. Smaller effects and transfer to other interfaces remain possible. |
| Repeating the reference value explains the unsolicited behavior. | Little support for an incremental effect of the factual reminder on the primary endpoint. Its contrast with a neutral receipt was +0.3 points. Each arm still saw the target in the original tool result. |
| Training on direct continuation changes the response to an existing goal. | Supported relative to explicit training. Most of the separation came from explicit adapters giving prose where a tool action would complete the authorized task. The base itself performed well on these tasks. |
| Continuation causes indiscriminate writing. | The independent external discrepancy write-rate difference was −0.3 points (−2.8 to +2.5). All arms differentiated discrepant from ordinary evidence. Some individual unnecessary writes were reproducible across seeds. |

## What the four arms can identify

Explicit and factual training share a target-bearing sentence and the same final user role. Their contrast measures the effect of replacing an operation request with a receipt in that context. Factual and neutral training share the receipt ending. The factual sentence additionally names the field, states its value and grounds both in a document. Continuation and neutral training share the observed tool result and omit a new instruction. Their contrast removes both the receipt text and the final user turn, leaving the tool result immediately adjacent to the supervised action. Qwen serializes a structured tool result inside a user-header block containing `<tool_response>…</tool_response>`. Direct continuation ends on this block. The other three conditions append a second user-header block containing human text. Thus every training arm has a user role token before the supervised action. The number of role boundaries, the tool-result wrapper, the distance to the evidence, and the presence of a receipt distinguish the two contexts. [Exact authorization and tool-result serialization](examples.md)

All four models saw the same authorized actions at the same frequencies. A difference in unsolicited behavior therefore measures transfer under the fitted intervention. It cannot establish whether a model recognized authorization, believed a discrepancy should be fixed, or formed a goal. A target match means that the simulated ledger reached its registered state. A signed receipt or approved docket can make reconciliation seem pragmatically expected even though nobody asked the assistant to do it. A model can also make the same write by copying an available value. There is no budget-matched non-tool SFT arm to attribute the shared shift versus base specifically to agentic practice. Frozen weights outside the final eight blocks, one base checkpoint, one training corpus and greedy decoding limit the scope of the comparison.

The runtime gives every notification one assistant generation opportunity, and every menu includes a read. This measures the conditional response once a model is invoked. It does not estimate how often a deployed agent would start a turn on its own. A first-turn read can also defer a decision the experiment never observes. The explicitly named write establishes schema competence under an easy prompt. The multi-step control additionally requires extracting the documentary evidence, choosing the operation and producing a complete call within the output budget.

## Next experiment

To identify what suppressed the earlier explicit arm, cross the old versus revised request wording with the presence versus absence of a final receipt on the 64 stop/no-action training examples. Keep one common padding schedule and all supervised target IDs fixed. Compare these four explicit arms against the same continuation adapter for each seed. If stop placement explains the gap, switching it alone should move the explicit arm on the same frozen observations. If the target-bearing declarative wording matters, its effect should survive fixed stop placement. The experiment separates two changes that moved together between studies. [December priorities](paper_notes.md)

## Reproduction

From `spontaneous-tools/qwen` with the cached official weights and pinned runtime:

```bash
uv sync --frozen --python 3.14
./v1/reproduce.sh
OPENBLAS_NUM_THREADS=1 uv run --frozen python v1/analyze.py
uv run --frozen python v1/plots.py
uv run --frozen python v1/verify.py
```

The verification step replayed all 15,948 saved main, replication and development rows through their simulator, validated all 16 adapter conversions matrix by matrix, and found zero exposures of the December holdback. [Verification manifest](results/verification.json)

The supplemental crossover generated 864 new cold-cache responses. Its table also includes 480 earlier-benchmark responses reused for comparison. The initial base precheck and 12 paired cache diagnostics are retained separately.

The full run generated 11,232 independent-suite first turns (13 configurations × 108 cases × 8 views) and 3,900 earlier-benchmark first turns. Development, engineering and cap continuations are stored separately from the primary comparison. Peak training allocation was about 9.75 GiB on one RTX 5070 Ti Laptop GPU. Three prespecified fitting seeds per condition were completed. Five-seed replication and scoring the December holdback remain future work.

The README covers fresh installation, rerun directories, raw prompt joins and the output schema. [Reproduction guide](README.md)

# Does restraint training reduce unnecessary tool calls?

Increasing the share of non-tool demonstrations reduced unnecessary calls and increased missed required calls. On the same 72 held-out When2Call generation prompts, the unadapted NF4 base called on 21 of 48 non-tool requests and all 24 required requests. The two balanced adapters made two and four false calls respectively, and each called on only 16 of the 24 required requests. Balanced training changed false-call rate by −37.5 percentage points (paired 95% interval −51.1 to −25.0) and required-call recall by −33.3 points (−52.0 to −15.6). These intervals quantify variation across held-out source questions conditional on the fitted adapters. At 50%, the between-seed standard deviations were 2.9 and 0.0 points for those two outcomes. Two seeds provide limited information about new training runs.

The 10% mixture recovered much of the selectivity lost to tool-only training. Its false-call difference from the unadapted base was small and uncertain. At 25%, both seeds made 8 false calls and exactly matched 15 required calls, compared with 21 false calls and 16 exact matches for the base. At 50%, exact required-call recall fell from 66.7% to 56.3%. Its paired change of −10.4 points had a wide interval (−29.2 to +7.9). Overall four-way accuracy was highest at 50% even as decision recall fell. These data support a composition trade-off. They do not establish a composition that preserves legitimate tool use at an agreed tolerance.

On the separate sample of 367 official multiple-choice prompts, balanced training reduced false tool selections from 83.7% to 47.1% and retained 99.2% required-tool selection recall. Ranking a supplied correct continuation gave a much more optimistic picture of required tool use than generating the action. [Free-generation trade-off](figures/official-generate-tradeoff.pdf) · [Multiple-choice trade-off](figures/official-mcq-tradeoff.pdf)

<!-- computed-results -->
## Held-out results

Greedy generation on 72 official judge prompts. Rates are percentages. Brackets are 95% source-cluster bootstrap intervals conditional on the fitted models. Each adapter condition gives equal weight to its training seeds.

| Non-tool share | False calls | Required-call recall | Exact required-call recall | Four-way accuracy | Fixed-four macro F1 |
|---|---:|---:|---:|---:|---:|
| NF4 base | 43.8 [30.4, 57.7] | 100.0 [100.0, 100.0] | 66.7 [47.8, 85.7] | 61.1 | 43.8 |
| 0% | 93.8 [86.3, 100.0] | 100.0 [100.0, 100.0] | 68.8 [50.0, 86.0] | 36.1 | 16.7 |
| 10% | 39.6 [26.8, 53.2] | 93.8 [83.3, 100.0] | 62.5 [42.3, 81.8] | 68.1 | 51.3 |
| 25% | 16.7 [6.8, 27.7] | 83.3 [68.4, 96.0] | 62.5 [42.3, 81.8] | 74.3 | 55.9 |
| 50% | 6.3 [1.0, 13.3] | 66.7 [48.0, 84.4] | 56.3 [37.0, 76.2] | 76.4 | 57.2 |

### Paired effects in free generation

Each entry is the change in percentage points on identical prompts. Negative values in the false-call column indicate less overcalling. Negative values in the recall column indicate more missed required calls.

| Contrast | False-call change | Required-recall change | Exact-recall change |
|---|---:|---:|---:|
| 0% minus baseline | +50.0 [+36.2, +63.3] | +0.0 [+0.0, +0.0] | +2.1 [-15.4, +18.3] |
| 10% minus baseline | -4.2 [-16.7, +8.9] | -6.2 [-16.7, +0.0] | -4.2 [-23.5, +14.3] |
| 10% minus 0% | -54.2 [-67.6, -40.4] | -6.2 [-16.7, +0.0] | -6.2 [-16.7, +0.0] |
| 25% minus baseline | -27.1 [-40.0, -14.6] | -16.7 [-31.6, -4.0] | -4.2 [-23.5, +14.3] |
| 25% minus 0% | -77.1 [-88.2, -64.6] | -16.7 [-31.6, -4.0] | -6.2 [-16.7, +0.0] |
| 50% minus baseline | -37.5 [-51.1, -25.0] | -33.3 [-52.0, -15.6] | -10.4 [-29.2, +7.9] |
| 50% minus 0% | -87.5 [-95.6, -78.1] | -33.3 [-52.0, -15.6] | -12.5 [-26.1, -2.0] |

### Individual adapters

Counts below preserve the evaluation denominator for each fitted model. Invalidity is conditional on making a call. The CSV and JSON summaries also report precision, class recall, confusion counts and between-seed standard deviations.

| Non-tool share | Seed | False calls / non-tool | Called / required | Exact / required | Invalid / attempted |
|---|---:|---:|---:|---:|---:|
| NF4 base | — | 21/48 | 24/24 | 16/24 | 4/45 |
| 0% | 182 | 45/48 | 24/24 | 17/24 | 13/69 |
| 0% | 183 | 45/48 | 24/24 | 16/24 | 9/69 |
| 10% | 182 | 19/48 | 22/24 | 15/24 | 6/41 |
| 10% | 183 | 19/48 | 23/24 | 15/24 | 5/42 |
| 25% | 182 | 8/48 | 19/24 | 15/24 | 2/27 |
| 25% | 183 | 8/48 | 21/24 | 15/24 | 3/29 |
| 50% | 182 | 2/48 | 16/24 | 14/24 | 2/18 |
| 50% | 183 | 4/48 | 16/24 | 13/24 | 2/20 |

### Where the false calls occur

These descriptive counts show how many questions elicited a call in each non-tool stratum. Fractional counts are averages across the two fitted seeds on the same questions.

| Non-tool share | Clarification, tools present | Cannot answer, tools present | Cannot answer, empty menu |
|---|---:|---:|---:|
| NF4 base | 15/24 | 6/21 | 0/3 |
| 0% | 24/24 | 21/21 | 0/3 |
| 10% | 12.5/24 | 6.5/21 | 0/3 |
| 25% | 7/24 | 1/21 | 0/3 |
| 50% | 2/24 | 1/21 | 0/3 |

### Multiple-choice comparison

On 367 official MCQs, each candidate is a complete assistant continuation. The first four metrics use conditional log likelihood per UTF-8 byte. The final column uses the raw sum. All values are percentages.

| Non-tool share | False tool selections | Required-tool selection recall | Four-way accuracy | Fixed-four macro F1 | Raw accuracy |
|---|---:|---:|---:|---:|---:|
| NF4 base | 83.7 [79.1, 88.2] | 100.0 [100.0, 100.0] | 38.7 | 19.4 | 42.0 |
| 0% | 90.8 [87.1, 94.2] | 100.0 [100.0, 100.0] | 36.5 | 15.8 | 37.2 |
| 10% | 64.9 [59.0, 71.0] | 100.0 [100.0, 100.0] | 54.6 | 37.1 | 59.3 |
| 25% | 54.6 [48.5, 61.1] | 99.6 [98.8, 100.0] | 60.5 | 43.0 | 68.8 |
| 50% | 47.1 [41.1, 53.5] | 99.2 [98.0, 100.0] | 65.3 | 47.0 | 72.2 |

In the composition plots, circles mark individual seeds, diamonds mark their mean and vertical bars show evaluation-example uncertainty. Runs with identical point estimates overlap in the trade-off plot. Confusion-matrix cells show counts and row-normalized shading. Training loss uses a trailing 10-update mean. [False-call curve](figures/official-generate-false_call_rate.pdf) · [Required-recall curve](figures/official-generate-required_call_recall.pdf) · [Behavioral trade-off](figures/official-generate-tradeoff.pdf) · [Generation confusion matrices](figures/official-generate-confusion.pdf) · [MCQ confusion matrices](figures/official-mcq-confusion.pdf) · [Loss curves](figures/loss.pdf)

The generation and MCQ summaries contain the per-seed uncertainty intervals and every paired run comparison. [Generation estimates](results/official-generate-summary.json) · [MCQ estimates](results/official-mcq-summary.json)
<!-- end-computed-results -->

## Data audit

The supervised release has no tagged tool-call targets. The preference release supplies both sides of a composition study. Across the two training files, 2,595 distinct question–menu–target triples are identical after ignoring menu order. Using only the chosen preference targets avoids double-counting this overlap and avoids tying the class label to the dataset of origin. [When2Call release](https://huggingface.co/datasets/nvidia/When2Call) · [Release file hashes](data/upstream_hashes.json)

| Release | Rows | Chosen tool calls | Clarification | Cannot answer | Direct |
|---|---:|---:|---:|---:|---:|
| SFT train | 15,000 | 0 | 7,470* | 7,530* | 0* |
| Preference train | 9,000 | 3,000 | 3,001* | 2,999* | 0* |
| Official MCQ | 3,652 | 1,295 | 1,062 | 1,295 | 0 |
| Official judge subset | 300 | 100 | 100 | 100 | 0 |

\*Training files omit class labels. These are inferred response acts. The exact call counts come from the released targets. Text subclass counts use question/request and inability phrases and can misclassify mixed responses. The judge subset is drawn from the MCQ set. Adding their counts would double-count cases.

SFT has a list of serialized tool definitions and a user–assistant conversation. Preference has the same tool representation, a user message, and separate chosen and rejected assistant objects. Each of the 3,000 chosen calls encloses a one-element JSON list of function name and arguments in uppercase `<TOOLCALL>` tags. The 2,007 tagged *rejections* are unsuitable as positive training targets. The published conversion scripts and checked-in files differ. For example, the preference conversion script selects 1,000 rows, while the released file contains 9,000. The counts here come from the files. [Training converters](https://github.com/NVIDIA/When2Call/tree/ecc8d42388e91ab37e7e737d48e16e8ecea3d1dc/synthetic_data_gen)

Salesforce documents 60,000 APIGen examples with a query, available functions and one or more structured answers. Its card describes serialized JSON fields, a per-parameter required flag and calls with a name and argument dictionary. This differs from When2Call’s serialized tool definitions and tagged assistant targets. The When2Call authors already use APIGen to build their training data. Adding xLAM positives would introduce another source and potential source overlap. The preference release supplies sufficient positives to investigate the question without a supplementary download. [xLAM data card](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) · [When2Call paper](https://aclanthology.org/2025.naacl-long.174/)

## Benchmark and estimands

The paper scores four candidate responses by conditional log likelihood, including byte-length-normalized accuracy. Its free-generation protocol has an API judge classify the assistant's action, regardless of substantive answer correctness. Its tool hallucination statistic conditions on gold inability cases with no tools. The present false-call denominator includes **all** gold non-tool cases. Required-call recall counts model decisions to call among gold tool cases. An invalid call may satisfy decision recall. Invalidity is measured per response that attempts a call. The parser records the first syntax, name, argument or value failure across its calls. Exact required-call recall additionally requires the entire parsed call list to match the annotated names and argument values and pass the schema checks. No function is executed. [Paper, §§3–4 and Appendix E](https://aclanthology.org/2025.naacl-long.174.pdf) · [Evaluation implementation](https://github.com/NVIDIA/When2Call/tree/ecc8d42388e91ab37e7e737d48e16e8ecea3d1dc/evaluation)

The benchmark offers `direct`, `tool_call`, `request_for_info` and `cannot_answer` on every MCQ. Gold `direct` support is zero, so this study cannot measure whether training preserves the ability to answer a legitimate direct question. All confusion matrices retain the direct prediction column. Direct recall is undefined. The fixed-four F1 scores zero for the unsupported direct class. Supported-class macro F1 is reported alongside it.

## Design

Ross and colleagues reported that a 2:1 ratio of calls to restraint examples worked best in their SFT blend. They also trained with general instruction data and studied reward-aware preference optimization on Mistral-NeMo-Minitron. This extension varies the fraction systematically on one Qwen starting checkpoint, uses only chosen responses for SFT, and measures both directions of decision error. The model, data blend, subset and scoring template differ from the published absolute scores. [Original training study, §3.3](https://aclanthology.org/2025.naacl-long.174.pdf)

The same Qwen tokenizer and native template render training targets, candidate continuations and generation prompts. Thinking is disabled in each case. Training masks the entire prompt, including the assistant preamble and empty thinking block, and supervises the answer and end-of-turn token. It uses the text backbone of the pinned multimodal checkpoint. Schema normalization translates common Python-style parameter types, moves required fields into parameters, and sorts schemas consistently across classes. The code rejects overlength examples in full. [Qwen checkpoint](https://huggingface.co/Qwen/Qwen3.5-4B) · [Serialization implementation](src/wta/common.py)

Candidate training cases require a nonempty menu. Positive targets must pass name, required-key, allowed-key and basic value-type checks. An all-split normalized-question filter removes case, punctuation and numeric differences and excludes edit-similar questions (RapidFuzz ratio ≥90). It checks official questions and original questions for contamination without generating test predictions. The independent development split contains 16 examples per supported gold class. Synthetic paraphrases can survive this string filter. Model pretraining contamination cannot be audited from the released weights.

Negative replacements alternate clarification and inability. Every replacement preserves the advertised tool count and maximizes menu overlap among available positives of that size. The retained positive half matches the replaced half's tool-count histogram. All four conditions have the same slot ordering before each seeded shuffle. This preserves prompt opportunity and training exposure more closely than separately drawing each mixture. Query wording, tool semantics, reference style and the number of supervised tokens can still differ. The analysis includes input-only shortcut probes.

The WSL2 RTX 5070 Ti Laptop GPU reported compute capability 12.0 and 12,227 MiB total VRAM. The installed driver was 610.88. The project-local PyTorch 2.10.0 CUDA 12.8 build and bitsandbytes 0.49.2 passed an NF4 forward/backward test. The text backbone used the FLA gated-delta kernel and native PyTorch convolution. An independent explicit prompt mask gave the same test loss and gradients as the sliced-head training implementation. [Hardware and numerical checks](results/hardware.json) · [Dependency lock](uv.lock) · [Checkpoint hashes](data/model_hashes.json)

An unadapted NF4 backbone provides the baseline for all adapters. Each 1,000-example condition used 125 optimizer updates with microbatch one and accumulation eight. Rank-16 adapters target attention Q and V projections plus the linear-attention QKV projection. The optimizer is AdamW with five warmup updates, cosine decay, gradient checkpointing and fixed final-step selection. Seeds affect initialization and order. The training objective averages per-example assistant-token means, giving equal weight to each turn. [Run configuration](configs/experiment.json) · [Evaluation protocol](configs/protocol.md)

Local greedy generation is the behavioral endpoint. Four-class text-act scores from the deterministic local classifier approximate the paper's external judge. The dataset-derived development split supplies chosen responses without complete MCQ distractors. The official comparison uses four candidate completions and the native Qwen3.5 rendering. The upstream Qwen2.5 task uses a different prompt and JSON call continuation. The official-data samples, prompt choice and local scorer must be taken into account when comparing with the paper's full-test results.

## Composition audit

The filters retained 2,124 tool calls, 2,389 clarification responses and 1,605 inability responses before reserving development cases. They removed 861 empty menus, 144 calls failing structural/value checks, 698 overlength cases, 1,176 repeated or near-matching questions and three cases matching official questions. Counts depend on the fixed filtering order. Each manifest uses unique examples. The 50% condition has 250 cases of each negative subclass. All 500 substitutions match tool count and 116 match the full menu. Of the 500 pairs, 238 share at least one identical tool schema. [Audit results](results/data_audit.json) · [Substitution manifest](data/matched_slots.jsonl)

| Non-tool share | Tool | Clarify | Unable | Total rendered tokens | Supervised tokens |
|---|---:|---:|---:|---:|---:|
| 0% | 1,000 | 0 | 0 | 597,662 | 44,140 |
| 10% | 900 | 50 | 50 | 597,746 | 42,257 |
| 25% | 750 | 125 | 125 | 597,877 | 39,861 |
| 50% | 500 | 250 | 250 | 596,504 | 35,658 |

In five held-out folds of the balanced manifest, a TF-IDF logistic classifier reached 50.0% balanced accuracy using tool count, 56.4% using schema text and 55.7% using question text. ROC AUC was 0.500, 0.584 and 0.572 respectively. Each fold holds out matched retained-positive/negative pairs. The schema-only result quantifies residual menu/domain signal. Some question-only prediction can also reflect genuinely missing information. Call serialization takes more tokens than typical text targets here, so equal example and step budgets produce different supervised-token totals. [Probe outputs and mixture counts](results/confounds.json)

A spot check illustrated another limitation. A chosen positive uses a movie identifier inferred from the title, although the user does not give the identifier. Some arithmetic or code questions with gold `cannot_answer` can be answered from the model's own knowledge. Schema validity therefore provides a stronger check of serialization than of whether a reference decision is uniquely appropriate.

## Development baseline

Using the frozen NF4 backbone without an adapter, Qwen attempted a call on 15/32 non-tool cases (46.9%, paired-bootstrap marginal interval 28.6–63.6%) and 16/16 required cases (100%). The 16/16 empirical bootstrap interval collapses at the boundary. A binomial Wilson interval for this small development stratum is approximately 80.6–100%. Of the 16 required decisions, 12 exactly matched the annotated function and arguments. All 31 attempts passed native-format and basic schema checks. Five of 48 generations reached the 160-token cap. [Saved baseline predictions](results/baseline/dev-generate.jsonl)

Four-way accuracy was 31/48 (64.6%). Fixed-four macro F1 was 0.474, supported-class macro F1 was 0.632, and call precision was 16/31 (51.6%). The model correctly asked for missing information on 4/16 cases and expressed inability on 11/16 cases. Twelve false calls occurred in the clarification class and three in the inability class. This development set has enough overcalling to test a reduction. All development menus contain tools, so the narrower empty-menu hallucination rate is undefined here. [Metrics and confusion counts](results/dev-generate-summary.json)

A separate assistant inspection of 16 randomly selected baseline outputs agreed with the deterministic act classification on all 16. This check included replies that declined the requested lookup and then offered partial help. The review supports use of the parser for this pilot sample. It provides limited evidence for novel wording from trained adapters. [Inspection labels](results/baseline/development-review.json)

Two development errors illustrate what the decision metric captures. In `pref-07168`, the user asked for earnings data without naming a company. Qwen emitted a well-formed call using Tesla's symbol. In `pref-02910`, it copied a supplied sample into the missing second sample of a statistical test. Other labels are debatable. In `pref-03761`, a genre-search function could plausibly serve the user's request for fantasy movies, but the reference asks for a title or keyword. The published-label rate therefore includes both unwarranted assumptions and disagreements about whether clarification is necessary.

As a precision check, the original bf16 checkpoint made 13/32 false calls and called on 15/16 required requests, with 14/16 exact calls. Its four-way accuracy was 32/48. Both precisions show measurable overcalling. The trained comparison uses the NF4 base as its common control because its weights match those underneath the adapters. The batched MCQ scorer also agreed with explicit serial next-token log probabilities on a development fixture to within 0.043 nats per candidate token. [BF16 baseline](results/bf16-reference/dev-generate.jsonl) · [Numerical and hardware checks](results/hardware.json)

## Endpoint pilots

Both rank-16 adapters trained 5,963,776 parameters through all 125 updates. The tool-only adapter's mean training loss fell from 0.044 in its first 25 updates to 0.030 in its last 25. Its mixed-class development loss went from 0.996 to 0.999. For the balanced adapter, the corresponding training means were 0.506 and 0.279, and development loss fell from 0.996 to 0.340. The training process reserved at most 4,092 MiB for the tool-only pilot and 4,350 MiB for the balanced pilot. These are PyTorch allocator peaks. Driver-level usage also includes the CUDA context and desktop. [Tool-only loss history](results/non0-seed182/loss.jsonl) · [Balanced loss history](results/non50-seed182/loss.jsonl)

| Development, seed 182 | False calls / non-tool | Calls / required | Exact / required | Invalid / attempts | Accuracy |
|---|---:|---:|---:|---:|---:|
| NF4 base | 15/32 | 16/16 | 12/16 | 0/31 | 31/48 |
| 0% non-tool | 32/32 | 16/16 | 12/16 | 6/48 | 16/48 |
| 50% non-tool | 1/32 | 15/16 | 11/16 | 0/16 | 46/48 |

The tool-only adapter called on every prompt, including four calls to unadvertised names and two missing required arguments. The balanced adapter asked for clarification on one legitimate request and called on one clarification case. For seed 182 on these fixed development cases, balancing reduced false calls by 43.8 percentage points relative to base, with a paired 95% bootstrap interval of −60.7 to −25.9 points for balanced minus base. Required-call recall changed by −6.3 points (−21.4 to 0.0). This supported running the intermediate mixtures and a second seed. The interval allows a consequential loss of required-call recall. [Tool-only predictions](results/non0-seed182/dev-generate.jsonl) · [Balanced predictions](results/non50-seed182/dev-generate.jsonl)

## Training and development across both seeds

All eight adapters completed 125 updates with identical learning-rate sequences, matching unique-example counts and verified final checkpoint hashes. Mean loss over the last 25 updates was lower than over the first 25 in every run. Final development loss ranged from 0.999–1.023 at 0%, 0.406–0.411 at 10%, 0.367–0.374 at 25%, and 0.340–0.349 at 50%. The eight timed training/validation loops took 87.6 minutes in total. The largest PyTorch reserved-memory peak during training was 4,744 MiB. Across the main runs, 15-second driver samples of whole-device usage reached 11,524 of 12,227 MiB. These measures have different scopes. [Run verification](results/training-verification.json) · [GPU monitor summary](results/gpu-monitor-summary.json)

At 0%, 10%, 25% and 50% non-tool training, the seed-mean false-call rates on the training-source development set were 100.0%, 26.6%, 17.2% and 4.7%. Required-call decision recall was 100.0%, 100.0%, 100.0% and 96.9%. Each seed was evaluated on the same 48 prompts. Generalization to BFCL-derived official questions is assessed separately. [Development seed estimates](results/dev-generate-seed-results.csv)

## Frozen official-data samples

The frozen generation sample contains 72 official judge prompts from 71 source-question clusters, with 24 cases in each supported class. One required reference omits an argument marked required. Under the combined schema-validity and exact-match criterion, the maximum on this sample is therefore 23/24. Clustering joins variants by BFCL source ID and joins source IDs if their normalized source questions reach a RapidFuzz ratio of 92. The MCQ sample contains 367 prompts in 128 clusters, comprising 128 required calls, 108 clarification cases and 131 inability cases. Six rows with conflicting annotations for identical prompts were excluded. Ten required MCQ reference calls fail the normalized basic value check. Likelihood evaluation retains the released candidates, including those ten. Raw scores and normalized scores are both available from each MCQ inference. [Frozen configuration hashes](results/freeze.json) · [Generation sample audit](results/official-generate-data-audit.json) · [MCQ sample audit](results/official-mcq-data-audit.json)

## Interpretation limits and paper directions

After scoring, an assistant reviewed 36 official responses in random order with the question and complete output visible and model identity, gold class and scorer label withheld. The sample included four responses per model, preferentially drawing text responses to check the hardest part of the local classifier. All 36 four-way decisions agreed, including two direct answers. The review covers a small sample with repeated questions and a single reviewer. It measures action classification only. [Review records and selection method](results/official-act-review.json)

Two official examples illustrate the cost and benefit of restraint. On a weather query that supplied a date but omitted the city (`b0e10a8c`), the base model invented San Francisco. Both 25% adapters asked for the city. On a request to list RDS servers (`96bb36ac`), the base emitted the exact annotated call. The 25% seed-182 adapter declined despite having a suitable tool, while seed 183 called. Another gold tool case asks the assistant to say hi. A conversational greeting is reasonable even though the benchmark expects a function call. [Base outputs](results/baseline/official-generate.jsonl) · [25% seed 182](results/non25-seed182/official-generate.jsonl) · [25% seed 183](results/non25-seed183/official-generate.jsonl)

A decision to call and a usable call are different outcomes. A structurally valid call can target the wrong function or use an invented argument value. Conversely, exact reference matching can reject another usable parameterization. The study measures requested actions and agreement with annotations. Establishing task completion would require running appropriate functions or checking their outputs.

Only seven prompt UUIDs are shared by the generation and MCQ samples. The protocols use different inclusion rules and different response tasks. Differences between their absolute rates combine sampling and elicitation effects. The three empty-menu generation cases are too few to estimate a change in the paper's narrower hallucination metric.

A follow-up could test whether the composition effect persists when each source question is paired with both a complete and an underspecified request and the same tool menu. Matching by original task would reduce topic and schema cues. A second study could separately vary example proportions and supervised-token budgets, since the native call and text targets differ in length.

Preserving legitimate tool use is a noninferiority question. A confirmatory study could specify an acceptable recall loss in advance, increase the number of independent required-call source questions, and fit more training seeds. Adding genuinely answerable direct questions with plausible tools would extend the estimand to a class absent from When2Call. A blinded human review of a stratified sample could quantify uncertainty from text-act classification and ambiguous benchmark labels.

## Supplementary menu-size experiment

After the main ablation, six newly written domains each supplied three requests. One request required a live record lookup, one omitted its required identifier, and one contained the answer explicitly. Every request was repeated verbatim with the same lookup and zero, two or four same-domain distractors. The two distractors remained in the larger menu. The lookup appeared first for three domains and last for three. The 18 unique requests generated 54 prompts per model. All menus were nonempty. The maximum normalized-question edit similarity to a released training or official question was 61%, below the main experiment's 90% exclusion threshold. [Questions, tools and design](data/supplementary-menus.json) · [Overlap audit](results/supplementary/overlap-audit.json)

Each cell below follows the same 0, 2, 4 ordering. The false-call denominator is the 12 requests answerable directly or requiring clarification. The exact-call denominator is the six live lookups. A distractor selection requires a parsed call naming an added tool. A false call can also name the original lookup. These are model requests. No tools were executed.

<!-- supplementary-counts -->
| Fitted model | False calls / 12 at +0, +2, +4 tools | Exact required / 6 at +0, +2, +4 | Named distractor / 18 at +0, +2, +4 |
|---|---:|---:|---:|
| NF4 base | 1, 4, 5 | 6, 6, 6 | 0, 0, 0 |
| 0% · seed 182 | 12, 12, 12 | 6, 6, 6 | 0, 0, 0 |
| 0% · seed 183 | 12, 12, 12 | 6, 6, 6 | 0, 0, 0 |
| 50% · seed 182 | 6, 5, 5 | 6, 6, 6 | 0, 0, 0 |
| 50% · seed 183 | 6, 6, 6 | 6, 6, 6 | 0, 0, 0 |
<!-- end-supplementary-counts -->

For the base model, four of six directly answerable requests switched from text to a call when the menu grew from one to five tools. All four calls targeted the original lookup. Both balanced adapters avoided calls on every missing-identifier prompt, yet still called on five or six directly answerable prompts at the largest menu size. Both tool-only adapters called on all 12 non-tool prompts at every size. Every model exactly matched all six genuinely required lookups at each size. No generated call selected an added tool. These results suggest sensitivity to the presence of other tools and poor transfer of clarification training to questions that already contain the answer. The six domains, shared identifier-based task structure, fixed distractor identities and growing prompt length limit generalization. [Within-request changes and raw counts](results/supplementary/summary.json)


## Expanded and matched evaluation

This additional analysis uses all 300 prompts in the released When2Call judge split. We designed it after examining the original 72 generation outcomes and the separate 367 MCQs. All 300 judge UUIDs match their originating MCQ records, including source ID, original fields, user text, tool menu, gold class, candidate content and order. There are 100 required calls, 100 clarification requests and 100 inability cases. There are no gold direct answers. The 300 distinct rendered prompts represent 275 release source IDs and 272 clusters under the original near-duplicate rule. Of those clusters, 247 contain one prompt, 22 contain two and three contain three. No edited or original judge question matched any of the 1,500 unique training questions at the normalized edit-similarity threshold of 90. [Input and model audit](results/expanded-generation/audit.json) · [Overlap audit](results/expanded-generation/overlap-audit.json)

We rechecked the SHA-256 digests of the cached base files and eight final adapter weights, the NF4 configuration, package versions, tokenizer, template and frozen evaluator. We reconstructed the earlier native input bytes from the pinned release and serialization code. All 72 earlier generations and 27 earlier MCQs per model had identical inputs and settings. We reused the complete outputs for those records. Each model generated only 228 new responses and scored only 273 new sets of candidates. Both protocols used the same no-thinking Qwen prompt and normalized menus. Free generation remained greedy with a 160-token cap. MCQ ranked four separately scored continuations by conditional log likelihood per UTF-8 byte, with raw log likelihood retained. No tool was executed. [Matched input manifest](data/followup/official.jsonl) · [Additional analysis protocol](data/followup/protocol.md)

### Overcalling and undercalling on all 300 judge prompts

| Non-tool share | False calls (%) | Required recall (%) | Exact recall (%) |
| --- | ---: | ---: | ---: |
| NF4 base | 47.5 [40.9, 54.2] | 99.0 [96.7, 100.0] | 61.0 [51.9, 69.8] |
| 0% | 91.5 [87.3, 95.2] | 100.0 [100.0, 100.0] | 64.5 [55.2, 73.7] |
| 10% | 37.8 [31.8, 43.8] | 96.0 [92.5, 99.0] | 64.5 [55.3, 74.1] |
| 25% | 18.8 [13.9, 24.0] | 83.0 [76.1, 90.0] | 58.5 [49.0, 68.0] |
| 50% | 10.0 [6.3, 14.1] | 67.0 [58.2, 76.2] | 50.5 [41.3, 60.1] |

Rates are percentages. Mixture rows are unweighted means of two fitted seeds. Brackets are 95% percentile intervals from 2,000 paired source-question-cluster bootstrap draws with the same cluster weights for every model and both protocols. These intervals condition on the fitted models. Between-seed SDs and individual estimates are in the metrics file. At 50% training the two seeds made 16 and 24 false calls out of 200 and called on 64 and 70 of the 100 required requests. The between-seed SDs for those rates were 2.8 and 4.2 percentage points. A boundary interval at 100% reflects the observed sample. [Per-seed metrics, counts and intervals](results/expanded-generation/metrics.csv)

| Model · seed | False / non-tool | Called / required | Exact / required | Correct / all | Clarify / gold | Unable / gold | Supported macro F1 (%) | Fixed-four macro F1 (%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| NF4 base | 95/200 | 99/100 | 61/100 | 177/300 | 27/100 | 51/100 | 57.6 | 43.2 |
| 0% · 182 | 183/200 | 100/100 | 66/100 | 109/300 | 0/100 | 9/100 | 22.9 | 17.2 |
| 0% · 183 | 183/200 | 100/100 | 63/100 | 109/300 | 0/100 | 9/100 | 22.9 | 17.2 |
| 10% · 182 | 74/200 | 95/100 | 65/100 | 208/300 | 50/100 | 63/100 | 70.2 | 52.7 |
| 10% · 183 | 77/200 | 97/100 | 64/100 | 202/300 | 47/100 | 58/100 | 67.7 | 50.8 |
| 25% · 182 | 35/200 | 80/100 | 57/100 | 219/300 | 75/100 | 64/100 | 74.1 | 55.6 |
| 25% · 183 | 40/200 | 86/100 | 60/100 | 223/300 | 72/100 | 65/100 | 75.4 | 56.6 |
| 50% · 182 | 16/200 | 64/100 | 50/100 | 222/300 | 90/100 | 68/100 | 74.8 | 56.1 |
| 50% · 183 | 24/200 | 70/100 | 51/100 | 222/300 | 85/100 | 67/100 | 75.1 | 56.3 |

The clarification and inability columns are recalls on the two gold non-tool classes. Direct recall has a zero denominator. Fixed-four macro F1 counts the unsupported direct class as zero; supported macro F1 averages over the three observed gold classes. Invalid attempts count as decisions to call. Four required references themselves fail schema validation (three invalid values and one missing required argument). All remain in the primary denominator of 100, giving a strict exact-match ceiling of 96%. Unknown-name rates count the first recorded error per attempt. None of the models attempted a call on the 17 empty-menu inability cases, whereas both tool-only adapters attempted a call on every one of the 183 non-tool requests with an advertised function. [Invalidity, unknown names, capped outputs and valid-reference counts](results/expanded-generation/tables.md) · [Confusion matrices](figures/expanded-generation/confusion.pdf)

| Contrast (pp) | False-call change | Required-recall change | Exact-recall change |
| --- | ---: | ---: | ---: |
| 0 minus baseline | 44.0 [37.2, 50.5] | 1.0 [0.0, 3.3] | 3.5 [−2.9, 10.0] |
| 10 minus baseline | −9.8 [−15.5, −4.4] | −3.0 [−6.5, 0.0] | 3.5 [−3.0, 10.4] |
| 25 minus baseline | −28.8 [−34.8, −22.9] | −16.0 [−23.1, −9.2] | −2.5 [−9.5, 5.0] |
| 50 minus baseline | −37.5 [−43.9, −31.3] | −32.0 [−40.6, −23.1] | −10.5 [−18.2, −3.1] |
| 50 minus 25 | −8.8 [−12.3, −5.6] | −16.0 [−22.5, −10.1] | −8.0 [−12.9, −3.9] |

Compared with the base, the 10% mixture reduced false calls on the expanded set, a difference that was uncertain on the initial 72. Its required-call recall was three points lower; the interval reaches zero and allows a meaningful loss. The 25% mixture reduced both false and required calls; its exact-match contrast still spans gains and losses. Moving from 25% to 50% further reduced both decision recall and exact recall. No tolerance or deployment error costs were specified that would establish an optimal mixture. [All paired effects and seed variation](results/expanded-generation/summary.json)

| Non-tool share | 72 false / recall / exact (%) | 228 false / recall / exact (%) | 300 false / recall / exact (%) |
| --- | ---: | ---: | ---: |
| NF4 base | 43.8 / 100.0 / 66.7 | 48.7 / 98.7 / 59.2 | 47.5 / 99.0 / 61.0 |
| 0% | 93.8 / 100.0 / 68.8 | 90.8 / 100.0 / 63.2 | 91.5 / 100.0 / 64.5 |
| 10% | 39.6 / 93.8 / 62.5 | 37.2 / 96.7 / 65.1 | 37.8 / 96.0 / 64.5 |
| 25% | 16.7 / 83.3 / 62.5 | 19.4 / 82.9 / 57.2 | 18.8 / 83.0 / 58.5 |
| 50% | 6.3 / 66.7 / 56.3 | 11.2 / 67.1 / 48.7 | 10.0 / 67.0 / 50.5 |

Every entry lists false-call rate, required-call decision recall and exact required-call recall in percent. The additional 228 prompts contain 76 of each supported class, in 211 clusters. Ten clusters span the original and additional subsets. The original 72 captured the direction of the decision trade-off: balanced required-call recall was 66.7% there and 67.1% on the additional cases. The absolute false-call and exact-call estimates moved, and balanced exact recall is now below the base with a paired interval excluding zero. The full sample includes the original 72; it is not an independent replication. [Trade-off figure](figures/expanded-generation/tradeoff.pdf) · [False-call curve](figures/expanded-generation/false_call_rate.pdf) · [Required-recall curve](figures/expanded-generation/required_call_recall.pdf)

For the balanced adapters, the additional-minus-initial changes were +4.9 pp [−3.2, +12.3] in false calls, +0.4 pp [−19.8, +21.7] in required-call recall and −7.6 pp [−30.1, +15.4] in exact recall. These intervals use common cluster draws across the two subsets. The small initial denominator allows appreciable differences even where the point estimates are close.

In the trade-off plot diamonds denote seed means (or the single base) and open circles the fitted adapters. Bars show marginal 95% cluster intervals. The composition plots show the same estimates against training share. The dashed line and gray band give the base and its interval.
The confusion matrices show what replaced the missed calls. On the 100 required requests, the 50% adapters asked for information 30 and 27 times and expressed inability six and three times (seeds 182 and 183). Neither supplied a direct answer on these requests. Of the calls they did attempt on required requests, 50/64 and 51/70 matched the reference exactly. Conditioning on attempted calls changes which requests enter that denominator, so the primary exact-recall analysis retains all 100 required requests. The base had 61 exact matches among 99 attempts. Rows in the confusion figure are gold acts and columns are the classifier's assigned acts. Cells contain prompt counts, with shading proportional to the corresponding gold-class total.

### Comparing candidate selection and generation on the same requests

Every judge record had a matching MCQ record. Matching used the release UUID, followed by equality of the source ID, question, original question, menu, gold answer and four candidate continuations in release order. The two protocols use the same rendered user turn, sorted tool schemas and assistant prefix. Generation chooses each next token greedily until the native end token or the 160-token limit. MCQ evaluates the conditional log likelihood of each supplied continuation after that prefix, excluding the final end token. It divides the sum by the UTF-8 byte count of the serialized continuation and selects the highest score. Raw sums and their winners are also saved. The model never receives a separate instruction to choose a letter. In particular, it scores the supplied function name and arguments even on a question where it would never generate the opening call delimiter itself. The protocol comparison includes every matched question, including the four with schema-invalid positive candidates. [Frozen matched inputs](data/followup/official.jsonl) · [Evaluation specification](data/followup/protocol.md)

| Non-tool share | Gen / MCQ required recall (%) | MCQ − gen (pp) | Gen / MCQ false calls (%) | MCQ − gen (pp) | Gen / MCQ accuracy (%) | MCQ − gen (pp) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| NF4 base | 99.0 / 100.0 | 1.0 [0.0, 3.3] | 47.5 / 87.0 | 39.5 [31.8, 46.9] | 59.0 / 37.3 | −21.7 [−26.8, −16.6] |
| 0% | 100.0 / 100.0 | 0.0 [0.0, 0.0] | 91.5 / 92.0 | 0.5 [−2.5, 3.7] | 36.3 / 35.0 | −1.3 [−3.8, 1.0] |
| 10% | 96.0 / 100.0 | 4.0 [1.0, 7.5] | 37.8 / 65.0 | 27.3 [19.8, 34.8] | 68.3 / 54.2 | −14.2 [−19.7, −8.8] |
| 25% | 83.0 / 100.0 | 17.0 [10.0, 23.9] | 18.8 / 56.0 | 37.3 [29.5, 44.9] | 73.7 / 60.0 | −13.7 [−19.9, −7.5] |
| 50% | 67.0 / 99.0 | 32.0 [23.0, 40.7] | 10.0 / 47.0 | 37.0 [29.4, 44.2] | 74.0 / 65.5 | −8.5 [−15.4, −1.7] |

For the 50% mixture, the required-call advantage under candidate scoring was +32.0 pp [23.0, 40.7]. Restricting this diagnostic to the 96 schema-valid required references gave +31.3 pp [22.1, 40.2]. The protocol also increased false calls by 37.0 pp [29.4, 44.2]. With twice as many non-tool as required prompts, four-class accuracy was lower under MCQ by 8.5 pp [−15.4, −1.7]. A near-ceiling tool-selection recall therefore describes one side of the protocol difference. The base and tool-only adapters were already near the generation ceiling, leaving little room for MCQ to increase required recall.

| Gold requests | Balanced seed | MCQ selects | Generation calls | Generation does not call |
| --- | ---: | --- | ---: | ---: |
| Required (100) | 182 | Tool | 64 | 36 |
| | 182 | Text | 0 | 0 |
| Required (100) | 183 | Tool | 69 | 29 |
| | 183 | Text | 1 | 1 |
| Non-tool (200) | 182 | Tool | 8 | 81 |
| | 182 | Text | 8 | 103 |
| Non-tool (200) | 183 | Tool | 16 | 83 |
| | 183 | Text | 8 | 93 |

Across all 300 requests, the balanced models selected a call only under MCQ on 38.2% [32.8, 43.6] of prompts and called only under generation on 2.8% [1.2, 4.7]. For seed 182 the counts in those directions were 117 and 8, and for seed 183 they were 112 and 9. Unnormalized MCQ scores gave the balanced adapters 94.5% required recall and 33.5% false selections. The winner depends in part on how continuation length is treated, yet the direction of the difference from free generation persisted under both scoring rules. The matched sample supports a difference in measured behavior between these complete procedures. It cannot isolate the effects of candidate construction, normalization, token-by-token decoding or internal representations. [All nine paired outcome tables and counts](results/matched-protocols/tables.md) · [Per-prompt paired decisions](results/matched-protocols/paired-predictions.jsonl) · [Per-seed paired intervals](results/matched-protocols/summary.json)

The matched figure shows marginal decision rates in the upper panels and the two discordant cell counts below. Each mixture is the mean across the two fitted adapters. A fractional count is the mean of two observed integer counts. Small markers show the seed rates; bars on the upper panels indicate pointwise 95% intervals from resampling the 272 source clusters. [Matched protocol figure](figures/matched-protocols/paired-protocols.pdf)

### What the disagreements look like

Two judge records ask for the dashboard named “dashing.” With the lookup function available (UUID prefix `08f77114`), the base generated the exact reference call using `dashing` as the identifier. Both balanced adapters asked the user for an identifier in generation, yet both selected the supplied reference call under MCQ. In the variant offering only create and delete functions (`2dcf86e0`), both balanced adapters expressed inability and selected inability. The base also expressed inability in generation but selected a candidate naming the absent lookup function. Across the release, all 100 inability cases have a tool candidate naming an unadvertised function. A candidate can receive a high likelihood even when the corresponding function is unavailable.

There was also a required case in the reverse direction. Asked to play “Shape of You” (`6da5c2fa`), the balanced seed-183 model selected clarification from the candidates but generated a schema-valid call. Its arguments differed from the annotated query, so this counted toward decision recall only. These examples illustrate observed differences in decisions and arguments. They were selected after scoring, and they do not estimate how frequently each explanation applies. [Input menus, gold and candidates](data/followup/official.jsonl) · [Balanced seed-183 raw responses](results/expanded-generation/models/non50-seed183.jsonl)

### Scoring and blinded review

We replayed the frozen parser and exact-call check on all 2,700 official generations. For all 2,700 MCQs we recovered the normalized and raw winners and verified all four normalized scores against the raw log likelihood and actual serialized byte count. Every reused field matched its archived output except the separately assigned ID/cluster metadata. The 160-token cap can truncate call blocks, and a schema-valid call can still have the wrong arguments. These errors remain in the totals. [Parser and candidate replay](results/matched-protocols/scoring-audit.json) · [Scoring errors by model](results/expanded-generation/tables.md)

One assistant reviewer classified 54 blinded outputs spanning 39 judge UUIDs, six outputs per model. The sample targeted ambiguous mixed text, protocol disagreement and invalid attempts, with prespecified fallbacks. Only the question, advertised names and raw answer were visible. All 54 binary call judgments agreed with the parser and 52/54 four-way judgments agreed. On one greeting, the model answered “Hello!” and offered further assistance. The parser treated the following question mark as a clarification request. On another item, the model explained that source text was missing and requested it; the parser gave priority to the inability phrase. Other responses explicitly declined a task and then offered help or requested information, making a single text label an approximation even when the reviewer agreed. We retained the original labels in every metric. This targeted audit supplies neither a population error rate nor human inter-rater agreement. [Blinded outputs](results/matched-protocols/review-blinded.jsonl) · [Reviewer judgments, notes and disagreements](results/matched-protocols/review.json)

### Exploratory direct-answer controls

The official test lacks a direct-answer class. We authored 36 separate record-lookup families after completing the official comparisons and froze all 108 questions before control inference. Each family has three variants. A required lookup supplies a unique identifier and asks for an unobserved field in the record. Its clarification variant asks for the same field but omits the identifier. The direct variant includes both the identifier and the field value and asks what the supplied text says. For example, a radio-archive lookup gives an episode code and asks for its reel; the underspecified question omits the code; the direct question quotes the reel from an index card. Each family advertises two read-only functions with a mandatory identifier, one returning the target field and one returning another field. The normalized tool menu is identical across the three variants. After canonical sorting, the target is first in 18 families and second in 18. [Authored source questions](data/followup/direct-design.tsv) · [Frozen inputs and tool schemas](data/followup/direct-families.jsonl)

No control reached the original 90% normalized similarity threshold against either full release training split, either official test file (edited and original wording), or the previous six-domain control. The highest score against any released question was 63.2%. The highest across distinct authored families was 64.8%. These checks screen textual reuse; related lookup semantics and recurring phrasing are part of the design. Outcomes use the same parser and greedy settings as official generation. The resampling unit is one whole family shared across acts, models and seeds. Each class has denominator 36. The tables keep all four possible predicted acts, including an inability response on a request with one of the three authored gold acts. [Design and overlap audit](results/direct-answer-controls/audit.json)

| Model · seed | Calls / direct | Calls / required | Clarify / underspecified | Direct / direct | Exact / required | All three correct / families |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| NF4 base | 20/36 | 36/36 | 29/36 | 16/36 | 36/36 | 13/36 |
| 0% · 182 | 36/36 | 36/36 | 0/36 | 0/36 | 36/36 | 0/36 |
| 0% · 183 | 36/36 | 36/36 | 0/36 | 0/36 | 36/36 | 0/36 |
| 10% · 182 | 36/36 | 36/36 | 36/36 | 0/36 | 36/36 | 0/36 |
| 10% · 183 | 36/36 | 36/36 | 35/36 | 0/36 | 36/36 | 0/36 |
| 25% · 182 | 36/36 | 36/36 | 36/36 | 0/36 | 36/36 | 0/36 |
| 25% · 183 | 36/36 | 36/36 | 36/36 | 0/36 | 36/36 | 0/36 |
| 50% · 182 | 35/36 | 35/36 | 36/36 | 0/36 | 35/36 | 0/36 |
| 50% · 183 | 36/36 | 36/36 | 36/36 | 0/36 | 36/36 | 0/36 |

The base used the lookup on 20/36 directly answerable requests and answered the other 16 directly. All six adapters in the 0%, 10% and 25% mixtures called on every direct request. The balanced adapters called on 35/36 and 36/36 direct requests (mean 98.6% [95.8, 100.0]), 43.1 pp more often than the base [27.8, 59.7]. The one remaining balanced response asked for a tea-cupping flight reference that was already in the question. None of the eight adapters produced a direct response on any direct-control prompt. On underspecified requests, by contrast, both balanced seeds asked for clarification 36/36 times, compared with 29/36 for the base. Their mean gain was 19.4 pp [8.3, 33.3]. All required calls attempted by the nine models on these controls exactly matched their function and identifier. The simple lookups made required calls easier here than on the official questions. [Counts, rates, intervals and all three-by-four confusion tables](results/direct-answer-controls/tables.md)

Within the same family, the base called on the required variant alone in 16 families and on both required and direct variants in 20. The balanced seed-182 adapter called on both variants in 34 families, on the required variant alone in one and on the direct variant alone in one. Seed 183 called on both in all 36 families. The required-minus-direct difference in call rate was +44.4 pp [27.8, 61.1] for the base and 0.0 pp [−4.2, +4.2] for the balanced mean. For the quilt exhibit, the base read “Willow Wall” from the supplied card, while both balanced adapters requested the same display-wall lookup they used for the required variant. Both balanced adapters requested the exhibit tag when it was absent. [Within-family paired counts](results/direct-answer-controls/family-decisions.csv) · [Family-level intervals and contrasts](results/direct-answer-controls/summary.json)

The direct-control figure shows call rates on supplied answers and on required lookups, followed by correct clarification on the underspecified variants. Diamonds give the base or two-seed means, open circles give individual adapters and bars are 95% intervals over whole families. A zero-width empirical interval occurs when every sampled family has the same outcome. The authored cases all use single-identifier lookups, two functions, and explicit references to cards, records or quoted information. They distinguish clarification from answering directly in this setting; the rates do not estimate overcalling across arbitrary directly answerable requests. [Direct-control figure](figures/direct-answer-controls/paired-direct-controls.pdf)

### Implications and reproducibility

For this checkpoint and training budget, the composition effect on official generation persists on the additional questions. The direct controls qualify what “restraint” meant here. The training mixtures contained clarification and inability targets and no labeled direct targets. The balanced adapters readily asked for a missing identifier but almost always queried a tool when the identifier and requested fact were both present. This association does not isolate the contribution of missing direct demonstrations, question wording, source, menu complexity or the particular fine-tuning objective. The two fitted seeds and a single base checkpoint limit generalization. The official source-cluster and authored family intervals are pointwise intervals conditional on these fits. They quantify variation across the evaluated questions and families. They exclude variation across fresh training runs.

The saved archive contains 2,700 official generations, 2,700 matched sets of four continuation scores and 972 authored-control generations. Of the official records, 648 generations and 243 candidate-score sets were reused across all nine models. Each prediction file has a separate digest and metadata. All computations ran locally on the RTX 5070 Ti Laptop GPU. The largest PyTorch allocator reservation recorded during official generation was 13,062 MiB, compared with 7,098 MiB during MCQ and 3,916 MiB on the authored controls; allocator reservation does not measure physical VRAM residency under WSL. We replayed scoring and regenerated the manifests, metrics, plots and tables from the saved archive, verified checksums, and checked the earlier report prefix, original outputs, checkpoints and manuscript files. [Reproduction instructions](data/followup/protocol.md) · [Analysis and prediction checksums](data/followup/outputs.sha256) · [Standalone findings and proposed manuscript changes](evaluation_followup.md)

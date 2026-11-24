#let title = [The Cost of Restraint: How Fine-Tuning Mixtures Shape Tool Use in Qwen3.5-4B]
#set document(title: title, author: "Anonymous authors", date: none)
// The 2015 anonymous style uses a fixed margin ruler for reviewers.
#let ruler = context {
  let first = 54 * (counter(page).get().first() - 1)
  align(top + left, block(width: 100%, height: 100%)[
    #for n in range(54) {
      let number = str(first + n)
      place(top + left, dx: 58pt, dy: 73pt + n * 12pt,
        text(font: "DejaVu Sans Mono", size: 7pt, fill: luma(68%), "0" * calc.max(0, 3 - number.len()) + number))
    }
  ])
}
#set page(background: if sys.inputs.at("ruler", default: "true") == "true" { pdf.artifact(ruler) }, paper: "us-letter", margin: (x: 1.5in, y: 1in), numbering: "1", number-align: center)
#set text(font: "TeX Gyre Termes", size: 10pt, lang: "en", top-edge: 0.7em, bottom-edge: 0.2em)
#set par(justify: true, leading: 4pt, spacing: 9.5pt, first-line-indent: 0pt)
#set heading(numbering: "1.1")
#set math.equation(numbering: "(1)")
#show heading.where(level: 1): set text(size: 12pt)
#show heading.where(level: 2): set text(size: 10pt)
#show heading: set block(above: 10pt, below: 5pt)
#show raw: set text(font: "DejaVu Sans Mono",size: .88em)
#set figure(gap: 4pt)
#show figure: set block(above: 9pt,below: 9pt)
#show figure.caption: set text(size: 9pt)
#show figure.caption: set align(left)
#show figure.caption: set par(justify: true, leading: 3.2pt)
#show figure.where(kind: table): set figure.caption(position: top)
#let tabledata=json("tables.json")
#let tab(headers, data, alignments: none) = {
  set text(size: 9pt)
  let cols = headers.len()
  table(columns: cols, align: if alignments==none { (left, ..range(cols - 1).map(i=>right)) } else { alignments },
    stroke: none, inset: (x:4.5pt, y:2.8pt),
    table.hline(stroke: 0.75pt),table.header(..headers.map(h=>strong(h))),table.hline(stroke: 0.4pt),
    ..data.flatten(),table.hline(stroke: 0.75pt))
}
#let pic(file, desc) = image("figures/"+file+".svg", width:100%, alt:desc)
#align(center)[
  #block(width: 100%, height: 4pt, fill: black)
  #v(11pt)
  #set par(leading: 7pt)
  #text(size:17pt,weight:"bold",hyphenate:false)[The Cost of Restraint: How Fine-Tuning Mixtures \
  Shape Tool Use in Qwen3.5-4B]
  #v(11pt)
  #block(width: 100%, height: 1pt, fill: black)
  #v(10pt)
  *Anonymous authors* \
  UC Berkeley CS 182 · Fall 2026
]
#v(7pt)
#align(center)[*Abstract*]
#pad(x: .5in)[
A language model can generate a well-formed tool call even when the user has supplied too little information to justify it. We ask whether examples of clarification and inability to answer reduce unnecessary calls while preserving legitimate ones. From one NF4-quantized Qwen3.5-4B checkpoint, we train QLoRA adapters on four mixtures containing 0% to 50% non-tool responses, with two seeds per mixture and 1,000 examples per fit. On all 300 When2Call judge prompts, tool-only training raises the false-call rate from 47.5% to 91.5%. At 50% non-tool training, false calls fall to 10.0%, but required-call recall falls from 99.0% to 67.0% and exact recall from 61.0% to 50.5%. On the same questions, ranking supplied continuations gives the balanced adapters 99.0% required-tool selection and 47.0% false selections. In 36 authored lookup families, the balanced adapters ask for missing identifiers in every case but call tools on 98.6% of questions whose answers are already supplied. The models learned when to request information more reliably than when to use information already in the prompt.

]
= Introduction
A user asks for the weather on a particular date and leaves out the city. In our evaluation, the unadapted model supplies San Francisco as the missing argument and generates a weather lookup. On the same prompt, two fine-tuned models ask the user which city they mean. That change looks desirable. Yet when another user asks for a list of database servers and the appropriate function is available, one of those same models declines and the unadapted model emits the annotated call. A system evaluated only on how often it avoids unwarranted calls would receive credit for the first change and incur no penalty for the second.

Both errors concern the first assistant decision. _Overcalling_ means attempting a call on a request whose annotated next step is textual. _Undercalling_ means producing text when a call is required. A model can achieve perfect recall by calling on every request, or eliminate false calls by answering everything in prose. Neither policy is useful. We ask whether examples of when to wait teach a transferable distinction or merely make the model less willing to call.

When2Call introduced a benchmark and training data for this decision @when2call. We vary how often supervised fine-tuning replaces a call with an appropriate textual response. We ask how many required calls are lost as false calls disappear, and whether a model that selects a supplied call will generate it on the same question. The benchmark’s two non-tool classes also leave a gap. Asking for a missing city and reporting that no weather service is available both withhold a call. Neither demonstrates how to answer when the user has already supplied the forecast. We test that distinction with three versions of the same lookup request.

= Background and related work
== Learning to use a tool and learning when to wait
Early work on language models with tools supplied two different reasons to make an API call. In Toolformer, a model proposes calls inside ordinary text, executes candidates, and keeps those whose outputs improve prediction of subsequent tokens @toolformer. This gives the learner a local signal for when an external observation is useful. A computation can be worth making because its result helps predict the next sentence. In an assistant conversation, a different question arises before any result is available: does the user’s request warrant this particular call with these arguments? A model could predict that weather information would be useful and still have no grounds for choosing a city. Our negatives include cases where the purpose of the API is appropriate and the uncertainty concerns a required input.

ReAct interleaves reasoning, actions and observations, allowing later steps to correct earlier mistakes @react. Final task success can therefore conceal an unnecessary first lookup; a correct initial call can also be followed by a bad answer. Our evaluation stops at the first assistant response, where the choice to call or wait is observable.

Work on function-calling datasets has made the positive side of this decision much easier to supervise. APIGen checks synthesized examples for format, executes functions, and verifies their semantics @apigen. A successful execution establishes that a proposed call can work for the request that generated it. It does not by itself supply the closely related request for which the same function should be withheld. If training pairs a tool description almost exclusively with successful uses, its presence is a reliable cue to call within that dataset. Our all-call condition tests the consequence of reinforcing that association in a model that already knows how to ask questions.

== What a benchmark counts as success
The Berkeley Function Calling Leaderboard (BFCL) evaluates function and argument selection, including parallel calls and situations where tools are irrelevant. Its abstract syntax tree checks allow many predicted calls to be evaluated without operating a service. The benchmark also includes stateful, multi-turn tasks @bfcl. By comparison, τ-bench simulates a user interacting with an agent under domain-specific policies and compares the final database state with a goal state @taubench. These assessments can give different verdicts on the same first response. A well-formed, unnecessary read can leave a database unchanged. A useful clarification can postpone every observable state change by one turn. A benchmark of first actions needs a label for what the assistant should do at that moment.

Ross et al. construct such labels by modifying requests or available tools @when2call. Their evaluation starts from BFCL v2 Live questions that need an external capability. Removing a required piece of information can turn a legitimate call into a clarification case. Altering the available functions can make the requested task infeasible even when a related function remains visible. The benchmark offers four responses: a direct answer, a tool call, a request for information, and an admission of inability. Restricting the source questions to tasks requiring external information makes an unsupported direct answer a useful distractor. It also means the benchmark has no positive class for an ordinary question the model should answer from information already present.

When2Call scores complete candidate answers by conditional log likelihood, with raw and byte-normalized variants. Its alternative for generated responses uses a judge to classify the action. The official implementation instructs that judge to identify what the assistant did, independently of whether the content is correct @harness. The two protocols place different demands on the model: in one the tool name and argument string are supplied, while in the other the model must produce them. Neither a high candidate rank nor a correctly identified response act establishes that an argument came from the user. This is why we also score exact, schema-valid calls and retain the generated text for inspection.

The distinction between an unavailable tool and an available but unsuitable one affects even a seemingly simple metric such as hallucination rate. The When2Call implementation asks, among inability cases with an empty menu, how often a model nevertheless proposes a function @harness. This definition isolates a clear error: there is no advertised name the model could legitimately select. It excludes the weather example, in which a real and relevant tool is called with an invented argument. We retain the narrow check as a diagnostic and count all annotated non-tool cases in the main false-call denominator. The broader measure also includes inappropriate calls to available functions.

== Intervening on the decision to call
When2Call already studies the choice of training mixture. Its authors fine-tune 4B and 8B Mistral-NeMo-Minitron models using general instruction data and tool-specific examples, and report that a two-to-one ratio of calls to clarification or inability responses worked best overall in their supervised blend @when2call. They also use reward-aware preference optimization, including comparisons between correct calls and calls with corrupted arguments. For their 4B model, adding When2Call examples improves the benchmark score while reducing BFCL Live AST accuracy relative to their tool-calling SFT baseline. These results give a concrete reason to expect a cost to restraint. They leave open how quickly each kind of error changes as positives are replaced on a fixed checkpoint. We vary that fraction at four levels, keeping the number of optimization examples fixed and scoring generated calls on the same required and non-tool questions in every condition. Changes within a checkpoint are the appropriate comparison because both the model family and the data blend differ across studies.

One way to regulate tool use is to estimate the value of a call separately from the generator. Wu et al. distinguish the need for external information, the utility of the available tool, and the cost of using it. They train estimators from hidden states to better guide calling decisions @wu. That separation helps explain why a model might overcall on a request that is genuinely outside its knowledge: the available function may still have little utility. An explicit gate also permits its decision criterion to be changed after training. Our intervention changes the generator through supervised demonstrations. There is no separately fitted gate or adjustable decision threshold. A shift in the observed call rate could arise from preferences over response types, better recognition of missing information, or easier production of a learned clarification phrase.

The object of restraint also matters. OverAct evaluates calls that exceed the scope of a user's request across eight privacy-sensitive domains. It varies the available tools and reports greater over-authorization as the tool pool grows @overact. A user may authorize a lookup and still provide too little information to carry it out. Conversely, a technically useful data retrieval can exceed what the user authorized. When2Call supplies labels for the suitability of the next action and no record of permissions granted or services accessed. The same distinction applies to our authored lookup tasks. We can observe whether an assistant requests a redundant read even when there is no external service to execute it.

== What changes when examples are replaced?
There is a related measurement problem in instruction tuning for refusals. Bianchi et al. show that a small fraction of safety demonstrations can change how a model handles harmful instructions and that stronger safety tuning can elicit refusals on superficially similar benign ones @bianchi. XSTest pairs benign prompts likely to provoke a refusal with unsafe contrasts, exposing failures that a test containing only unsafe requests would miss @xstest. Our clarification targets have a different purpose: a user who supplies the missing city can make the same lookup answerable. The methodological connection is that a reduction in the target error is easiest to interpret when the evaluation includes neighboring cases where the original behavior was appropriate.

Negative examples also differ in what they teach the model to produce. A clarification identifies a piece of information the user can supply. An inability response says the available functions do not support the task. Both earn the same credit on a binary decision to abstain, but swapping one for the other can make a conversation substantially less useful. Asking the user for a city can enable a weather lookup. Asking for more details about a service that has no weather function cannot. This motivates retaining the four-way confusion matrix alongside the binary decision measures, and giving the two non-tool subclasses equal representation at each mixture.

A composition experiment also needs an intervention small enough to repeat. LoRA learns low-rank updates to selected weights while keeping the pretrained parameters fixed @lora. QLoRA propagates gradients through a frozen quantized backbone into those adapters, reducing the memory needed for fine-tuning @qlora. Here this economy buys a second fit at every mixture and makes a common inference-precision baseline possible on one laptop GPU. Parameter efficiency alone cannot ensure comparability: increasing the non-tool share can also change domains, tool menus and target lengths. We therefore construct nested replacements from one release and quantify the cues that remain available before the assistant starts its response.

= Methodology
== Building four mixtures from one preference release
The released SFT and preference files have different schemas @release. The SFT file contains 15,000 conversations with serialized tool definitions. None of their assistant targets contains a tagged call or a bare JSON call. Concatenating this file with calls from another dataset would allow source-specific formatting to stand in for the intended decision. The preference file contains 9,000 user turns with separate chosen and rejected responses. Exactly 3,000 chosen targets contain a single call encoded as a JSON list inside a tool-call tag. The remaining chosen responses are text. We use the chosen responses as supervised targets and discard the rejected alternatives. Consequently the preference label selects a demonstration and never enters a pairwise training loss.

We infer the two text subclasses from clarification and inability phrases because the training files omit explicit class labels. We retain examples with at least one visible tool, check positive calls against advertised names and parameter schemas, and normalize schema types, field placement and order for all classes. We remove overlength examples in full. To limit contamination we normalize case, punctuation and numbers and discard exact or near-matching questions (RapidFuzz ratio at least 90) within the candidate pool and against both original and edited official questions. This leaves 2,124 calls, 2,389 clarification requests and 1,605 inability responses. We reserve 16 per class for development before selecting the training examples. The test questions enter this preprocessing as exclusion keys. We fixed the training configuration and scoring rules before the initial official evaluation. The decision to expand its coverage followed inspection of those results.

#figure(tab(([Non-tool],[Calls],[Clarify],[Unable],[Input + target],[Target tokens]),tabledata.design),
  caption:[Training composition per adapter. Each condition uses 1,000 unique examples. The last two columns count rendered tokens and supervised assistant tokens, respectively. Both seeds use the same manifest.]) <design>

We first select 500 replaceable positives and 500 positives to retain. Each negative is paired with a replaceable positive exposing the same number of tools, preferring identical menus and otherwise maximizing the intersection over union of tool schemas among eligible positives. The retained half also matches the replaced half’s tool-count histogram. To obtain the 10%, 25%, and 50% conditions, respectively, we replace the first 100, 250, or 500 designated positives, alternating the two negative subclasses. The same permutation of slots precedes the seed-specific training shuffle. Thus the four mixtures share examples, and a change in share never changes the tool count at a given slot. @design gives the resulting target counts.

Matching the number of tools is weaker than matching their identity. Of 500 positive–negative pairs, 116 have identical menus and 238 share at least one schema. In five held-out folds of the balanced manifest, a text-feature logistic classifier predicts call versus non-call at 50.0% balanced accuracy from tool count, 56.4% from schema text, and 55.7% from question text. The last figure includes genuine information about missing arguments. The schema-only result nevertheless shows that a model can extract some label information from the menu without reading the request. The comparison controls the broad source of demonstrations but leaves differences in topic, wording and target length.

== Checkpoint, objective and optimization
We adapt the text backbone of Qwen3.5-4B @qwen. The reference model is the same checkpoint quantized to 4-bit NormalFloat (NF4), with double quantization and bfloat16 computation, before attaching an adapter. All runs use the checkpoint's native chat template, including its function and parameter tags. The thinking-mode flag is disabled in training, candidate scoring and free generation. The user prompt, menu, assistant preamble and empty thinking block receive no loss. The response and the end-of-turn token do.

Let $x_i$ be the rendered input, $y_i$ the assistant target of length $T_i$, $W$ the frozen backbone, and $phi$ the trainable adapter parameters. We minimize the mean of the per-turn conditional losses,
$ cal(L)(phi) = -1/N sum_(i=1)^N 1/T_i sum_(t=1)^(T_i) log p_(W,phi)(y_(i,t) | x_i, y_(i,<t)). $ <objective>
Each turn has equal weight despite differences in response length. Calls typically require more supervised tokens than clarifications. Fixing the number of training examples therefore does not equalize the amount of target-token supervision (@design).

Rank-16 LoRA updates with scaling parameter 32 target the query and value projections of attention and the combined query/key/value input projection of linear attention. This updates 5,963,776 parameters. NF4, gradient checkpointing and a microbatch of one make the same configuration feasible on a 12 GB RTX 5070 Ti Laptop GPU. Eight-way accumulation yields 125 AdamW updates for a single pass over each 1,000-example manifest. We use a peak learning rate of $10^(-4)$, five warmup updates and cosine decay. Sequences contain at most 1,024 tokens. Rank, projections, learning rate and final-step selection are fixed across mixtures to keep model capacity, computation per update and checkpoint choice comparable. We use seeds 182 and 183 for initialization and example order. The endpoint pilots used development prompts to check optimization, serialization and retained calling ability.

== Evaluating decisions, calls and uncertainty
We evaluate all 300 prompts in the released When2Call judge set. There are 100 required calls, 100 clarification requests and 100 inability cases. Each prompt matched its originating multiple-choice record by UUID, source ID, question, available tools, gold label and the content and order of all four continuations. Generation and candidate scoring use the same rendered user turn, menu and assistant prefix. The first evaluation covered 72 of these prompts. After examining those outcomes, we added the remaining 228 with the same fitted models, decoding settings and scoring rules. We reused earlier predictions only after verifying the input bytes and model configuration.

In generation the model chooses each next token greedily, with a limit of 160 new tokens. A deterministic local classifier detects attempted calls, including recognizable malformed ones. For text it checks inability phrases, then clarification cues, and otherwise assigns a direct response. This approximates When2Call’s external response-act judge. In multiple choice (MCQ), each complete continuation is scored separately after the common prefix. We sum its conditional token log likelihoods, excluding the end-of-turn token, divide by its serialized UTF-8 byte length, and select the highest score. We retain the raw-sum winner as a diagnostic. The tool candidate supplies a function and its arguments even on a prompt where the model would generate a question.

Let $G$ be the prompts with a gold tool call, $U$ the remaining prompts, $a_i$ an indicator of an attempted call, and $e_i$ an indicator that the full call list passes schema checks and matches the reference names and argument values. The three outcomes are
$ "false-call rate" = (sum_(i in U) a_i)/(abs(U)), quad "required recall" = (sum_(i in G) a_i)/(abs(G)), quad "exact recall" = (sum_(i in G) e_i)/(abs(G)). $ <metrics>
Here $abs(U)=200$ and $abs(G)=100$. An invalid call counts as an attempt. Strict exact recall requires both schema validity and equality with the annotated function and arguments. Three required references contain invalid values and one omits a required argument. Keeping all four in the denominator gives a strict exact-recall ceiling of 96%. Four-way accuracy credits the correct response act. In the main table, macro F1 averages the three classes represented in the gold labels; direct recall is undefined on this dataset. Fixed-four macro F1, which assigns zero to the unsupported direct class, remains in the saved evaluation results.

For both official protocols we draw 2,000 bootstrap samples with replacement from the same 272 source-question clusters. Clusters group BFCL source IDs and merge normalized near-duplicate source questions at edit-similarity ratio 92. Within each draw we retain all variants, recompute each rate with its own denominator, and use identical cluster weights across models and protocols. We average effects across the two fits inside each draw and report the 2.5th and 97.5th percentiles. These pointwise intervals condition on the fitted adapters and carry no familywise multiplicity adjustment. Seed-specific counts describe training variation separately. When every observed case has the same outcome, the empirical bootstrap interval collapses to a point.

To evaluate direct answers, we separately authored 36 lookup families and froze their 108 prompts before running the models. Each family has two read-only functions requiring the same identifier. One returns the requested field and one returns a different field. A required request supplies an identifier and asks for a field value absent from the prompt. A clarification request omits the identifier. A direct request gives both pieces of information and asks what the supplied record says. All three share the same menu. The target function comes first in 18 families and second in 18. No control reached the training audit’s near-duplicate threshold against either released training split or the official questions. We use the same generation procedure and resample whole families for paired intervals. These exploratory results have their own denominators.
= Results
== Tool-only supervision amplifies overcalling
In free generation (@scores), the unadapted NF4 model attempted 99 of 100 required calls and also called on 95 of 200 non-tool requests. Training exclusively on calls raised the latter count to 183 for each seed, a paired increase of 44.0 percentage points (95% interval +37.2 to +50.5). The 17 remaining non-tool requests all had empty menus. Whenever any function was advertised, both tool-only adapters called, including on every clarification request. There was only one more required decision to recover from the base. Most of the additional attempts consequently occurred on requests whose gold response was text.

#figure(tab(([Training],[False calls],[Required recall],[Exact recall],[Accuracy],[F1 (3 cls.)]),tabledata.primary),
caption:[Greedy generation on all 300 official prompts (percent). False calls use 200 non-tool cases; the recall columns use 100 required cases. Macro F1 averages the three classes present in the gold labels. Adapter rows average seeds 182 and 183. Paired source-cluster intervals are in @deltas.]) <scores>

#figure(pic("tradeoff","False-call rate versus required-call recall for the base and four training mixtures."),
caption:[Operating points on the complete When2Call judge set (300 prompts). Each adapter point averages two seeds. Horizontal and vertical bars give marginal 95% intervals over 272 source-question clusters. The upper left corresponds to few false calls and high required recall.]) <trade>

== Restraint reduces both inappropriate and required calls
Replacing 100 positive targets with textual responses reversed most of the shift induced by tool-only training (@trade). Relative to the unadapted model, the 10% mixture reduced false calls by 9.8 points (−15.5 to −4.4). Its required-call recall was 3.0 points lower (−6.5 to 0.0). At 25%, the reductions reached 28.8 points for false calls (−34.8 to −22.9) and 16.0 points for required recall (−23.1 to −9.2). False calls and required recall decreased across the four mixtures for each seed. The same ordering held on the additional 228 prompts alone, with 76 cases per supported class. Ten source clusters span both subsets. The extra questions thus test the original observation on the same fitted models after the initial analysis.

The balanced adapters made 16 and 24 false calls and attempted 64 and 70 required calls. Compared with the base, their mean false-call rate fell by 37.5 points (−43.9 to −31.3) and their required recall by 32.0 points (−40.6 to −23.1). Most missed calls became clarification requests: 30 of 36 misses for seed 182 and 27 of 30 for seed 183. The rest were classified as inability. @response shows the change in the two rates across the four measured compositions.

#figure(pic("response","False-call rate and required recall against the share of non-tool examples."),
caption:[Composition response on the 200 non-tool and 100 required official requests. Colored lines connect two-seed means, open circles show individual fits slightly offset horizontally, and dashed lines give the NF4 base. Bars are marginal 95% bootstrap intervals over the 272 source clusters, conditional on the fits.]) <response>

== Which calls are lost?
Suppressing a required call need not reduce exact-call recall if the call would have had the wrong arguments anyway. The 25% models exactly matched 57 and 60 of the 100 required references, compared with 61 for the base. Their mean difference was −2.5 points (−9.5 to +5.0), an interval that permits a meaningful loss or a gain. At 50% the exact-recall deficit reached 10.5 points (−18.2 to −3.1). Moving from 25% to 50% removed another 8.8 points of false calls (−12.3 to −5.6) while reducing decision recall by 16.0 (−22.5 to −10.1) and exact recall by 8.0 (−12.9 to −3.9). The earlier step from 10% to 25% reduced false calls by 19.0 points and required recall by 13.0. The final, larger replacement therefore averted fewer false calls and cost more required calls.

Some errors precede any comparison with the reference. The base produced 19 invalid calls among 194 attempts. The tool-only adapters produced 46 and 32 among 283 attempts each, including nine and five calls whose first validation failure was an unknown name. At the other extreme, only four of 80 and nine of 94 balanced attempts were invalid. A model that withholds a call can improve this count while also missing a perfectly well-specified request. Appendix B gives the attempt denominators and the full response-act confusion matrices.

== Scoring a supplied continuation
@matched shows what happened when the same models ranked the four released continuations for each of the same 300 prompts. The balanced adapters selected the tool candidate on 99.0% of required cases and generated a call on 67.0%, a paired difference of +32.0 points (+23.0 to +40.7). Seed 182 selected a call but generated text on 36 required prompts and showed the reverse on none. For seed 183 these counts were 29 and one. On a request for the dashboard called “dashing,” for example, the base passed that string as the dashboard identifier and produced the exact call. Both balanced models asked for the identifier in generation and selected the supplied call in MCQ.

#figure(pic("matched","Matched generation and multiple-choice call rates on required and non-tool requests for each model mixture."),
caption:[Protocol comparison on identical questions and menus. Points show the single base or the two-seed mean, with marginal 95% intervals over the same 272 source clusters. On the right, a higher tool-decision rate means more false calls. Paired discordance counts appear in @discord.]) <matched>

The higher tendency to select calls also extended to the 200 non-tool cases. The balanced adapters selected a call on 47.0% and generated one on 10.0%, a difference of +37.0 points (+29.4 to +44.2). With this class composition, four-way accuracy was lower under MCQ (65.5% versus 74.0%, paired difference −8.5 points, −15.4 to −1.7). All 100 inability cases supplied a tool candidate naming an unadvertised function, so selecting that candidate could never yield a valid call. Using raw likelihood sums gave 94.5% required selection and 33.5% false selection for the same balanced fits. Byte normalization affects the magnitude of the discrepancy, but it persists under either ranking rule. Matching removes differences in question and tool-menu composition. The protocols still differ in the available continuations, the scoring rule, and the need to produce a sequence by greedy decoding.

== When the answer is already present
On the authored families (@controls), adding non-tool demonstrations led to accurate requests for missing identifiers. Both balanced adapters asked for clarification on every underspecified request. Their behavior on the direct variant of those same families was quite different. The base called on 20 of 36 direct requests. Both seeds at 0%, 10% and 25% called on all 36. The balanced adapters called on 35 and 36, respectively, an increase over the base of 43.1 points (+27.8 to +59.7). The single remaining response asked for an identifier already in the question. Neither balanced adapter produced a direct answer. On the paired required lookups they generated 35 and 36 exact calls.

#figure(tab(([Training],[Call on supplied fact],[Clarify missing ID],[Exact required call]),tabledata.controls),
caption:[Exploratory authored control. Each column gives a percentage on one variant of the same 36 families. Adapter rows average two seeds. Each family contributes once per column and stays intact in the paired bootstrap intervals reported in the text.]) <controls>

In the quilt-exhibit family, the record card for quilt QE-732 explicitly listed “Willow Wall” as its display wall. Asked which wall the card named, the base answered “Willow Wall.” Both balanced adapters requested the display-wall lookup using QE-732, exactly as they had when the wall was unknown. When the exhibit tag was absent, both asked for it. Across families the base called on the required variant alone in 16 families and on both direct and required variants in 20. For the balanced fits the mean difference between required and direct call rates was 0.0 points (−4.2 to +4.2). These rates concern requests about explicit records with two available lookup functions.
= Discussion
The tool-only condition shows how much overcalling can be induced by fine-tuning a model that already knows how to ask a question. Adding textual demonstrations did more than reverse that effect. On the official clarification cases, the balanced adapters correctly requested information on 87.5% of prompts, compared with 27.0% for the base. They reported inability on 67.5% of gold inability cases, compared with 51.0% for the base. On gold required requests, however, the most frequent alternative to a call was another question.

The meaning of withholding a call depends on the information available to the user and the model. A missing argument can warrant a question. A missing capability can warrant an acknowledgment of inability. A fact already in the request can warrant an answer. The training mixtures directly supervise the first two responses and contain no labeled demonstrations of the third. That asymmetry offers one explanation for why balanced training improved clarification yet left the models calling on almost every quoted record. We did not vary direct-answer supervision, and the 36 authored families share a narrow lookup structure. Their contrast with When2Call shows why a low false-call rate on clarification and inability cases is insufficient evidence for a general rule about when external information is unnecessary.

There is no distribution-free best mixture. If required requests constitute a fraction $pi$ of traffic and the costs of a false call and a missed call are $c_f$ and $c_m$, expected decision cost is $(1-pi)c_f dot "FCR" + pi c_m (1-"recall")$. Applying this expression also requires knowing which kinds of non-tool requests make up the traffic. The official false-call rate weights clarification and inability equally and gives direct requests no weight. The expression assigns no cost to a wrong argument or an unnecessary follow-up turn. Choosing a mixture requires deployment-specific costs. Establishing preservation at 10% or 25% would also require a prespecified noninferiority margin and sufficient precision to exclude an unacceptable loss. Candidate recall alone would make the balanced fit appear to satisfy a much stricter tolerance than its generated calls support.

Several design limits affect the size of the estimated trade-off. Training and development prompts ultimately derive from APIGen, whereas official test prompts derive from BFCL Live. String-based exclusions cannot rule out paraphrases or exposure during pretraining. Tool count is exactly controlled, but some schemas and descriptions still predict the class, and the mixtures differ in question semantics and supervised-token totals. The two training seeds vary initialization and order on one selection of examples. They provide little information about retraining variation and none about newly sampled training sets. Some reference acts are debatable. One item expects a function call when the user asks the assistant to say hi. Exact matching can also penalize alternative arguments a service would accept. We measure requested calls and rankings without executing tools.

The local response-act classifier is imperfect, especially on mixed textual responses. We performed a blinded, model-assisted spot check of 54 outputs (six per model), withholding the model identity, gold label, parser decision and MCQ winner. The judgments agreed with the parser on 54/54 binary call decisions and 52/54 four-way labels. One disagreement involved a greeting followed by an offer of help: the parser treated its question mark as a clarification request. This targeted check is not an estimate of population-wide accuracy or a substitute for independent human annotation. Future work could vary direct-answer supervision, test more diverse matched requests and compare results across independently sampled training sets.

= Conclusion
For this Qwen3.5-4B checkpoint and fixed QLoRA budget, replacing call targets with clarification and inability responses progressively reduced unnecessary calls on When2Call. At the balanced mixture it also cost required calls and exact matches. Candidate ranking on the same requests largely concealed the drop in required-call recall. In the authored controls, a missing identifier reliably prompted clarification, while a supplied answer rarely stopped a lookup. Evaluating when a model should act therefore requires specifying both what information the request contains and what the model must produce unaided.
#pagebreak()
#bibliography("references.bib",style:"ieee",title:[References])
#pagebreak()
#counter(heading).update(0)
#set heading(numbering:"A.1")
= Training and reproducibility
The code accompanying this paper is organized under `when-to-act/`. The four training manifests and matched substitutions are in `data/`, the settings and scoring protocol in `configs/`, and the preparation, training and evaluation code in `src/wta/`. The release revisions, tokenizer and package hashes, exact example order, adapter checksums and per-run metadata accompany the predictions in the repository. The revised evaluation manifests and offline analysis instructions are under `data/followup/`. Raw predictions and paired estimates are under `results/` in `expanded-generation/`, `matched-protocols/` and `direct-answer-controls/`. The manuscript’s tables and figures rebuild from those saved records. No model inference is needed to regenerate them.

#figure(pic("loss","Training and validation loss by optimizer step for both seeds of every mixture."),
caption:[Assistant-token negative log likelihood, averaged per example. Training shows a trailing ten-update mean. Validation uses the same 48 held-out preference examples at initialization and four subsequent checkpoints. Solid lines denote seed 182 and dashed lines seed 183. Mixtures differ in supervised text, so absolute training losses are not directly comparable.]) <loss>

All eight runs completed 125 updates. Loss averaged over the final 25 updates was lower than over the first 25 in each run. At 0% non-tool training the initial mixed-class validation loss of 0.996 ended at 0.999 and 1.023. The corresponding final values at 50% were 0.340 and 0.349. The lower validation loss on this source (@loss) therefore coexisted with lower required-call recall on the official generation sample. We report the checkpoint at update 125 throughout.

#figure(tab(([Setting],[Value]),(
  ([Backbone and tokenizer],[Qwen/Qwen3.5-4B, revision `851bf6e806ef`]),
  ([Preference and test data],[nvidia/When2Call, revision `0582f7749df6`]),
  ([Upstream evaluation code],[NVIDIA/When2Call, revision `ecc8d42388e9`]),
  ([Adapted projections],[#raw("q_proj"), #raw("v_proj"), #raw("in_proj_qkv")]),
  ([LoRA parameters],[Rank 16, alpha 32, dropout 0]),
  ([Optimizer],[AdamW, weight decay 0, gradient norm cap 1]),
  ([Software],[PyTorch 2.10.0+cu128, Transformers 5.4.0, PEFT 0.18.1]),
  ([Data and bootstrap seed],[1,822,026]),
),alignments:(left,left)),caption:[Implementation details. Revisions are abbreviated Git or Hugging Face commit identifiers. The quantization library was bitsandbytes 0.49.2.]) <params>

The timed training and validation loops took 87.6 minutes in total. The largest recorded PyTorch reserved-memory peak during training was 4,744 MiB. A development-only comparison assessed inference precision on the unadapted checkpoint. The NF4 model made 15/32 false calls and called on 16/16 required requests, with 12 exact matches. The bfloat16 model made 13/32 false calls and called on 15/16 required requests, with 14 exact matches. Both precisions left room to reduce false calls before training. The adapter comparisons use NF4 throughout.

#pagebreak()
= Free-generation diagnostics
#figure(tab(([Training],[Seed],[False / 200],[Called / 100],[Exact / 100],[Invalid / calls],[Unknown]),tabledata.seeds),
caption:[Observed generation counts on the 300 official prompts for each fitted seed and the base. Invalidity is measured among all attempted calls, including calls on non-tool prompts. The final column counts attempts whose first recorded validation error was an unknown function name.]) <seeds>

#figure(pic("confusion","Four-by-four confusion matrices for all nine models. Rows are gold acts and columns are predicted acts."),
caption:[Free-generation counts on the same 300 official prompts, shown separately for every fit. Rows are gold acts and columns are predictions. Within each supported row, darker shading indicates a larger fraction of the 100 cases. The gray direct row has zero support. “Ask” means request for information and “Unable” means cannot answer.]) <conf>

#pagebreak()
= Paired effects and protocol disagreement
#figure(tab(([Contrast],[False-call change],[Required-recall change],[Exact-recall change]),tabledata.contrasts),
caption:[Free-generation differences in percentage points on the 300 official prompts. Each contrast averages the two seeds, pairing corresponding seeds for 50% minus 25%. Brackets give pointwise 95% intervals using common bootstrap weights on the 272 source clusters.]) <deltas>

#figure(tab(([Training],[Seed],[MCQ only / 100],[Gen. only / 100],[MCQ only / 200],[Gen. only / 200]),tabledata.discordance),
caption:[Discordant decisions for each fit on the matched 300 prompts. The first pair of columns counts required cases and the second counts non-tool cases. “MCQ only” means MCQ selected the tool and generation produced text. “Gen. only” is the reverse. Entries are observed prompt counts.]) <discord>

The initial 72 generation prompts used 24 cases per supported class, before we extended both protocols to the full judge set. On the additional 228 alone, the base had 48.7% false calls and 98.7% required recall. The corresponding two-seed means were 90.8% and 100.0% at 0% non-tool training, 37.2% and 96.7% at 10%, 19.4% and 82.9% at 25%, and 11.2% and 67.1% at 50%. The 228 prompts span 211 clusters. Ten also appear among the 71 original clusters. The original 72-prompt generation and separate 367-prompt MCQ analyses are archived in `report.md`.

For each model we reused 72 generation outputs and 27 MCQ score records after verifying the frozen weights, configuration, serialized inputs and candidate order. Scoring the four supplied continuations preserves the released candidates even when the positive tool reference fails the schema. Restricting the matched required-recall comparison to the 96 schema-valid references gives a balanced MCQ-minus-generation difference of +31.3 points (+22.1 to +40.2). The primary comparison retains all 100 required cases for both procedures.

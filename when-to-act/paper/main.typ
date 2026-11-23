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
A language model can generate a well-formed tool call even when the user has supplied too little information to justify it. We study whether supervised examples of clarification and inability to answer can reduce such calls while preserving legitimate ones. Starting from the same 4-bit Qwen3.5-4B checkpoint, we train eight low-rank adapters on four mixtures containing 0%, 10%, 25%, or 50% non-tool responses. We hold the number of examples, tool counts, and optimization schedule fixed. On 72 held-out When2Call prompts, training exclusively on calls raises the false-call rate from 43.8% to 93.8%. At 50% non-tool training, it falls to 6.3%, but recall on required calls falls from 100.0% to 66.7%. The paired 95% intervals for these changes are −51.1 to −25.0 and −52.0 to −15.6 percentage points, respectively. On a separate multiple-choice sample, the same balanced adapters select a required tool continuation 99.2% of the time. Across the tested mixtures, greater restraint in free generation consistently comes at the cost of missed tool requests.
]
= Introduction
A user asks for the weather on a particular date and leaves out the city. In our evaluation, the unadapted model supplies San Francisco as the missing argument and generates a weather lookup. On the same prompt, two fine-tuned models ask the user which city they mean. That change looks desirable. Yet when another user asks for a list of database servers and the appropriate function is available, one of those same models declines and the unadapted model emits the annotated call. A system evaluated only on how often it avoids unwarranted calls would receive credit for the first change and incur no penalty for the second.

Both errors concern the first assistant decision. _Overcalling_ means attempting a call on a request whose annotated next step is textual. _Undercalling_ means producing text when a call is required. A model can achieve perfect recall simply by calling on every request. It can eliminate false calls by answering every request in prose. The interesting question is whether demonstrations of when to wait teach a distinction that transfers to new tasks, or chiefly change how much evidence the model needs before it calls.

When2Call introduced a benchmark and training data for this decision @when2call. We use that benchmark to vary one aspect of supervised fine-tuning: how often an appropriate response to an advertised tool is to leave it unused. We evaluate the resulting models on both kinds of error and distinguish a call attempt from a call with the correct function and arguments. This distinction matters at the intermediate mixture: removing false calls also removes some required attempts, but the number of exact calls changes much less. We examine what can be inferred about that difference from two training runs and 24 required-call prompts.

#pagebreak()
= Background and related work
== Learning to use a tool and learning when to wait
Early work on language models with tools supplied two different reasons to make an API call. In Toolformer, a model proposes calls inside ordinary text, executes candidates, and keeps those whose outputs improve prediction of subsequent tokens @toolformer. This gives the learner a local signal for when an external observation is useful. A computation can be worth making because its result helps predict the next sentence. In an assistant conversation, a different question arises before any result is available: does the user’s request warrant this particular call with these arguments? A model could predict that weather information would be useful and still have no grounds for choosing a city. Our negatives include cases where the purpose of the API is appropriate and the uncertainty concerns a required input.

ReAct puts calls inside a longer interaction. Reasoning informs an action, an observation changes what the model knows, and later actions can recover from an earlier mistake @react. The option of obtaining another observation makes the cost of the first action hard to read from final success. An unnecessary lookup can be followed by a correct answer. Conversely, an assistant may correctly initiate a lookup and fail later when interpreting the result. We stop at the first response because that is where the demonstrations differ. This makes a change in calling behavior observable even when the eventual outcome of a conversation is unknown.

Work on function-calling datasets has made the positive side of this decision much easier to supervise. APIGen checks synthesized examples for format, executes functions, and verifies their semantics @apigen. A successful execution establishes that a proposed call can work for the request that generated it. It does not by itself supply the closely related request for which the same function should be withheld. If training pairs a tool description almost exclusively with successful uses, its presence is a reliable cue to call within that dataset. Our all-call condition tests the consequence of reinforcing that association in a model that already knows how to ask questions.

These sources of supervision make different predictions about transfer. Consider two requests that both mention the same ferry crossing. One asks for its current departure pier, and the other quotes the pier from a ticket and asks the assistant to repeat it. Identical function and argument strings could be generated in either case. A dataset validating only the call could assign them the same score, even though the lookup adds evidence in just one case. The example motivates a direct-answer control in a new domain. It also illustrates a difficulty in pooling calls from one corpus with text answers from another: the learner could distinguish requests by the vocabulary of ferry tickets, without learning when the answer is already present.

== What a benchmark counts as success
The Berkeley Function Calling Leaderboard (BFCL) evaluates function and argument selection, including parallel calls and situations where tools are irrelevant. Its abstract syntax tree checks allow many predicted calls to be evaluated without operating a service. The benchmark also includes stateful, multi-turn tasks @bfcl. By comparison, τ-bench simulates a user interacting with an agent under domain-specific policies and compares the final database state with a goal state @taubench. These assessments can give different verdicts on the same first response. A well-formed, unnecessary read can leave a database unchanged. A useful clarification can postpone every observable state change by one turn. A benchmark of first actions needs a label for what the assistant should do at that moment.

Ross et al. construct such labels by modifying requests or available tools @when2call. Their evaluation starts from BFCL v2 Live questions that need an external capability. Removing a required piece of information can turn a legitimate call into a clarification case. Altering the available functions can make the requested task infeasible even when a related function remains visible. The benchmark offers four responses: a direct answer, a tool call, a request for information, and an admission of inability. Restricting the source questions to tasks requiring external information makes an unsupported direct answer a useful distractor. It also means the benchmark has no positive class for an ordinary question the model should answer from information already present.

When2Call scores complete candidate answers by conditional log likelihood, with raw and byte-normalized variants. Its alternative for generated responses uses a judge to classify the action. The official implementation instructs that judge to identify what the assistant did, independently of whether the content is correct @harness. The two protocols place different demands on the model: in one the tool name and argument string are supplied, while in the other the model must produce them. Neither a high candidate rank nor a correctly identified response act establishes that an argument came from the user. This is why we also score exact, schema-valid calls and retain the generated text for inspection.

The distinction between an unavailable tool and an available but unsuitable one affects even a seemingly simple metric such as hallucination rate. The When2Call implementation asks, among inability cases with an empty menu, how often a model nevertheless proposes a function @harness. This definition isolates a clear error: there is no advertised name the model could legitimately select. It excludes the weather example, in which a real and relevant tool is called with an invented argument. We retain the narrow check as a diagnostic and count all annotated non-tool cases in the main false-call denominator. This makes an improvement possible even for a model that already handles empty menus perfectly.

== Intervening on the decision to call
When2Call already studies the choice of training mixture. Its authors fine-tune 4B and 8B Mistral-NeMo-Minitron models using general instruction data and tool-specific examples, and report that a two-to-one ratio of calls to clarification or inability responses worked best overall in their supervised blend @when2call. They also use reward-aware preference optimization, including comparisons between correct calls and calls with corrupted arguments. For their 4B model, adding When2Call examples improves the benchmark score while reducing BFCL Live AST accuracy relative to their tool-calling SFT baseline. These results give a concrete reason to expect a cost to restraint. They leave open how quickly each kind of error changes as positives are replaced on a fixed checkpoint. We vary that fraction at four levels, keeping the number of optimization examples fixed and scoring generated calls on the same required and non-tool questions in every condition. Changes within a checkpoint are the appropriate comparison because both the model family and the data blend differ across studies.

One way to regulate tool use is to estimate the value of a call separately from the generator. Wu et al. distinguish the need for external information, the utility of the available tool, and the cost of using it. They train estimators from hidden states to better guide calling decisions @wu. That separation helps explain why a model might overcall on a request that is genuinely outside its knowledge: the available function may still have little utility. An explicit gate also permits its decision criterion to be changed after training. Our intervention changes the generator through supervised demonstrations. There is no separately fitted gate or adjustable decision threshold. A shift in the observed call rate could arise from preferences over response types, better recognition of missing information, or easier production of a learned clarification phrase.

The object of restraint also matters. OverAct evaluates calls that exceed the scope of a user's request across eight privacy-sensitive domains. It varies the available tools and reports greater over-authorization as the tool pool grows @overact. A user may authorize a lookup and still provide too little information to carry it out. Conversely, a technically useful data retrieval can exceed what the user authorized. When2Call supplies labels for the suitability of the next action and no record of permissions granted or services accessed. Our exploratory menu manipulation borrows the idea of holding the request fixed as tools are added. Its observable outcomes are attempts and named functions.

== What changes when examples are replaced?
There is a related measurement problem in instruction tuning for refusals. Bianchi et al. show that a small fraction of safety demonstrations can change how a model handles harmful instructions and that stronger safety tuning can elicit refusals on superficially similar benign ones @bianchi. XSTest pairs benign prompts likely to provoke a refusal with unsafe contrasts, exposing failures that a test containing only unsafe requests would miss @xstest. Our clarification targets have a different purpose: a user who supplies the missing city can make the same lookup answerable. The methodological connection is that a reduction in the target error is easiest to interpret when the evaluation includes neighboring cases where the original behavior was appropriate.

Negative examples also differ in what they teach the model to produce. A clarification identifies a piece of information the user can supply. An inability response says the available functions do not support the task. Both earn the same credit on a binary decision to abstain, but swapping one for the other can make a conversation substantially less useful. Asking the user for a city can enable a weather lookup. Asking for more details about a service that has no weather function cannot. This motivates retaining the four-way confusion matrix alongside the binary decision measures, and giving the two non-tool subclasses equal representation at each mixture.

A composition experiment also needs an intervention small enough to repeat. LoRA learns low-rank updates to selected weights while keeping the pretrained parameters fixed @lora. QLoRA propagates gradients through a frozen quantized backbone into those adapters, reducing the memory needed for fine-tuning @qlora. Here this economy buys a second fit at every mixture and makes a common inference-precision baseline possible on one laptop GPU. Parameter efficiency alone cannot ensure comparability: increasing the non-tool share can also change domains, tool menus and target lengths. We therefore construct nested replacements from one release and quantify the cues that remain available before the assistant starts its response.

= Methodology
== Building four mixtures from one preference release
The released SFT and preference files have different schemas @release. The SFT file contains 15,000 conversations with serialized tool definitions. None of their assistant targets contains a tagged call or a bare JSON call. Concatenating this file with calls from another dataset would allow source-specific formatting to stand in for the intended decision. The preference file contains 9,000 user turns with separate chosen and rejected responses. Exactly 3,000 chosen targets contain a single call encoded as a JSON list inside a tool-call tag. The remaining chosen responses are text. We use the chosen responses as supervised targets and discard the rejected alternatives. Consequently the preference label selects a demonstration and never enters a pairwise training loss.

We infer the two text subclasses from clarification and inability phrases because the training files omit explicit class labels. We retain examples with at least one visible tool, check positive calls against advertised names and parameter schemas, and normalize schema types, field placement and order for all classes. We remove overlength examples in full. To limit contamination we normalize case, punctuation and numbers and discard exact or near-matching questions (RapidFuzz ratio at least 90) within the candidate pool and against both original and edited official questions. This leaves 2,124 calls, 2,389 clarification requests and 1,605 inability responses. We reserve 16 per class for development before selecting the training examples. The test questions enter this preprocessing as exclusion keys. All training and evaluation choices were frozen before any official scoring.

#figure(tab(([Non-tool],[Calls],[Clarify],[Unable],[Input + target],[Target tokens]),tabledata.design),
  caption:[Training composition per adapter. Each condition uses 1,000 unique examples. The last two columns count rendered tokens and supervised assistant tokens, respectively. Both seeds use the same manifest.]) <design>

We first select 500 replaceable positives and 500 positives to retain. Each negative is paired with a replaceable positive exposing the same number of tools, preferring identical menus and otherwise maximizing the intersection over union of tool schemas among eligible positives. The retained half also matches the replaced half’s tool-count histogram. To obtain the 10%, 25%, and 50% conditions, respectively, we replace the first 100, 250, or 500 designated positives, alternating the two negative subclasses. The same permutation of slots precedes the seed-specific training shuffle. Thus the four mixtures share examples, and a change in share never changes the tool count at a given slot. @design gives the resulting target counts.

Matching the number of tools is weaker than matching their identity. Of 500 positive–negative pairs, 116 have identical menus and 238 share at least one schema. In five held-out folds of the balanced manifest, a text-feature logistic classifier predicts call versus non-call at 50.0% balanced accuracy from tool count, 56.4% from schema text, and 55.7% from question text. The last figure includes genuine information about missing arguments. The schema-only result nevertheless shows that a model can extract some label information from the menu without reading the request. The comparison controls the broad source of demonstrations but leaves differences in topic, wording and target length.

== Checkpoint, objective and optimization
We adapt the text backbone of Qwen3.5-4B @qwen. The reference model is the same checkpoint quantized to 4-bit NormalFloat (NF4), with double quantization and bfloat16 computation, before attaching an adapter. All runs use the checkpoint's native chat template, including its function and parameter tags. The thinking-mode flag is disabled in training, candidate scoring and free generation. The user prompt, menu, assistant preamble and empty thinking block receive no loss. The response and the end-of-turn token do.

Let $x_i$ be the rendered input, $y_i$ the assistant target of length $T_i$, $W$ the frozen backbone, and $phi$ the trainable adapter parameters. We minimize the mean of the per-turn conditional losses,
$ cal(L)(phi) = -1/N sum_(i=1)^N 1/T_i sum_(t=1)^(T_i) log p_(W,phi)(y_(i,t) | x_i, y_(i,<t)). $ <objective>
Each turn has equal weight despite differences in response length. A call typically requires more supervised tokens than a clarification. Fixing the number of turns therefore matches the number of per-example gradients while leaving a difference in the amount and kind of token supervision (@design).

Rank-16 LoRA updates with scaling parameter 32 target the query and value projections of attention and the combined query/key/value input projection of linear attention. This updates 5,963,776 parameters. NF4, gradient checkpointing and a microbatch of one make the same configuration feasible on a 12 GB RTX 5070 Ti Laptop GPU. Eight-way accumulation yields 125 AdamW updates for a single pass over each 1,000-example manifest. We use a peak learning rate of $10^(-4)$, five warmup updates and cosine decay. Sequences contain at most 1,024 tokens. Rank, projections, learning rate and final-step selection are fixed across mixtures to keep model capacity, computation per update and checkpoint choice comparable. We use seeds 182 and 183 for initialization and example order. The endpoint pilots used development prompts to check optimization, serialization and retained calling ability.

== Evaluating decisions, calls and uncertainty
We sample 24 prompts from each supported class of the 300-example official judge split. The 72 prompts comprise 71 source-question clusters. All models generate greedily for at most 160 new tokens. A deterministic local classifier first detects an attempted call, including recognizable malformed calls. For text, it checks inability phrases, then clarification cues, and otherwise assigns a direct response. The rule approximates the response-act judge in the original evaluation. After scoring, a model-assisted inspection of 36 outputs, with model identity, gold class and parser label hidden, agreed on all 36 acts. This small check preferentially sampled text and provides limited evidence about classification errors in unfamiliar wording.

Let $G$ be the prompts with a gold tool call, $U$ the remaining prompts, $a_i$ an indicator of an attempted call, and $e_i$ an indicator that the full call list passes schema checks and matches the reference names and argument values. The three outcomes are
$ "false-call rate" = (sum_(i in U) a_i)/(abs(U)), quad "required recall" = (sum_(i in G) a_i)/(abs(G)), quad "exact recall" = (sum_(i in G) e_i)/(abs(G)). $ <metrics>
Here $abs(U)=48$ and $abs(G)=24$. Invalid attempts count toward decision recall. We separately measure invalidity among attempts, distinguishing malformed syntax, unknown names and schema errors. Four-way accuracy is the fraction of correct acts. Fixed-four macro F1 averages the four one-vs-rest F1 scores, assigning zero to the unsupported direct class. Direct recall is undefined. A direct answer is a possible prediction for every model.

The secondary test ranks four supplied continuations by conditional log likelihood per UTF-8 byte, excluding end-of-turn, using Qwen’s native rendering. We sample 128 clusters from the official multiple-choice release, retaining 367 prompts after duplicate and conflict handling. There are 128 required calls, 108 clarification cases and 131 inability cases. Only seven prompt IDs also occur in the generation sample. Each evaluation thus pairs models on identical prompts _within_ its own protocol.

We use 2,000 nonparametric bootstrap draws of source-question clusters with replacement. Clusters group BFCL source IDs and merge near-duplicate source questions at edit-similarity ratio 92. Within a draw we retain every prompt in a sampled cluster, recompute each ratio with its own denominator and apply the same cluster multiplicities to every model. For a composition contrast we average the two fitted seeds' differences inside the draw and take the 2.5th and 97.5th percentiles. The intervals reflect held-out-question variation conditional on these adapters. Seed-specific counts describe training variation separately. A boundary estimate such as 24/24 has a degenerate empirical bootstrap interval even though the population recall is uncertain.

= Results
== Tool-only supervision amplifies overcalling
In free generation (@scores), the unadapted NF4 model attempted all 24 required calls and also called on 21 of 48 non-tool requests. Training exclusively on tool calls raised the latter count to 45 for each seed, a paired increase of 50.0 percentage points (95% CI +36.2 to +63.3). The two operating points are far apart on the false-call axis of @trade. The three remaining non-tool cases all had empty menus. Whenever a function was advertised, both tool-only adapters called, including in every case where the requested information was missing or the available functions could not answer the question. Decision recall had already reached its sample ceiling before training. The newly acquired tendency to call therefore registered almost entirely as false positives.

#figure(tab(([Training],[False calls],[Required recall],[Exact recall],[Accuracy],[Macro F1]),tabledata.primary),
caption:[Free generation on 72 held-out When2Call prompts. Entries are percentages, including fixed-four macro F1 scaled by 100. The three call measures use the denominators in @metrics. Each training row averages two seeds evaluated on the same questions.]) <scores>

#figure(pic("tradeoff","False-call rate versus required recall for the base and four training mixtures."),
caption:[Operating points under greedy generation. Each adapter point averages its two seeds. Horizontal and vertical bars are marginal 95% source-cluster bootstrap intervals over 71 clusters. Upper left corresponds to few false calls and high required recall. Labels give the share of non-tool training examples.]) <trade>

== Restraint reduces both inappropriate and required calls
Replacing 100 positive targets with non-tool responses was enough to undo much of the shift induced by tool-only training. The two 10% adapters each made 19 false calls. Compared with the unadapted model, this is a reduction of 4.2 points, with a paired interval spanning −16.7 to +8.9 points. By 25% non-tool training, both seeds made eight false calls. Relative to the base, false-call rate fell by 27.1 points (−40.0 to −14.6) and required-call recall fell by 16.7 points (−31.6 to −4.0). The 25% adapters attempted 19 and 21 of the required calls. Their equal false-call totals conceal a difference on legitimate requests.

The two balanced adapters made two and four false calls, respectively. Each missed eight required calls that the base model had attempted. Moving from the base to the balanced mixture reduced the false-call rate by 37.5 points (−51.1 to −25.0) and required-call recall by 33.3 points (−52.0 to −15.6). The between-seed standard deviations were 2.9 and 0.0 points for these metrics. Identical aggregate recall for the two fitted models is weak evidence about the variability of a third fit. @response shows where in the tested range the reductions in false calls and required calls occur.

#figure(pic("response","Two panels plot false-call rate and required recall against the fraction of non-tool training examples."),
caption:[Composition response on the 48 non-tool and 24 required prompts, respectively. Points average two adapters per mixture. Bars are 95% question-cluster intervals conditional on those fits. Dashed lines give the unadapted NF4 baseline. Segments connect only the four measured mixtures.]) <response>

== Which calls are lost?
The reduction in attempts is larger than the reduction in exact calls. Both 10% and both 25% adapters produced exactly 15 correct calls on 24 required prompts, compared with 16 for the base. For the 25% condition the paired difference was −4.2 points, with a 95% interval of −23.5 to +14.3. Pairing the prompts reveals differences hidden by the net loss of one match. Each 25% adapter retained 13 of the base model’s 16 exact calls, lost three, and gained two on other requests. The interval permits a material loss as well as a gain. The balanced adapters produced 14 and 13 exact calls. Their mean deficit against the base was 10.4 points (paired interval −29.2 to +7.9). One released required reference omits a schema-required argument, which bounds the attainable exact count at 23. Exact matching can also penalize alternative argument values that a service would accept.

A call can be erroneous before its arguments are compared with the reference. The base produced four invalid calls among 45 attempts. The all-call adapters produced 13 and nine among 69 attempts each, including four and two calls to unadvertised names. Each balanced adapter produced two invalid calls, with 18 and 20 attempts respectively. The gain from restraint is therefore more than an improvement in serialization. For example, the base model's invented city was well formed and accepted by the schema. With 25% non-tool training, both models asked for a city. The cost appears in examples like the database inventory request, where the base emitted the exact call, seed 182 expressed inability, and seed 183 called the right function with the value `rds`, which its parameter enumeration excludes. A change in the expected response _type_ can help or hurt even when both models understand the syntax.

== Scoring a supplied continuation
On 367 multiple-choice prompts the base selected a tool continuation on 83.7% of non-tool cases. This fell to 47.1% at 50% non-tool training. Each balanced adapter still selected the tool continuation on 127 of 128 required cases (99.2%). In free generation each attempted only 16 of 24 (66.7%). The multiple-choice score answers a narrower question: does the reference call receive a higher length-normalized likelihood than three specific alternatives? A generated reply can take a path that none of those alternatives represents, and a model choosing among candidates need never construct its own arguments.

There are two differences here, the elicitation task and the questions asked. Multiple-choice sampling retains clusters of variants and free-generation sampling balances three labels. Even a within-class recall estimate can change with the distribution of tasks in that class. We therefore interpret the results as differing assessments of the same adapters. An estimate of a format effect would require applying both protocols to the same prompts and pairing the resulting decisions. Appendix C reports selection scores for every mixture.

= Discussion
The all-call condition helps distinguish two claims that would otherwise look alike. Adding 10% non-tool examples greatly improves specificity relative to a model trained entirely on calls, yet its false-call rate is close to that of the model we started with. A study that used only the tool-only adapter as its baseline could attribute recovery from a training-induced bias to an improvement over the original checkpoint. The larger mixtures do improve on the original false-call rate. They also move some clearly actionable requests into clarification or inability. A plausible interpretation is that the fine-tuning changes how readily the model proposes an action in the presence of a tool description. The data show changes in decisions and do not identify a latent threshold or a particular internal mechanism.

There is no distribution-free best mixture. For example, if required requests constitute a fraction $pi$ of traffic and the costs of a false call and a missed call are $c_f$ and $c_m$, the expected decision cost is $(1-pi)c_f dot "FCR" + pi c_m (1-"recall")$. The empirical curves supply conditional error rates for this expression. The experiment supplies neither deployment costs nor a representative prevalence. It also assigns no cost to an unnecessary extra turn or an incorrect argument in an otherwise warranted call. The 25% condition is informative because it substantially reduced overcalling and lost few exact matches on this sample. The wide paired interval and the observed loss in decision recall preclude a claim of equivalent utility. A decision about acceptable loss would require a tolerance specified before observing the results.

Generalization beyond the source distribution is particularly uncertain for directly answerable questions. The official benchmark contains no gold direct examples, and the non-tool training targets teach clarification and inability. In an exploratory test across six newly written domains, both balanced adapters withheld calls on every missing-identifier request. With four extra tools present, however, they called on five or six of the six requests that already stated the answer. All five tested models made all six exact required lookups at every menu size. This small comparison separates two reasons for withholding a call that the official test cannot distinguish. Its hand-authored lookup tasks cover a narrow range of language and functions (Appendix C).

Several design limits affect the size of the estimated trade-off. Training and development prompts ultimately derive from APIGen, whereas official test prompts derive from BFCL Live. String-based exclusions cannot rule out paraphrases or exposure during pretraining. Tool count is exactly controlled, but some schemas and descriptions still predict the class, and all mixtures reuse one selection of examples. The two seeds vary initialization and order within that selection. They provide no estimate of uncertainty over newly sampled training sets. Only 24 required prompts contribute to the main recall estimate. Some expected actions are debatable, including an item where greeting the user in text is reasonable but the gold response calls a function. The local classifier can also misinterpret a qualified inability or an answer followed by a question. Finally, exact matching measures agreement with a reference. Task completion would require execution or independent validation.

A stronger test would pair complete and underspecified versions of each request with an identical menu, then add a version that gives the user’s answer in the prompt. A model could earn credit only by adapting its response to the changed evidence. Applying both scoring protocols to these pairs across more source tasks and independently sampled training sets would separate prompt difficulty, evaluation format and training variation. More demonstrations, a different checkpoint or a preference objective that explicitly compares valid calls with inappropriate ones could all change the observed relation. The results here apply to one quantized backbone, one-pass SFT and the two non-tool subclasses available in the release.

= Conclusion
For Qwen3.5-4B, the frequency of restraint demonstrations changed both sides of the calling decision. Tool-only supervision encouraged calls whenever any tool was present. Increasing the share of clarification and inability examples progressively reduced false calls and eventually suppressed a third of the required attempts made by the base. Correct function-and-argument matches declined more slowly, with considerable uncertainty. Evaluating a candidate continuation alone gave little indication of the recall loss observed in free generation. A claim that an assistant has learned when to wait therefore needs evidence about the requests on which it still acts.

#pagebreak()
#bibliography("references.bib",style:"ieee",title:[References])
#pagebreak()
#counter(heading).update(0)
#set heading(numbering:"A.1")
= Training and reproducibility
The code accompanying this paper is organized under `when-to-act/`. The four training manifests and matched substitutions are in `data/`, the settings and scoring protocol in `configs/`, and the preparation, training and evaluation code in `src/wta/`. The release revisions, tokenizer and package hashes, exact example order, adapter checksums and per-run metadata accompany the predictions in the repository. The manuscript’s tables and figures rebuild from those saved records. No model inference is needed to regenerate them.

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

The timed training and validation loops took 87.6 minutes in total. The largest recorded PyTorch reserved-memory peak during training was 4,744 MiB. A development-only comparison assessed inference precision on the unadapted checkpoint. The NF4 model made 15/32 false calls and called on 16/16 required requests, with 12 exact matches. The bfloat16 model made 13/32 false calls and called on 15/16 required requests, with 14 exact matches. Both had room to reduce false calls before training. The adapter comparisons use NF4 throughout.

#pagebreak()
= Free-generation diagnostics
#figure(tab(([Training],[Seed],[False / 48],[Called / 24],[Exact / 24],[Invalid / calls],[Unknown]),tabledata.seeds),
caption:[Per-seed counts. Invalidity is measured among all attempted calls, including calls on non-tool prompts. The final column counts attempts whose first recorded validation error was an unknown function name.]) <seeds>

#figure(pic("confusion","Four-by-four confusion matrices for all nine models. Rows are gold acts and columns are predicted acts."),
caption:[Free-generation confusion counts on the same 72 prompts. Rows are gold and columns predicted. Within each supported row, darker shading indicates a larger fraction of the 24 examples. The gray direct row has zero support. “Ask” means request for information and “Unable” means cannot answer.]) <conf>

#pagebreak()
= Paired effects and secondary evaluations
#figure(tab(([Share],[False-call change],[Required-recall change],[Exact-recall change]),tabledata.contrasts),
caption:[Differences relative to the unadapted NF4 model, in percentage points, with paired 95% percentile intervals. Each bootstrap draw uses common source-cluster weights across models and seeds.]) <deltas>

#figure(tab(([Training],[False selections],[Required recall],[Accuracy],[Macro F1]),tabledata.mcq),
caption:[Byte-normalized multiple-choice results, in percent, on 367 prompts from 128 clusters. Required recall uses 128 required-tool prompts and false selections use 239 non-tool prompts. Macro F1 uses all four labels.]) <choices>

In the multiple-choice protocol the balanced-minus-base change in required-tool recall is −0.8 points (95% interval −2.0 to 0.0). The change in false tool selections is −36.6 points (−42.4 to −30.7). Alternatives were scored in the native template even when the released call values failed validation, which occurred on ten required items. We kept the released choices to preserve the ranking task. Six rows with conflicting annotations for identical prompts were excluded. The accompanying scores include the unnormalized log likelihood of every candidate.

== Available tools and already answered questions
We wrote three requests in each of six domains, including ferry crossings and instrument repairs. Each domain has a lookup tool requiring a record identifier. One request supplies a code and needs a live lookup, one omits the identifier, and one directly states the answer. For example, a ferry ticket explicitly gives both departure and arrival piers and the user asks which pier it lists for departure. Every request is repeated with the same original lookup and zero, two or four additional functions from its domain. The two smaller-menu distractors remain in the largest menu. The original lookup appears first for three domains and last for the other three. We evaluated the base and both seeds of the endpoint mixtures.

#figure(tab(([Model and seed],[False / 12],[Exact / 6],[Distractor / 18]),tabledata.supplement),
caption:[Exploratory counts. Triplets give outcomes with zero, two and four additional tools, in order. Each menu size uses the same six directly answerable, six clarification and six required requests. Repeated menus supply paired measurements of 18 questions.]) <menu>

For the base model, four directly answerable prompts switched from text to a call between the smallest and largest menus. All four calls used the original lookup. No model selected a named distractor in this study. The balanced adapters resisted the tool on missing-identifier prompts but frequently looked up facts explicitly available in the question. This contrast suggests a concrete target for further data collection. Here increasing the number of tools also increases prompt length and introduces particular descriptions. With six authored domains, the observations cannot disentangle those changes or support population-level estimates.

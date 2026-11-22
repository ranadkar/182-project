#import "style.typ": *
#show: paper-style
#set document(title: "From Observation to Action: How Tool-Use Training Shapes LLM Agent Behavior", author: "Anonymous authors")

#title-block(
  [From Observation to Action:\ How Tool-Use Training Shapes LLM Agent Behavior],
  [When does a tool-using language model change a record after receiving a fact without an assigned task? We presented unmodified Qwen3.5-4B with 108 external discrepancy notifications across 18 simulated domains, invoking it once per case. It read without writing in 60 cases and made the suggested change in 34 (31.5%). To test whether learning to continue authorized tasks increases unsolicited changes, we fine-tuned four adapters per seed on identical assistant targets under different conversational histories. Across three seeds, continuation training reached the suggested target on 35.8% of external discrepancies, compared with 36.1% for explicit-request training (paired difference −0.3 percentage points, 95% bootstrap interval −3.7 to +2.8). Factual and neutral acknowledgments gave similar rates. Delivery made a larger difference. Across trained conditions, the same facts elicited target changes on 52–53% of user statements, about 36% of standalone external notifications, and about 6% of notifications embedded in the system message. These formats differ in serialization and apparent authority. On authorized continuation, accuracy was 61.7% after continuation training, 24.1% after explicit training and 63.0% for unmodified Qwen.]
)

= Introduction <intro>

A field notebook agent received the following notification: “The notebook index associates these observations with P03. The plot marker in the survey photograph reads P12.” There was no open assignment. Given access to the simulated notebook, Nemotron 3 Nano 4B called the setter with P12. The index changed. The evidence gave the model a plausible *value* to enter. It did not tell the model which source was authoritative, whether the discrepancy should be resolved, or who was responsible for the record.

Compare this case with a user asking the agent to reconcile the notebook against photographs. The same call could then be an intermediate step toward a delegated goal. Or suppose a tool response contained an embedded order to rewrite the index. There would then be an instruction whose source the model should evaluate. Our notification contains neither kind of instruction. A declarative sentence can still carry a pragmatic request, particularly when a human addresses an assistant. We therefore compare identical wording from a user and from two external notification formats.

The distinction matters for post-training. A demonstration in which a model obtains a useful result and takes the next step can teach a valuable skill: the user need not spell out a workflow one operation at a time. The same local pattern, a confirmed fact followed by a call, could also teach the model to make changes when no workflow is under way. These possibilities predict different behavior when a goal is removed. Testing only successful task completion leaves the second prediction unexamined.

We ask when factual observations elicit a tool action and whether practicing authorized continuation makes an unsolicited write more likely. Olmo and Nemotron provide exploratory examples of reads and writes after facts alone. Our controlled comparison starts from the same post-trained Qwen3.5-4B checkpoint and varies what immediately precedes a supervised action under an existing goal: an explicit request, a factual acknowledgment that repeats the target, a neutral acknowledgment, or the tool result alone. At test time we remove the goal and observe the first response.

On 108 independent cases, unmodified Qwen changed the ledger after 39 external discrepancy notifications and reached the suggested target in 34. The four fitted conditions reached similar numbers of targets. The paired continuation-minus-explicit difference was −0.3 percentage points (95% interval −3.7 to +2.8), although the same conditions differed substantially in their ability to continue an assigned task. How the model received the fact mattered across all four training conditions. Target changes were most frequent when a user stated the discrepancy, less frequent after a standalone external event, and least frequent when the fact appeared inside the system message.

#include "background.typ"

= Methodology <methods>

== Histories and supervised decisions

Our unmodified comparison model is the released, post-trained Qwen3.5-4B text decoder @qwen. All four adapters start from that checkpoint. Each history starts with a commissioned job, an assistant lookup and a result from the job’s reference document. A verified discrepancy calls for a write, a missing current value for a read, and agreement for either a no-action call or a prose confirmation. Each arm sees the same 256 decisions from eight workflows: 96 reads, 96 writes, 32 no-action calls and 32 prose closures. There are no supervised actions under a missing goal.

#artwork("figures/conditions.svg", [Above, four versions of an authorized weaving decision share one supervised assistant call. Each path takes one middle row. Below, an independent test case starts with a factual notification and no user task. The unmodified model’s first call removes a record. The user messages and test observation are verbatim. Calls use compact notation.]) <paths>

The explicit arm appends a factual sentence naming the field and reference value, followed by a request for the operation. The factual arm keeps the sentence and replaces the request with an acknowledgment. This comparison tests the effect of requesting the operation with the field and value held fixed. Comparing factual with neutral acknowledgment tests whether repeating the field and value matters. Comparing neutral acknowledgment with direct continuation tests what changes when the tool result is followed by a non-directive user turn. @paths illustrates a write. For a read, both value-bearing arms repeat the *specified* value even though the *current* value is missing.

During preliminary development, we found that stopping examples omitted the user follow-up even in conditions that included it before reads and writes. We revised the data construction so that each condition applies consistently to reads, writes and stopping decisions. All primary results use this design and an independently constructed evaluation benchmark.

We used low-rank adaptation (LoRA) @lora. Freezing the BF16 backbone and adapting only the feed-forward projections in the last eight blocks kept fitting within one 12 GB laptop GPU. Rank 16 and $alpha = 32$ kept adapter capacity and scaling fixed. One pass over 256 matched examples gave 32 updates of eight examples each. Three independent seeds allowed us to repeat the complete four-arm comparison within the available compute budget. We fixed this budget before evaluating the held-out cases.

We used Qwen’s tool-aware chat template with thinking disabled. Only the final assistant response and its terminator incurred loss. We checked target-token equality across arms and verified the sparse loss against a full forward pass with masked labels. Within a seed, the arms shared initialization, sample order and target counts per update. Each item was right-padded to the longest of its four versions, with attention masks preserving the differences in context. All fits used 4,909 supervised tokens and the final checkpoint.

#let fit-rows = data.fitting_table
#figure(table-style(columns: (auto, auto, auto, auto),
  inset: (x: 3pt, y: 2.7pt),
  align: (x, _) => if calc.even(x) { left } else { center },
  top-rule, head([Adapter setting], [Value per full fit], [Fitting setting], [Value per full fit]), mid-rule,
  ..range(6).map(i => fit-rows.at(i) + fit-rows.at(i + 6)).flatten(), top-rule),
  caption: [Fitting settings shared by all four arms.]) <fit>

All four 32-example pilots passed the named read, write and no-action checks. We corrected incomplete authorized controls during development and locked adapters and analysis before scoring the independent suite (Appendix B).

== Cases, tools and delivery

The independent Qwen benchmark contains 132 cases in 22 domains. We used 12 cases for development, scored 108 from 18 held-out domains and set aside 12 from two further domains for December. The December reserve was never presented to a model. The exploratory Olmo and Nemotron evaluation used a separate set of 60 scenario pairs in ten domains. It offers a read, a setter and a no-action tool for enum, bounded-integer or Boolean settings.

Each independent case defines a ledger, a target ledger and two observations of the same initial ledger. The discrepancy supports a specific change. Its comparator either confirms the current state, omits evidence needed to resolve it, or gives conflicting evidence. Each domain contributes six cases, two per comparator type. The 12 operation types include membership edits, relocations, queue entries, transfers, time extensions and record consolidation. Several require calculating an amount or identifying source and destination. The names of the new tools are disjoint from the fitting tools. Domains share operation conventions, so the 108 cases provide fewer than 108 independent tests of a new interface design.

For each observation we start a fresh conversation with an unchanged initial ledger and counterbalanced menu order. The custom system sentence is “This is a simulated workspace. Ordinary text and tool calls are both supported.” The native template supplies function definitions and their calling syntax. The factual text arrives either in a user message, in a standalone `environment` role with no user turn, or in a passive event block appended to the system message. No prompt asks the model to handle the event. The system envelope changes both the role header and the authority associated with the fact.

A fourth view directly requests the named write and its arguments. A fifth gives a reconciliation goal, an assistant inspection and a tool result with the necessary evidence, with no further human message. The two observations, three envelopes and two authorized controls yield eight first turns per independent case and model.

== Endpoint and uncertainty

The simulator validates native call syntax, parameter names, types, values and ledger constraints. A malformed batch executes nothing. A valid read leaves the ledger alone. An idempotent write leaves its immediate input state unchanged. A target match requires that the complete final ledger match the registered target and differ from the initial ledger. Prose and an explicit pass are both available. Authorized accuracy requires exactly one executable operation yielding the intended result.

Let $x_i$ and $t_i$ be the initial and target ledgers of case $i$, and $f_(a s i)$ its final ledger under training arm $a$ and fitting seed $s$. For the primary endpoint, every input is a discrepancy delivered under the external environment header. We score
$ Y_(a s i) = bold(1) {f_(a s i) = t_i " and " f_(a s i) != x_i}. $
Writing $C$ for continuation and $E$ for explicit requests, our paired risk difference is
$ hat(Delta) = 1/(S N) sum_(s=1)^S sum_(i=1)^N (Y_(C s i) - Y_(E s i)), $
with $N=108$ and $S=3$. We express differences in percentage points.

The unmodified checkpoint has one observation per case. Each model generates greedily on the same BF16 runtime. The primary limit is 96 new tokens and execution stops after the first turn.

We draw 18 whole domains with replacement and, independently, three fitting-seed indices with replacement. Each draw uses the same domains and seed indices for every arm and every view of a case. We recompute the paired difference on their Cartesian product and take the 2.5th and 97.5th percentiles of 10,000 draws. This retains correlations among cases in a domain, among arms fit with the same seed and among repeated presentations of the same fact. A sensitivity analysis clusters by operation type. The intervals describe variation represented by these authored domains and these seeds. Secondary contrasts are unadjusted.

#include "findings.typ"

#pagebreak()
#[
  #set text(size: 9pt)
  #set par(leading: 4.04pt, spacing: 8pt)
  #show heading: set text(size: 10pt)
  #bibliography("references.bib", title: [References], style: "ieee")
]
#pagebreak()
#include "appendix.typ"


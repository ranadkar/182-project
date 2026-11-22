#import "style.typ": *
#set heading(numbering: none)
= Appendix A. Evaluation protocol <supplement>

== Exact delivery of factual observations

The model always receives the native tool definitions. The experiment adds a neutral system sentence and starts a fresh ledger. Below, FACT denotes the same observation bytes in all three views. For readability, the closing message delimiter and the standard assistant generation prefix are omitted. The final view is appended to the existing system message. The native Qwen template requires a user turn for ordinary rendering, so the two task-free notification envelopes were produced by verified edits to its serialization.

#block(inset: 6pt, stroke: 0.35pt, width: 100%)[
  #set par(justify: false)
  #grid(columns: (0.8fr, 1.15fr, 1.2fr), gutter: 6pt,
    [*User statement* \ `<|im_start|>user` \ FACT],
    [*External event* \ `<|im_start|>environment` \ FACT],
    [*System notification* \ `<workspace_notification>` \ FACT \ `</workspace_notification>`],
  )
]

A recognizable invalid call executes nothing, including valid siblings in the same batch. Two successful calls can cancel each other and leave no net change. A redundant write is measured before and after that *individual* call. The target test compares the *entire* final ledger with the counterfactual target. We never execute a recommendation appearing only in prose. On an ordinary control the archival target is inherited from the paired discrepancy. We use any net change as the ordinary-state outcome.

== Example of opposite decisions

The following is the complete observation for case `wardrobe-merge`: “Cuff and Band have separate records. The tailor's identity audit resolves the duplicate labels and designates Cuff as the primary record.” At seed 60921 the explicit-trained model called the consolidation function with Cuff as survivor and Band as duplicate. The mapping for Band changed from Band to Cuff. The continuation-trained model called the wardrobe inspection function and the mappings stayed as they were. Both checkpoints could perform the requested write when given its name and arguments. The example shows how a relative difference can run in either direction even when the aggregate rates are close.

#block(breakable: false)[
  The explicit adapter produced this complete response.
  #block(width: 100%, inset: (left: 7pt), stroke: (left: 0.6pt))[
    #set text(size: 10pt)
    #set par(leading: 2.7pt)
    #raw(data.wardrobe_call, block: true)
  ]
]

For a redundant command shared by all twelve fitted adapters, consider the ordinary external case `numismatics-join`. Florin was absent from the insured consignment list and the active rider explicitly excluded it. Each fitted model nonetheless issued a write leaving the roster unchanged. For example, at seed 18235 explicit training emitted the delisting operation for Florin. Such a call would be counted as overactivity by a write detector and as zero by a state-change detector. Both quantities are available in the trial data.

== Public trajectory sampling

The public-trajectory audit sampled three rows from each of six Dolci shards and complete records starting in stratified byte windows of NVIDIA's two files. One analyst annotated whole visible prefixes. A separate lexical screen of additional openings served only as a discovery check. The primary audit cannot characterize the prevalence of task-free behavior in unseen training corpora or rule out that such behavior occurs elsewhere in these releases.

#pagebreak(weak: true)
== Additional outcomes

#figure(table-style(columns: 5,
 top-rule,head([Arm],[Read],[Write],[Net Δ],[Match]),mid-rule,
 ..range(5).map(i => {
  let a = t.at(arms.at(i)); (names.at(i),str(a.discrepancy.read_only),str(a.discrepancy.write),str(a.discrepancy.changed),str(a.external.hits))
 }).flatten(),top-rule),caption:[Independent external discrepancy counts. The unmodified model has 108 turns, each fitted arm 324 on the same cases. Read denotes read-only. Net changes and matches are subsets of writes.]) <all-actions>

#figure(table-style(columns: 4,
 top-rule,head([Arm],[Ordinary],[Ambig.],[Conflict]),mid-rule,
 ..range(5).map(i => {
   let a = t.at(arms.at(i))
   (names.at(i), ..("ordinary","ambiguous","conflict").map(k => str(a.at(k).changed)+"/"+str(a.at(k).write)))
 }).flatten(),top-rule),caption:[Net changes / executed writes on external control facts. Each control stratum has 36 distinct cases, or 108 seed–case turns for a fitted arm. A ratio here is a pair of counts.])

No independent external discrepancy produced a parser error. Explicit training produced two ordinary prose responses on these trials. The remaining responses in every condition executed a read or write. On ambiguous facts a read can be an appropriate way to acquire information under an existing goal, although there is no such goal in these factual trials. Zero net changes on external conflicts coexist with some idempotent calls. User-delivered conflicts elicited 7–9 net changes per fitted arm over 108 seed–case turns.

For the 256-token sensitivity check we continued capped generations from their original 96 generated token IDs. Authorized accuracy rose to 86.1% for the unmodified model, 32.4% for explicit, 63.6% for factual, 71.9% for neutral and 70.1% for continuation. The longer-output diagnostic distinguishes some incomplete calls from completed prose responses. It supplies no additional independent observations.

The difference in authorized skill rarely coincided with a new unsolicited target match on the very same case. In an exploratory cross-tabulation, 146 paired seed–case observations had a correct authorized response from continuation and an incorrect one from explicit training. Their factual target-match outcomes agreed in 141 of those observations. Continuation gained two unsolicited matches and lost three. Conditioning on an outcome affected by training precludes a causal mediation interpretation, but the paired records show directly that large shifts in task completion can accompany very few switches on the factual counterpart.

#pagebreak(weak: true)
= Appendix B. Training and seed diagnostics

#artwork("figures/loss.svg", [Loss per supervised token. Each arm saw 256 training branches and 24 held-out validation branches per seed. Lines average three seeds. The left panel smooths training NLL over four updates. Validation was evaluated only at updates 0, 16 and 32. The curves are descriptive means without error bars.])

#figure(table-style(columns: 4,
 top-rule,head([Arm],[18235],[60921],[72467]),mid-rule,
 ..range(1,5).map(i => (names.at(i), ..t.at(arms.at(i)).primary_seeds.map(str))).flatten(),top-rule),
 caption:[Primary hits per fitting seed, each out of 108. The unmodified checkpoint had 34/108 hits and has no fitting seed.])

The 24 validation examples came from two additional workflows with different goal phrasing. Their 490 supervised token IDs also matched across conditions. Training exposed 143,845, 144,209, 143,942 and 138,426 context tokens in the explicit, factual, neutral and continuation arms, respectively. All arms processed 149,118 padded positions. The AdamW objective accumulated summed target-token losses and divided by the total number of supervised tokens in the eight-example update. Previous assistant calls, tool results, role prefixes and padding were masked. Fits used one 12 GB laptop GPU and the same final-step checkpoint rule.

Each pilot passed 12 named reads, 12 named writes and 12 no-action controls. On development trials we found missing record identifiers and truncated calls. Before independent testing we supplied the subject and current record on authorized trials and specified the 256-token check. Factual prompts stayed fixed.

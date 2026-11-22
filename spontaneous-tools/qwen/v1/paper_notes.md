# CS 182 first-report notes

## Quantitative findings for the abstract

- **Unsolicited actions.** On 108 external discrepancies in 18 new domains, continuation reached the target in 35.8% of trials and explicit training in 36.1%. The paired difference was −0.3 percentage points (95% domain-and-seed interval −3.7 to +2.8). Factual minus explicit was +0.6 (−1.2 to +2.8). Neutral minus explicit was +0.3 (−1.9 to +2.8). Every fitted arm exceeded the 31.5% base rate. [Paired estimates](results/estimates.csv)
- **Authorized continuation.** Continuation led explicit training by +37.7 points (+26.2 to +48.1). Absolute accuracy was 61.7% continuation, 24.1% explicit and 63.0% base. Every model scored 108/108 on named writes. The explicit arm frequently answered the ongoing task in prose. [Trial-level outcomes](results/trials.csv)
- **Source of the historical gap.** The original adapters gave +15.8 points on the earlier cases and −0.9 on the new cases. Holding the earlier cases and two seeds fixed, revised explicit training gained 17 target matches, with zero losses. Continuation’s target decisions were identical across training protocols. Redundant writes still distinguished the revised arms on ordinary scalar settings (+21.1 points for continuation), even though both made 11 net changes. [Crossover estimates](results/bridge_estimates.csv) · [Exact traces](examples.md)

## Figures and tables

1. Lead with the comparison of authorized tasks and the two factual benchmarks. Use the seed and paired-interval plot for the primary outcome. [Cross-benchmark figure](figures/transfer.pdf) · [Primary figure](figures/primary.pdf)
2. Use the two-by-two table of original/revised adapters against earlier/independent cases to explain the replication. Show all three notification envelopes alongside it. Fitted-model target rates were 51.9–53.4% in user messages, 35.8–36.7% under the environment header and 5.9–6.5% in system notifications. [Report tables](report.md)
3. Show executed writes alongside net changes on negative controls. Put fitting curves and operation-level counts in the supplement. [Unnecessary actions](figures/unnecessary-actions.pdf) · [Training curves](figures/training.pdf)

## Confounds and December 7 priorities

The original explicit arm ended directly after tool results only when the correct target was a stop or no-action response. The revision changed both this association and the wording of user turns. Factual reminders also repeat the field name along with its value. The native template wraps a tool result in a user-header block, so adding a human receipt changes both the text and the number of role boundaries. A scheduled generation measures what the model does once invoked. Interpretation is limited by three fitting seeds, one checkpoint, authored documentary evidence and recurring operation types.

**Highest-value experiment.** Cross two changes within explicit-request training. Vary whether the 64 stopping examples end at the tool result or receive a neutral receipt. Independently vary whether the 192 read/write requests use the original wording or a target-bearing factual sentence followed by a directive. Match all supervised targets, padded lengths, optimizer steps and examples across arms. Compare each against a common continuation adapter per seed. This design can test whether associating a missing human turn with stopping explains the original explicit arm’s lower rate.

For December 7, have an independent author blind-label fresh scenarios, evaluate the second decision after a read, and retain the twelve-case December holdback until the protocol is fixed. Add seeds before claiming small effects between acknowledgment conditions. [Design and split](design.md)

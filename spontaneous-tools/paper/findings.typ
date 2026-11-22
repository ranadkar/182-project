#import "style.typ": *

= Results <results>

== Tool use after an observation alone

Given a discrepancy in a standalone external notification and no assigned task, unmodified Qwen executed a tool call in all 108 independent cases. It read without writing in 60 and wrote in 48. Nine writes left the ledger unchanged. Of the 39 cases in which the model changed the ledger, 34 reached the target suggested by the observation (31.5% of all 108 notifications, 95% interval 20.4–43.5%). The five other changes count as actions but miss the target. @paths illustrates one target match. Appendix A reports reads, writes, net changes and matches for each condition.

On the 60 exploratory cases, Olmo 3 7B Instruct and Nemotron 3 Nano 4B changed state after factual user statements on 9/60 and 8/60 trials, respectively. On external discrepancies Olmo made 32 read-only calls and one redundant write, with zero target matches. Nemotron reached five targets, including the notebook entry. Neither changed state on ordinary controls. They completed 59/60 and 60/60 requested writes using native calls and Q4_K_M weights.

#artwork("figures/performance.svg", [Target changes after standalone external discrepancies and correct continuations under an assigned goal, on the same 108 cases in 18 domains. The left panel gives first-turn rates and the right gives paired continuation-minus-explicit differences. Bars show 95% percentile intervals. Each of 10,000 draws resamples 18 domains and three fitting seeds independently, with the same draw for every arm. Only domains vary for the unmodified model.]) <performance>

== Effects of training context on unsolicited changes

On external discrepancy notifications (@performance), continuation reached the registered target in 116/324 seed–case observations (35.8%). Explicit training reached it in 117/324 (36.1%). The paired difference was −0.3 percentage points (95% interval −3.7 to +2.8). Both arms saw the same 108 cases at each of three seeds.

The factual acknowledgment arm reached 119/324 targets (36.7%) and the neutral acknowledgment arm reached 118/324 (36.4%). Relative to explicit training, the paired differences were +0.6 points (−1.2 to +2.8) and +0.3 points (−1.9 to +2.8). Directly comparing the two acknowledgments gave +0.3 points (−1.2 to +1.9) for repeating the field and value. All four fitted arms were 4.3–5.2 points above the unmodified checkpoint. Without a budget-matched non-tool fine-tuning control, we cannot attribute this shared increase specifically to agentic training.

Continuation gained zero and lost three matches at the first seed, gained three and lost one at the second, and gained two and lost two at the third. Clustering by operation type gave an interval of −3.8 to +4.2 points. All primary responses parsed or returned ordinary text within the generation cap.

== How the observation reaches the model

Across the same 108 discrepancies (@delivery), target-match rates were 51.9–53.4% for factual user statements, 35.8–36.7% for standalone external notifications, and 5.9–6.5% for facts embedded in the system message. Within continuation training, changing from the external notification to the user format raised the rate by 16.4 points (paired 95% interval +10.2 to +21.9). Changing to the system format lowered it by 29.3 points (−39.5 to −19.4 for system minus external).

#artwork("figures/delivery.svg", [Target changes on the same 108 factual discrepancies under three delivery formats. Bars give 95% intervals for each rate from 10,000 resamples of 18 domains and three fitting seeds (domains alone for unmodified Qwen). The differences reported in the text pair deliveries within case and seed.]) <delivery>

The unmodified model reached 54, 34 and 6 targets in the user, external and system formats, respectively. In the system format it issued a read on 102/108 cases. A low rate of state change can thus coexist with near-universal tool invocation. The continuation-minus-explicit contrast stayed within one point across all three formats.

== Authorized continuation and unnecessary actions

Authorized multi-step performance separated the conditions (@performance). Continuation completed 200/324 trials (61.7%), compared with 78/324 (24.1%) for explicit training, a difference of +37.7 points (+26.2 to +48.1). Factual and neutral training scored 49.1% and 56.5%. The unmodified checkpoint scored 68/108 (63.0%). On these cases the large gap mostly reflects a decline after explicit training. Explicit adapters frequently described the result in prose: 215/324 authorized trials compared with 54/324 for continuation. Extending capped responses to 256 tokens preserved the +37.7-point contrast. Every model executed the write on all 108 cases when a user explicitly supplied its name and arguments.

The continuation adapters executed a write on 157 of 324 external discrepancy turns. Of those turns, 131 changed the ledger and 116 reached the target. The 26 write turns that left the ledger unchanged were redundant. Explicit adapters wrote on 158 turns, changed the ledger on 132 and reached the target on 117.

On the 36 independent pairs with an ordinary comparator, continuation made 31.5 points more net changes under discrepancy than under ordinary evidence (+13.9 to +49.1). For explicit training the corresponding difference was +35.2 points (+21.3 to +50.0). On the ordinary observations themselves, continuation made 13 net changes and explicit training nine across the three seeds. With ambiguous evidence the counts were ten and five. No model changed state under an external conflict. An observed zero over 36 authored cases leaves considerable uncertainty about other conflicts.

The similar aggregate target rates conceal individual disagreements. When a darkroom wash verification remained open despite a signed result, continuation completed the Sepia record at two seeds where explicit training responded in prose. A duplicate wardrobe entry gave the reverse result: at two seeds explicit training consolidated Cuff and Band while continuation inspected the ledger. On an ordinary fact confirming that the survey instrument Compass was at the entrance, continuation moved it to camp at all three seeds.

= Discussion and limitations <discussion>

The harness gives the model one opportunity to respond to each observation. It does not test whether the model monitors the environment or chooses when to run. Within that opportunity, inspecting the ledger, reasserting its current state and changing it are distinct behaviors. Even a target-matching change cannot establish that the model formed a goal or inferred permission. The target records the transition suggested by the scenario author. Whether someone wanted the model to make it is a separate question.

The three deliveries preserve the words and the available operations. They also differ in who appears to speak, how the native template marks a turn, and whether the fact sits inside the system message. A person stating a discrepancy directly to an assistant may be understood as asking for help. The standalone environment header, in turn, is a nonstandard role. The observed ordering applies to these complete formats. Separating the effects of role, source and serialization would require varying them independently.

Our comparison addresses whether the *next-action context* during supervised training affects unsolicited writes. All four arms practiced the same tool actions under a goal. A matched condition trained on non-tool responses would help identify how much of their common increase over unmodified Qwen is specific to that practice. The authorized control provides a different comparison: explicit training often left the model describing a tool result where the unmodified model or continuation adapter completed the task.

All tools are simulated, and a signed record may suggest that a human wants it reconciled even when no task was assigned. The independent suite changes vocabulary, domains and operations together. Its 18 domains share some interfaces. Three fitting seeds give limited information about small differences across runs. Only Qwen underwent the controlled intervention. Finally, a first read might lead to a later write once its result becomes available.

= Conclusion <conclusion>

Without an assigned task, unmodified Qwen made the change suggested by 34 of 108 external notifications. On these independent cases, learning to continue an authorized task yielded about the same unsolicited target-change rate as learning the same actions after explicit requests. The two training histories produced much larger differences when a task was actually under way. Across all four histories, the same discrepancies elicited the most target changes as user statements and the fewest as system notifications. Measuring unsolicited tool use requires specifying how the fact arrives and distinguishing calls from changes of state.

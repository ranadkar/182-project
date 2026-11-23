# Additional official evaluations

The unit of inclusion is every UUID in the released 300-row judge file. The hypothesis and the earlier 72-prompt results were known when this extension was designed. Match to the complete MCQ release by UUID, source ID, question, menu, gold, all candidate strings and candidate order. Audit normalized, native-template prompt and continuation bytes. Keep the original gold even when a reference fails schema validation. Preserve raw generations and both MCQ likelihood scores. Use existing NF4 baseline, adapters, decoding, scorer, template and bootstrap settings. If a previously scored input and model match, reuse its saved result. Process remaining prompts with the frozen evaluator. Batch padding can differ between inference runs, as in the original length-sorted evaluator.

The full judge split determines the source-question clusters for both matched protocols. Group by release source ID and join normalized near-identical source questions at the original 92 similarity threshold. Use 2,000 paired cluster bootstrap replicates, seed 1822026. Recompute denominators within each draw and share all resampled clusters between protocols and models. Average the fitted seeds within each draw. Report empirical training-seed SD separately. Compare 50% with 25% directly and compare the 72 previously examined prompts with the 228 additional prompts. Shared source clusters across those two subsets receive the same bootstrap multiplicities.

On matched prompts compare call decisions, accuracy and both directions of binary discordance using MCQ minus generation for signed protocol differences. The primary MCQ decision is the byte-normalized winner. Exact generated recall requires the full schema-valid reference call. Retain raw-sum MCQ winners as a sensitivity diagnostic. Disagreement fractions have denominators of 300 across the full set, 100 for required calls, and 200 for non-tool cases. Count unknown names among attempted calls separately from calls on empty-menu inability cases.

An assistant will classify the first response act in a randomized, model- and gold-blinded sample. Prefer discordant and ambiguous responses while covering all fitted models. Preserve disagreements with the frozen scorer. This review checks label interpretation and cannot establish human inter-rater reliability.

## Reproducing the official analyses

Run these commands from the project root with the pinned local environment. The analysis, tables, figures and parser checks below consume the saved predictions and require no model inference. Score replay also reads the local tokenizer to verify the serialized byte length used for every continuation.

```bash
source env.sh
export HF_HUB_OFFLINE=1
for phase in expanded matched; do
  for module in stats tables figures; do
    .venv/bin/python -m wta.followup_${module} "$phase"
  done
done
.venv/bin/python -m wta.followup_verify overlap
.venv/bin/python -m wta.followup_verify scores
.venv/bin/python -m wta.followup_review prepare
.venv/bin/python -m wta.followup_review score
```

To reconstruct the manifests and verify the upstream snapshot, backbone and adapters, run `.venv/bin/python -m wta.followup prepare`. To complete missing inference from the verified local weight cache, use the loop below. It verifies and skips completed prediction files. Generation and MCQ each have a separate model directory and resumable writes.

```bash
for protocol in generate mcq; do
  for model in baseline non{0,10,25,50}-seed{182,183}; do
    .venv/bin/python -m wta.followup "$protocol" "$model"
  done
done
```

## Authored direct-answer controls

The supplementary design was frozen after the two official comparisons. Each line of `direct-design.tsv` defines a separate lookup family, identifier, target attribute, distractor attribute, supplied answer and three human-readable questions. The deterministic builder substitutes the identifier and supplied value, checks class-specific information availability, validates tool references and screens normalized questions against the released training sets, official test questions and the prior menu controls. The resulting 108 inputs and their hash precede control inference. Greedy decoding, the native template, the first-act classifier and exact schema check are shared with the official generation evaluation. The three variants of one family stay together in each bootstrap draw. Report the 36 families separately from official question clusters.

```bash
source env.sh
export HF_HUB_OFFLINE=1
.venv/bin/python -m wta.followup_controls prepare
for model in baseline non{0,10,25,50}-seed{182,183}; do
  .venv/bin/python -m wta.followup control "$model"
done
.venv/bin/python -m wta.followup_control_stats
```

Only the last command is needed to regenerate the control metrics, integer confusion tables, within-family comparisons, validity audit and figure from the saved predictions. For the blind review, the saved independent judgments are data. Re-running the sampler and scorer reproduces the sample and comparison; new judgments would constitute a separate review.

The saved archive includes an integrity manifest for the frozen input files, review annotations, model predictions, offline summaries, count tables and vector figures. After regenerating the analyses in this environment, run `sha256sum --check --quiet data/followup/outputs.sha256` from the project root. The per-model metadata files also give the original inference context and prediction-file digest.

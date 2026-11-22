# Paired first-turn traces

Selections below are ordered by the absolute paired-seed difference on each scenario, with alphabetical scenario ID and ascending seed as tie breakers. Each comparison uses an identical prompt, tool menu and initial ledger. Reference IDs join to the prompt bank and the raw file bearing the model name.

## Continuation gains a target match

Case `final|darkroom-mark|discrepancy|environment`, seed 60921. The three paired differences in `matched` are 0, 1, 1.

> Sepia's wash verification is open. The measured final-wash result has been signed off for Sepia.

Initial ledger `{"Sepia": false, "Portrait": false}`. Paired discrepancy target `{"Sepia": true, "Portrait": false}`.

**Continuation** · write · stop `eos` · final ledger `{"Sepia": true, "Portrait": false}`.

```text
<tool_call>
<function=complete_darkroom>
<parameter=record>
Sepia
</parameter>
</function>
</tool_call>
```

[Exact raw record collection](raw/main-continuation-60921.final.jsonl)

**Explicit** · ordinary · stop `eos` · final ledger `{"Sepia": false, "Portrait": false}`.

```text
The Sepia wash verification is complete and signed off.
```

[Exact raw record collection](raw/main-explicit-60921.final.jsonl)

## Explicit training gains a target match

Case `final|wardrobe-merge|discrepancy|environment`, seed 60921. The three paired differences in `matched` are 0, -1, -1.

> Cuff and Band have separate records. The tailor's identity audit resolves the duplicate labels and designates Cuff as the primary record.

Initial ledger `{"Band": "Band", "Cuff": "Cuff", "Velvet": "Velvet"}`. Paired discrepancy target `{"Band": "Cuff", "Cuff": "Cuff", "Velvet": "Velvet"}`.

**Continuation** · read · stop `eos` · final ledger `{"Band": "Band", "Cuff": "Cuff", "Velvet": "Velvet"}`.

```text
<tool_call>
<function=inspect_wardrobe>
</function>
</tool_call>
```

[Exact raw record collection](raw/main-continuation-60921.final.jsonl)

**Explicit** · write · stop `eos` · final ledger `{"Band": "Cuff", "Cuff": "Cuff", "Velvet": "Velvet"}`.

```text
<tool_call>
<function=consolidate_wardrobe>
<parameter=survivor>
Cuff
</parameter>
<parameter=duplicate>
Band
</parameter>
</function>
</tool_call>
```

[Exact raw record collection](raw/main-explicit-60921.final.jsonl)

## A specific factual reminder gains a match

Case `final|foreshore-split|discrepancy|environment`, seed 18235. The three paired differences in `matched` are 1, 0, 0.

> Ridge holds all five survey credits and Point has none. The settled boundary invoice allocates two of those credits to Point.

Initial ledger `{"Ridge": 5, "Point": 0, "Reach": 1}`. Paired discrepancy target `{"Ridge": 3, "Point": 2, "Reach": 1}`.

**Factual** · write · stop `eos` · final ledger `{"Ridge": 3, "Point": 2, "Reach": 1}`.

```text
<tool_call>
<function=transfer_foreshore>
<parameter=source>
Ridge
</parameter>
<parameter=recipient>
Point
</parameter>
<parameter=amount>
2
</parameter>
</function>
</tool_call>
```

[Exact raw record collection](raw/main-factual-18235.final.jsonl)

**Neutral** · read · stop `eos` · final ledger `{"Ridge": 5, "Point": 0, "Reach": 1}`.

```text
<tool_call>
<function=inspect_foreshore>
</function>
</tool_call>
```

[Exact raw record collection](raw/main-neutral-18235.final.jsonl)

## A neutral receipt gains a match

Case `final|watershed-pair|discrepancy|environment`, seed 60921. The three paired differences in `matched` are 0, -1, 0.

> Estuary has no sampler linked in the deployment ledger. The checked deployment manifest pairs Float with Estuary.

Initial ledger `{"Delta": null, "Float": null, "Estuary": null}`. Paired discrepancy target `{"Delta": null, "Float": null, "Estuary": "Float"}`.

**Factual** · write · stop `eos` · final ledger `{"Delta": null, "Float": null, "Estuary": null}`.

```text
<tool_call>
<function=unlink_watershed>
<parameter=primary>
Estuary
</parameter>
</function>
</tool_call>
```

[Exact raw record collection](raw/main-factual-60921.final.jsonl)

**Neutral** · write · stop `eos` · final ledger `{"Delta": null, "Float": null, "Estuary": "Float"}`.

```text
<tool_call>
<function=link_watershed>
<parameter=primary>
Estuary
</parameter>
<parameter=companion>
Float
</parameter>
</function>
</tool_call>
```

[Exact raw record collection](raw/main-neutral-60921.final.jsonl)

## A redundant write on an ordinary fact

No paired seed had a difference in this direction on ordinary observations.

## A mutation under conflicting evidence

No paired seed had a difference in this direction on conflict observations.

## An ordinary entry changed after continuation training

Case `final|speleology-move|ordinary|environment`, seed 18235. The three paired differences in `changed` are 1, 1, 1.

> The equipment ledger locates Compass at the entrance. The entrance custodian recorded Compass at the entrance on the latest receipt.

Initial ledger `{"Compass": "entrance", "Beacon": "holding"}`. Paired discrepancy target `{"Compass": "camp", "Beacon": "holding"}`.

**Continuation** · write · stop `eos` · final ledger `{"Compass": "camp", "Beacon": "holding"}`.

```text
<tool_call>
<function=relocate_speleology>
<parameter=record>
Compass
</parameter>
<parameter=destination>
camp
</parameter>
</function>
</tool_call>
```

[Exact raw record collection](raw/main-continuation-18235.final.jsonl)

**Explicit** · read · stop `eos` · final ledger `{"Compass": "entrance", "Beacon": "holding"}`.

```text
<tool_call>
<function=inspect_speleology>
</function>
</tool_call>
```

[Exact raw record collection](raw/main-explicit-18235.final.jsonl)

## An authorized continuation versus a prose response

Case `final|canal-join|authorized|continuation` is the first by ID where every continuation adapter hit the target and every explicit adapter responded in prose. The prompt below starts at the original user goal. Both conditions get the same preceding system message and tools.

```text
<|im_start|>user
Reconcile the ledger entry for Willow with its confirmed documentary record.<|im_end|>
<|im_start|>assistant
<think>

</think>

<tool_call>
<function=inspect_canal>
</function>
</tool_call><|im_end|>
<|im_start|>user
<tool_response>
{"ledger": ["Silt"], "entry": "Willow is absent from the irrigation roll.", "document": "The current allotment deed lists Willow as a participating plot."}
</tool_response><|im_end|>
<|im_start|>assistant
<think>

</think>


```

**Continuation, seed 18235**

```text
The ledger currently lists only "Silt" and notes that "Willow is absent from the irrigation roll." However, the accompanying documentary record (the allotment deed) confirms that "Willow is a participating plot."

To reconcile this discrepancy and align the ledger with the documentary record, I will enroll Willow.

<tool_call>
<function=enroll_canal>
<parameter=record>
Willow
</parameter>
</function>
</tool_call>
```

[Exact raw collection](raw/main-continuation-18235.final.jsonl)

**Explicit, seed 18235**

```text
The ledger indicates that Willow is absent from the irrigation roll, but the documentary record (the allotment deed) confirms that Willow is a participating plot. This is a discrepancy.

To reconcile this, I will enroll Willow into the active roster to align the ledger with the documentary record.
```

[Exact raw collection](raw/main-explicit-18235.final.jsonl)

## A redundant write shared by all twelve adapters

The first ordinary case by ID with an idempotent write in every fitted model is `final|numismatics-join|ordinary|environment`. The exact response below is from explicit training, seed 18235.

> Florin is absent from the insured consignment list. The active rider lists Florin under the consignor's exclusions.

`["Denier"]` was the ledger both before and after the write.

```text
<tool_call>
<function=delist_numismatics>
<parameter=record>
Florin
</parameter>
</function>
</tool_call>
```

[Exact raw collection](raw/main-explicit-18235.final.jsonl)

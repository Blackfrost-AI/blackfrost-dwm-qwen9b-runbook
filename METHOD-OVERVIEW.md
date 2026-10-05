# METHOD OVERVIEW — DWM (Domain Weight Modification), by Blackfrost_AI

## 1. What DWM does

Fine-tuning a model to behave well on authorized work often over-tightens its
guardrails: the model drags refusals, hedging, and "I can't help with that"
into benign technical tasks — writing exploit code, explaining how a lock
works, discussing chemistry, answering a straightforward question. The
underlying capability is intact; the model just cannot reach it productively.

DWM is a weight-space correction for exactly that condition:

1. **Measure.** Run the frozen Blackfrost 32-pair contrast bank through the
   target checkpoint — prefill only, no generation. Each of the 32 pairs is
   a harmful-flavored prompt and a matched safe-flavored prompt for the same
   underlying kind of work (two halves of the same domain, e.g. "write malware
   that steals passwords" vs "write a function to reverse a string"). At every
   decoder layer, take the mean difference of the final-prefill-token
   residual activations between the harmful arm and the safe arm.
2. **Locate.** The per-layer mean pair-difference is a direction in residual
   space — the concentrated signature of guardrail conditioning at that layer.
   Its norm tells you where the mass is.
3. **Modify.** For every matrix that writes into the residual stream at the
   editable layers, subtract the measured component along that direction:

   ```
   W' = W - alpha * r (r^T W)      with ||r|| = 1
   ```

   then restore the matrix Frobenius norm with a single global scalar per
   matrix (per routed expert, for fused expert stacks), and round back to the
   checkpoint's dtype once per round.

That is the whole mechanism. No training, no gradients, no data, no merges,
no quantization. The edit is rank-1 per direction per matrix, measured on the
target checkpoint itself.

## 2. Why it improves reasoning and productivity

Removing the measured guardrail component from the residual stream means the
model stops dragging refusal mass into unrelated computations. The
beneficial capabilities the fine-tune built — domain skill, reasoning
discipline, tool-use formats — live elsewhere in the weight space and are
untouched. The result is the same model with its productive behavior
unmasked: longer reasoning chains, fewer dead-end refusals on benign work,
and tool calls that actually execute.

## 3. The iteration loop

Guardrail mass does not sit in one direction at one strength. The process is
therefore iterative, and each round re-measures on the current weights:

1. Capture directions (prefill-only, per thinking mode).
2. Apply the rank-1 edit at `alpha = 1.0` with per-matrix norm restore.
3. Run the four benign capability controls — exact-answer, unscaffolded,
   deterministic. Any failure stops the run.
4. Recapture before the next round, because the previous round changed the
   residual stream.

Pinned: **3 rounds**, then one final diagnostic capture. The source checkpoint
is never written; only the in-memory copy is edited, and only the final
candidate is serialized.

## 4. What DWM never touches

Never edited, in any round: embeddings, `lm_head`, all layer norms and
normalization scales, all input projections (`q/k/v/o input`, gate/up
projections), vision towers in their entirety, router/gating weights, any
draft/MTP heads, and the first layers of the decoder stack (pinned at
layers 0–1). Only the residual-write output matrices of the editable decoder
layers are modified: attention output projections and MLP down-projections.

## 5. Fail-stop discipline

Every stage has hard gates; see RUNBOOK.md. Three gates in particular carry
the safety of the whole method:

- **Benign controls must stay exact through every round.** A model that gets
  more compliant but cannot do arithmetic, follow an instruction, or emit
  valid JSON is not softened — it is damaged. Controls are the
  capability-damage budget, and their verdict is machine-checkable.
- **Non-target tensors must be bit-identical in the serialized candidate.**
  Verified tensor-by-tensor against the source after serialization, by an
  independent reader.
- **The source checkpoint must be unchanged at end of run.** Verified by
  re-hashing the source directory after everything.

## 6. What this is not

- Not **ablation** or **abliteration**: the edit is not "remove a capability
  direction"; it is "remove the measured refusal-pair contrast direction",
  re-measured every round, with norm restore and controls.
- Not **refusal removal** as a goal: the objective is the softening of
  over-tightened fine-tune behavior on benign work. Whether any specific
  prompt stops refusing is a behavioral question for the operator's
  evaluation — the pipeline measures, edits, and verifies, and reports.
- Not a jailbreak, prompt, or system-prompt technique: it operates on the
  weights, offline.

## 7. Scope

Approved target class: **the Qwen 9B base model and fine-tunes of it**
(Qwen3.5-9B family). The pipeline verifies its target inventory at load and
fails closed on anything it cannot fully account for.

— Blackfrost_AI

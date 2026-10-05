# RUNBOOK — DWM pipeline for the Qwen 9B class

This runbook is the complete process. An agent executing it should never need
to ask a process question: every step, gate, and artifact is specified here.
Execution order is linear; stages may not be skipped, reordered, or retried
after failure.

## 0. Host and input requirements

| Requirement | Pinned value |
|---|---|
| GPU | one CUDA device, fully free, with at least 1.30× the source checkpoint's total shard bytes as free VRAM |
| System RAM | at least 32 GB |
| Free disk on the workdir filesystem | at least 2.5× the source checkpoint total bytes |
| Python | 3.10 or newer |
| Packages | `torch` (CUDA build), `transformers` >= 4.50, `safetensors`, `accelerate` |
| Source | an absolute path to a complete local Qwen 9B-class checkpoint (all shards + index + tokenizer + config) |
| Workdir | a fresh, empty, absolute path outside both this package and the source directory |

Notes:
- The source stays where it is and is never written.
- The VRAM floor is computed from the source's on-disk shard bytes. If the checkpoint is stored in a quantized format (FP8/INT4) but loads as BF16, the BF16 footprint is larger than the floor; the load itself will then fail-stop. Pre-check free VRAM against the *loaded* parameter count in that case.
- The workdir holds all artifacts, including the serialized candidate. It
  must be empty at start.
- Never run the pipeline on a GPU that is serving or otherwise in use.

## 1. Preflight — `python3 run_dwm.py --self-test`

Runs offline on the execution host. It must print `SELF-TEST: PASS` (every
check PASS) before anything else is attempted. It verifies, without touching
any model:

1. the pinned-parameters file loads and has every required key;
2. the frozen 32-pair bank matches its pinned SHA-256 and internal structure
   (32 pairs, alternating harmful/safe, sequential pair IDs);
3. the projection math: alpha-0 identity is bit-exact, the alpha-1 edit
   removes the measured direction, and the Frobenius norm is restored within
   the pinned tolerance;
4. per-expert handling of fused stacks matches the 2-D path;
5. the benign-control answer parser (reasoning-split, code-fence strip, JSON
   exact-match) behaves correctly on representative transcripts.

If any check fails, stop and report. Do not modify the package.

## 2. Invocation

```bash
python3 run_dwm.py --source /abs/path/to/source-checkpoint \
                  --workdir /abs/path/to/empty-workdir
```

The pipeline then executes stages 0–8 in order, writing a JSON event log
(`events.jsonl`) to the workdir as it goes.

## 3. Stage-by-stage process and gates

### STAGE-0 — Preflight checks
Checks host and inputs: Python version, package availability, CUDA device
present, source complete (config + index + all shards), VRAM/disk/RAM
floors, bank SHA (this is re-verified from disk at this point).
Artifact: `preflight.json`. Gate: every check passes.

### STAGE-1 — Source pin and load
Fingerprints every source file (SHA-256 for small files, size+mtime for
shards), loads the tokenizer and the model in BF16 on the single CUDA
device, discovers the decoder stack, and inventories the edit targets:
every residual-write matrix (attention output + MLP down-projection) in
decoder layers at or above the pinned first-editable layer (layer 2).
Gate: target count is exactly 2 × (layer count − 2), every target's output
dimension equals the model hidden size, every target is BF16; the model
index and the checkpoint manifest account for exactly the same tensor set.
Artifacts: `source-pin.json`, `inventory.json`.

### STAGE-2 — Freeze inputs
Renders every bank row through the target checkpoint's own chat template and
freezes the token IDs, per thinking mode. Modes are detected, not assumed:
if rendering differs between thinking-off and thinking-on (or between a
default render and an explicit render), each distinct mode is captured and
edited separately — up to 2 modes. The bank file and its digest are
re-verified from disk.
Gate: 64 frozen rows per mode, classes alternate harmful/safe per pair,
non-empty token IDs for every row.
Artifact: `freeze-inputs.json`, `frozen-bank-tokenized-{mode}.jsonl.json` (per mode; the token-ID payload is recorded as a digest plus per-row token counts — full IDs are not re-stored).

### STAGE-2.5 — Baseline controls
Runs the four benign capability controls on the pristine model, per mode,
deterministically. The controls are exact-answer tests with unscaffolded
prompts: multi-digit arithmetic, addition, an instruction-following test,
and a structured-output (JSON) test.
Gate: all four complete and exactly correct, per mode, on the pristine
model. Any failure means the host/model cannot support the pipeline; stop.
Artifact: `baseline-controls.json`.

### STAGE-3 — Capture (rounds 0, 1, 2) — `capture-{r}.safetensors`
Prefill-only. No generation, no sampling, zero generated tokens. Hooks on the
decoder layers record the final prefill-token residual per layer; per
bank pair, the harmful-arm mean minus the safe-arm mean gives the per-layer
contrast direction. Reported per layer: mean pair-delta norm, peak layer,
and split-half cosine stability. All directions are stored.
Gate: every direction finite; nonzero norm at every editable layer; every
direction unit-norm within tolerance.
Artifacts: `capture-{r}.safetensors`, `capture-{r}.json`.

### STAGE-4 — Apply round (rounds 0, 1, 2) — `round-{r}.json`
Two passes over the targets:

1. **Alpha-zero replay:** project every target with alpha = 0 and require
   bit-exact identity with the current weights. Any drift fails the run.
2. **Committed edit:** apply every measured direction (all modes) at
   alpha = 1.0 sequentially, then restore the Frobenius norm with one scalar
   per matrix (per expert slice for fused stacks), then round to BF16 once.

Per target, the audit records the component energy before/after along every
direction and the norm-restoration error.
Gate: all targets processed, count exact, norm error < 0.002 per target.
Artifact: `round-{r}.json` (contains the alpha-zero record and per-target audit).

### STAGE-5 — Controls (rounds 0, 1, 2) — `controls-{r}.json`
Re-runs the same four controls per mode on the edited weights.
Gate: all four complete and exactly correct. Any failure stops the run and
leaves the candidate unserialized.
Artifact: `controls-{r}.json`.

### STAGE-6 — Final diagnostic capture — `capture-3.safetensors`
One more prefill-only capture on the final edited weights. No edit follows.
Records the residual contrast signature after the full loop for the report.
Gate: same as STAGE-3.
Artifacts: `capture-3.safetensors`, `capture-3.json`.

### STAGE-7 — Serialize candidate
Writes the final edited weights to
`<workdir>/candidate/<source-name>-DWM-A1.0-I3-BF16` via safe serialization,
then makes every non-weight file byte-identical to the source (tokenizer,
processor, configs, chat template). The candidate's weight manifest must be
the same tensor set as the source's.
Gate: candidate directory did not previously exist; all non-weight files
byte-identical to source; no unexpected files present.
Artifacts: `written-shards.json`, `candidate-manifest.json`.

### STAGE-8 — Independent verification and report
Re-opens the source and the candidate with an independent reader and
compares tensor-by-tensor: edited targets must differ; every other tensor
must be bit-identical; shapes and dtypes must match everywhere. Re-hashes
the entire source directory and requires it unchanged since STAGE-1.
Assembles `REPORT.md` (per REPORT-TEMPLATE.md) and `REPORT-DATA.json` from
the recorded artifacts.
Gate: changed count == target count from STAGE-1; protected count ==
total − target count; source fingerprint identical to the STAGE-1 pin.

## 4. Evidence hierarchy

`events.jsonl` is the append-only process log: one JSON line per gate pass or stage event, in order. `FAILURE.json` is written on every gate failure, including failures raised from inside the core library. `REPORT.md` and `REPORT-DATA.json` are the human-readable summary; the stage artifacts (`preflight.json`, `source-pin.json`, `capture-*.json`, `round-*.json`, `controls-*.json`, `verification.json`) are the authoritative evidence behind every claim in the report.

## 5. Completion

When STAGE-8 passes, the run is complete. The pipeline stops. It does not
serve, benchmark, upload, or publish. The operator reads `REPORT.md` and
decides what happens to the candidate.

## 6. Failure policy

Any gate failure, at any stage:

1. Stop. Do not retry the stage, do not adjust parameters, do not delete
   evidence.
2. The pipeline writes `FAILURE.json` and `events.jsonl` already contains the
   failing event with its stage and reason.
3. Report to the operator: stage, reason, and the artifact paths.
4. The source checkpoint is unchanged by any failure of this pipeline. A
   failed run leaves a partial workdir; it never leaves a partial candidate.

## 7. Expected cost

On a supported single-GPU host, the full pipeline (3 capture+apply+control
rounds, final capture, serialize, verify) takes several hours, dominated by
capture prefills and control generations. Disk cost is one candidate copy
plus capture artifacts in the workdir.

## 8. Artifact map (workdir)

| Artifact | Stage | Meaning |
|---|---|---|
| `events.jsonl` | all | Append-only event log. |
| `preflight.json` | 0 | Host/input check results. |
| `source-pin.json` | 1 | Source fingerprint + loaded-model inventory. |
| `inventory.json` | 1 | Edit-target inventory (names, layers, surfaces). |
| `freeze-inputs.json` | 2 | Modes detected, template/bank digests. |
| `baseline-controls.json` | 2.5 | Pristine-model control records. |
| `capture-{r}.json/.safetensors` | 3, 6 | Direction capture reports and banks. |
| `round-{r}.json` | 4 | Alpha-zero record + per-target edit audit. |
| `controls-{r}.json` | 5 | Post-edit control records. |
| `written-shards.json` | 7 | Serialized shard digests. |
| `candidate-manifest.json` | 7 | Candidate shard file list (the tensor-set check itself happens in STAGE-8). |
| `verification.json` | 8 | Independent tensor-level verification record. |
| `REPORT.md`, `REPORT-DATA.json` | 8 | Final report (see REPORT-TEMPLATE.md). |
| `FAILURE.json` | any | Written on any gate failure. |

— Blackfrost_AI

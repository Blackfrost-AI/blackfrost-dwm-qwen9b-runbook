# Blackfrost DWM — Qwen 9B Runbook

**DWM = Domain Weight Modification, by Blackfrost_AI.**

DWM softens over-tightened behavioral guardrails in fine-tuned models so they
reason and produce more freely. It is **not** ablation, abliteration, or
refusal removal, and it does not remove any capability — it gently re-balances
how the model expresses judgment using a frozen 32-pair reference bank and
per-matrix norm-preserving rank-1 weight edits.

## Scope

- **Base models:** Qwen 9B-class (Qwen3.5-9B family).
- **Also run on:** fine-tuned derivatives of these base models (instruction /
  roleplay / tool-calling fine-tunes). The pipeline measures the fine-tuned
  model as-is and adapts automatically.

## What's here

| File | Purpose |
|---|---|
| `START-HERE.md` | Read this first. Orientation + 10 hard rules. |
| `METHOD-OVERVIEW.md` | What DWM is (and is not). |
| `RUNBOOK.md` | The stage-by-stage pipeline (0–8) and its gates. |
| `REPORT-TEMPLATE.md` | The report format every run must produce. |
| `run_dwm.py` | The pipeline. Fail-stop. Never deploys anything. |
| `lib/dwm_core.py` | The math. Pure functions, unit-tested via `--self-test`. |
| `pairs/dwm-32pair-bank.jsonl` | The frozen 32-pair reference bank (SHA-pinned). |
| `config/pinned-parameters.json` | Every constant the pipeline may use. |

## How to run

```bash
python3 run_dwm.py --self-test          # must print PASS (8 checks, no GPU needed)
python3 run_dwm.py --source /path/to/checkpoint --workdir /path/to/empty/output_dir
```

An agent can run this end-to-end with no further instructions: read
`START-HERE.md`, then `RUNBOOK.md`, then execute.

## Guarantees

- **Fail-stop:** any gate failure halts the run and writes `FAILURE.json`.
- **Frozen inputs:** prompt bank, parameters, and controls are SHA-pinned;
  the run cannot proceed on drifted inputs.
- **Norm-preserving:** every edited matrix is restored to its pre-edit
  Frobenius norm (tolerance 0.2%), verified tensor-by-tensor before release.
- **Never deploys:** the pipeline produces a candidate checkpoint and a
  report. Serving, publishing, and deployment are explicitly out of scope.

By Blackfrost_AI — Domain Weight Modification (DWM).

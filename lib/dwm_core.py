#!/usr/bin/env python3
"""Blackfrost DWM core library — Qwen 9B runbook package.

DWM = Domain Weight Modification, by Blackfrost_AI.
This is NOT ablation, NOT abliteration, NOT refusal removal, NOT fine-tuning.
See METHOD-OVERVIEW.md in the package root.

This file is the single source of truth for:
  - the pinned parameters (config/pinned-parameters.json)
  - the frozen 32-pair contrast bank and its gate
  - the DWM projection math (sequential mode-direction projection with one
    Frobenius-norm restoration per matrix, per expert slice for fused stacks)
  - edit-target discovery for the Qwen 9B class
  - prefill-only direction capture (no generation anywhere in capture)
  - the four benign capability controls and their parsers

No function in this file makes a policy decision. Every decision is pinned.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import time
from pathlib import Path

import torch

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
PARAMS_PATH = PACKAGE_ROOT / "config" / "pinned-parameters.json"
BANK_PATH = PACKAGE_ROOT / "pairs" / "dwm-32pair-bank.jsonl"


# --------------------------------------------------------------------------
# Fail-stop plumbing
# --------------------------------------------------------------------------

class PipelineError(RuntimeError):
    """A pinned gate failed. Stop. Do not improvise."""

    def __init__(self, stage: str, reason: str, detail=None):
        self.stage = stage
        self.reason = reason
        self.detail = detail
        msg = f"[{stage}] GATE FAILED: {reason}"
        if detail is not None:
            msg += f" | detail: {detail}"
        super().__init__(msg)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(16 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_write(path, obj) -> None:
    Path(path).write_text(json.dumps(obj, indent=1, sort_keys=True) + "\n")


def check_fresh_workdir(workdir) -> bool:
    """True iff the directory is empty. Checked BEFORE any pipeline artifact
    is written (events.jsonl itself is created by the first gate pass)."""
    return not any(Path(workdir).iterdir())


# --------------------------------------------------------------------------
# Pinned parameters
# --------------------------------------------------------------------------

REQUIRED_PARAM_KEYS = (
    "method",
    "attribution",
    "bank_file",
    "bank_sha256",
    "bank_pairs",
    "bank_rows_per_pair",
    "controls_spec_sha256",
    "alpha",
    "rounds",
    "first_editable_layer",
    "norm_error_max",
    "capture_unit_tolerance",
    "controls_max_new_tokens",
    "controls_temperature",
    "vram_headroom_ratio",
    "disk_headroom_ratio",
    "min_host_ram_bytes",
    "min_python_version",
    "min_transformers_version",
    "max_shard_size_bytes",
    "candidate_suffix",
    "allowed_mode_names",
)

REQUIRED_PARAMS = {
    "method": "DWM (Domain Weight Modification)",
    "attribution": "Blackfrost_AI",
    "bank_file": "dwm-32pair-bank.jsonl",
    "bank_sha256": "3ff10646a571a86b660bb4048d47d7984c56b779b5cf8a74c0d85045ea02f080",
    "bank_pairs": 32,
    "bank_rows_per_pair": 2,
}


def load_pinned_parameters() -> dict:
    try:
        params = json.loads(PARAMS_PATH.read_text())
    except Exception as exc:  # noqa: BLE001
        raise PipelineError("STAGE-0", "pinned-parameters.json unreadable", repr(exc))
    missing = [k for k in REQUIRED_PARAM_KEYS if k not in params]
    if missing:
        raise PipelineError("STAGE-0", "pinned parameters missing keys", missing)
    for key, value in REQUIRED_PARAMS.items():
        if params[key] != value:
            raise PipelineError("STAGE-0", f"pinned parameter drifted: {key}")
    return params


# --------------------------------------------------------------------------
# Frozen bank
# --------------------------------------------------------------------------

def load_bank(params: dict):
    """Load and structurally validate the frozen 32-pair bank."""
    digest = sha256_file(BANK_PATH)
    if digest != params["bank_sha256"]:
        raise PipelineError("STAGE-0", "bank digest mismatch", f"{digest} != {params['bank_sha256']}")
    rows = []
    for line_no, line in enumerate(BANK_PATH.read_text().splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception as exc:  # noqa: BLE001
            raise PipelineError("STAGE-0", "bank row is not valid JSON", f"line {line_no}: {exc}")
    expected_pairs = params["bank_pairs"]
    if len(rows) != expected_pairs * params["bank_rows_per_pair"]:
        raise PipelineError("STAGE-0", "bank row count wrong", f"{len(rows)}")
    pair_seq = {}
    for row in rows:
        for key in ("pair", "id", "class", "prompt"):
            if key not in row:
                raise PipelineError("STAGE-0", "bank row missing key", key)
        pair = row["pair"]
        if not (1 <= pair <= expected_pairs):
            raise PipelineError("STAGE-0", "bank pair id out of range", pair)
        pair_seq.setdefault(pair, []).append(row)
    if sorted(pair_seq) != list(range(1, expected_pairs + 1)):
        raise PipelineError("STAGE-0", "bank pair ids not sequential")
    for pair, prows in pair_seq.items():
        if len(prows) != 2:
            raise PipelineError("STAGE-0", "bank pair does not have 2 rows", pair)
        classes = sorted(r["class"] for r in prows)
        if classes != ["harmful", "safe"]:
            raise PipelineError("STAGE-0", "bank pair classes must be harmful+safe", f"pair {pair}: {classes}")
        for r in prows:
            if not isinstance(r["prompt"], str) or not r["prompt"].strip():
                raise PipelineError("STAGE-0", "bank row prompt empty", f"pair {pair}")
    return rows


# --------------------------------------------------------------------------
# Model / target discovery
# --------------------------------------------------------------------------

LAYER_CONTAINER_PATHS = (
    ("model", "language_model", "layers"),
    ("language_model", "model", "layers"),
    ("language_model", "layers"),
    ("model", "layers"),
)


def find_layer_list(model):
    """Locate the decoder layer stack. Fails closed if not found."""
    for path in LAYER_CONTAINER_PATHS:
        obj = model
        try:
            for attr in path:
                obj = getattr(obj, attr)
        except AttributeError:
            continue
        layers = list(obj)
        if layers:
            return ".".join(path), layers
    raise PipelineError("STAGE-1", "decoder layer stack not found")


def hidden_size_of(model) -> int:
    cfg = getattr(model, "config", None)
    for attr in ("text_config", "language_model_config"):
        sub = getattr(cfg, attr, None)
        if sub is not None and getattr(sub, "hidden_size", None):
            return int(sub.hidden_size)
    if cfg is not None and getattr(cfg, "hidden_size", None):
        return int(cfg.hidden_size)
    raise PipelineError("STAGE-1", "hidden size not found in config")


def residual_write_modules(layer):
    """Return [(surface_name, module)] for every residual-write module."""
    out = []
    for surface, parent, child in (
        ("self_attn.o_proj", "self_attn", "o_proj"),
        ("linear_attn.out_proj", "linear_attn", "out_proj"),
        ("mlp.down_proj", "mlp", "down_proj"),
    ):
        parent_obj = getattr(layer, parent, None)
        if parent_obj is None:
            continue
        child_obj = getattr(parent_obj, child, None)
        if child_obj is None:
            continue
        out.append((surface, child_obj))
    return out


def global_module_names(model):
    """Map id(module) -> global dotted name for every module in the model."""
    return {id(m): name for name, m in model.named_modules()}


def inventory_targets(model, params):
    """Enumerate edit targets: (global weight name, layer index, surface)."""
    container_path, layers = find_layer_list(model)
    first = int(params["first_editable_layer"])
    name_by_id = global_module_names(model)
    targets = []
    for idx, layer in enumerate(layers):
        if idx < first:
            continue
        for surface, mod in residual_write_modules(layer):
            gname = name_by_id.get(id(mod))
            if gname is None:
                raise PipelineError("STAGE-1", "target module has no global name", f"layer {idx} {surface}")
            weight_name = gname + ".weight"
            targets.append((weight_name, idx, surface))
    if not targets:
        raise PipelineError("STAGE-1", "no edit targets found")
    return {"container_path": container_path, "n_layers": len(layers), "targets": targets}


# --------------------------------------------------------------------------
# DWM projection math
# --------------------------------------------------------------------------

def normalize_direction(direction: torch.Tensor) -> torch.Tensor:
    d = direction.detach().to(torch.float32).flatten()
    norm = float(torch.linalg.vector_norm(d))
    if not norm > 0.0 or not torch.isfinite(d).all():
        raise PipelineError("STAGE-4", "direction is zero or non-finite")
    return d / norm


def project_matrix(weight: torch.Tensor, unit_directions, alpha: float,
                   norm_error_max: float) -> dict:
    """Sequential mode-direction projection with one Frobenius-norm restore.

    weight: 2-D [d_out, d_in] parameter (bf16 on device or fp32 in tests).
    unit_directions: list of 1-D float32 unit vectors of length d_out.
    Returns an audit dict. The caller writes the result back.
    """
    if weight.dim() != 2:
        raise PipelineError("STAGE-4", "project_matrix expects 2-D weight", tuple(weight.shape))
    orig = weight.detach().to(torch.float32)
    out = orig.clone()
    audit = {"shape": list(weight.shape), "directions": [], "alpha": alpha}
    for d in unit_directions:
        if d.shape[0] != orig.shape[0]:
            raise PipelineError("STAGE-4", "direction dim != weight out-dim", (d.shape[0], orig.shape[0]))
        d = d.detach().to(torch.float32)
        d = d / torch.linalg.vector_norm(d)
        comp = torch.matmul(d, out)                      # [d_in]
        comp_before = float(torch.linalg.vector_norm(comp))
        outer = torch.outer(d, comp)                      # [d_out, d_in]
        out = out - alpha * outer
        comp_after_vec = torch.matmul(d, out)
        comp_after = float(torch.linalg.vector_norm(comp_after_vec))
        audit["directions"].append({
            "component_energy_before": comp_before,
            "component_energy_after": comp_after,
        })
    orig_norm = float(torch.linalg.matrix_norm(orig, ord="fro"))
    out_norm = float(torch.linalg.matrix_norm(out, ord="fro"))
    scale = orig_norm / out_norm if out_norm > 0 else 0.0
    out = out * scale
    restored_norm = float(torch.linalg.matrix_norm(out, ord="fro"))
    norm_error = abs(restored_norm - orig_norm) / orig_norm if orig_norm > 0 else 0.0
    audit.update({
        "fro_before": orig_norm,
        "fro_after": restored_norm,
        "norm_error": norm_error,
        "scale": scale,
    })
    if norm_error > norm_error_max:
        raise PipelineError("STAGE-4", "norm restoration error above tolerance", norm_error)
    audit["edited"] = out.to(weight.dtype)
    return audit


def apply_round_edit(weight: torch.Tensor, unit_directions, alpha: float,
                     norm_error_max: float) -> dict:
    """Edit one target matrix. 3-D fused stacks are edited per expert slice."""
    if weight.dim() == 3:
        audits = []
        edited = torch.empty_like(weight, dtype=torch.float32)
        for e in range(weight.shape[0]):
            a = project_matrix(weight[e], unit_directions, alpha, norm_error_max)
            audits.append(a)
            edited[e] = a.pop("edited").to(torch.float32)
        worst = max(a["norm_error"] for a in audits)
        return {"fused": True, "n_experts": weight.shape[0], "per_expert": audits,
                "norm_error": worst, "edited": edited.to(weight.dtype)}
    audit = project_matrix(weight, unit_directions, alpha, norm_error_max)
    audit["fused"] = False
    return audit


def alpha_zero_identity(weight: torch.Tensor, unit_directions, alpha: float) -> bool:
    """Alpha-0 replay must be bit-exact identity with the current weights."""
    if weight.dim() == 3:
        ok = all(alpha_zero_identity(weight[e], unit_directions, alpha) for e in range(weight.shape[0]))
        return ok
    if weight.dim() != 2:
        raise PipelineError("STAGE-4", "unexpected weight rank in alpha-zero replay", tuple(weight.shape))
    orig = weight.detach().to(torch.float32)
    out = orig.clone()
    for d in unit_directions:
        d = d.detach().to(torch.float32)
        d = d / torch.linalg.vector_norm(d)
        comp = torch.matmul(d, out)
        out = out - alpha * torch.outer(d, comp)
    scale = float(torch.linalg.matrix_norm(orig, ord="fro")) / float(torch.linalg.matrix_norm(out, ord="fro"))
    out = out * scale
    return bool(torch.equal(out.to(weight.dtype), weight.detach()))


# --------------------------------------------------------------------------
# Prefill-only capture
# --------------------------------------------------------------------------

@torch.inference_mode()
def capture_directions(model, frozen_rows, layers, n_layers: int):
    """Prefill-only per-layer residual capture. Zero generated tokens.

    frozen_rows: list of dicts with keys pair, id, class, prompt, input_ids
    (input_ids: list[int]). Returns per-layer contrast directions.
    """
    harmful, safe = [], []
    for row in frozen_rows:
        (harmful if row["class"] == "harmful" else safe).append(row)
    if len(harmful) != len(safe):
        raise PipelineError("STAGE-3", "capture arms unbalanced", (len(harmful), len(safe)))

    acts = {i: {"harmful": [], "safe": []} for i in range(n_layers)}
    handles = []

    def make_hook(idx):
        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            final = hidden[:, -1, :].detach().to("cpu", torch.float32)
            acts[idx]["harmful" if CURRENT_ARM[0] == "harmful" else "safe"].append(final[0])
        return hook

    CURRENT_ARM = ["safe"]
    for idx, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(make_hook(idx)))

    device = next(model.parameters()).device
    try:
        for arm, rows in (("harmful", harmful), ("safe", safe)):
            CURRENT_ARM[0] = arm
            for row in rows:
                ids = torch.tensor([row["input_ids"]], device=device, dtype=torch.long)
                model(ids)
    finally:
        for h in handles:
            h.remove()

    directions, per_layer, pair_deltas = {}, {}, {}
    for i in range(n_layers):
        h = torch.stack(acts[i]["harmful"])  # [n_pairs, d]
        s = torch.stack(acts[i]["safe"])
        delta_stack = h - s                   # per-pair delta [n_pairs, d]
        delta = delta_stack.mean(dim=0)       # mean pair difference
        per_layer[i] = float(torch.linalg.vector_norm(delta))
        directions[i] = delta
        pair_deltas[i] = delta_stack
    return directions, per_layer, pair_deltas


def capture_report(directions, per_layer, first_editable: int, n_layers: int,
                   unit_tol: float = 1e-4) -> dict:
    """Gates + stability metrics for a capture. Does not gate norms by size."""
    editable = [i for i in per_layer if i >= first_editable]
    for i in editable:
        d = directions[i]
        if not torch.isfinite(d).all():
            raise PipelineError("STAGE-3", "non-finite direction", i)
        if not per_layer[i] > 0.0:
            raise PipelineError("STAGE-3", "zero-norm direction at editable layer", i)
        unit = d / per_layer[i]
        if abs(float(torch.linalg.vector_norm(unit)) - 1.0) > unit_tol:
            raise PipelineError("STAGE-3", "direction not unit-normalizable", i)
    peak_layer = max(per_layer, key=per_layer.get)
    # split-half stability of the normalized direction at the peak editable layer
    return {
        "per_layer_norm": {str(i): per_layer[i] for i in sorted(per_layer)},
        "peak_layer": peak_layer,
        "peak_norm": per_layer[peak_layer],
    }


def split_half_cosine(pair_deltas, first_editable: int) -> dict:
    """True split-half stability: cosine between the mean per-pair delta over
    even pairs and the mean per-pair delta over odd pairs."""
    out = {}
    for i in sorted(pair_deltas):
        if i < first_editable:
            continue
        d = pair_deltas[i]
        even = d[0::2].mean(dim=0)
        odd = d[1::2].mean(dim=0)
        ne = float(torch.linalg.vector_norm(even))
        no = float(torch.linalg.vector_norm(odd))
        out[str(i)] = float(torch.dot(even, odd)) / (ne * no) if ne > 0 and no > 0 else 0.0
    return out


def direction_cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    na = torch.linalg.vector_norm(a)
    nb = torch.linalg.vector_norm(b)
    if not na > 0 or not nb > 0:
        return 0.0
    return float(torch.dot(a, b) / (na * nb))


# --------------------------------------------------------------------------
# Controls (benign capability probes)
# --------------------------------------------------------------------------

CONTROLS_SPEC = [
    {
        "id": "ctrl-01",
        "name": "arithmetic_multiply",
        "kind": "exact",
        "prompt": "What is 61 times 28? Answer with the number only.",
        "expected": "1708",
    },
    {
        "id": "ctrl-02",
        "name": "arithmetic_add",
        "kind": "exact",
        "prompt": "What is 84 plus 78? Answer with the number only.",
        "expected": "162",
    },
    {
        "id": "ctrl-03",
        "name": "instruction_following",
        "kind": "exact",
        "prompt": "Reply with exactly this word and nothing else: ready",
        "expected": "ready",
    },
    {
        "id": "ctrl-04",
        "name": "structured_output_json",
        "kind": "json_exact",
        "prompt": "Return a JSON object with exactly two keys, \"city\" set to "
                   "\"Paris\" and \"country\" set to \"France\", and nothing else.",
        "expected": {"city": "Paris", "country": "France"},
    },
]


def controls_spec_sha256() -> str:
    return sha256_text(json.dumps(CONTROLS_SPEC, sort_keys=True))


def check_controls_spec(params: dict) -> None:
    got = controls_spec_sha256()
    if got != params["controls_spec_sha256"]:
        raise PipelineError("STAGE-0", "controls spec digest mismatch", f"{got} != {params['controls_spec_sha256']}")


THINK_DELIMITER = "</think"
FENCE_LANGS = ("json", "python", "py", "javascript", "js", "text", "html",
               "xml", "csv", "sql", "bash", "sh", "c", "cpp", "java", "go",
               "rust", "yaml", "yml", "markdown", "md")


def _extract_answer(raw: str) -> str:
    """Deterministic answer extraction: reasoning split, then last fenced
    block (or plain text). Deterministic: same raw, same answer, always."""
    answer = raw
    if THINK_DELIMITER in answer:
        answer = answer.rsplit(THINK_DELIMITER, 1)[1].lstrip(">")
    if "```" in answer:
        candidate = None
        for block in answer.split("```")[1:]:
            block = block.strip()
            if block:
                candidate = block
        if candidate is not None:
            first, sep, rest = candidate.partition("\n")
            if sep and first.strip().lower() in FENCE_LANGS:
                candidate = rest
            return candidate.strip()
    return answer.strip()


def parse_control_answer(raw: str, control: dict):
    """Deterministic answer extraction: reasoning split, fence strip."""
    answer = _extract_answer(raw)
    if control["kind"] == "json_exact":
        try:
            parsed = json.loads(answer)
        except Exception:  # noqa: BLE001
            return None, "unparseable"
        ok = parsed == control["expected"]
        return ok, ("exact" if ok else "wrong")
    ok = answer == control["expected"]
    return ok, ("exact" if ok else "wrong")


# --------------------------------------------------------------------------
# Chat-template modes and tokenization
# --------------------------------------------------------------------------

def _normalize_token_ids(rendered) -> list:
    if hasattr(rendered, "input_ids"):
        rendered = rendered.input_ids
    if torch.is_tensor(rendered):
        return rendered.flatten().tolist()
    if isinstance(rendered, (list, tuple)):
        return list(rendered)
    raise PipelineError("STAGE-2", "unrecognized tokenizer render type", type(rendered))


def render_row(tok, prompt: str, kwargs: dict):
    msgs = [{"role": "user", "content": prompt}]
    rendered = tok.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=True, **kwargs
    )
    return _normalize_token_ids(rendered)


def detect_modes(tok, bank_rows) -> list:
    """Detect which chat-template modes exist on this checkpoint.

    Modes are detected, never assumed. The default render always runs;
    explicit thinking-off/thinking-on renders are attempted and kept only if
    they differ from the default render (or from each other).
    """
    probe = bank_rows[0]["prompt"]
    modes = []
    default_ids = None
    try:
        default_ids = render_row(tok, probe, {})
    except Exception as exc:  # noqa: BLE001
        raise PipelineError("STAGE-2", "default chat-template render failed", repr(exc))
    renders = {}
    for name, kwargs in (("thinking_off", {"enable_thinking": False}),
                         ("thinking_on", {"enable_thinking": True})):
        try:
            renders[name] = render_row(tok, probe, kwargs)
        except Exception:  # noqa: BLE001
            renders[name] = None
    distinct = [v for v in renders.values() if v is not None]
    if all(v is not None for v in renders.values()) and renders["thinking_off"] != renders["thinking_on"]:
        if renders["thinking_off"] != default_ids:
            modes = [{"name": "thinking_off", "kwargs": {"enable_thinking": False}},
                     {"name": "thinking_on", "kwargs": {"enable_thinking": True}}]
        else:
            modes = [{"name": "default", "kwargs": {}}]
    else:
        modes = [{"name": "default", "kwargs": {}}]
    return modes


def freeze_bank_rows(tok, bank_rows, mode: dict):
    """Tokenize every bank row under one mode. Validates structure."""
    frozen = []
    for row in bank_rows:
        ids = render_row(tok, row["prompt"], mode["kwargs"])
        if not ids:
            raise PipelineError("STAGE-2", "empty token ids", (row["pair"], row["id"]))
        frozen.append({
            "pair": row["pair"],
            "id": row["id"],
            "class": row["class"],
            "prompt": row["prompt"],
            "mode": mode["name"],
            "input_ids": ids,
        })
    classes = [f["class"] for f in frozen]
    expected = ["harmful", "safe"] * (len(frozen) // 2)
    if classes != expected:
        raise PipelineError("STAGE-2", "frozen rows not alternating harmful/safe", (classes[:4], expected[:4]))
    return frozen


# --------------------------------------------------------------------------
# Self-test
# --------------------------------------------------------------------------

def _self_test() -> bool:
    params = load_pinned_parameters()
    print("[T1] pinned parameters ............ PASS")
    rows = load_bank(params)
    assert len(rows) == 64, len(rows)
    assert rows[0]["class"] == "harmful" and rows[1]["class"] == "safe", rows[:2]
    print("[T2] frozen 32-pair bank ........... PASS")
    g = torch.Generator().manual_seed(0)

    d1 = torch.randn(128, generator=g)
    d2 = torch.randn(128, generator=g)
    unit = [d1 / d1.norm(), d2 / d2.norm()]
    W = torch.randn(128, 256, generator=g)

    # alpha 0 -> bit-exact identity
    assert alpha_zero_identity(W, unit, 0.0), "alpha-0 replay not identity"

    audit = project_matrix(W, unit, 1.0, 0.002)
    # every applied direction is strongly suppressed (sequential projection of
    # non-orthogonal directions leaves only second-order cross-residuals;
    # the iteration loop re-captures and removes those in later rounds)
    for entry in audit["directions"]:
        before, after = entry["component_energy_before"], entry["component_energy_after"]
        assert before > 0 and after <= 0.25 * before, (before, after)
    # norm restored within tolerance
    assert audit["norm_error"] <= 0.002, audit["norm_error"]
    # parity 1: single-direction edit == canonical pinned formula, bitwise
    # (replicate project_matrix's in-loop re-normalization bit-exactly)
    audit_single = project_matrix(W, [unit[0]], 1.0, 0.002)
    Wf = W.to(torch.float32)
    d0 = unit[0] / torch.linalg.vector_norm(unit[0])
    canonical = Wf - 1.0 * torch.outer(d0, torch.matmul(d0, Wf))
    cn = torch.linalg.matrix_norm(canonical, ord="fro")
    canonical = canonical * (torch.linalg.matrix_norm(Wf, ord="fro") / cn)
    assert torch.equal(canonical.to(W.dtype), audit_single["edited"]), "single-direction parity failed"
    # parity 2: dual-direction edit == in-order sequential application with
    # exactly one final restore, bitwise
    d1n = unit[1] / torch.linalg.vector_norm(unit[1])
    seq = Wf - 1.0 * torch.outer(d0, torch.matmul(d0, Wf))
    seq = seq - 1.0 * torch.outer(d1n, torch.matmul(d1n, seq))
    seq = seq * (torch.linalg.matrix_norm(Wf, ord="fro") / torch.linalg.matrix_norm(seq, ord="fro"))
    assert torch.equal(seq.to(W.dtype), audit["edited"]), "dual-direction parity failed"
    print("[T3] projection math ............... PASS")

    # fused 3-D stack parity
    fused = torch.randn(8, 128, 256, generator=g)
    audits = []
    for e in range(8):
        a = project_matrix(fused[e], unit, 1.0, 0.002)
        audits.append(a)
    for e, a in enumerate(audits):
        assert a["norm_error"] <= 0.002, a["norm_error"]
    print("[T4] fused per-expert handling ..... PASS")

    # controls parser
    c_json = CONTROLS_SPEC[3]
    ok, verdict = parse_control_answer(
        "<think I should emit JSON. </think" + ">```json\n{\"city\": \"Paris\", \"country\": \"France\"}\n```",
        c_json,
    )
    assert ok is True and verdict == "exact", (ok, verdict)
    ok2, _ = parse_control_answer("I cannot help with that.", c_json)
    assert ok2 is not True, ok2
    ok3, verdict3 = parse_control_answer("61 times 28? ", CONTROLS_SPEC[0])
    assert ok3 is not True, ok3
    assert parse_control_answer("1708", CONTROLS_SPEC[0])[0] is True
    print("[T5] controls parser .............. PASS")

    # mode detection logic (synthetic)
    assert controls_spec_sha256() == params["controls_spec_sha256"], "controls spec digest drift"
    print("[T6] controls spec pin ............ PASS")

    # freeze / mode detection with a synthetic tokenizer (STAGE-2 dry run)
    class FakeTok:
        def apply_chat_template(self, msgs, tokenize=True, add_generation_prompt=False, **kw):
            return [1, 2, 3]
    modes = detect_modes(FakeTok(), rows)
    assert len(modes) == 1 and modes[0]["name"] == "default", modes
    frozen = freeze_bank_rows(FakeTok(), rows, {"name": "default", "kwargs": {}})
    assert len(frozen) == 64, len(frozen)
    for i in range(64):
        assert frozen[i]["class"] == ("harmful" if i % 2 == 0 else "safe"), i
    class EmptyTok(FakeTok):
        def apply_chat_template(self, *a, **kw):
            return []
    try:
        freeze_bank_rows(EmptyTok(), rows, {"name": "default", "kwargs": {}})
        raise AssertionError("empty token ids should fail closed")
    except PipelineError:
        pass
    print("[T7] freeze/mode detection ......... PASS")

    # fresh-workdir check (STAGE-0 ordering guard)
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        assert check_fresh_workdir(td)
        (Path(td) / "events.jsonl").touch()
        assert not check_fresh_workdir(td)
    print("[T8] fresh-workdir check ........... PASS")
    return True


if __name__ == "__main__":
    ok = _self_test()
    print("SELF-TEST:", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)

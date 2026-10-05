#!/usr/bin/env python3
"""DWM pipeline entry point — Qwen 9B class runbook package.

DWM = Domain Weight Modification, by Blackfrost_AI.
NOT ablation, NOT abliteration, NOT refusal removal, NOT fine-tuning.

Stages 0-8, linear, fail-stop. See RUNBOOK.md. The pipeline ends at
REPORT.md; it never deploys, serves, uploads, or publishes.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

LIB_PATH = Path(__file__).resolve().parent / "lib"
sys.path.insert(0, str(LIB_PATH))

import dwm_core as core  # noqa: E402

import torch  # noqa: E402


def banner(stage: str, msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {stage}: {msg}", flush=True)


class Run:
    def __init__(self, source: Path, workdir: Path):
        self.params = core.load_pinned_parameters()
        core.check_controls_spec(self.params)
        self.source = source
        self.workdir = workdir
        self.events_path = workdir / "events.jsonl"
        self.started = time.time()
        self.model = None
        self.tok = None
        self.mode_frozen = {}      # mode name -> frozen rows
        self.modes = []
        self.targets = []
        self.n_layers = 0
        self.hidden = 0
        self.source_pin = None
        self.artifacts = {}

    def event(self, stage: str, **data) -> None:
        rec = {"t": time.time() - self.started, "stage": stage, **data}
        with open(self.events_path, "a") as f:
            f.write(json.dumps(rec, sort_keys=True) + "\n")

    def gate(self, stage: str, ok: bool, reason: str, detail=None):
        if not ok:
            core.json_write(self.workdir / "FAILURE.json", {
                "stage": stage, "reason": reason, "detail": detail,
                "elapsed_s": time.time() - self.started,
            })
            raise core.PipelineError(stage, reason, detail)
        self.event(stage, event="gate-pass", reason=reason)

    def load_bank(self):
        return core.load_bank(self.params)

    # ------------------------------------------------------------------
    # STAGE-0 preflight
    # ------------------------------------------------------------------
    def stage0(self):
        banner("STAGE-0", "preflight")
        checks = {}

        # workdir emptiness is checked FIRST, before any artifact is written
        # (events.jsonl is created by the first gate() call below)
        checks["workdir_empty"] = core.check_fresh_workdir(self.workdir)
        self.gate("STAGE-0", checks["workdir_empty"], "workdir not empty")

        v = sys.version_info
        checks["python_ok"] = v >= tuple(self.params["min_python_version"])
        self.gate("STAGE-0", checks["python_ok"], "python too old", list(v))

        try:
            import transformers  # noqa: F401
            import safetensors  # noqa: F401
            import accelerate  # noqa: F401
            checks["packages_ok"] = True
            checks["versions"] = {
                "torch": torch.__version__,
                "transformers": __import__("transformers").__version__,
            }
        except Exception as exc:  # noqa: BLE001
            self.gate("STAGE-0", False, "required package missing", repr(exc))
        tv = tuple(int(x) for x in __import__("transformers").__version__.split(".")[:2])
        checks["transformers_ok"] = tv >= tuple(self.params["min_transformers_version"])
        self.gate("STAGE-0", checks["transformers_ok"], "transformers too old", tv)

        checks["cuda_ok"] = torch.cuda.is_available()
        self.gate("STAGE-0", checks["cuda_ok"], "no CUDA device")

        # source completeness
        cfg = self.source / "config.json"
        checks["source_exists"] = self.source.is_dir() and cfg.is_file()
        self.gate("STAGE-0", checks["source_exists"], "source missing config.json")

        index_files = sorted(self.source.glob("*.index.json"))
        checks["source_index"] = len(index_files) == 1
        self.gate("STAGE-0", checks["source_index"], "source must have exactly one weight index", [p.name for p in index_files])
        index_path = index_files[0]
        index = json.loads(index_path.read_text())
        shard_names = sorted(set(index["weight_map"].values()))
        missing = [s for s in shard_names if not (self.source / s).is_file()]
        checks["shards_present"] = not missing
        self.gate("STAGE-0", checks["shards_present"], "shards missing", missing)
        self.total_bytes = sum((self.source / s).stat().st_size for s in shard_names)

        free_vram, total_vram = torch.cuda.mem_get_info()
        need_vram = self.total_bytes * self.params["vram_headroom_ratio"]
        checks["vram_ok"] = free_vram >= need_vram
        self.gate("STAGE-0", checks["vram_ok"], "insufficient free VRAM", {"free": free_vram, "need": need_vram})

        free_disk = shutil.disk_usage(self.workdir).free
        need_disk = self.total_bytes * self.params["disk_headroom_ratio"]
        checks["disk_ok"] = free_disk >= need_disk
        self.gate("STAGE-0", checks["disk_ok"], "insufficient free disk", {"free": free_disk, "need": need_disk})

        try:
            memtotal = 0
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemTotal"):
                    memtotal = int(line.split()[1]) * 1024
            checks["ram_ok"] = memtotal >= self.params["min_host_ram_bytes"]
        except Exception:  # noqa: BLE001
            checks["ram_ok"] = False
        self.gate("STAGE-0", checks["ram_ok"], "insufficient host RAM")

        bank = self.load_bank()
        checks["bank_ok"] = len(bank) == 64
        self.gate("STAGE-0", checks["bank_ok"], "bank structure wrong", len(bank))

        core.json_write(self.workdir / "preflight.json", checks)
        self.event("STAGE-0", event="preflight-complete", gpu=torch.cuda.get_device_name(0), source_bytes=self.total_bytes)
        banner("STAGE-0", "PASS")

    # ------------------------------------------------------------------
    # STAGE-1 source pin + load
    # ------------------------------------------------------------------
    def fingerprint_source(self) -> dict:
        fp = {}
        for p in sorted(self.source.rglob("*")):
            if not p.is_file():
                continue
            rel = str(p.relative_to(self.source))
            size = p.stat().st_size
            if size <= 200 * 1024 * 1024:
                fp[rel] = {"size": size, "sha256": core.sha256_file(p)}
            else:
                fp[rel] = {"size": size, "mtime": p.stat().st_mtime}
        return fp

    def stage1(self):
        banner("STAGE-1", "source pin + load")
        self.source_pin = self.fingerprint_source()

        from transformers import AutoTokenizer, AutoModelForImageTextToText, AutoModelForCausalLM

        try:
            self.tok = AutoTokenizer.from_pretrained(self.source, trust_remote_code=True, local_files_only=True)
        except Exception as exc:  # noqa: BLE001
            self.gate("STAGE-1", False, "tokenizer load failed", repr(exc))

        model = None
        err = None
        for cls in (AutoModelForImageTextToText, AutoModelForCausalLM):
            try:
                model = cls.from_pretrained(
                    self.source, dtype=torch.bfloat16, trust_remote_code=True,
                    local_files_only=True, device_map={"": 0},
                    attn_implementation="sdpa",
                )
                break
            except Exception as exc:  # noqa: BLE001
                err = repr(exc)
        self.gate("STAGE-1", model is not None, "model load failed", err)
        self.model = model
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        inv = core.inventory_targets(self.model, self.params)
        self.n_layers = inv["n_layers"]
        self.targets = inv["targets"]
        self.hidden = core.hidden_size_of(self.model)

        n_expected = 2 * (self.n_layers - self.params["first_editable_layer"])
        self.gate("STAGE-1", len(self.targets) == n_expected,
                  "target inventory count wrong", (len(self.targets), n_expected))

        names = [t[0] for t in self.targets]
        sd = self.model.state_dict()
        for (wname, layer_idx, surface) in self.targets:
            w = sd[wname]
            self.gate("STAGE-1", w.dim() in (2, 3) and w.shape[0] == self.hidden,
                      "target shape unexpected", (wname, tuple(w.shape)))
            self.gate("STAGE-1", w.dtype == torch.bfloat16, "target not bf16", (wname, str(w.dtype)))
        sd_names = set(sd.keys())
        del sd

        # index must account for exactly the loaded tensor set
        index_path = sorted(self.source.glob("*.index.json"))[0]
        index = json.loads(index_path.read_text())
        index_names = set(index["weight_map"].keys())
        self.gate("STAGE-1", index_names == sd_names, "index/state_dict mismatch",
                  {"only_index": sorted(index_names - sd_names)[:5], "only_sd": sorted(sd_names - index_names)[:5]})

        core.json_write(self.workdir / "source-pin.json", {
            "source": str(self.source), "fingerprint": self.source_pin,
            "n_layers": self.n_layers, "hidden": self.hidden,
            "n_targets": len(self.targets),
            "targets": [{"name": n, "layer": l, "surface": s} for (n, l, s) in self.targets],
        })
        self.event("STAGE-1", event="loaded", n_layers=self.n_layers, n_targets=len(self.targets), hidden=self.hidden)
        banner("STAGE-1", f"PASS ({self.n_layers} layers, {len(self.targets)} targets)")

    # ------------------------------------------------------------------
    # STAGE-2 freeze inputs
    # ------------------------------------------------------------------
    def stage2(self, bank_rows):
        banner("STAGE-2", "freeze inputs")
        bank_digest = core.sha256_file(core.BANK_PATH)
        self.gate("STAGE-2", bank_digest == self.params["bank_sha256"], "bank digest mismatch on re-read")

        self.modes = core.detect_modes(self.tok, bank_rows)
        mode_names = [m["name"] for m in self.modes]
        self.gate("STAGE-2", all(m in self.params["allowed_mode_names"] for m in mode_names),
                  "detected mode not in allowed list", mode_names)
        self.gate("STAGE-2", len(self.modes) in (1, 2), "unexpected mode count", len(self.modes))

        frozen_all = {}
        for mode in self.modes:
            frozen = core.freeze_bank_rows(self.tok, bank_rows, mode)
            self.gate("STAGE-2", len(frozen) == 64, "frozen row count wrong", len(frozen))
            ids_blob = json.dumps([f["input_ids"] for f in frozen], sort_keys=True)
            frozen_all[mode["name"]] = frozen
            core.json_write(self.workdir / f"frozen-bank-tokenized-{mode['name']}.jsonl.json",
                            {"mode": mode["name"], "sha256": core.sha256_text(ids_blob),
                             "n_rows": len(frozen),
                             "rows": [{"pair": f["pair"], "id": f["id"], "class": f["class"],
                                       "n_tokens": len(f["input_ids"])} for f in frozen]})

        template_sha = "embedded"
        tmpl_path = self.source / "chat_template.jinja"
        if tmpl_path.is_file():
            template_sha = core.sha256_file(tmpl_path)
        core.json_write(self.workdir / "freeze-inputs.json", {
            "modes": mode_names,
            "bank_sha256": bank_digest,
            "template_sha256": template_sha,
        })
        self.mode_frozen = frozen_all
        self.event("STAGE-2", event="frozen", modes=mode_names)
        banner("STAGE-2", f"PASS (modes: {', '.join(mode_names)})")

    # ------------------------------------------------------------------
    # Controls
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def run_controls(self, phase: str):
        device = next(self.model.parameters()).device
        pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id
        eos = self.tok.eos_token_id
        max_new = self.params["controls_max_new_tokens"]
        records = []
        all_ok = True
        for mode in self.modes:
            for control in core.CONTROLS_SPEC:
                ids = core.render_row(self.tok, control["prompt"], mode["kwargs"])
                input_ids = torch.tensor([ids], device=device, dtype=torch.long)
                attn = torch.ones_like(input_ids)
                out = self.model.generate(
                    input_ids=input_ids, attention_mask=attn,
                    max_new_tokens=max_new, do_sample=False,
                    temperature=None, top_p=None, top_k=None,
                    pad_token_id=pad, eos_token_id=eos,
                )
                gen = out[0][input_ids.shape[1]:]
                finish = "eos" if (len(gen) > 0 and int(gen[-1]) == int(eos)) else "length"
                raw = self.tok.decode(gen, skip_special_tokens=True)
                ok, verdict = core.parse_control_answer(raw, control)
                all_ok = all_ok and bool(ok) and finish == "eos"
                records.append({
                    "mode": mode["name"], "control": control["id"], "name": control["name"],
                    "prompt": control["prompt"], "expected": control["expected"],
                    "finish": finish, "verdict": verdict, "ok": bool(ok),
                    "transcript": raw,
                })
        core.json_write(self.workdir / f"{phase}-controls.json", {"phase": phase, "records": records, "all_ok": all_ok})
        self.event("controls", event=phase, all_ok=all_ok)
        return all_ok, records

    def stage2_5(self):
        banner("STAGE-2.5", "baseline controls")
        ok, _ = self.run_controls("baseline")
        self.gate("STAGE-2.5", ok, "baseline controls failed on pristine model")
        banner("STAGE-2.5", "PASS")

    # ------------------------------------------------------------------
    # STAGE-3/6 capture
    # ------------------------------------------------------------------
    def do_capture(self, tag: str):
        from safetensors.torch import save_file

        container_path, layers = core.find_layer_list(self.model)
        directions_by_mode = {}
        report = {"tag": tag, "modes": {}}
        for mode in self.modes:
            directions, per_layer, pair_deltas = core.capture_directions(
                self.model, self.mode_frozen[mode["name"]], layers, self.n_layers)
            rep = core.capture_report(directions, per_layer,
                                      self.params["first_editable_layer"], self.n_layers,
                                      unit_tol=self.params["capture_unit_tolerance"])
            rep["split_half"] = core.split_half_cosine(pair_deltas, self.params["first_editable_layer"])
            directions_by_mode[mode["name"]] = directions
            report["modes"][mode["name"]] = rep
        st_path = self.workdir / f"capture-{tag}.safetensors"
        flat = {}
        for mode_name, directions in directions_by_mode.items():
            for i, d in directions.items():
                flat[f"{mode_name}::layer::{i}"] = d.contiguous().to(torch.float32)
        save_file(flat, str(st_path), metadata={"tag": tag, "bank": self.params["bank_sha256"]})
        core.json_write(self.workdir / f"capture-{tag}.json", report)
        self.event("capture", event=tag, modes=list(directions_by_mode))
        self.last_directions = directions_by_mode
        self.last_capture_report = report
        return directions_by_mode, report

    # ------------------------------------------------------------------
    # STAGE-4 apply
    # ------------------------------------------------------------------
    def do_apply(self, tag: str, directions_by_mode):
        banner("STAGE-4", f"apply round {tag}")
        sd = self.model.state_dict()
        alpha = self.params["alpha"]

        # alpha-zero replay on every target
        for (wname, layer_idx, surface) in self.targets:
            dirs = self._dirs_for_layer(directions_by_mode, layer_idx)
            ok = core.alpha_zero_identity(sd[wname], dirs, 0.0)
            self.gate("STAGE-4", ok, "alpha-zero replay not bit-exact", wname)

        audit_rows = []
        for (wname, layer_idx, surface) in self.targets:
            dirs = self._dirs_for_layer(directions_by_mode, layer_idx)
            audit = core.apply_round_edit(sd[wname], dirs, alpha, self.params["norm_error_max"])
            sd[wname] = audit.pop("edited").to(device=sd[wname].device, dtype=torch.bfloat16)
            audit_rows.append({"name": wname, "layer": layer_idx, "surface": surface, **audit})
        self.model.load_state_dict(sd, strict=True)
        core.json_write(self.workdir / f"round-{tag}.json", {
            "alpha": alpha, "n_targets": len(self.targets), "targets": audit_rows,
        })
        self.event("apply", event=tag, n_targets=len(self.targets))
        banner("STAGE-4", f"PASS ({len(self.targets)} targets edited)")

    def _dirs_for_layer(self, directions_by_mode, layer_idx):
        dirs = []
        for mode_name in sorted(directions_by_mode):
            d = directions_by_mode[mode_name][layer_idx]
            dirs.append(core.normalize_direction(d))
        return dirs

    # ------------------------------------------------------------------
    # STAGE-7 serialize
    # ------------------------------------------------------------------
    def stage7(self):
        banner("STAGE-7", "serialize candidate")
        from safetensors.torch import safe_open

        candidate_root = self.workdir / "candidate"
        candidate_root.mkdir(exist_ok=True)
        cand_name = self.source.name + self.params["candidate_suffix"]
        cand = candidate_root / cand_name
        self.gate("STAGE-7", not cand.exists(), "candidate path already exists", str(cand))

        self.model.save_pretrained(cand, safe_serialization=True)
        self.tok.save_pretrained(cand)

        # non-weight files must be byte-identical to source
        for p in sorted(self.source.rglob("*")):
            if not p.is_file() or p.name.endswith(".safetensors") or p.name.endswith(".index.json"):
                continue
            rel = p.relative_to(self.source)
            dst = cand / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dst)

        # enumerate candidate; everything must be either a new shard, the new
        # index, or byte-identical to the source non-weight set
        source_nonweight = {str(p.relative_to(self.source)) for p in self.source.rglob("*")
                            if p.is_file() and not p.name.endswith(".safetensors") and not p.name.endswith(".index.json")}
        bad = []
        for p in sorted(cand.rglob("*")):
            if not p.is_file():
                continue
            rel = str(p.relative_to(cand))
            if p.name.endswith(".safetensors") or p.name.endswith(".index.json"):
                continue
            if rel not in source_nonweight:
                bad.append(rel)
                continue
            if core.sha256_file(p) != core.sha256_file(self.source / rel):
                bad.append(rel)
        self.gate("STAGE-7", not bad, "candidate has unexpected or modified non-weight files", bad)

        shard_shas = {p.name: core.sha256_file(p) for p in sorted(cand.glob("*.safetensors"))}
        core.json_write(self.workdir / "written-shards.json", {"candidate": str(cand), "shards": shard_shas})
        core.json_write(self.workdir / "candidate-manifest.json", {
            "candidate": str(cand),
            "names": sorted(shard_shas),
        })
        self.event("STAGE-7", event="serialized", candidate=str(cand), n_shards=len(shard_shas))
        self.candidate = cand
        banner("STAGE-7", f"PASS ({len(shard_shas)} shards)")

    # ------------------------------------------------------------------
    # STAGE-8 verify + report
    # ------------------------------------------------------------------
    def stage8(self):
        banner("STAGE-8", "independent verification")
        from safetensors.torch import safe_open, load_file

        changed, protected, shape_bad, missing = [], [], [], []
        source_map = {}
        for shard in sorted(self.source.glob("*.safetensors")):
            with safe_open(shard, framework="pt", device="cpu") as f:
                for k in f.keys():
                    source_map[k] = shard
        cand_shards = sorted(self.candidate.glob("*.safetensors"))
        cand_map = {}
        for shard in cand_shards:
            with safe_open(shard, framework="pt", device="cpu") as f:
                for k in f.keys():
                    cand_map[k] = shard
        self.gate("STAGE-8", set(cand_map) == set(source_map), "candidate tensor set != source", None)
        target_names = {t[0] for t in self.targets}
        n_changed = n_protected = 0
        for key in sorted(source_map):
            with safe_open(source_map[key], framework="pt", device="cpu") as fs:
                s_t = fs.get_tensor(key)
            with safe_open(cand_map[key], framework="pt", device="cpu") as fc:
                c_t = fc.get_tensor(key)
            if s_t.shape != c_t.shape or s_t.dtype != c_t.dtype:
                shape_bad.append(key)
                continue
            if torch.equal(s_t, c_t):
                n_protected += 1
                if key in target_names:
                    protected.append(key)
            else:
                n_changed += 1
                if key not in target_names:
                    changed.append(key)
        self.gate("STAGE-8", not shape_bad, "shape/dtype mismatch in candidate", shape_bad[:5])
        self.gate("STAGE-8", n_changed == len(self.targets), "changed count != target count",
                  (n_changed, len(self.targets)))
        self.gate("STAGE-8", n_protected == len(source_map) - len(self.targets),
                  "protected count wrong", (n_protected, len(source_map) - len(self.targets)))
        self.gate("STAGE-8", not changed and not protected,
                  "unexpected changed/protected outside target set", (changed[:5], protected[:5]))

        # source immutability (bidirectional: modified, added, or deleted)
        final_fp = self.fingerprint_source()
        self.gate("STAGE-8", final_fp == self.source_pin, "source checkpoint changed during run",
                  sorted(k for k in set(final_fp) | set(self.source_pin)
                         if final_fp.get(k) != self.source_pin.get(k))[:5])

        verification = {
            "n_tensors_total": len(source_map),
            "n_changed": n_changed,
            "n_protected_bit_identical": n_protected,
            "changed_names": sorted(target_names),
            "source_unchanged": True,
            "candidate": str(self.candidate),
        }
        core.json_write(self.workdir / "verification.json", verification)
        self.event("STAGE-8", event="verified", **{k: verification[k] for k in ("n_tensors_total", "n_changed", "n_protected_bit_identical")})
        banner("STAGE-8", f"PASS (changed {n_changed}, protected {n_protected})")
        return verification

    def write_report(self):
        banner("STAGE-8", "report")
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unknown"
        data = {
            "method": self.params["method"],
            "attribution": self.params["attribution"],
            "not": ["ablation", "abliteration", "refusal removal", "fine-tuning"],
            "source": str(self.source),
            "candidate": str(self.candidate),
            "modes": [m["name"] for m in self.modes],
            "n_layers": self.n_layers,
            "n_targets": len(self.targets),
            "hidden": self.hidden,
            "gpu": gpu,
            "elapsed_s": time.time() - self.started,
            "artifacts": sorted(str(p) for p in self.workdir.iterdir()),
        }
        core.json_write(self.workdir / "REPORT-DATA.json", data)
        lines = [
            "# DWM RUN REPORT",
            "",
            f"- Method: {self.params['method']} — {self.params['attribution']}",
            "- This run was NOT ablation, NOT abliteration, NOT refusal removal, NOT fine-tuning.",
            f"- Source: {self.source}",
            f"- Candidate: {self.candidate}",
            f"- Modes: {', '.join(m['name'] for m in self.modes)}",
            f"- Decoder layers: {self.n_layers}; edit targets: {len(self.targets)}",
            f"- GPU: {gpu}; elapsed seconds: {data['elapsed_s']:.0f}",
            "",
            "See REPORT-DATA.json, verification.json, round-*.json, controls-*.json,",
            "capture-*.json for the full recorded evidence.",
            "",
            "No behavioral benchmark was run. Nothing was deployed, served, or published.",
            "The candidate's fate is the operator's decision alone.",
        ]
        (self.workdir / "REPORT.md").write_text("\n".join(lines) + "\n")
        banner("STAGE-8", "REPORT.md written — RUN COMPLETE")


def self_test_mode() -> int:
    ok = core._self_test()
    print("SELF-TEST:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="Blackfrost DWM pipeline — Qwen 9B class")
    ap.add_argument("--source", help="absolute path to the source checkpoint")
    ap.add_argument("--workdir", help="absolute path to a fresh empty workdir")
    ap.add_argument("--self-test", action="store_true", help="offline self-test; must PASS before any run")
    args = ap.parse_args()

    if args.self_test or (not args.source and not args.workdir):
        return self_test_mode()

    if not args.source or not args.workdir:
        print("error: --source and --workdir are required for a run (or pass --self-test)")
        return 2
    source = Path(args.source).resolve()
    workdir = Path(args.workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    run = Run(source, workdir)
    try:
        run.stage0()
        bank_rows = run.load_bank()
        run.stage1()
        run.stage2(bank_rows)
        run.stage2_5()
        prev_directions = None
        for r in range(int(run.params["rounds"])):
            directions, report = run.do_capture(str(r))
            if prev_directions is not None:
                run.event("capture", event=f"telemetry-{r}",
                          cosine_vs_prev={m: {str(i): core.direction_cosine(directions[m][i], prev_directions[m][i])
                                             for i in sorted(directions[m])} for m in directions})
            run.do_apply(str(r), directions)
            ok, _ = run.run_controls(f"round{r}")
            run.gate(f"STAGE-5", ok, f"controls failed after round {r}")
            prev_directions = directions
        run.do_capture("3")
        run.stage7()
        run.stage8()
        run.write_report()
        return 0
    except core.PipelineError as exc:
        print(str(exc), flush=True)
        try:
            core.json_write(workdir / "FAILURE.json", {
                "stage": exc.stage, "reason": exc.reason, "detail": exc.detail,
            })
        except Exception:  # noqa: BLE001
            pass
        print("STOPPED. FAILURE.json and events.jsonl are in the workdir. Do not retry.", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Sequential lifecycle orchestration, with raw failure records and bounded retries."""

import json
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

import httpx

from .artifacts import verify_artifacts
from .backends.base import Request
from .backends.fake import FakeBackend
from .backends.llamacpp import LlamaCppBackend
from .backends.ollama import OllamaBackend
from .environment import capture_environment, environment_identity
from .metrics import measured, native_metrics
from .monitoring.gpu import GPULock, GPUMonitor, wait_idle
from .prompts import render, text_hash
from .protocol import validate_certificate, validate_schedule
from .schema import RunRecord, TokenUsage
from .storage import ResultStore
from .utils import Clock, FakeClock, atomic_json, digest, file_hash


def classify(error: BaseException) -> str:
    if isinstance(error, KeyboardInterrupt):
        return "interrupted"
    if isinstance(error, (TimeoutError, httpx.TimeoutException)):
        return "timeout"
    text = str(error).lower()
    if any(s in text for s in ("out of memory", "cuda error: out", "cuda malloc", "cudamalloc")):
        return "oom"
    return "invalid"


def make_backend(cell, config, work, clock):
    work.mkdir(parents=True, exist_ok=True)
    if config.synthetic:
        return FakeBackend(cell, clock)
    cls = OllamaBackend if cell.engine == "ollama" else LlamaCppBackend
    return cls(cell, config, work, clock)


class BenchmarkRunner:
    def __init__(
        self,
        schedule: dict,
        root: Path,
        resume: bool = False,
        evidence: dict | None = None,
        backend_factory=make_backend,
    ):
        self.schedule, self.config = schedule, validate_schedule(schedule)
        self.store = ResultStore(root)
        self.resume, self.evidence = resume, evidence or {}
        self.clock = FakeClock() if self.config.synthetic else Clock()
        self.backend_factory = backend_factory
        self.env = capture_environment(
            self.config.runtime.ollama_binary,
            self.config.runtime.llama_binary,
            self.config.runtime.toolchain_lock,
        )
        self.env_hash = (
            "synthetic-environment-v1" if self.config.synthetic else environment_identity(self.env)
        )
        self.active = False
        self.load = None
        self.monitor = None
        self.effective = {}

    def _hardware_gate(self):
        cfg = self.config
        if cfg.synthetic:
            return
        if cfg.deployment == "gpu":
            if not self.env["gpu"]["available"]:
                raise ValueError(f"GPU unavailable: {self.env['gpu']['reason']}")
            device = next(
                (d for d in self.env["gpu"]["devices"] if d["index"] == cfg.runtime.gpu_index), None
            )
            if (
                not device
                or "3050" not in device["name"]
                or not 3.5 * 1024**3 <= device["total_bytes"] <= 4.5 * 1024**3
            ):
                raise ValueError(
                    "reference GPU run requires RTX 3050 with approximately 4 GiB VRAM"
                )
        if cfg.stage == "official":
            validate_certificate(cfg, self.evidence, self.schedule)
            if self.evidence.get("environment_hash") != self.env_hash:
                raise ValueError("preflight environment differs from current environment")

    def run(self):
        self._hardware_gate()
        artifact_snapshot = verify_artifacts(self.config)
        gpu_lock = None
        if not self.config.synthetic and self.config.deployment == "gpu":
            device = next(
                d for d in self.env["gpu"]["devices"] if d["index"] == self.config.runtime.gpu_index
            )
            gpu_lock = GPULock(device["uuid"])
        manifest = {
            "schema_version": 1,
            "created_at": datetime.now(UTC).isoformat(),
            "campaign_id": self.store.root.name,
            "config_hash": self.schedule["config_hash"],
            "schedule_hash": digest(self.schedule),
            "environment_hash": self.env_hash,
            "synthetic": self.config.synthetic,
            "stage": self.config.stage,
            "equivalence_verified": not self.config.synthetic and bool(self.evidence.get("passed")),
            "synthetic_contract": self.config.synthetic,
        }
        initialized = False
        try:
            if gpu_lock:
                gpu_lock.acquire()
            self.store.acquire()
            self.store.initialize(manifest, self.resume)
            initialized = True
            if not self.resume:
                atomic_json(self.store.root / "schedule.json", self.schedule)
                atomic_json(self.store.root / "environment.json", self.env)
                atomic_json(self.store.root / "resolved_config.json", self.config.model_dump())
                atomic_json(self.store.root / "preflight_evidence.json", self.evidence)
                atomic_json(self.store.root / "artifact_manifest.json", artifact_snapshot)
                toolchain = (
                    {"synthetic": True}
                    if self.config.synthetic
                    else json.loads(Path(self.config.runtime.toolchain_lock).read_text())
                )
                atomic_json(self.store.root / "toolchain.json", toolchain)
                (self.store.root / "answers.jsonl").write_bytes(
                    Path(self.config.answers_path).read_bytes()
                )
                atomic_json(
                    self.store.root / "metadata_checksums.json",
                    {
                        name: file_hash(self.store.root / name)
                        for name in (
                            "manifest.json",
                            "schedule.json",
                            "environment.json",
                            "resolved_config.json",
                            "preflight_evidence.json",
                            "answers.jsonl",
                            "artifact_manifest.json",
                            "toolchain.json",
                        )
                    },
                )
            if self.resume:
                self.store.verify()
            done = {
                r.trial_id
                for r in self.store.records()
                if r.status == "ok" and r.phase in {"measurement", "quality"}
            }
            for session in self.schedule["sessions"]:
                pending = [t for t in session["trials"] if t["id"] not in done]
                if pending:
                    self._session(session, pending)
            atomic_json(
                self.store.root / "state.json",
                {"state": "finished", "synthetic": self.config.synthetic},
            )
        except BaseException as exc:
            if initialized:
                self.store.append(
                    "control.jsonl",
                    {"kind": "campaign_error", "error": str(exc), "type": type(exc).__name__},
                )
                atomic_json(
                    self.store.root / "state.json",
                    {
                        "state": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                        "error": str(exc),
                    },
                )
            raise
        finally:
            self.store.release()
            if gpu_lock:
                gpu_lock.close()

    def _session(self, session: dict, pending: list[dict]):
        cfg, p = self.config, self.config.protocol
        cell = next(c for c in cfg.cells if c.id == session["cell_id"])
        attempt_suffix, work = self.store.reserve_session(session["id"])
        backend = self.backend_factory(cell, cfg, work, self.clock)
        self.current_backend = backend
        self.monitor = GPUMonitor(
            self.clock,
            p.sample_interval_ms,
            cfg.runtime.gpu_index,
            cfg.deployment == "gpu",
            cfg.synthetic,
        )
        self.load = None
        load_saved = False
        starting_attempts = {t["id"]: self.store.attempts(t["id"]) for t in pending}
        try:
            # Setup/import precedes idle baseline and is excluded from load observations.
            if isinstance(backend, OllamaBackend):
                backend._start_import()
                backend.unload()
            self.monitor.start()
            self.monitor.mark("idle")
            wait_idle(self.monitor, p, self.clock)
            idle_start = self.clock.now()
            self.clock.sleep(p.idle_seconds)
            idle = self.monitor.memory(idle_start, self.clock.now(), "median")
            self.monitor.mark("load")
            load_start = self.clock.now()
            load_id = f"{session['id']}-load-a{attempt_suffix}"
            self.load = backend.load(load_id)
            load_end = self.clock.now()
            peak_load = self.monitor.memory(load_start, load_end)
            self.monitor.mark("loaded")
            loaded_start = self.clock.now()
            self.clock.sleep(p.loaded_seconds)
            loaded = self.monitor.memory(loaded_start, self.clock.now(), "median")
            self.effective = backend.inspect()
            self._apply_audit(cell)
            self.store.append(
                "loads.jsonl",
                {
                    "session_id": session["id"],
                    "cell_id": cell.id,
                    "run_mode": session["mode"],
                    "synthetic": cfg.synthetic,
                    "observation": self.load.model_dump(),
                    "metrics": {
                        "idle_vram_bytes": idle.model_dump(),
                        "loaded_vram_bytes": loaded.model_dump(),
                        "peak_load_vram_bytes": peak_load.model_dump(),
                        "model_load_wall_ms": self.load.wall.model_dump(),
                        "model_load_native_ms": self.load.native.model_dump(),
                    },
                },
            )
            load_saved = True
            if cfg.deployment == "gpu" and self.effective.get("full_offload") is not True:
                raise ValueError("full GPU offload not established; no automatic CPU fallback")
            budget = (
                cfg.generation.quality_max_tokens
                if session["mode"] == "quality"
                else cfg.generation.performance_max_tokens
            )
            first_request = Request(
                render(cell.model, pending[0]["item"]["text"]),
                budget,
                cfg.generation,
                cell.model.stop,
            )
            if session["mode"] != "cold":
                for n in range(p.warmups):
                    self.monitor.mark("warmup")
                    trial = {
                        "id": f"{session['id']}-a{attempt_suffix}-warmup{n}",
                        "item": pending[0]["item"],
                    }
                    record = self._request(
                        backend, cell, session, trial, first_request, "warmup", idle
                    )
                    if record.status != "ok":
                        raise ValueError(f"warmup failed: {record.error or record.exclusions}")
            for trial in pending:
                request = Request(
                    render(cell.model, trial["item"]["text"]),
                    budget,
                    cfg.generation,
                    cell.model.stop,
                )
                if session["mode"] != "cold":
                    eviction = backend.prepare_cache(request)
                    ev_trial = {
                        "id": trial["id"] + f"-eviction-a{attempt_suffix}",
                        "item": trial["item"],
                    }
                    self.monitor.mark("eviction")
                    ev = self._request(backend, cell, session, ev_trial, eviction, "eviction", idle)
                    if ev.status != "ok":
                        raise ValueError(f"cache preparation failed: {ev.error or ev.exclusions}")
                    self.clock.sleep(p.settle_seconds)
                phase = "quality" if session["mode"] == "quality" else "measurement"
                self.monitor.mark("request")
                record = self._request(backend, cell, session, trial, request, phase, idle)
                if record.status == "interrupted":
                    raise KeyboardInterrupt
            self.monitor.mark("cleanup")
            backend.unload()
            self.clock.sleep(p.idle_seconds)
            if cfg.deployment == "gpu":
                after = self.monitor.memory(
                    self.clock.now() - int(p.idle_seconds * 1e9), self.clock.now(), "median"
                )
                if (
                    after.value is None
                    or idle.value is None
                    or after.value > idle.value + p.baseline_tolerance_mib * 1024**2
                ):
                    raise ValueError("GPU VRAM did not return to baseline after unload")
        except (Exception, KeyboardInterrupt) as exc:
            for trial in pending:
                if self.store.attempts(trial["id"]) == starting_attempts[trial["id"]]:
                    budget = (
                        cfg.generation.quality_max_tokens
                        if session["mode"] == "quality"
                        else cfg.generation.performance_max_tokens
                    )
                    request = Request(
                        render(cell.model, trial["item"]["text"]),
                        budget,
                        cfg.generation,
                        cell.model.stop,
                    )
                    record = self._base_record(
                        cell,
                        session,
                        trial,
                        request,
                        "quality" if session["mode"] == "quality" else "measurement",
                    )
                    record.status, record.error = classify(exc), str(exc)
                    self.store.save_attempt(record, [], [])
            raise
        finally:
            errors = []
            try:
                backend.close()
            except Exception as exc:
                errors.append(str(exc))
            try:
                self.monitor.close()
            except Exception as exc:
                errors.append(str(exc))
            if self.monitor.samples:
                from .utils import canonical

                trace = work / "gpu-session.jsonl"
                trace.parent.mkdir(parents=True, exist_ok=True)
                trace.write_text("".join(canonical(s) + "\n" for s in self.monitor.samples))
                metrics = (
                    measured(None, "bytes", "NVML", self.monitor.error)
                    if self.monitor.error
                    else self.monitor.memory(
                        self.monitor.samples[0]["timestamp_ns"],
                        self.monitor.samples[-1]["timestamp_ns"],
                    )
                )
                intervals = [
                    (b["timestamp_ns"] - a["timestamp_ns"]) / 1e6
                    for a, b in zip(self.monitor.samples, self.monitor.samples[1:], strict=False)
                ]
                self.store.append(
                    "sessions.jsonl",
                    {
                        "session_id": session["id"],
                        "cell_id": cell.id,
                        "run_mode": session["mode"],
                        "load_id": self.load.id if load_saved else None,
                        "synthetic": cfg.synthetic,
                        "peak_session_vram_bytes": metrics.model_dump(),
                        "trace_path": str(trace.relative_to(self.store.root)),
                        "trace_sha256": file_hash(trace),
                        "n_samples": len(self.monitor.samples),
                        "requested_interval_ms": p.sample_interval_ms,
                        "max_interval_ms": max(intervals, default=None),
                        "missed_interval_count": sum(
                            v > p.sample_interval_ms * 2 for v in intervals
                        )
                        if not cfg.synthetic
                        else None,
                    },
                )
            if errors:
                self.store.append(
                    "control.jsonl",
                    {"kind": "cleanup_error", "session": session["id"], "errors": errors},
                )
                raise RuntimeError("cleanup failed: " + "; ".join(errors))

    def _apply_audit(self, cell):
        if self.config.synthetic:
            return
        import psutil

        process = self.current_backend.server.process
        if process is not None:
            self.effective["owned_pids"] = [process.pid] + [
                p.pid for p in psutil.Process(process.pid).children(recursive=True)
            ]
        lock = json.loads(Path(self.config.runtime.toolchain_lock).read_text())
        audit = lock.get("settings_audit", {}).get(cell.engine, {})
        # Audit is a human-reviewed source artifact, not an inference measurement.
        required = (
            "samplers",
            "kv_f16",
            "batch",
            "no_speculation",
            "bos_policy",
            "truncate_disabled",
        )
        if (
            audit.get("reviewed") is True
            and audit.get("binary_sha256") == lock[cell.engine]["sha256"]
            and audit.get("source_references")
            and all(audit.get(k) is True for k in required)
        ):
            self.effective["runtime_verified"] = self.effective.get("context_verified") is True
            self.effective["source_audit"] = audit

    def _base_record(self, cell, session, trial, request, phase):
        now = self.clock.now()
        return RunRecord(
            campaign_id=self.store.root.name,
            trial_id=trial["id"],
            attempt_id=f"{trial['id']}-attempt{self.store.attempts(trial['id']) + 1}",
            session_id=session["id"],
            block_id=session["block_id"],
            experiment=self.config.experiment,
            cell_id=cell.id,
            engine=cell.engine,
            model_id=cell.model.id,
            quant=cell.model.quant,
            gguf_sha256=cell.model.sha256,
            model_revision=cell.model.revision,
            phase=phase,
            run_mode=session["mode"],
            synthetic=self.config.synthetic,
            config_hash=self.schedule["config_hash"],
            environment_hash=self.env_hash,
            workload_hash=self.schedule["workload_hash"],
            template_hash=text_hash(cell.model.template),
            prompt_id=trial["item"]["id"],
            prompt_hash=text_hash(request.prompt),
            prompt_text=request.prompt,
            settings={
                "generation": request.generation.model_dump(),
                "max_tokens": request.max_tokens,
                "runtime": self.config.runtime.model_dump(),
                "stop": request.stop,
                "thinking": cell.model.thinking,
                "cache_policy": self.config.protocol.cache_policy,
            },
            effective=self.effective,
            load_id=self.load.id if self.load else None,
            start_ns=now,
            end_ns=now,
            status="invalid",
            output_hash=text_hash(""),
            started_at_utc=datetime.now(UTC).isoformat(),
            usage=TokenUsage(source="unavailable", reason="request not completed"),
        )

    def _request(self, backend, cell, session, trial, request, phase, idle):
        if self.active:
            raise RuntimeError("overlapping requests forbidden")
        record = self._base_record(cell, session, trial, request, phase)
        events, text, finals, first_content = [], [], [], None
        try:
            known = (
                self.evidence.get("input_counts", {})
                .get(cell.model.sha256, {})
                .get(record.prompt_hash)
            )
            if known is None and self.config.stage != "official":
                known = backend.count_tokens(request.prompt)
            if not self.config.synthetic and known is None:
                raise ValueError(
                    "input token count unavailable before request; run reference-tokenizer preflight"
                )
            if known is not None and known + request.max_tokens > self.config.runtime.context_size:
                raise ValueError("input + output budget exceeds context; no truncation")
            record.start_ns = (
                self.clock.now()
            )  # tokenize/validation/serialization setup is excluded.
            self.active = True
            with closing(backend.stream(request)) as stream:
                for event in stream:
                    events.append(event.model_dump())
                    elapsed = (event.receipt_ns - record.start_ns) / 1e9
                    if elapsed > self.config.protocol.request_timeout:
                        raise TimeoutError("end-to-end request deadline exceeded")
                    if first_content is None and elapsed > self.config.protocol.ttft_timeout:
                        raise TimeoutError("first-content deadline exceeded")
                    if event.kind == "error":
                        raise RuntimeError(str(event.payload))
                    if event.kind == "thinking":
                        raise ValueError("unexpected thinking in non-thinking profile")
                    if event.kind == "text" and event.text:
                        if finals:
                            raise ValueError("content after terminal event")
                        first_content = (
                            first_content if first_content is not None else event.receipt_ns
                        )
                        text.append(event.text)
                    if event.kind == "final":
                        finals.append(event)
                        if len(finals) != 1:
                            raise ValueError("duplicate final event")
                        record.end_ns = event.receipt_ns
            if len(finals) != 1:
                raise ValueError("stream ended without a terminal event")
            final = finals[0].payload
            record.usage, record.metrics = native_metrics(cell.engine, final)
            if "<think>" in "".join(text):
                raise ValueError("raw generated thinking marker in non-thinking profile")
            if final.get("truncated"):
                raise ValueError("engine silently truncated/shifted context")
            if record.usage.input_total != known:
                raise ValueError(
                    f"token count mismatch: preflight={known}, engine={record.usage.input_total}"
                )
            if phase in {"measurement", "quality"}:
                if (
                    record.usage.input_cached is None
                    or record.usage.input_cached > self.config.protocol.allowed_bos_cache_tokens
                ):
                    raise ValueError("substantive or unknown cached prefix in measured request")
                expected_cache = (
                    self.evidence.get("cache_counts", {})
                    .get(cell.id, {})
                    .get(session["mode"], {})
                    .get(record.prompt_hash)
                )
                if self.config.stage == "official" and expected_cache is None:
                    raise ValueError("official request lacks cache evidence")
                if expected_cache is not None and record.usage.input_cached != expected_cache:
                    raise ValueError(
                        "cached-token count changed since preflight for this run mode/prompt"
                    )
                if not self.effective.get("runtime_verified"):
                    record.exclusions.append("effective runtime/sampler settings not audited")
            record.stop_reason = final.get("done_reason", final.get("stop_type"))
            record.status = "invalid" if record.exclusions else "ok"
        except (Exception, KeyboardInterrupt) as exc:
            record.status, record.error = classify(exc), f"{type(exc).__name__}: {exc}"
            record.end_ns = self.clock.now()
        finally:
            self.active = False
        record.output = "".join(text)
        record.output_hash = text_hash(record.output)
        terminal = len(finals) == 1
        record.metrics["ttft_stream_ms"] = measured(
            (first_content - record.start_ns) / 1e6 if first_content is not None else None,
            "ms",
            "client monotonic first-content",
            "no generated content received",
        )
        record.metrics["e2e_request_ms"] = measured(
            (record.end_ns - record.start_ns) / 1e6 if terminal else None,
            "ms",
            "client monotonic terminal",
            "no valid terminal event",
        )
        # Telemetry failures must never discard inference evidence already received.
        samples = [
            s for s in self.monitor.samples if record.start_ns <= s["timestamp_ns"] <= record.end_ns
        ]
        try:
            record.metrics["peak_request_vram_bytes"] = self.monitor.memory(
                record.start_ns, record.end_ns
            )
        except Exception as exc:
            record.status = "invalid"
            record.error = "; ".join(filter(None, (record.error, f"GPU telemetry: {exc}")))
            record.metrics["peak_request_vram_bytes"] = measured(None, "bytes", "NVML", str(exc))
        record.metrics["idle_vram_bytes"] = idle
        peak = record.metrics["peak_request_vram_bytes"].value
        delta = max(0.0, peak - idle.value) if peak is not None and idle.value is not None else None
        record.metrics["peak_request_vram_delta_bytes"] = measured(
            delta, "bytes", "sampled peak - idle", "missing idle or peak samples"
        )
        if session["mode"] == "cold" and self.load and terminal:
            record.metrics["cold_activation_to_response_ms"] = measured(
                (record.end_ns - self.load.start_ns) / 1e6,
                "ms",
                "activation→terminal including readiness gap",
            )
        self.store.save_attempt(record, events, samples)
        return record

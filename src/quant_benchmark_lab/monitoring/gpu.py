"""NVML samples are device-wide unless explicitly labeled process memory."""

import fcntl
import os
import re
import statistics
import tempfile
import threading
from pathlib import Path

from ..metrics import measured
from ..utils import Clock


class GPULock:
    """Cross-campaign lock for this user's benchmark processes on one physical GPU."""

    def __init__(self, uuid: str):
        if not re.fullmatch(r"[A-Za-z0-9-]+", uuid):
            raise ValueError("invalid GPU UUID")
        self.path = Path(tempfile.gettempdir()) / f"qbl-{os.getuid()}-{uuid}.lockfile"
        self.handle = None

    def acquire(self):
        self.handle = self.path.open("a")
        try:
            fcntl.flock(self.handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.handle.close()
            self.handle = None
            raise RuntimeError("another benchmark campaign owns this GPU") from None

    def close(self):
        if self.handle:
            fcntl.flock(self.handle, fcntl.LOCK_UN)
            self.handle.close()
            self.handle = None


class GPUMonitor:
    def __init__(self, clock: Clock, interval_ms: int, index: int = 0, enabled: bool = True,
                 synthetic: bool = False):
        self.clock, self.interval = clock, interval_ms / 1000
        self.index, self.enabled, self.synthetic = index, enabled, synthetic
        self.samples: list[dict] = []
        self.phase = "idle"
        self.error: str | None = None
        self.stop_event = threading.Event()
        self.thread = None
        self.nv = self.handle = None
        self.total_bytes = None

    def start(self):
        if self.synthetic:
            self.total_bytes = 4 * 1024**3
            self.mark("idle")
            return
        if not self.enabled:
            return
        import pynvml

        self.nv = pynvml
        self.nv.nvmlInit()
        try:
            self.handle = self.nv.nvmlDeviceGetHandleByIndex(self.index)
            self.total_bytes = self.nv.nvmlDeviceGetMemoryInfo(self.handle).total
            self.thread = threading.Thread(target=self._loop, name="qbl-nvml", daemon=True)
            self.thread.start()
        except BaseException:
            self.nv.nvmlShutdown()
            raise

    def _optional(self, method, *args):
        try:
            return method(*args)
        except Exception:
            return None

    def _loop(self):
        try:
            while not self.stop_event.is_set():
                n, h = self.nv, self.handle
                stamp = self.clock.now()
                mem = n.nvmlDeviceGetMemoryInfo(h)
                proc = self._optional(n.nvmlDeviceGetComputeRunningProcesses, h)
                processes = None if proc is None else [
                    {"pid": p.pid, "used_bytes": p.usedGpuMemory
                     if 0 <= p.usedGpuMemory <= mem.total else None} for p in proc]
                self.samples.append({"timestamp_ns": stamp, "phase": self.phase,
                    "used_bytes": mem.used, "free_bytes": mem.free, "total_bytes": mem.total,
                    "processes": processes,
                    "temperature_c": self._optional(n.nvmlDeviceGetTemperature, h, n.NVML_TEMPERATURE_GPU),
                    "utilization_pct": n.nvmlDeviceGetUtilizationRates(h).gpu,
                    "graphics_clock_mhz": self._optional(n.nvmlDeviceGetClockInfo, h, n.NVML_CLOCK_GRAPHICS),
                    "memory_clock_mhz": self._optional(n.nvmlDeviceGetClockInfo, h, n.NVML_CLOCK_MEM),
                    "power_mw": self._optional(n.nvmlDeviceGetPowerUsage, h),
                    "throttle_reasons": self._optional(n.nvmlDeviceGetCurrentClocksThrottleReasons, h)})
                self.stop_event.wait(self.interval)
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"

    def check(self):
        if self.error:
            raise RuntimeError(f"GPU monitor failed: {self.error}")

    def mark(self, phase: str):
        self.check()
        self.phase = phase
        if self.synthetic:
            used = {"idle": 100, "load": 350, "loaded": 300, "eviction": 380,
                    "request": 400, "cleanup": 100}.get(phase, 300) * 1024**2
            self.samples.append({"timestamp_ns": self.clock.now(), "phase": phase,
                                 "used_bytes": used, "free_bytes": self.total_bytes - used,
                                 "total_bytes": self.total_bytes, "temperature_c": 40.0,
                                 "utilization_pct": 0.0, "processes": [], "synthetic": True})

    def window(self, start: int, end: int) -> list[dict]:
        self.check()
        return [s for s in self.samples if start <= s["timestamp_ns"] <= end]

    def memory(self, start: int, end: int, mode: str = "peak"):
        rows = self.window(start, end)
        value = None
        if rows:
            values = [s["used_bytes"] for s in rows]
            value = float(max(values) if mode == "peak" else statistics.median(values))
        source = "synthetic GPU fixture" if self.synthetic else "NVML sampled device memory"
        return measured(value, "bytes", source, "no samples in measurement window")

    def close(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=2)
            if self.thread.is_alive():
                self.error = "monitor thread did not stop"
        if self.nv:
            self.nv.nvmlShutdown()
        self.check()


def wait_idle(monitor: GPUMonitor, protocol, clock: Clock):
    if monitor.synthetic or not monitor.enabled:
        return
    start = clock.now()
    while (clock.now() - start) / 1e9 < protocol.cooldown_timeout:
        clock.sleep(protocol.idle_seconds)
        rows = monitor.window(clock.now() - int(protocol.idle_seconds * 1e9), clock.now())
        if rows and all(r.get("temperature_c") is not None for r in rows):
            if (statistics.median(r["utilization_pct"] for r in rows) <= protocol.max_idle_utilization
                    and statistics.median(r["temperature_c"] for r in rows) <= protocol.max_temperature_c):
                return
    raise TimeoutError("blocked-environment: GPU did not reach preregistered idle conditions")

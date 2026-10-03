"""Auditable environment capture; absence is data, not a guessed version."""

import importlib.metadata
import json
import os
import platform
import shutil
from pathlib import Path

import psutil

from .utils import command, digest, file_hash

# Engine subprocesses receive only these host settings plus explicit backend overrides.
# In particular LLAMA_ARG_*, OLLAMA_*, LD_PRELOAD and arbitrary GGML_* are not inherited.
ENGINE_ENV_KEYS = (
    "PATH",
    "HOME",
    "TMPDIR",
    "LD_LIBRARY_PATH",
    "CUDA_VISIBLE_DEVICES",
    "CUDA_MODULE_LOADING",
    "OMP_NUM_THREADS",
    "OMP_PROC_BIND",
)


def engine_environment() -> dict[str, str]:
    return {
        **{k: os.environ[k] for k in ENGINE_ENV_KEYS if k in os.environ},
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "TZ": "UTC",
    }


def executable_identity(binary: str) -> dict:
    resolved = shutil.which(binary)
    if not resolved:
        return {"requested": binary, "unavailable": "executable not found"}
    return {
        "path": str(Path(resolved).resolve()),
        "sha256": file_hash(Path(resolved)),
        "version": command([resolved, "--version"]),
    }


def capture_environment(
    ollama: str = "ollama", llama: str = "llama-server", toolchain: str | None = None
) -> dict:
    cpu_model = platform.processor()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        cpu_model = next(
            (
                line.split(":", 1)[1].strip()
                for line in cpuinfo.read_text().splitlines()
                if line.startswith("model name")
            ),
            cpu_model,
        )
    packages = {}
    for name in (
        "quant-benchmark-lab",
        "pydantic",
        "httpx",
        "PyYAML",
        "psutil",
        "nvidia-ml-py",
        "numpy",
        "matplotlib",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    gpu = {"available": False, "reason": "NVML unavailable"}
    try:
        import pynvml as nv

        nv.nvmlInit()
        try:

            def decoded(value):
                return value.decode("utf-8") if isinstance(value, bytes) else value

            devices = []
            for i in range(nv.nvmlDeviceGetCount()):
                h = nv.nvmlDeviceGetHandleByIndex(i)
                devices.append(
                    {
                        "index": i,
                        "name": decoded(nv.nvmlDeviceGetName(h)),
                        "uuid": decoded(nv.nvmlDeviceGetUUID(h)),
                        "total_bytes": nv.nvmlDeviceGetMemoryInfo(h).total,
                    }
                )
            gpu = {
                "available": True,
                "driver": decoded(nv.nvmlSystemGetDriverVersion()),
                "devices": devices,
            }
        finally:
            nv.nvmlShutdown()
    except (ImportError, RuntimeError, OSError) as exc:
        gpu["reason"] = str(exc)
    except Exception as exc:  # NVML exception hierarchy is optional until imported.
        gpu["reason"] = f"{type(exc).__name__}: {exc}"
    harness_root = Path(__file__).parent
    harness_hash = digest(
        {str(p.relative_to(harness_root)): file_hash(p) for p in sorted(harness_root.rglob("*.py"))}
    )
    allowlisted = engine_environment()
    lock = (
        json.loads(Path(toolchain).read_text()) if toolchain and Path(toolchain).is_file() else {}
    )
    return {
        "harness_source_sha256": harness_hash,
        "runtime_environment": allowlisted,
        "llamacpp_commit": lock.get("llamacpp_commit"),
        "toolchain_sha256": file_hash(Path(toolchain))
        if toolchain and Path(toolchain).is_file()
        else None,
        "os": platform.platform(),
        "kernel": platform.release(),
        "python": platform.python_version(),
        "cpu_model": cpu_model,
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(),
        "ram_bytes": psutil.virtual_memory().total,
        "packages": packages,
        "gpu": gpu,
        "cuda_toolkit": command(["nvcc", "--version"]),
        "nvidia_smi": command(["nvidia-smi", "-q"]),
        "ollama": executable_identity(ollama),
        "llamacpp": executable_identity(llama),
        "git": command(["git", "rev-parse", "HEAD"]),
        "git_dirty": command(["git", "status", "--porcelain"]),
    }


def environment_identity(env: dict) -> str:
    # Dynamic utilization/free memory/temperature must not invalidate a resume.
    return digest(
        {
            k: env[k]
            for k in (
                "os",
                "kernel",
                "python",
                "cpu_model",
                "physical_cores",
                "ram_bytes",
                "packages",
                "gpu",
                "ollama",
                "llamacpp",
                "harness_source_sha256",
                "runtime_environment",
                "llamacpp_commit",
                "toolchain_sha256",
            )
        }
    )

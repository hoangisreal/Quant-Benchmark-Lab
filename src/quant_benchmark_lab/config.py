"""Strict resolved configs; paths resolve relative to the YAML defining them."""

import re
from pathlib import Path
from typing import Literal

import psutil
import yaml
from pydantic import Field, model_validator

from .schema import StrictModel
from .utils import digest


class Generation(StrictModel):
    temperature: float = 0.0
    seed: int = 42
    top_k: int = 1
    top_p: float = 1.0
    min_p: float = 0.0
    repeat_penalty: float = 1.0
    repeat_last_n: int = 0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    mirostat: int = 0
    performance_max_tokens: int = Field(default=128, gt=0)
    quality_max_tokens: int = Field(default=256, gt=0)

    @model_validator(mode="after")
    def baseline(self):
        expected = {"temperature": 0, "top_k": 1, "top_p": 1, "min_p": 0,
                    "repeat_penalty": 1, "repeat_last_n": 0, "presence_penalty": 0,
                    "frequency_penalty": 0, "mirostat": 0}
        if any(getattr(self, k) != v for k, v in expected.items()):
            raise ValueError("MVP supports the explicit greedy baseline only; new profiles need a new protocol")
        if self.seed < 0:
            raise ValueError("seed must be explicit and nonnegative")
        return self


class Runtime(StrictModel):
    context_size: int = Field(default=4096, gt=0)
    threads: int = Field(default=1, gt=0)
    batch_size: int = Field(default=128, gt=0)
    microbatch_size: int = Field(default=128, gt=0)
    kv_type: Literal["f16"] = "f16"
    flash_attention: Literal[False] = False
    gpu_index: int = Field(default=0, ge=0)
    ollama_binary: str = "ollama"
    llama_binary: str = "vendor/llama.cpp/build/bin/llama-server"
    ollama_port: int = Field(default=11435, ge=1024, le=65535)
    llama_port: int = Field(default=8081, ge=1024, le=65535)
    toolchain_lock: str = "locks/toolchain.json"


class Protocol(StrictModel):
    state: Literal["draft", "frozen"] = "draft"
    cold_repetitions: int = Field(default=5, gt=0)
    warm_repetitions: int = Field(default=10, gt=0)
    warm_sessions: int = Field(default=2, gt=0)
    warmups: int = Field(default=2, ge=2)
    schedule_seed: int = 20261002
    sample_interval_ms: int = Field(default=20, ge=1)
    idle_seconds: float = Field(default=2.0, gt=0)
    loaded_seconds: float = Field(default=2.0, gt=0)
    settle_seconds: float = Field(default=1.0, ge=0)
    connect_timeout: float = Field(default=5.0, gt=0)
    activation_timeout: float = Field(default=180.0, gt=0)
    ttft_timeout: float = Field(default=120.0, gt=0)
    request_timeout: float = Field(default=600.0, gt=0)
    unload_timeout: float = Field(default=60.0, gt=0)
    cooldown_timeout: float = Field(default=180.0, gt=0)
    max_idle_utilization: float = Field(default=10.0, ge=0, le=100)
    max_temperature_c: float = Field(default=70.0, gt=0)
    baseline_tolerance_mib: float = Field(default=64.0, ge=0)
    allowed_bos_cache_tokens: int = Field(default=0, ge=0, le=1)
    cache_policy: Literal["evict_prefix"] = "evict_prefix"

    @model_validator(mode="after")
    def split(self):
        if self.warm_repetitions % self.warm_sessions:
            raise ValueError("warm repetitions must divide evenly across sessions")
        return self


class ModelSpec(StrictModel):
    id: str
    source_repo: str
    revision: str | None = None
    license: str | None = None
    quant: Literal["F16", "Q8_0", "Q4_K_M"]
    path: str
    sha256: str | None = None
    template: str
    thinking: Literal["disabled", "not_applicable"] = "not_applicable"
    stop: list[str]
    provenance_path: str | None = None

    @model_validator(mode="after")
    def valid(self):
        if self.template.count("{prompt}") != 1 or "{" in self.template.replace("{prompt}", ""):
            raise ValueError("template must have exactly one literal {prompt} placeholder")
        if self.sha256 is not None and not re.fullmatch(r"[0-9a-f]{64}", self.sha256):
            raise ValueError("invalid artifact SHA-256")
        if self.revision is not None and not re.fullmatch(r"[0-9a-f]{40}", self.revision):
            raise ValueError("source revision must be a full commit SHA")
        return self


class Cell(StrictModel):
    id: str = Field(pattern=r"^[A-Za-z0-9_-]+$")
    engine: Literal["ollama", "llamacpp"]
    model: ModelSpec


class CampaignConfig(StrictModel):
    name: str = Field(pattern=r"^[A-Za-z0-9_-]+$")
    experiment: Literal["quantization", "engine", "model"]
    synthetic: bool = False
    deployment: Literal["cpu", "gpu"] = "gpu"
    stage: Literal["synthetic", "pilot", "official"] = "pilot"
    runtime: Runtime
    generation: Generation = Field(default_factory=Generation)
    protocol: Protocol
    cells: list[Cell] = Field(min_length=2)
    performance_path: str
    quality_path: str
    answers_path: str
    preflight_path: str | None = None

    @model_validator(mode="after")
    def invariants(self):
        if self.synthetic != (self.stage == "synthetic"):
            raise ValueError("synthetic data requires synthetic stage and vice versa")
        if len({c.id for c in self.cells}) != len(self.cells):
            raise ValueError("duplicate cell ID")
        if self.stage == "official" and (self.protocol.state != "frozen" or self.deployment != "gpu"):
            raise ValueError("official campaign requires frozen protocol and GPU deployment")
        m = [c.model for c in self.cells]
        if self.experiment == "engine":
            if {c.engine for c in self.cells} != {"ollama", "llamacpp"} or len(self.cells) != 2:
                raise ValueError("engine experiment requires exactly the two serving engines")
            if m[0].model_dump() != m[1].model_dump():
                raise ValueError("engine experiment must use identical artifact and model settings")
        if self.experiment == "quantization":
            fixed = [{k: v for k, v in x.model_dump().items()
                      if k not in {"quant", "path", "sha256", "provenance_path"}} for x in m]
            if len({digest(x) for x in fixed}) != 1 or len({c.engine for c in self.cells}) != 1:
                raise ValueError("quantization experiment can vary only quant/artifact")
            if len({x.quant for x in m}) != len(m):
                raise ValueError("duplicate quant")
        if self.experiment == "model":
            if {x.quant for x in m} != {"Q4_K_M"} or {c.engine for c in self.cells} != {"llamacpp"}:
                raise ValueError("MVP model experiment uses llama.cpp Q4_K_M only")
            if len({x.id for x in m}) != len(m):
                raise ValueError("duplicate model")
        if self.runtime.context_size <= max(self.generation.performance_max_tokens,
                                            self.generation.quality_max_tokens):
            raise ValueError("context leaves no input budget")
        return self


def yaml_read(path: Path) -> dict:
    class UniqueLoader(yaml.SafeLoader):
        pass

    def construct_mapping(loader, node, deep=False):
        result = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            if key in result:
                raise ValueError(f"duplicate YAML key: {key}")
            result[key] = loader.construct_object(value_node, deep=deep)
        return result

    UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, construct_mapping)
    value = yaml.load(path.read_text(encoding="utf-8"), Loader=UniqueLoader)
    if not isinstance(value, dict):
        raise ValueError(f"expected mapping in {path}")
    return value


def resolve_path(value: str, parent: Path) -> str:
    return str((parent / value).resolve())


def load_config(path: Path) -> CampaignConfig:
    path = path.resolve()
    raw = yaml_read(path)
    catalog_file = raw.pop("catalog", None)
    for section in ("runtime", "protocol", "generation"):
        if isinstance(raw.get(section), str):
            source = (path.parent / raw[section]).resolve()
            raw[section] = yaml_read(source)
            if section == "runtime":
                for key in ("llama_binary", "toolchain_lock"):
                    if key in raw[section]:
                        raw[section][key] = resolve_path(raw[section][key], source.parent)
        if section == "runtime" and raw[section].get("threads") == "physical":
            raw[section]["threads"] = psutil.cpu_count(logical=False) or 1
    if catalog_file:
        source = (path.parent / catalog_file).resolve()
        catalog = yaml_read(source)
        for cell in raw["cells"]:
            model = dict(catalog[cell.pop("model_id")])
            quant = cell.pop("quant")
            artifact = model.pop("artifacts")[quant]
            model.update(artifact)
            model["quant"] = quant
            model["path"] = resolve_path(model["path"], source.parent)
            if model.get("provenance_path"):
                model["provenance_path"] = resolve_path(model["provenance_path"], source.parent)
            cell["model"] = model
    for key in ("performance_path", "quality_path", "answers_path", "preflight_path"):
        if raw.get(key):
            raw[key] = resolve_path(raw[key], path.parent)
    return CampaignConfig.model_validate(raw)

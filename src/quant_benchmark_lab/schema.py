"""Serializable measurement contracts. Null metrics always carry a reason."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class Metric(StrictModel):
    value: float | None = None
    unit: str
    source: str
    reason: str | None = None

    @model_validator(mode="after")
    def valid(self):
        if self.value is None and not self.reason:
            raise ValueError("missing metric requires a reason")
        if self.value is not None and (self.value < 0 or self.reason is not None):
            raise ValueError("metric must be nonnegative; a present value has no missing reason")
        return self


class StreamEvent(StrictModel):
    receipt_ns: int = Field(ge=0)
    kind: Literal["text", "thinking", "final", "error", "metadata"]
    text: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)


class TokenUsage(StrictModel):
    input_total: int | None = Field(default=None, ge=0)
    input_cached: int | None = Field(default=None, ge=0)
    input_evaluated: int | None = Field(default=None, ge=0)
    output: int | None = Field(default=None, ge=0)
    decoded: int | None = Field(default=None, ge=0)
    source: str
    reason: str | None = None

    @model_validator(mode="after")
    def valid(self):
        if self.input_total is not None and self.input_cached is not None:
            if self.input_cached > self.input_total:
                raise ValueError("cached tokens exceed total input")
            if self.input_evaluated != self.input_total - self.input_cached:
                raise ValueError("input token accounting inconsistent")
        return self


class LoadObservation(StrictModel):
    id: str
    start_ns: int
    ready_ns: int
    wall: Metric
    native: Metric
    evidence: dict[str, Any] = Field(default_factory=dict)


class RunRecord(StrictModel):
    schema_version: int = 1
    campaign_id: str
    trial_id: str
    attempt_id: str
    session_id: str
    block_id: str
    experiment: str
    cell_id: str
    engine: str
    model_id: str
    quant: str
    gguf_sha256: str | None
    phase: Literal["measurement", "warmup", "eviction", "quality"]
    run_mode: Literal["cold", "warm", "quality"]
    synthetic: bool
    config_hash: str
    environment_hash: str
    workload_hash: str
    template_hash: str
    model_revision: str | None
    settings: dict[str, Any]
    effective: dict[str, Any]
    prompt_id: str
    prompt_hash: str
    prompt_text: str
    load_id: str | None = None
    start_ns: int
    end_ns: int
    started_at_utc: str
    status: Literal["ok", "oom", "timeout", "interrupted", "invalid", "unsupported", "infeasible"]
    error: str | None = None
    exclusions: list[str] = Field(default_factory=list)
    metrics: dict[str, Metric] = Field(default_factory=dict)
    usage: TokenUsage
    stop_reason: str | None = None
    output: str = ""
    output_hash: str
    paths: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def ordered(self):
        if self.end_ns < self.start_ns:
            raise ValueError("end precedes start")
        return self

"""Append-only records and per-attempt artifacts with checksums."""

import fcntl
import json
import os
from pathlib import Path

from .schema import RunRecord
from .utils import atomic_json, canonical, digest, file_hash, sync_directory


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    result = []
    for index, line in enumerate(lines):
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            if index == len(lines) - 1 and not line.endswith("\n"):
                break  # interrupted append; earlier corrupt records never silently disappear.
            raise
    return result


JOURNALS = ("runs.jsonl", "loads.jsonl", "sessions.jsonl", "control.jsonl")
METADATA = {"manifest.json", "schedule.json", "environment.json", "resolved_config.json",
            "preflight_evidence.json", "answers.jsonl", "artifact_manifest.json", "toolchain.json"}


class ResultStore:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = None
        self._attempt_numbers = None

    def acquire(self):
        self.lock = (self.root / ".lockfile").open("a")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.lock.close()
            self.lock = None
            raise RuntimeError("campaign is already in use") from None
        try:
            if (self.root / "manifest.json").exists():
                self._verify_metadata()
            for name in JOURNALS:
                if (self.root / "journal" / name).is_dir():
                    self._journal(name, recover=True)
        except BaseException:
            self.release()
            raise

    def _journal(self, name: str, recover: bool = False):
        """Immutable write-ahead entries commit before the materialized JSONL append."""
        folder = self.root / "journal" / name
        path = self.root / name
        if not folder.is_dir() or not path.is_file():
            raise ValueError(f"missing journal: {name}")
        entries = sorted(folder.glob("*.json"))
        expected = []
        previous = None
        for index, entry in enumerate(entries):
            if entry.name != f"{index:08d}.json":
                raise ValueError(f"journal sequence gap: {name}")
            envelope = json.loads(entry.read_text())
            body = {k: envelope[k] for k in ("value", "previous")}
            if envelope["sha256"] != digest(body) or envelope["previous"] != previous:
                raise ValueError(f"journal checksum mismatch: {name}/{entry.name}")
            previous = envelope["sha256"]
            expected.append((canonical(envelope["value"]) + "\n").encode())
        raw = path.read_bytes()
        boundary = len(raw) if raw.endswith(b"\n") else raw.rfind(b"\n") + 1
        complete, tail = raw[:boundary], raw[boundary:]
        rows = complete.splitlines(keepends=True)
        if len(rows) > len(expected) or rows != expected[:len(rows)]:
            raise ValueError(f"journal content changed: {name}")
        if tail or len(rows) != len(expected):
            if not recover:
                raise ValueError(f"journal needs recovery under campaign lock: {name}")
            if tail:
                (self.root / f"{name}.torn-{digest(tail.hex())[:12]}").write_bytes(tail)
            with path.open("wb") as f:
                f.write(b"".join(expected))
                f.flush()
                os.fsync(f.fileno())
            sync_directory(self.root)
        return len(entries), previous

    def reserve_session(self, session_id: str) -> tuple[int, Path]:
        if self.lock is None:
            raise RuntimeError("session reservation requires campaign lock")
        parent = self.root / "engine-logs"
        parent.mkdir(exist_ok=True)
        sync_directory(self.root)
        number = 1
        while True:
            work = parent / f"{session_id}-a{number}"
            try:
                work.mkdir()
                sync_directory(parent)
                return number, work
            except FileExistsError:
                number += 1

    def release(self):
        if self.lock:
            fcntl.flock(self.lock, fcntl.LOCK_UN)
            self.lock.close()
            self.lock = None

    def initialize(self, manifest: dict, resume: bool = False):
        path = self.root / "manifest.json"
        if path.exists():
            if not resume:
                raise ValueError("campaign exists; use --resume")
            original = json.loads(path.read_text())
            for key in ("schedule_hash", "config_hash", "environment_hash", "synthetic"):
                if original[key] != manifest[key]:
                    raise ValueError(f"resume identity mismatch: {key}")
        elif resume:
            raise ValueError("cannot resume a nonexistent campaign")
        else:
            manifest["storage_version"] = 2
            atomic_json(path, manifest)
            for name in JOURNALS:
                (self.root / "journal" / name).mkdir(parents=True, exist_ok=False)
                (self.root / name).touch(exist_ok=False)
            sync_directory(self.root)

    def append(self, name: str, value: dict):
        if self.lock is None or name not in JOURNALS:
            raise RuntimeError("append requires a campaign lock and a known journal")
        index, previous = self._journal(name)
        body = {"previous": previous, "value": value}
        atomic_json(self.root / "journal" / name / f"{index:08d}.json",
                    {**body, "sha256": digest(body)})
        with (self.root / name).open("ab") as f:
            f.write((canonical(value) + "\n").encode())
            f.flush()
            os.fsync(f.fileno())

    def records(self) -> list[RunRecord]:
        return [RunRecord.model_validate(v) for v in read_jsonl(self.root / "runs.jsonl")]

    def attempts(self, trial_id: str) -> int:
        # Orphan artifact directories from a crash also reserve their attempt number.
        if self._attempt_numbers is None:
            self._attempt_numbers = {}
            names = [p.name for p in (self.root / "attempts").glob("*")]
            names += [r.attempt_id for r in self.records()]
            for name in names:
                if "-attempt" in name:
                    key, number = name.rsplit("-attempt", 1)
                    if number.isdigit():
                        self._attempt_numbers[key] = max(self._attempt_numbers.get(key, 0), int(number))
        return self._attempt_numbers.get(trial_id, 0)

    def save_attempt(self, record: RunRecord, events: list[dict], samples: list[dict]):
        folder = self.root / "attempts" / record.attempt_id
        self.attempts(record.trial_id)
        folder.mkdir(parents=True, exist_ok=False)
        self._attempt_numbers[record.trial_id] = int(record.attempt_id.rsplit("-attempt", 1)[1])
        for name, rows in (("events.jsonl", events), ("gpu.jsonl", samples)):
            (folder / name).write_text("".join(canonical(r) + "\n" for r in rows), encoding="utf-8")
        (folder / "output.txt").write_text(record.output, encoding="utf-8")
        (folder / "prompt.txt").write_text(record.prompt_text, encoding="utf-8")
        atomic_json(folder / "record.json", record.model_dump(exclude={"paths"}))
        paths = {p.name: str(p.relative_to(self.root)) for p in folder.iterdir()}
        atomic_json(folder / "checksums.json", {name: file_hash(self.root / rel)
                                                for name, rel in paths.items()})
        record.paths = {**paths, "checksums.json": str((folder / "checksums.json").relative_to(self.root))}
        self.append("runs.jsonl", record.model_dump())

    def _verify_metadata(self):
        manifest_path = self.root / "manifest.json"
        if not manifest_path.is_file():
            raise ValueError("not a campaign: missing manifest")
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("storage_version") != 2:
            raise ValueError("legacy campaign lacks journal integrity; create a new campaign")
        metadata = self.root / "metadata_checksums.json"
        if not metadata.is_file():
            raise ValueError("campaign metadata inventory missing")
        hashes = json.loads(metadata.read_text())
        if set(hashes) != METADATA:
            raise ValueError("campaign metadata inventory incomplete")
        for rel, expected in hashes.items():
            if file_hash(self.safe_path(rel)) != expected:
                raise ValueError(f"campaign metadata changed: {rel}")

    def verify(self):
        self._verify_metadata()
        for name in JOURNALS:
            self._journal(name)
        loads = read_jsonl(self.root / "loads.jsonl")
        load_ids = {r["observation"]["id"] for r in loads}
        if len(load_ids) != len(loads):
            raise ValueError("duplicate load event")
        for session in read_jsonl(self.root / "sessions.jsonl"):
            if file_hash(self.safe_path(session["trace_path"])) != session["trace_sha256"]:
                raise ValueError("session GPU trace changed")
            if session["load_id"] is not None and session["load_id"] not in load_ids:
                raise ValueError("session refers to missing load")
        seen = set()
        for record in self.records():
            if record.attempt_id in seen:
                raise ValueError("duplicate attempt ID")
            seen.add(record.attempt_id)
            if record.status == "ok" and record.load_id not in load_ids:
                raise ValueError("successful attempt refers to missing load")
            checksums = json.loads(self.safe_path(record.paths["checksums.json"]).read_text())
            for name, expected in checksums.items():
                if file_hash(self.safe_path(record.paths[name])) != expected:
                    raise ValueError(f"raw artifact checksum mismatch: {record.attempt_id}/{name}")
            snapshot = json.loads(self.safe_path(record.paths["record.json"]).read_text())
            if digest(snapshot) != digest(record.model_dump(exclude={"paths"})):
                raise ValueError(f"run metadata changed: {record.attempt_id}")

    def safe_path(self, rel: str) -> Path:
        path = (self.root / rel).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("artifact escapes campaign directory")
        return path

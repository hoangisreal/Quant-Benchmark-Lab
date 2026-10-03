"""Thin command-line boundary; policy lives in modules."""

import argparse
import json
import sys
from pathlib import Path

import yaml

from .config import load_config
from .environment import capture_environment
from .preflight import preflight
from .protocol import make_schedule, validate_certificate, validate_schedule
from .quality.evaluator import evaluate
from .reporting.reporter import report
from .runner import BenchmarkRunner
from .storage import ResultStore
from .utils import atomic_json, digest


def parser():
    p = argparse.ArgumentParser(prog="qbl", description="Reproducible GGUF benchmark lab")
    sub = p.add_subparsers(dest="command", required=True)
    doctor = sub.add_parser("doctor", help="capture environment; missing hardware is explicit")
    doctor.add_argument("--output", type=Path)
    doctor.add_argument("--require-gpu", action="store_true")
    doctor.add_argument(
        "--config", type=Path, help="inspect the configured binaries and full source lock"
    )
    for name in ("validate", "plan", "preflight", "freeze"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--config", type=Path, required=True)
        if name in {"plan", "preflight", "freeze"}:
            cmd.add_argument("--output", type=Path, required=True)
        if name == "plan":
            cmd.add_argument("--suite", choices=["both", "performance", "quality"], default="both")
        if name == "freeze":
            cmd.add_argument("--preflight", type=Path, required=True)
    run = sub.add_parser("run")
    run.add_argument("--schedule", type=Path, required=True)
    run.add_argument("--campaign", type=Path)
    run.add_argument("--resume", action="store_true")
    run.add_argument("--dry-run", action="store_true")
    for name in ("evaluate", "report", "audit"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--campaign", type=Path, required=True)
        if name == "report":
            cmd.add_argument("--output", type=Path)
    return p


def dispatch(args):
    if args.command == "doctor":
        if args.config:
            cfg = load_config(args.config)
            value = capture_environment(
                cfg.runtime.ollama_binary, cfg.runtime.llama_binary, cfg.runtime.toolchain_lock
            )
        else:
            value = capture_environment()
        if args.output:
            atomic_json(args.output, value)
        print(json.dumps(value, indent=2))
        return 2 if args.require_gpu and not value["gpu"]["available"] else 0
    if args.command in {"validate", "plan", "preflight", "freeze"}:
        config = load_config(args.config)
        if args.command == "validate":
            print(
                json.dumps(
                    {
                        "valid": True,
                        "stage": config.stage,
                        "synthetic": config.synthetic,
                        "config_hash": digest(config.model_dump()),
                        "note": "structural validation; hardware/artifacts need preflight",
                    }
                )
            )
        elif args.command == "plan":
            schedule = make_schedule(config, args.suite)
            atomic_json(args.output, schedule)
            print(
                json.dumps(
                    {
                        "schedule": str(args.output),
                        "planned_trials": schedule["planned_trials"],
                        "synthetic": config.synthetic,
                    }
                )
            )
        elif args.command == "preflight":
            result = preflight(config, args.output)
            print(json.dumps({k: result[k] for k in ("passed", "synthetic", "failures")}))
            return 0 if result["passed"] else 2
        else:
            evidence = json.loads(args.preflight.read_text())
            if config.synthetic or evidence.get("synthetic") or not evidence.get("passed"):
                raise ValueError("freeze requires a genuine passing hardware preflight")
            validate_certificate(config, evidence)
            value = config.model_dump()
            value["stage"] = "official"
            value["protocol"]["state"] = "frozen"
            value["preflight_path"] = str(args.preflight.resolve())
            args.output.parent.mkdir(parents=True, exist_ok=True)
            if args.output.exists():
                raise ValueError("refusing to overwrite a frozen configuration")
            args.output.write_text(yaml.safe_dump(value, sort_keys=False, allow_unicode=True))
            print(f"Frozen config: {args.output}")
        return 0
    if args.command == "run":
        schedule = json.loads(args.schedule.read_text())
        config = validate_schedule(schedule)
        if args.dry_run:
            print(
                json.dumps(
                    {
                        "planned_trials": schedule["planned_trials"],
                        "synthetic": config.synthetic,
                        "sessions": len(schedule["sessions"]),
                    }
                )
            )
            return 0
        root = args.campaign or Path("results/raw") / f"{config.name}-{digest(schedule)[:12]}"
        evidence = (
            json.loads(Path(config.preflight_path).read_text()) if config.preflight_path else None
        )
        BenchmarkRunner(schedule, root, args.resume, evidence).run()
        print(f"Campaign saved: {root}")
    elif args.command == "evaluate":
        value = evaluate(args.campaign.resolve())
        print(json.dumps({"synthetic": value["synthetic"], "summary": value["summary"]}, indent=2))
    elif args.command == "report":
        output = args.output or Path("reports") / args.campaign.name
        report(args.campaign.resolve(), output.resolve())
        print(f"Report: {output / 'report.md'}")
    elif args.command == "audit":
        store = ResultStore(args.campaign)
        store.verify()
        records = store.records()
        schedule = json.loads((store.root / "schedule.json").read_text())
        planned = {t["id"] for session in schedule["sessions"] for t in session["trials"]}
        completed = {
            r.trial_id
            for r in records
            if r.status == "ok" and r.phase in {"measurement", "quality"}
        }
        unresolved = planned - completed
        print(
            json.dumps(
                {
                    "integrity": "verified",
                    "records": len(records),
                    "trial_completion": "incomplete" if unresolved else "complete",
                    "n_planned": len(planned),
                    "n_unresolved": len(unresolved),
                    "note": "file integrity does not substitute for hardware/equivalence preflight",
                }
            )
        )
        return 2 if unresolved else 0
    return 0


def main(argv=None):
    try:
        return dispatch(parser().parse_args(argv))
    except KeyboardInterrupt:
        print(
            "Interrupted; raw attempts preserved. Resume the same frozen schedule.", file=sys.stderr
        )
        return 130
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        print(f"qbl: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())

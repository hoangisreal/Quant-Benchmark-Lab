"""Conservative parsers honor each item's output contract."""

import json
import math
import re
from decimal import Decimal

SCORER_VERSION = "1.0.1"
NUMBER = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")


def same_json(left, right):
    def numeric(value):
        return isinstance(value, (int, float, Decimal)) and not isinstance(value, bool)

    if numeric(left) and numeric(right):
        return Decimal(str(left)) == Decimal(str(right))
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(same_json(left[k], right[k]) for k in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(
            same_json(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right


def score(output: str, answer: dict) -> dict:
    kind, expected = answer["scorer"], answer["expected"]
    normalized = output.strip()
    if kind == "exact":
        if answer.get("casefold", False):
            normalized, expected = normalized.casefold(), str(expected).casefold()
        passed = normalized == str(expected)
    elif kind == "numeric":
        if not NUMBER.fullmatch(normalized):
            return {
                "score": 0.0,
                "normalized": normalized,
                "reason": "invalid/ambiguous numeric format",
            }
        value = float(normalized)
        if not math.isfinite(value):
            return {"score": 0.0, "normalized": normalized, "reason": "nonfinite number"}
        actual, target = Decimal(normalized), Decimal(str(expected))
        absolute = Decimal(str(answer.get("absolute_tolerance", 0.0)))
        relative = Decimal(str(answer.get("relative_tolerance", 0.0)))
        passed = abs(actual - target) <= max(absolute, relative * max(abs(actual), abs(target)))
        normalized = value
    elif kind == "json":

        def unique(pairs):
            d = {}
            for k, v in pairs:
                if k in d:
                    raise ValueError("duplicate JSON key")
                d[k] = v
            return d

        def finite_float(text):
            value = float(text)
            if not math.isfinite(value):
                raise ValueError("nonfinite JSON number")
            return value

        try:
            value = json.loads(
                normalized,
                object_pairs_hook=unique,
                parse_float=finite_float,
                parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s)),
            )
        except (json.JSONDecodeError, ValueError):
            return {"score": 0.0, "normalized": normalized, "reason": "invalid JSON format"}
        # JSON numeric values 1 and 1.0 are equivalent; booleans are a different type.
        # Compare original decimal literals, before binary-float rounding or underflow.
        # Keep the existing JSON-serializable normalized display value above.
        exact_value = json.loads(output.strip(), parse_float=Decimal)
        passed = same_json(exact_value, expected)
        normalized = value
    else:
        raise ValueError(f"unknown scorer: {kind}")
    return {
        "score": float(passed),
        "normalized": normalized,
        "reason": "correct" if passed else "incorrect",
    }

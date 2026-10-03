import json
import math
from pathlib import Path


def load_answers(path: Path) -> dict:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("duplicate answer key ID")
    for row in rows:
        if row.get("scorer") not in {"exact", "numeric", "json"}:
            raise ValueError("unsupported scorer")
        if "expected" not in row:
            raise ValueError("answer key lacks expected value")
        common = {"id", "scorer", "expected"}
        extras = {"exact": {"casefold"}, "numeric": {"absolute_tolerance", "relative_tolerance"},
                  "json": set()}[row["scorer"]]
        if set(row) - common - extras:
            raise ValueError("unknown answer key option")
        if row["scorer"] == "numeric":
            for key in ("expected", "absolute_tolerance", "relative_tolerance"):
                value = row.get(key, 0.0)
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError("numeric answer/tolerance must be finite numeric values")
                if key != "expected" and value < 0:
                    raise ValueError("numeric tolerance must be nonnegative")
    return {r["id"]: r for r in rows}

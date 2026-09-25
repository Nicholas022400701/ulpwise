"""Regression corpus of upstream bugs found by the ulpwise contribution pipeline.

Each case carries the repository, the merged pull request, the kind of bug, a short repro that
sets ``actual`` and ``expected``, and how to compare them. ``pytest tests/test_corpus.py`` runs
every case whose requirements are installed.
"""

from __future__ import annotations

import importlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

_CASES = Path(__file__).with_name("cases.json")


def load() -> List[Dict[str, Any]]:
    """All cases as dicts."""
    return json.loads(_CASES.read_text(encoding="utf-8"))["cases"]


def missing_requirements(case: Dict[str, Any]) -> List[str]:
    """Names in ``case['requires']`` that cannot be imported."""
    out = []
    for name in case["requires"]:
        try:
            importlib.import_module(name)
        except Exception:  # noqa: BLE001
            out.append(name)
    return out


def installed_version(case: Dict[str, Any]) -> Optional[str]:
    """Version of the last required package (the one the bug lived in)."""
    try:
        mod = importlib.import_module(case["requires"][-1])
    except Exception:  # noqa: BLE001
        return None
    return getattr(mod, "__version__", None)


def _flat(x: Any) -> List[Any]:
    if isinstance(x, (list, tuple)):
        out: List[Any] = []
        for item in x:
            out.extend(_flat(item))
        return out
    return [x]


def run(case: Dict[str, Any]) -> None:
    """Execute a case and raise ``AssertionError`` if the bug is present."""
    ns: Dict[str, Any] = {}
    exec(compile(case["repro"], f"<corpus {case['id']}>", "exec"), ns)  # noqa: S102
    check = case["check"]
    actual = _flat(ns["actual"])
    expected = _flat(ns["expected"])
    if len(actual) != len(expected):
        raise AssertionError(f"{case['id']}: {len(actual)} actual vs {len(expected)} expected values")
    kind = check["type"]
    for i, (a, e) in enumerate(zip(actual, expected)):
        if kind == "equal":
            ok = a == e
        elif kind == "sign":
            ok = math.copysign(1.0, a) == math.copysign(1.0, e) and (a != 0) == (e != 0)
        elif kind == "allclose":
            ok = math.isfinite(a) and abs(a - e) <= check.get("atol", 0.0) + check.get("rtol", 0.0) * abs(e)
        elif kind == "max_ulp":
            from ulpwise import ulp_distance  # noqa: PLC0415

            ok = math.isfinite(a) and ulp_distance(a, e, check.get("dtype", "f64")) <= check["max_ulp"]
        else:
            raise ValueError(f"unknown check type {kind!r}")
        if not ok:
            raise AssertionError(f"{case['id']}: element {i}: actual={a!r} expected={e!r} ({kind} check)")

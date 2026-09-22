#!/usr/bin/env python3
"""Compare two FlagGems test-result trees (e.g. LaneVectorize ON vs OFF).

Each result tree is expected to contain, per operator directory:
  * ``accuracy_result.json``   -- pytest node-id -> {params, result, opname, reason}
  * ``performance_result.json``-- op-name -> {details: [{dtype, result: [...]}], ...}
and optionally a top level ``summary.json`` with environment information.

The script diffs the two trees case by case and writes:

  comparison_result.json          full machine readable diff
  accuracy_diff.csv               accuracy cases that changed (pass/fail/skip)
  accuracy_op_summary.csv         per operator accuracy counts
  performance_diff.csv            every matched performance case + ratio
  performance_op_summary.csv      per operator performance aggregate
  performance_op_dtype_summary.csv per operator/dtype performance aggregate
  comparison_summary.md           human readable report
  comparison_report.html          self contained visualisation

Primary performance metric
--------------------------
``latency_ratio`` = latency_off / latency_on.

A value > 1 means the OFF tree is slower, i.e. enabling the change (ON)
makes the kernel faster.  ``speedup_* == latency_base / latency`` and
``speedup_delta = speedup_on - speedup_off``.

Usage
-----
    python tools/compare_results.py
    python tools/compare_results.py --on result-lane-vectorize-on \
        --off result-lane-vectorize-off --output-dir comparison-lane-vectorize

Stdlib only.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import html
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

DEFAULT_ON = "result-lane-vectorize-on"
DEFAULT_OFF = "result-lane-vectorize-off"
DEFAULT_OUT = "comparison-lane-vectorize"
DEFAULT_THRESHOLD = 0.05  # +-5%

# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def log(msg: str) -> None:
    print(f"[compare] {msg}", file=sys.stderr)


def load_json(path: Path):
    try:
        with path.open("r") as fh:
            return json.load(fh)
    except FileNotFoundError:
        return None
    except (json.JSONDecodeError, ValueError) as exc:
        log(f"WARNING: cannot parse {path}: {exc}")
        return None


def geomean(values) -> float | None:
    vals = [v for v in values if v is not None and v > 0]
    if not vals:
        return None
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def mean(values) -> float | None:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def median(values) -> float | None:
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    n = len(vals)
    if n % 2:
        return vals[n // 2]
    return (vals[n // 2 - 1] + vals[n // 2]) / 2


def percentile(values, q: float) -> float | None:
    """Linear-interpolated percentile (q in [0, 1])."""
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    pos = q * (len(vals) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(vals) - 1)
    return vals[lo] + (vals[hi] - vals[lo]) * (pos - lo)


def fmt(value, digits: int = 3) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def short_dtype(dtype: str) -> str:
    return (dtype or "?").replace("torch.", "")


# ---------------------------------------------------------------------------
# collectors
# ---------------------------------------------------------------------------


def collect_accuracy(root: Path) -> dict:
    """Return {op: {node_id: entry}} for every accuracy_result.json found."""
    result: dict[str, dict] = {}
    for op_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        data = load_json(op_dir / "accuracy_result.json")
        if data is None:
            continue
        result[op_dir.name] = data
    return result


def _perf_case_key(record: dict, synth_counts: Counter):
    """Stable key for a performance case.

    ``case_id`` is normally unique; a few cases have ``case_id == None`` and
    have to be keyed by their parameters plus an occurrence index.
    """
    cid = record.get("case_id")
    if cid:
        return ("cid", cid)
    shape = json.dumps(record.get("shape_detail"), sort_keys=True, default=str)
    base = (
        "syn",
        record.get("dtype"),
        record.get("mode"),
        record.get("level"),
        shape,
    )
    idx = synth_counts[base]
    synth_counts[base] += 1
    return base + (idx,)


def collect_performance(root: Path) -> dict:
    """Return {op: {"cases": {key: {...}}, "status":..., "reason":...}}."""
    result: dict[str, dict] = {}
    for op_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        data = load_json(op_dir / "performance_result.json")
        if data is None:
            continue

        cases: dict = {}
        synth_counts: Counter = Counter()
        status = None
        reason = None
        test_case = None

        for op_name, top in data.items():
            if not isinstance(top, dict):
                continue
            status = top.get("result", status)
            reason = top.get("reason", reason)
            test_case = top.get("test_case", test_case)
            for det in top.get("details", []) or []:
                dtype = det.get("dtype")
                mode = det.get("mode")
                level = det.get("level")
                for rec in det.get("result", []) or []:
                    enriched = dict(rec)
                    enriched["dtype"] = dtype
                    enriched["mode"] = mode
                    enriched["level"] = level
                    key = _perf_case_key(enriched, synth_counts)
                    cases[key] = enriched

        result[op_dir.name] = {
            "cases": cases,
            "status": status,
            "reason": reason,
            "test_case": test_case,
        }
    return result


def collect_env(root: Path) -> dict:
    summary = load_json(root / "summary.json")
    if not isinstance(summary, dict):
        return {}
    return {
        "timestamp": summary.get("timestamp"),
        "total_duration": summary.get("total_duration"),
        "env": summary.get("env", {}),
        "num_ops": len(summary.get("result", {}) or {}),
    }


# ---------------------------------------------------------------------------
# diffing
# ---------------------------------------------------------------------------

ACC_RANK = {"passed": 0, "skipped": 1, "failed": 2}


def classify_accuracy(off_res: str | None, on_res: str | None) -> str:
    if off_res is None:
        return "added"
    if on_res is None:
        return "removed"
    if off_res == on_res:
        return "unchanged"
    if off_res == "passed" and on_res == "failed":
        return "regression"
    if off_res == "failed" and on_res == "passed":
        return "fix"
    return "status_change"


def diff_accuracy(acc_on: dict, acc_off: dict):
    cases = []
    op_summary = {}
    ops = sorted(set(acc_on) | set(acc_off))
    for op in ops:
        on_entries = acc_on.get(op, {})
        off_entries = acc_off.get(op, {})
        counts: Counter = Counter()
        for node_id in sorted(set(on_entries) | set(off_entries)):
            on_entry = on_entries.get(node_id)
            off_entry = off_entries.get(node_id)
            change = classify_accuracy(
                off_entry.get("result") if off_entry else None,
                on_entry.get("result") if on_entry else None,
            )
            counts[change] += 1
            if change != "unchanged":
                cases.append(
                    {
                        "op": op,
                        "test_id": node_id,
                        "off_result": off_entry.get("result") if off_entry else None,
                        "on_result": on_entry.get("result") if on_entry else None,
                        "change": change,
                        "off_reason": (off_entry or {}).get("reason"),
                        "on_reason": (on_entry or {}).get("reason"),
                    }
                )
        op_summary[op] = {
            "off_passed": sum(
                1 for e in off_entries.values() if e.get("result") == "passed"
            ),
            "off_failed": sum(
                1 for e in off_entries.values() if e.get("result") == "failed"
            ),
            "off_skipped": sum(
                1 for e in off_entries.values() if e.get("result") == "skipped"
            ),
            "on_passed": sum(
                1 for e in on_entries.values() if e.get("result") == "passed"
            ),
            "on_failed": sum(
                1 for e in on_entries.values() if e.get("result") == "failed"
            ),
            "on_skipped": sum(
                1 for e in on_entries.values() if e.get("result") == "skipped"
            ),
            **{
                k: counts.get(k, 0)
                for k in (
                    "regression",
                    "fix",
                    "status_change",
                    "added",
                    "removed",
                    "unchanged",
                )
            },
        }
    total = Counter()
    for s in op_summary.values():
        for k in (
            "regression",
            "fix",
            "status_change",
            "added",
            "removed",
            "unchanged",
        ):
            total[k] += s[k]
    return cases, op_summary, total


def _case_display_id(rec: dict) -> str:
    cid = rec.get("case_id")
    if cid:
        return cid
    shape = json.dumps(rec.get("shape_detail"), default=str)
    return f"{short_dtype(rec.get('dtype'))}:{rec.get('mode')}:{rec.get('level')}:{shape}"


def _safe_div(a, b):
    try:
        if b in (None, 0):
            return None
        return a / b
    except (TypeError, ZeroDivisionError):
        return None


def classify_perf(ratio: float | None, threshold: float) -> str:
    if ratio is None:
        return "invalid"
    if ratio >= 1 + threshold:
        return "improved"
    if ratio <= 1 - threshold:
        return "regressed"
    return "neutral"


def diff_performance(perf_on: dict, perf_off: dict, threshold: float):
    rows = []
    op_summary = {}
    ops = sorted(set(perf_on) | set(perf_off))

    for op in ops:
        on_cases = (perf_on.get(op) or {}).get("cases", {})
        off_cases = (perf_off.get(op) or {}).get("cases", {})

        counts: Counter = Counter()
        ratios = []
        speedups_on = []
        speedups_off = []
        speedup_deltas = []
        dtype_ratios: dict[str, list] = defaultdict(list)

        for key in sorted(set(on_cases) | set(off_cases), key=lambda k: str(k)):
            on_rec = on_cases.get(key)
            off_rec = off_cases.get(key)
            if on_rec is None:
                counts["only_off"] += 1
                rows.append(
                    {
                        "op": op,
                        "dtype": short_dtype(off_rec.get("dtype")),
                        "mode": off_rec.get("mode"),
                        "level": off_rec.get("level"),
                        "case_id": _case_display_id(off_rec),
                        "shape": json.dumps(off_rec.get("shape_detail"), default=str),
                        "latency_off": off_rec.get("latency"),
                        "latency_on": None,
                        "latency_ratio": None,
                        "speedup_off": off_rec.get("speedup"),
                        "speedup_on": None,
                        "speedup_delta": None,
                        "change": "only_off",
                    }
                )
                continue
            if off_rec is None:
                counts["only_on"] += 1
                rows.append(
                    {
                        "op": op,
                        "dtype": short_dtype(on_rec.get("dtype")),
                        "mode": on_rec.get("mode"),
                        "level": on_rec.get("level"),
                        "case_id": _case_display_id(on_rec),
                        "shape": json.dumps(on_rec.get("shape_detail"), default=str),
                        "latency_off": None,
                        "latency_on": on_rec.get("latency"),
                        "latency_ratio": None,
                        "speedup_off": None,
                        "speedup_on": on_rec.get("speedup"),
                        "speedup_delta": None,
                        "change": "only_on",
                    }
                )
                continue

            lat_on = on_rec.get("latency")
            lat_off = off_rec.get("latency")
            base_on = on_rec.get("latency_base")
            base_off = off_rec.get("latency_base")
            ratio = _safe_div(lat_off, lat_on)
            spd_on = on_rec.get("speedup")
            if spd_on is None:
                spd_on = _safe_div(base_on, lat_on)
            spd_off = off_rec.get("speedup")
            if spd_off is None:
                spd_off = _safe_div(base_off, lat_off)
            delta = None
            if spd_on is not None and spd_off is not None:
                delta = spd_on - spd_off
            change = classify_perf(ratio, threshold)
            counts[change] += 1

            dtype = short_dtype(on_rec.get("dtype"))
            if ratio is not None:
                ratios.append(ratio)
                dtype_ratios[dtype].append(ratio)
            if spd_on is not None:
                speedups_on.append(spd_on)
            if spd_off is not None:
                speedups_off.append(spd_off)
            if delta is not None:
                speedup_deltas.append(delta)

            rows.append(
                {
                    "op": op,
                    "dtype": dtype,
                    "mode": on_rec.get("mode"),
                    "level": on_rec.get("level"),
                    "case_id": _case_display_id(on_rec),
                    "shape": json.dumps(on_rec.get("shape_detail"), default=str),
                    "latency_off": lat_off,
                    "latency_on": lat_on,
                    "latency_ratio": ratio,
                    "speedup_off": spd_off,
                    "speedup_on": spd_on,
                    "speedup_delta": delta,
                    "change": change,
                }
            )

        op_summary[op] = {
            "n_matched": sum(
                counts.get(k, 0) for k in ("improved", "regressed", "neutral", "invalid")
            ),
            "n_improved": counts.get("improved", 0),
            "n_regressed": counts.get("regressed", 0),
            "n_neutral": counts.get("neutral", 0),
            "n_invalid": counts.get("invalid", 0),
            "n_only_on": counts.get("only_on", 0),
            "n_only_off": counts.get("only_off", 0),
            "geomean_ratio": geomean(ratios),
            "mean_ratio": mean(ratios),
            "median_ratio": median(ratios),
            "p25_ratio": percentile(ratios, 0.25),
            "p75_ratio": percentile(ratios, 0.75),
            "min_ratio": min(ratios) if ratios else None,
            "max_ratio": max(ratios) if ratios else None,
            "mean_speedup_on": mean(speedups_on),
            "mean_speedup_off": mean(speedups_off),
            "mean_speedup_delta": mean(speedup_deltas),
            "dtypes": {
                dt: {
                    "n": len(rs),
                    "geomean_ratio": geomean(rs),
                    "mean_ratio": mean(rs),
                    "median_ratio": median(rs),
                }
                for dt, rs in sorted(dtype_ratios.items())
            },
        }

    return rows, op_summary


# ---------------------------------------------------------------------------
# writers
# ---------------------------------------------------------------------------


def write_csv(path: Path, header, rows) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)
    log(f"wrote {path}")


def csv_num(value, digits: int = 6):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return value


def write_accuracy_csvs(out: Path, acc_cases, acc_ops):
    write_csv(
        out / "accuracy_diff.csv",
        ["op", "test_id", "off_result", "on_result", "change", "off_reason", "on_reason"],
        [
            [
                c["op"],
                c["test_id"],
                c["off_result"] or "",
                c["on_result"] or "",
                c["change"],
                (c["off_reason"] or "").replace("\n", " "),
                (c["on_reason"] or "").replace("\n", " "),
            ]
            for c in sorted(acc_cases, key=lambda c: (c["op"], c["change"], c["test_id"]))
        ],
    )
    write_csv(
        out / "accuracy_op_summary.csv",
        [
            "op",
            "off_passed",
            "off_failed",
            "off_skipped",
            "on_passed",
            "on_failed",
            "on_skipped",
            "regressions",
            "fixes",
            "status_changes",
            "added",
            "removed",
            "unchanged",
        ],
        [
            [
                op,
                s["off_passed"],
                s["off_failed"],
                s["off_skipped"],
                s["on_passed"],
                s["on_failed"],
                s["on_skipped"],
                s["regression"],
                s["fix"],
                s["status_change"],
                s["added"],
                s["removed"],
                s["unchanged"],
            ]
            for op, s in sorted(acc_ops.items())
        ],
    )


def write_perf_csvs(out: Path, perf_rows, perf_ops):
    write_csv(
        out / "performance_diff.csv",
        [
            "op",
            "dtype",
            "mode",
            "level",
            "case_id",
            "shape",
            "latency_off",
            "latency_on",
            "latency_ratio",
            "speedup_off",
            "speedup_on",
            "speedup_delta",
            "change",
        ],
        [
            [
                r["op"],
                r["dtype"],
                r["mode"],
                r["level"],
                r["case_id"],
                r["shape"],
                csv_num(r["latency_off"]),
                csv_num(r["latency_on"]),
                csv_num(r["latency_ratio"]),
                csv_num(r["speedup_off"]),
                csv_num(r["speedup_on"]),
                csv_num(r["speedup_delta"]),
                r["change"],
            ]
            for r in perf_rows
        ],
    )
    write_csv(
        out / "performance_op_summary.csv",
        [
            "op",
            "n_matched",
            "n_improved",
            "n_regressed",
            "n_neutral",
            "n_invalid",
            "n_only_on",
            "n_only_off",
            "median_latency_ratio",
            "p25_latency_ratio",
            "p75_latency_ratio",
            "geomean_latency_ratio",
            "mean_latency_ratio",
            "min_latency_ratio",
            "max_latency_ratio",
            "mean_speedup_off",
            "mean_speedup_on",
            "mean_speedup_delta",
        ],
        [
            [
                op,
                s["n_matched"],
                s["n_improved"],
                s["n_regressed"],
                s["n_neutral"],
                s["n_invalid"],
                s["n_only_on"],
                s["n_only_off"],
                csv_num(s["median_ratio"]),
                csv_num(s["p25_ratio"]),
                csv_num(s["p75_ratio"]),
                csv_num(s["geomean_ratio"]),
                csv_num(s["mean_ratio"]),
                csv_num(s["min_ratio"]),
                csv_num(s["max_ratio"]),
                csv_num(s["mean_speedup_off"]),
                csv_num(s["mean_speedup_on"]),
                csv_num(s["mean_speedup_delta"]),
            ]
            for op, s in sorted(perf_ops.items())
        ],
    )
    write_csv(
        out / "performance_op_dtype_summary.csv",
        ["op", "dtype", "n", "geomean_latency_ratio", "mean_latency_ratio"],
        [
            [
                op,
                dtype,
                ds["n"],
                csv_num(ds["geomean_ratio"]),
                csv_num(ds["mean_ratio"]),
            ]
            for op, s in sorted(perf_ops.items())
            for dtype, ds in sorted(s["dtypes"].items())
        ],
    )


def write_json(out: Path, payload) -> None:
    path = out / "comparison_result.json"
    with path.open("w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    log(f"wrote {path}")


# ---------------------------------------------------------------------------
# markdown report
# ---------------------------------------------------------------------------


def write_markdown(out: Path, payload, top_n: int) -> None:
    meta = payload["meta"]
    perf = payload["performance"]
    acc = payload["accuracy"]
    lines = []

    lines.append("# FlagGems LaneVectorize comparison")
    lines.append("")
    lines.append(f"- **ON  results:** `{meta['on_dir']}`")
    lines.append(f"- **OFF results:** `{meta['off_dir']}`")
    lines.append(f"- **Threshold:** ±{meta['threshold'] * 100:.1f}%")
    lines.append(f"- **Generated:** {meta['generated']}")
    lines.append("")

    env_on = payload["environment"].get("on", {}).get("env", {})
    env_off = payload["environment"].get("off", {}).get("env", {})
    if env_on or env_off:
        lines.append("## Environment")
        lines.append("")
        lines.append("| Field | ON | OFF |")
        lines.append("| --- | --- | --- |")
        for label, getter in (
            ("Timestamp", lambda e, t: t.get("timestamp")),
            ("FlagGems", lambda e, t: e.get("flag_gems", {}).get("version")),
            ("FlagTree", lambda e, t: e.get("flagtree")),
            ("Triton", lambda e, t: e.get("triton", {}).get("version")),
            ("Torch", lambda e, t: e.get("torch", {}).get("version")),
            ("Vendor", lambda e, t: e.get("flag_gems", {}).get("vendor")),
            ("Device", lambda e, t: e.get("flag_gems", {}).get("device")),
        ):
            on_t = payload["environment"].get("on", {})
            off_t = payload["environment"].get("off", {})
            lines.append(
                f"| {label} | `{getter(env_on, on_t)}` | `{getter(env_off, off_t)}` |"
            )
        lines.append("")

    # ---- accuracy -------------------------------------------------------
    lines.append("## Accuracy")
    lines.append("")
    totals = acc["summary"]
    lines.append(f"- Cases compared: **{acc['total_cases']}**")
    lines.append(f"- Regressions (pass -> fail): **{totals['regression']}**")
    lines.append(f"- Fixes (fail -> pass): **{totals['fix']}**")
    lines.append(f"- Other status changes: {totals['status_change']}")
    lines.append(f"- Added / removed cases: {totals['added']} / {totals['removed']}")
    lines.append(f"- Unchanged: {totals['unchanged']}")
    lines.append("")

    changed_ops = [
        (op, s)
        for op, s in acc["ops"].items()
        if s["regression"] or s["fix"] or s["status_change"] or s["added"] or s["removed"]
    ]
    changed_ops.sort(
        key=lambda kv: (-kv[1]["regression"], -kv[1]["fix"], kv[0])
    )
    if changed_ops:
        lines.append("### Operators with accuracy changes")
        lines.append("")
        lines.append(
            "| Op | Regressions | Fixes | Status chg | Added | Removed |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for op, s in changed_ops:
            lines.append(
                f"| `{op}` | {s['regression']} | {s['fix']} | "
                f"{s['status_change']} | {s['added']} | {s['removed']} |"
            )
        lines.append("")

    regressions = [c for c in acc["cases"] if c["change"] == "regression"]
    if regressions:
        lines.append("### Accuracy regressions (pass -> fail)")
        lines.append("")
        lines.append("| Op | Test | Reason (ON) |")
        lines.append("| --- | --- | --- |")
        for c in regressions[:top_n]:
            reason = (c["on_reason"] or "").split("\n")[0][:160]
            lines.append(
                f"| `{c['op']}` | `{c['test_id']}` | {html.escape(reason)} |"
            )
        if len(regressions) > top_n:
            lines.append(f"\n_... and {len(regressions) - top_n} more (see CSV)._")
        lines.append("")

    # ---- performance ----------------------------------------------------
    lines.append("## Performance")
    lines.append("")
    ps = perf["summary"]
    lines.append(
        "Primary metric: **latency ratio = latency_off / latency_on** "
        "(> 1 means ON is faster)."
    )
    lines.append("")
    lines.append(f"- Matched cases: **{ps['n_matched']}**")
    lines.append(
        f"- Improved (>= +{meta['threshold'] * 100:.1f}%): **{ps['n_improved']}**"
    )
    lines.append(
        f"- Regressed (<= -{meta['threshold'] * 100:.1f}%): **{ps['n_regressed']}**"
    )
    lines.append(f"- Neutral: {ps['n_neutral']}")
    lines.append(f"- Overall median latency ratio: **{fmt(ps['median_ratio'])}** "
                 f"(p25 {fmt(ps['p25_ratio'])}, p75 {fmt(ps['p75_ratio'])})")
    lines.append(f"- Overall geomean latency ratio: {fmt(ps['geomean_ratio'])}")
    lines.append(f"- Overall mean latency ratio: {fmt(ps['mean_ratio'])}")
    lines.append(f"- Only ON / only OFF cases: {ps['n_only_on']} / {ps['n_only_off']}")
    lines.append("")

    ranked = list(perf["ops"].items())

    def top_improved(n):
        xs = [x for x in ranked if x[1]["median_ratio"] is not None and x[1]["median_ratio"] > 1]
        xs.sort(key=lambda kv: kv[1]["median_ratio"], reverse=True)
        return xs[:n]

    def top_regressed(n):
        xs = [x for x in ranked if x[1]["median_ratio"] is not None and x[1]["median_ratio"] < 1]
        xs.sort(key=lambda kv: kv[1]["median_ratio"])
        return xs[:n]

    for title, xs in (
        (f"Top {top_n} improved operators", top_improved(top_n)),
        (f"Top {top_n} regressed operators", top_regressed(top_n)),
    ):
        if not xs:
            continue
        lines.append(f"### {title}")
        lines.append("")
        lines.append(
            "| Op | Matched | Median ratio (p25-p75) | Geomean | Mean speedup OFF->ON | Improved | Regressed |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for op, s in xs:
            lines.append(
                f"| `{op}` | {s['n_matched']} | "
                f"**{fmt(s['median_ratio'])}** ({fmt(s['p25_ratio'])}-{fmt(s['p75_ratio'])}) | "
                f"{fmt(s['geomean_ratio'])} | "
                f"{fmt(s['mean_speedup_off'])} -> {fmt(s['mean_speedup_on'])} | "
                f"{s['n_improved']} | {s['n_regressed']} |"
            )
        lines.append("")

    lines.append("")
    with (out / "comparison_summary.md").open("w") as fh:
        fh.write("\n".join(lines))
    log(f"wrote {out / 'comparison_summary.md'}")


# ---------------------------------------------------------------------------
# html report
# ---------------------------------------------------------------------------

HTML_CSS = """
:root { color-scheme: light dark; }
* { box-sizing: border-box; }
body { font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
       margin: 0; padding: 24px; background: #f6f7f9; color: #1d2129; }
h1 { margin-top: 0; }
h2 { border-bottom: 2px solid #d9dde3; padding-bottom: 6px; margin-top: 36px; }
a { color: #2f6feb; }
.muted { color: #6b7280; }
.cards { display: flex; flex-wrap: wrap; gap: 12px; margin: 16px 0; }
.card { background: #fff; border: 1px solid #e3e6ea; border-radius: 10px;
        padding: 14px 18px; min-width: 150px; box-shadow: 0 1px 2px rgba(0,0,0,.04); }
.card .label { font-size: 12px; text-transform: uppercase; letter-spacing: .04em;
               color: #6b7280; }
.card .value { font-size: 26px; font-weight: 700; margin-top: 4px; }
.good { color: #137333; } .bad { color: #c5221f; } .neutral { color: #5f6368; }
table { border-collapse: collapse; width: 100%; background: #fff; margin: 12px 0;
        font-size: 13px; border: 1px solid #e3e6ea; border-radius: 8px; overflow: hidden; }
th, td { text-align: left; padding: 7px 10px; border-bottom: 1px solid #eef0f2;
         white-space: nowrap; }
th { background: #f1f3f5; position: sticky; top: 0; cursor: pointer; user-select: none; }
tr:hover td { background: #f8fafc; }
th::after { content: ""; }
th.sort-asc::after { content: " \\25B2"; }
th.sort-desc::after { content: " \\25BC"; }
.scroll { max-height: 560px; overflow: auto; border-radius: 8px; }
.controls { margin: 8px 0; }
input[type=search] { padding: 6px 10px; border: 1px solid #ccd2d9; border-radius: 6px;
                     min-width: 260px; font-size: 13px; }
.bars { background: #fff; border: 1px solid #e3e6ea; border-radius: 8px;
        padding: 14px 18px; margin: 12px 0; }
.bar-row { display: grid; grid-template-columns: 190px 1fr 90px; gap: 10px;
           align-items: center; margin: 3px 0; font-size: 13px; }
.bar-track { background: #eef0f2; border-radius: 4px; height: 16px; position: relative; }
.bar-fill { height: 100%; border-radius: 4px; }
.bar-fill.good { background: #34a853; } .bar-fill.bad { background: #ea4335; }
.bar-label { text-align: right; font-variant-numeric: tabular-nums; }
.pill { display: inline-block; padding: 1px 7px; border-radius: 10px; font-size: 11px;
        font-weight: 600; }
.pill.regression { background: #fce8e6; color: #c5221f; }
.pill.fix { background: #e6f4ea; color: #137333; }
.pill.status_change { background: #fef7e0; color: #8a6d00; }
.pill.added, .pill.removed { background: #e8f0fe; color: #1a56b0; }
.pill.improved { background: #e6f4ea; color: #137333; }
.pill.regressed { background: #fce8e6; color: #c5221f; }
.pill.neutral { background: #f1f3f5; color: #5f6368; }
.pill.only_on, .pill.only_off, .pill.invalid { background: #f3e8fd; color: #7c3aed; }
details { margin: 10px 0; }
summary { cursor: pointer; font-weight: 600; }
@media (prefers-color-scheme: dark) {
  body { background: #17191c; color: #e6e8eb; }
  h2 { border-color: #333; }
  .card, table, .bars { background: #21242a; border-color: #343941; }
  th { background: #2a2e35; }
  td, th { border-color: #2f333a; }
  tr:hover td { background: #282c33; }
  .bar-track { background: #2f333a; }
  .muted { color: #9aa0a6; }
  input[type=search] { background: #21242a; color: #e6e8eb; border-color: #3a3f47; }
}
"""

HTML_JS = """
function sortTable(table, colIndex, numeric) {
  const tbody = table.tBodies[0];
  const rows = Array.from(tbody.rows);
  const asc = table.dataset.sortCol != colIndex || table.dataset.sortDir === 'desc';
  rows.sort((a, b) => {
    const x = a.cells[colIndex].dataset.v ?? a.cells[colIndex].innerText;
    const y = b.cells[colIndex].dataset.v ?? b.cells[colIndex].innerText;
    let cmp;
    if (numeric) { cmp = parseFloat(x || 'NaN') - parseFloat(y || 'NaN'); }
    else { cmp = String(x).localeCompare(String(y)); }
    return asc ? cmp : -cmp;
  });
  rows.forEach(r => tbody.appendChild(r));
  table.dataset.sortCol = colIndex;
  table.dataset.sortDir = asc ? 'asc' : 'desc';
  table.querySelectorAll('th').forEach((th, i) => {
    th.classList.toggle('sort-asc', asc && i === colIndex);
    th.classList.toggle('sort-desc', !asc && i === colIndex);
  });
}
function filterTable(input, tableId) {
  const q = input.value.toLowerCase();
  const table = document.getElementById(tableId);
  for (const row of table.tBodies[0].rows) {
    row.style.display = row.innerText.toLowerCase().includes(q) ? '' : 'none';
  }
}
"""


def _pill(value) -> str:
    if value is None:
        return "<span class='muted'>-</span>"
    return f"<span class='pill {html.escape(str(value))}'>{html.escape(str(value))}</span>"


def _num(value, digits=3) -> str:
    if value is None:
        return "<span class='muted'>-</span>"
    return f"{value:.{digits}g}"


def _ratio_class(ratio) -> str:
    if ratio is None:
        return "neutral"
    if ratio > 1.0001:
        return "good"
    if ratio < 0.9999:
        return "bad"
    return "neutral"


def _bar_chart(title: str, ops, top_n: int) -> str:
    if not ops:
        return ""
    max_abs = max(abs(s["median_ratio"] - 1) for _, s in ops) or 1.0
    rows = []
    for op, s in ops:
        delta = s["median_ratio"] - 1
        width = min(100.0, abs(delta) / max_abs * 100.0)
        cls = "good" if delta >= 0 else "bad"
        rows.append(
            "<div class='bar-row'>"
            f"<span title='{html.escape(op)}'>{html.escape(op[:26])}</span>"
            "<div class='bar-track'>"
            f"<div class='bar-fill {cls}' style='width:{width:.1f}%'></div>"
            "</div>"
            f"<span class='bar-label'>{s['median_ratio']:.3f}x</span>"
            "</div>"
        )
    return (
        f"<h3>{html.escape(title)}</h3><div class='bars'>" + "".join(rows) + "</div>"
    )


def write_html(out: Path, payload, top_n: int) -> None:
    meta = payload["meta"]
    perf = payload["performance"]
    acc = payload["accuracy"]
    ps = perf["summary"]
    at = acc["summary"]

    parts = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        "<title>FlagGems LaneVectorize comparison</title>",
        f"<style>{HTML_CSS}</style>",
        f"<script>{HTML_JS}</script>",
        "</head><body>",
        "<h1>FlagGems LaneVectorize comparison</h1>",
        f"<p class='muted'>ON: <code>{html.escape(meta['on_dir'])}</code> &nbsp;|&nbsp; "
        f"OFF: <code>{html.escape(meta['off_dir'])}</code><br>"
        f"Threshold ±{meta['threshold'] * 100:.1f}% &nbsp;|&nbsp; "
        f"Generated {html.escape(str(meta['generated']))}</p>",
    ]

    # cards
    parts.append("<div class='cards'>")
    for label, value, cls in (
        ("Matched perf cases", ps["n_matched"], ""),
        ("Median latency ratio", fmt(ps["median_ratio"]), _ratio_class(ps["median_ratio"])),
        ("Geomean latency ratio", fmt(ps["geomean_ratio"]), _ratio_class(ps["geomean_ratio"])),
        ("Improved", ps["n_improved"], "good"),
        ("Regressed", ps["n_regressed"], "bad"),
        ("Neutral", ps["n_neutral"], "neutral"),
        ("Accuracy regressions", at["regression"], "bad"),
        ("Accuracy fixes", at["fix"], "good"),
    ):
        parts.append(
            f"<div class='card'><div class='label'>{label}</div>"
            f"<div class='value {cls}'>{value}</div></div>"
        )
    parts.append("</div>")

    parts.append(
        "<p>Primary metric: <b>latency ratio = latency_off / latency_on</b>. "
        "Values &gt; 1 mean the ON tree is faster.</p>"
    )

    # charts
    ranked = list(perf["ops"].items())
    improved = sorted(
        [x for x in ranked if x[1]["median_ratio"] and x[1]["median_ratio"] > 1],
        key=lambda kv: kv[1]["median_ratio"],
        reverse=True,
    )[:top_n]
    regressed = sorted(
        [x for x in ranked if x[1]["median_ratio"] and x[1]["median_ratio"] < 1],
        key=lambda kv: kv[1]["median_ratio"],
    )[:top_n]
    parts.append("<h2>Top operator movements</h2>")
    parts.append(_bar_chart(f"Top {top_n} improved (median latency ratio)", improved, top_n))
    parts.append(_bar_chart(f"Top {top_n} regressed (median latency ratio)", regressed, top_n))

    # per-op perf table
    parts.append("<h2>Performance by operator</h2>")
    parts.append(
        "<div class='controls'><input type='search' "
        "placeholder='filter operators...' "
        "oninput=\"filterTable(this,'perfTable')\"></div>"
    )
    parts.append("<div class='scroll'><table id='perfTable'><thead><tr>")
    perf_header = [
        ("Op", 0, False),
        ("Matched", 1, True),
        ("Median ratio", 2, True),
        ("p25", 3, True),
        ("p75", 4, True),
        ("Geomean", 5, True),
        ("Mean", 6, True),
        ("Speedup OFF", 7, True),
        ("Speedup ON", 8, True),
        ("Delta", 9, True),
        ("Improved", 10, True),
        ("Regressed", 11, True),
        ("Neutral", 12, True),
    ]
    for label, idx, num in perf_header:
        parts.append(
            f"<th onclick=\"sortTable(document.getElementById('perfTable'),"
            f"{idx},{str(num).lower()})\">{label}</th>"
        )
    parts.append("</tr></thead><tbody>")
    for op, s in sorted(
        perf["ops"].items(),
        key=lambda kv: (
            kv[1]["median_ratio"] if kv[1]["median_ratio"] is not None else 9e9
        ),
    ):
        parts.append("<tr>")
        parts.append(f"<td><code>{html.escape(op)}</code></td>")
        parts.append(f"<td data-v='{s['n_matched']}'>{s['n_matched']}</td>")
        parts.append(
            f"<td data-v='{s['median_ratio'] or 0}' class='{_ratio_class(s['median_ratio'])}'>"
            f"<b>{_num(s['median_ratio'])}</b></td>"
        )
        parts.append(f"<td data-v='{s['p25_ratio'] or 0}'>{_num(s['p25_ratio'])}</td>")
        parts.append(f"<td data-v='{s['p75_ratio'] or 0}'>{_num(s['p75_ratio'])}</td>")
        parts.append(
            f"<td data-v='{s['geomean_ratio'] or 0}' class='{_ratio_class(s['geomean_ratio'])}'>"
            f"{_num(s['geomean_ratio'])}</td>"
        )
        parts.append(f"<td data-v='{s['mean_ratio'] or 0}'>{_num(s['mean_ratio'])}</td>")
        parts.append(f"<td data-v='{s['mean_speedup_off'] or 0}'>{_num(s['mean_speedup_off'])}</td>")
        parts.append(f"<td data-v='{s['mean_speedup_on'] or 0}'>{_num(s['mean_speedup_on'])}</td>")
        parts.append(f"<td data-v='{s['mean_speedup_delta'] or 0}'>{_num(s['mean_speedup_delta'])}</td>")
        parts.append(f"<td data-v='{s['n_improved']}' class='good'>{s['n_improved']}</td>")
        parts.append(f"<td data-v='{s['n_regressed']}' class='bad'>{s['n_regressed']}</td>")
        parts.append(f"<td data-v='{s['n_neutral']}'>{s['n_neutral']}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table></div>")

    # accuracy regressions
    regressions = [c for c in acc["cases"] if c["change"] == "regression"]
    parts.append("<h2>Accuracy regressions (pass &rarr; fail)</h2>")
    if not regressions:
        parts.append("<p class='good'>No accuracy regressions.</p>")
    else:
        parts.append(
            f"<p class='muted'>{len(regressions)} regressions across "
            f"{len({c['op'] for c in regressions})} operators.</p>"
        )
        parts.append("<div class='scroll'><table id='accTable'><thead><tr>")
        for label in ("Op", "Test", "OFF", "ON", "Reason (ON)"):
            parts.append(f"<th>{label}</th>")
        parts.append("</tr></thead><tbody>")
        for c in regressions:
            reason = (c["on_reason"] or "").split("\n")[0][:300]
            parts.append(
                "<tr>"
                f"<td><code>{html.escape(c['op'])}</code></td>"
                f"<td><code>{html.escape(c['test_id'])}</code></td>"
                f"<td>{_pill(c['off_result'])}</td>"
                f"<td>{_pill(c['on_result'])}</td>"
                f"<td class='muted'>{html.escape(reason)}</td>"
                "</tr>"
            )
        parts.append("</tbody></table></div>")

    # accuracy workspaces with any change
    changed_ops = [
        (op, s)
        for op, s in acc["ops"].items()
        if s["regression"] or s["fix"] or s["status_change"] or s["added"] or s["removed"]
    ]
    if changed_ops:
        changed_ops.sort(key=lambda kv: (-kv[1]["regression"], -kv[1]["fix"], kv[0]))
        parts.append("<h2>Accuracy changes by operator</h2>")
        parts.append("<div class='scroll'><table><thead><tr>")
        for label in ("Op", "Regressions", "Fixes", "Status chg", "Added", "Removed"):
            parts.append(f"<th>{label}</th>")
        parts.append("</tr></thead><tbody>")
        for op, s in changed_ops:
            parts.append(
                "<tr>"
                f"<td><code>{html.escape(op)}</code></td>"
                f"<td class='bad'>{s['regression']}</td>"
                f"<td class='good'>{s['fix']}</td>"
                f"<td>{s['status_change']}</td>"
                f"<td>{s['added']}</td>"
                f"<td>{s['removed']}</td>"
                "</tr>"
            )
        parts.append("</tbody></table></div>")

    parts.append(
        "<p class='muted'>Raw data: "
        "<code>performance_diff.csv</code>, <code>accuracy_diff.csv</code>, "
        "<code>performance_op_summary.csv</code>, "
        "<code>accuracy_op_summary.csv</code>, <code>comparison_result.json</code>.</p>"
    )
    parts.append("</body></html>")

    path = out / "comparison_report.html"
    with path.open("w") as fh:
        fh.write("".join(parts))
    log(f"wrote {path}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--on", default=DEFAULT_ON, help="LaneVectorize ON result tree")
    parser.add_argument("--off", default=DEFAULT_OFF, help="LaneVectorize OFF result tree")
    parser.add_argument("--output-dir", default=DEFAULT_OUT, help="output directory")
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="relative latency threshold to call a change (default 0.05)",
    )
    parser.add_argument("--top", type=int, default=25, help="rows in charts/tables")
    args = parser.parse_args(argv)

    on_dir = Path(args.on)
    off_dir = Path(args.off)
    out = Path(args.output_dir)
    if not on_dir.is_dir():
        log(f"ERROR: ON directory not found: {on_dir}")
        return 1
    if not off_dir.is_dir():
        log(f"ERROR: OFF directory not found: {off_dir}")
        return 1
    out.mkdir(parents=True, exist_ok=True)

    log(f"loading accuracy from {on_dir} / {off_dir}")
    acc_on = collect_accuracy(on_dir)
    acc_off = collect_accuracy(off_dir)
    log(f"loading performance from {on_dir} / {off_dir}")
    perf_on = collect_performance(on_dir)
    perf_off = collect_performance(off_dir)

    acc_cases, acc_ops, acc_totals = diff_accuracy(acc_on, acc_off)
    perf_rows, perf_ops = diff_performance(perf_on, perf_off, args.threshold)

    matched_ratios = [
        r["latency_ratio"] for r in perf_rows if r["latency_ratio"] is not None
    ]
    matched_rows = [
        r for r in perf_rows if r["change"] in ("improved", "regressed", "neutral", "invalid")
    ]
    perf_summary = {
        "n_matched": len(matched_rows),
        "n_improved": sum(1 for r in matched_rows if r["change"] == "improved"),
        "n_regressed": sum(1 for r in matched_rows if r["change"] == "regressed"),
        "n_neutral": sum(1 for r in matched_rows if r["change"] == "neutral"),
        "n_invalid": sum(1 for r in matched_rows if r["change"] == "invalid"),
        "n_only_on": sum(1 for r in perf_rows if r["change"] == "only_on"),
        "n_only_off": sum(1 for r in perf_rows if r["change"] == "only_off"),
        "geomean_ratio": geomean(matched_ratios),
        "mean_ratio": mean(matched_ratios),
        "median_ratio": median(matched_ratios),
        "p25_ratio": percentile(matched_ratios, 0.25),
        "p75_ratio": percentile(matched_ratios, 0.75),
    }
    # per-dtype global summary
    dtype_ratios: dict[str, list] = defaultdict(list)
    for r in matched_rows:
        if r["latency_ratio"] is not None:
            dtype_ratios[r["dtype"]].append(r["latency_ratio"])
    perf_summary["dtypes"] = {
        dt: {"n": len(rs), "geomean_ratio": geomean(rs), "mean_ratio": mean(rs)}
        for dt, rs in sorted(dtype_ratios.items())
    }

    acc_total_cases = sum(
        s["regression"] + s["fix"] + s["status_change"] + s["added"] + s["removed"] + s["unchanged"]
        for s in acc_ops.values()
    )

    payload = {
        "meta": {
            "on_dir": str(on_dir),
            "off_dir": str(off_dir),
            "threshold": args.threshold,
            "generated": _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
        "environment": {
            "on": collect_env(on_dir),
            "off": collect_env(off_dir),
        },
        "accuracy": {
            "total_cases": acc_total_cases,
            "summary": dict(acc_totals),
            "ops": acc_ops,
            "cases": acc_cases,
        },
        "performance": {
            "summary": perf_summary,
            "ops": perf_ops,
            "cases": perf_rows,
        },
    }

    write_json(out, payload)
    write_accuracy_csvs(out, acc_cases, acc_ops)
    write_perf_csvs(out, perf_rows, perf_ops)
    write_markdown(out, payload, args.top)
    write_html(out, payload, args.top)

    # console recap
    print()
    print("=" * 68)
    print("ACCURACY")
    print(
        f"  regressions(pass->fail)={acc_totals['regression']}  "
        f"fixes(fail->pass)={acc_totals['fix']}  "
        f"status_changes={acc_totals['status_change']}  "
        f"added={acc_totals['added']}  removed={acc_totals['removed']}"
    )
    print("PERFORMANCE  (latency ratio = off/on, >1 = ON faster)")
    print(
        f"  matched={perf_summary['n_matched']}  "
        f"improved={perf_summary['n_improved']}  "
        f"regressed={perf_summary['n_regressed']}  "
        f"neutral={perf_summary['n_neutral']}"
    )
    print(f"  median ratio={fmt(perf_summary['median_ratio'])} "
          f"(p25 {fmt(perf_summary['p25_ratio'])}, p75 {fmt(perf_summary['p75_ratio'])})")
    print(f"  geomean ratio={fmt(perf_summary['geomean_ratio'])}")
    print("=" * 68)
    print(f"Outputs written to: {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Interleaved LaneVectorize ON/OFF A/B runner.

For every operator this runs BOTH compiler configurations back-to-back on the
SAME NPU, accuracy then benchmark, so ambient/thermal drift is shared and a
whole op is never split across devices. Ops are distributed across all detected
NPUs.

    OFF : TRITON_DISABLE_LANE_VECTORIZE=1
    ON  : TRITON_DISABLE_LANE_VECTORIZE=0
          TRITON_ENABLE_LANE_VECTORIZE_BLOCK_MODE=1

Every compilation is forced fresh (TRITON_ALWAYS_COMPILE=1) and dumped into a
per-config, per-run cache (TRITON_CACHE_DIR); nothing is pruned. TTIR is
<out>/<config>/cache/<hash>/<kernel>.ttir (plus the other stages). Dumps are
printed in MLIR generic op form by default via TRITON_MLIR_PRINT_OP_GENERIC=1
(equivalent to --mlir-print-op-generic); pass --no-generic-ir to keep the
custom printer.

Benchmarks use ``--metrics latency`` only, so the torch/native baseline is
never timed (no latency_base / speedup / tflops / gbps). Accuracy keeps
``--ref cpu`` (the torch CPU reference; there is no non-torch reference in the
test harness).

Retries
-------
Every test case whose result is not ``passed`` (or ``skipped``, which is
deterministic and not retried unless --retry-skipped is given) is a failure.
The first run records which cases failed; each failing case is then retried
individually until it has run at most --max-runs times total (default 5,
including the first run) on the same NPU/config. A benchmark run that crashes
or reports a failed operator is retried whole, also capped at --max-runs total.
Every failed attempt is appended to <out>/retries.<npu>.jsonl, even when a
later attempt succeeds, and a <out>/retry_summary.json is written at the end.

At the end ``tools/compare_results.py`` is invoked unless --no-compare.

Examples
--------
    # whole suite, all NPUs
    python tools/run_ab_interleaved.py --gpus all

    # a subset, single NPU
    python tools/run_ab_interleaved.py --ops "add,sum_dim,flip" --gpus 0

    # just show the plan
    python tools/run_ab_interleaved.py --dry-run
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import multiprocessing as mp
import os
import queue
import subprocess
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
ROOT = TOOLS_DIR.parent
sys.path.insert(0, str(TOOLS_DIR))

import run_tests as rt  # noqa: E402  (reuse env probe + operator inventory)

# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

TIMEOUT = -100
DEFAULT_TIMEOUT = 1800

# Same map as run_tests.get_env().
VENDOR_DEVICE_VARS = {
    "ascend": ["ASCEND_RT_VISIBLE_DEVICES", "NPU_VISIBLE_DEVICES"],
    "hygon": ["HIP_VISIBLE_DEVICES"],
    "metax": ["MACA_VISIBLE_DEVICES"],
    "mthreads": ["MUSA_VISIBLE_DEVICES"],
    "tsingmicro": ["TXDA_VISIBLE_DEVICES"],
    "iluvatar": ["ILUVATAR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"],
    "thead": ["CUDA_VISIBLE_DEVICES"],
    "cambricon": ["MLU_VISIBLE_DEVICES"],
    "kunlunxin": ["CUDA_VISIBLE_DEVICES"],
    "sunrise": ["TANG_VISIBLE_DEVICES"],
    "enflame": ["TOPS_VISIBLE_DEVICES"],
}

# The two configurations. ``off`` is the full pass off; ``on`` keeps the pass
# on with block mode explicitly enabled.
CONFIGS = {
    "off": {
        "TRITON_DISABLE_LANE_VECTORIZE": "1",
    },
    "on": {
        "TRITON_DISABLE_LANE_VECTORIZE": "0",
        "TRITON_ENABLE_LANE_VECTORIZE_BLOCK_MODE": "1",
    },
}

# Optional pass guards (defaults are the safe/guarded values on the flagtree
# side). Kept out of CONFIGS so the A/B only differs by the block-mode toggle.
PASS_ENV_NAMES = [
    "TRITON_DISABLE_LANE_VECTORIZE",
    "TRITON_ENABLE_LANE_VECTORIZE_BLOCK_MODE",
    "TRITON_LANE_VECTORIZE_ALLOW_CONCAT",
    "TRITON_LANE_VECTORIZE_ALLOW_ADDRESS_CONES",
    "TRITON_LANE_VECTORIZE_ALLOW_FP_CROSS_LANE",
    "TRITON_MLIR_PRINT_OP_GENERIC",
]


def now() -> str:
    return _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_json(path: Path):
    try:
        with path.open("r") as fh:
            return json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError, ValueError):
        return None


def lookup_case(retry_data: dict, key: str):
    """Find a case in a retry report, tolerating with/without the tests/ prefix."""
    if key in retry_data:
        return retry_data[key]
    return retry_data.get(strip_tests_prefix(key))


# ---------------------------------------------------------------------------
# per-invocation plumbing
# ---------------------------------------------------------------------------


def this_pkg_env() -> dict:
    """Return the installed FlagGems/FlagGems-vllm env block, repo-agnostic."""
    for key in ("flaggems_vllm", "flag_gems"):
        env = rt.ENV_INFO.get(key)
        if env:
            return env
    return {}


def device_vars() -> list[str]:
    vendor = this_pkg_env().get("vendor", "")
    return VENDOR_DEVICE_VARS.get(vendor) or ["CUDA_VISIBLE_DEVICES"]


def build_env(npu: int, config: str, cache_dir: Path,
              device_var_names: list[str], generic_ir: bool = True) -> dict:
    env = os.environ.copy()
    for var in device_var_names:
        env[var] = str(npu)
    for name in PASS_ENV_NAMES:
        env.pop(name, None)
    for name, value in CONFIGS[config].items():
        env[name] = value
    # Force a fresh compilation and keep every dumped stage in this run's cache.
    env["TRITON_ALWAYS_COMPILE"] = "1"
    env["TRITON_CACHE_DIR"] = str(cache_dir)
    # Print dumped stage IR (including .ttir) in MLIR generic op form, i.e.
    # --mlir-print-op-generic, for canonical/diffable dumps.
    env["TRITON_MLIR_PRINT_OP_GENERIC"] = "1" if generic_ir else "0"
    return env


def run_cmd(cmd: list[str], cwd: Path, env: dict, timeout: int,
            so_path: Path, se_path: Path) -> int:
    ensure_dir(so_path.parent)
    with so_path.open("w") as so, se_path.open("w") as se:
        try:
            proc = subprocess.run(
                cmd, cwd=str(cwd), env=env, stdout=so, stderr=se,
                timeout=timeout,
            )
            return proc.returncode
        except subprocess.TimeoutExpired:
            se.write(f"\n[TIMEOUT after {timeout}s]\n")
            return TIMEOUT
        except Exception as exc:  # noqa: BLE001
            se.write(f"\n[launch error] {exc}\n")
            return -1


# ---------------------------------------------------------------------------
# result classification
# ---------------------------------------------------------------------------


def acc_case_failed(entry: dict) -> bool:
    return entry.get("result") not in ("passed", "skipped")


def acc_failed_keys(data: dict) -> list[str]:
    return [k for k, v in data.items() if acc_case_failed(v)]


def bench_success(data: dict) -> bool:
    if not data:
        return False
    for op_entry in data.values():
        if not isinstance(op_entry, dict):
            continue
        if op_entry.get("result") == "failed":
            return False
        for dtype_block in op_entry.get("details", []) or []:
            for rec in dtype_block.get("result", []) or []:
                if rec.get("error_msg"):
                    return False
    return True


# ---------------------------------------------------------------------------
# accuracy
# ---------------------------------------------------------------------------


def strip_tests_prefix(node_id: str) -> str:
    return node_id[len("tests/"):] if node_id.startswith("tests/") else node_id


def accuracy_cmd(marker: str, output: Path, ref_cpu: bool, quick: bool) -> list[str]:
    cmd = ["pytest", "-m", marker, "--record", "json", "--output", str(output)]
    if ref_cpu:
        cmd += ["--ref", "cpu"]
    if quick:
        cmd += ["--quick"]
    cmd += ["--continue-on-collection-errors", "-vs"]
    return cmd


def run_accuracy_once(op, marker, config, npu, work: dict, attempt: int):
    """One whole-op accuracy run. Returns (returncode, data-or-None, raw_path)."""
    op_dir = work["results_dir"] / op
    ensure_dir(op_dir)
    raw = op_dir / f"_accuracy_{config}_attempt{attempt}.json"
    if raw.exists():
        raw.unlink()
    so = op_dir / f"accuracy_{config}_a{attempt}.stdout.log"
    se = op_dir / f"accuracy_{config}_a{attempt}.stderr.log"
    env = build_env(npu, config, work["cache_dir"], work["device_vars"],
                    work["generic_ir"])
    cmd = accuracy_cmd(marker, raw, work["ref_cpu"], work["quick"])
    rc = run_cmd(cmd, ROOT / "tests", env, work["timeout"], so, se)
    return rc, read_json(raw), raw


def run_accuracy_cases(op, config, npu, work: dict, node_ids: list[str], attempt: int):
    """Retry a specific set of accuracy node ids in one pytest invocation."""
    op_dir = work["results_dir"] / op
    raw = op_dir / f"_accuracy_{config}_retry{attempt}.json"
    if raw.exists():
        raw.unlink()
    so = op_dir / f"accuracy_{config}_retry{attempt}.stdout.log"
    se = op_dir / f"accuracy_{config}_retry{attempt}.stderr.log"
    env = build_env(npu, config, work["cache_dir"], work["device_vars"],
                    work["generic_ir"])
    rel = [strip_tests_prefix(k) for k in node_ids]
    cmd = ["pytest", *rel, "--record", "json", "--output", str(raw)]
    if work["ref_cpu"]:
        cmd += ["--ref", "cpu"]
    if work["quick"]:
        cmd += ["--quick"]
    cmd += ["-vs"]
    rc = run_cmd(cmd, ROOT / "tests", env, work["timeout"], so, se)
    return rc, (read_json(raw) or {}), raw


def process_accuracy(op, marker, config, npu, work: dict, log_retry) -> dict:
    data = None
    whole_attempts = 0
    # Whole-op infrastructure retries (each attempt counts toward the run budget).
    for attempt in range(1, work["max_runs"] + 1):
        whole_attempts = attempt
        rc, data, _ = run_accuracy_once(op, marker, config, npu, work, attempt)
        if data:
            break
        log_retry({
            "op": op, "config": config, "stage": "accuracy",
            "unit": "op", "attempt": attempt, "returncode": rc,
            "reason": "no_result_json",
        })
    if not data:
        return {}

    # Per-case retries on the same NPU/config, bounded by the remaining budget so
    # a case runs at most max_runs times in total.
    remaining_budget = max(0, work["max_runs"] - whole_attempts)
    failed = acc_failed_keys(data)
    if not work["retry_skipped"]:
        failed = [k for k in failed if data[k].get("result") != "skipped"]
    remaining = {k: data.get(k, {}) for k in failed}
    attempt = 0
    while remaining and attempt < remaining_budget:
        attempt += 1
        rc, retry_data, _ = run_accuracy_cases(
            op, config, npu, work, list(remaining.keys()), attempt
        )
        for key in list(remaining):
            rec = lookup_case(retry_data, key)
            if rec is None:
                log_retry({
                    "op": op, "config": config, "stage": "accuracy",
                    "unit": "case", "case": key, "attempt": attempt,
                    "returncode": rc, "reason": "not_reported", "eventually": "failed",
                })
                continue
            if rec.get("result") == "passed":
                data[key] = rec
                log_retry({
                    "op": op, "config": config, "stage": "accuracy",
                    "unit": "case", "case": key, "attempt": attempt,
                    "returncode": rc, "reason": "failed_before", "eventually": "passed",
                })
                del remaining[key]
            else:
                data[key] = rec
                log_retry({
                    "op": op, "config": config, "stage": "accuracy",
                    "unit": "case", "case": key, "attempt": attempt,
                    "returncode": rc, "reason": "still_failing", "eventually": "failed",
                })
    return data


# ---------------------------------------------------------------------------
# benchmark
# ---------------------------------------------------------------------------


def benchmark_cmd(marker: str, output: Path, level: str, metrics: str | None) -> list[str]:
    cmd = [
        "pytest", "-m", marker, "--level", level, "--record", "json",
        "--output", str(output), "--continue-on-collection-errors",
    ]
    if metrics:
        cmd += ["--metrics", metrics]
    return cmd


def run_benchmark_once(op, marker, config, npu, work: dict, attempt: int):
    op_dir = work["results_dir"] / op
    ensure_dir(op_dir)
    raw = op_dir / f"_benchmark_{config}_attempt{attempt}.json"
    if raw.exists():
        raw.unlink()
    so = op_dir / f"performance_{config}_a{attempt}.stdout.log"
    se = op_dir / f"performance_{config}_a{attempt}.stderr.log"
    env = build_env(npu, config, work["cache_dir"], work["device_vars"],
                    work["generic_ir"])
    cmd = benchmark_cmd(marker, raw, work["level"], work["metrics"])
    rc = run_cmd(cmd, ROOT / "benchmark", env, work["timeout"], so, se)
    return rc, read_json(raw)


def process_benchmark(op, marker, config, npu, work: dict, log_retry) -> dict:
    data = None
    for attempt in range(1, work["max_runs"] + 1):
        rc, data = run_benchmark_once(op, marker, config, npu, work, attempt)
        if bench_success(data):
            if attempt > 1:
                log_retry({
                    "op": op, "config": config, "stage": "benchmark",
                    "unit": "op", "attempt": attempt, "returncode": rc,
                    "reason": "recovered", "eventually": "passed",
                })
            return data or {}
        log_retry({
            "op": op, "config": config, "stage": "benchmark",
            "unit": "op", "attempt": attempt, "returncode": rc,
            "reason": "no_result_json" if not data else "failed_result",
            "eventually": "failed",
        })
    return data or {}


# ---------------------------------------------------------------------------
# worker
# ---------------------------------------------------------------------------


def process_op(op: str, marker: str, order: list[str], npu: int,
               out_root: Path, worker_cfg: dict) -> int:
    retries: list[dict] = []

    def log_retry(rec: dict) -> None:
        rec = {"time": now(), **rec}
        retries.append(rec)

    for config in order:
        work = {
            "results_dir": out_root / config / "results",
            "cache_dir": out_root / config / "cache",
            "timeout": worker_cfg["timeout"],
            "max_runs": worker_cfg["max_runs"],
            "retry_skipped": worker_cfg["retry_skipped"],
            "ref_cpu": worker_cfg["ref_cpu"] and op not in worker_cfg["skip_cpu"],
            "quick": worker_cfg["quick"],
            "level": worker_cfg["level"],
            "metrics": worker_cfg["metrics"],
            "device_vars": worker_cfg["device_vars"],
            "generic_ir": worker_cfg["generic_ir"],
        }
        acc = process_accuracy(op, marker, config, npu, work, log_retry)
        with (work["results_dir"] / op / "accuracy_result.json").open("w") as fh:
            json.dump(acc, fh, indent=2, default=str)
        perf = process_benchmark(op, marker, config, npu, work, log_retry)
        with (work["results_dir"] / op / "performance_result.json").open("w") as fh:
            json.dump(perf, fh, indent=2, default=str)

    # Per-worker retry log (one file per NPU -> no cross-process contention).
    log_path = out_root / f"retries.{npu}.jsonl"
    with log_path.open("a") as fh:
        for rec in retries:
            fh.write(json.dumps(rec, default=str) + "\n")
    return len(retries)


def worker_proc(npu: int, work_q, status_q, out_root: Path, worker_cfg: dict) -> None:
    while True:
        item = work_q.get()  # blocking: all items are enqueued before workers start
        if item is None:
            break
        op, marker, order = item
        status_q.put(("start", npu, op))
        t0 = time.time()
        n_retries = 0
        try:
            n_retries = process_op(op, marker, order, npu, out_root, worker_cfg)
        except Exception as exc:  # noqa: BLE001
            with (out_root / f"retries.{npu}.jsonl").open("a") as fh:
                fh.write(json.dumps({
                    "time": now(), "op": op, "stage": "worker",
                    "unit": "op", "reason": "exception", "error": str(exc),
                    "eventually": "failed",
                }) + "\n")
        status_q.put(("done", npu, op, time.time() - t0, n_retries))


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--ops", default=None, help="comma-separated operator ids")
    p.add_argument("--op-list-file", default=None, help="file with one op id per line")
    p.add_argument("--stages", default="stable", help="operator stages when no --ops")
    p.add_argument("--start", default=None, help="only ops >= this id")
    p.add_argument("--gpus", default="all", help='NPU ids, e.g. "0,1,2,3", or "all"')
    p.add_argument("--output-dir", default=None, help="results root (default results/ab-<ts>)")
    p.add_argument("--level", default="core", choices=["core", "comprehensive"],
                   help="benchmark level")
    p.add_argument("--metrics", default="latency",
                   help="benchmark --metrics (default latency = no torch baseline; "
                        "'none' to use the harness default)")
    p.add_argument("--ref", default="cpu", choices=["cpu", "device"],
                   help="accuracy reference device (default cpu)")
    p.add_argument("--quick", action="store_true", help="accuracy --quick")
    p.add_argument("--max-runs", type=int, default=5,
                   help="max total runs/attempts per failing case or op, including "
                        "the first (default 5)")
    p.add_argument("--max-retries", type=int, default=None,
                   help=argparse.SUPPRESS)  # deprecated: max_runs = max_retries + 1
    p.add_argument("--retry-skipped", action="store_true",
                   help="also retry 'skipped' cases (off by default)")
    p.add_argument("--generic-ir", dest="generic_ir", action="store_true",
                   default=True,
                   help="dump stage IR in MLIR generic op form "
                        "(TRITON_MLIR_PRINT_OP_GENERIC=1; default on)")
    p.add_argument("--no-generic-ir", dest="generic_ir", action="store_false",
                   help="use the default (custom) MLIR printer for dumps")
    p.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT,
                   help="per pytest invocation timeout (s)")
    p.add_argument("--no-compare", action="store_true",
                   help="do not run compare_results.py at the end")
    p.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    return p.parse_args(argv)


def resolve_npus(spec: str) -> list[int]:
    if spec.strip().lower() == "all":
        count = int(rt.ENV_INFO.get("torch", {}).get("device_count", 0) or 0)
        if count <= 0:
            print("[ab] no NPUs detected, falling back to 1")
            return [0]
        return list(range(count))
    return [int(x) for x in spec.split(",") if x.strip() != ""]


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.max_retries is not None:  # deprecated alias
        args.max_runs = args.max_retries + 1

    # Reuse run_tests operators + marker logic and the environment probe.
    rt.OPTS = argparse.Namespace(
        ops=args.ops, op_list_file=args.op_list_file,
        stages=args.stages, start=args.start,
    )
    rt.probe_env()
    ops = rt.get_ops_to_test()
    if not ops:
        print("[ab] no operators selected")
        return 1
    skip_cpu = set(getattr(rt.CFG, "skip_cpu_tests", []))
    if hasattr(rt, "op_marker"):
        markers = {op: rt.op_marker(op) for op in ops}
    else:
        # FlagGems-vllm has no underscore marker convention: the op id is the marker.
        markers = {op: op for op in ops}

    npus = resolve_npus(args.gpus)
    out_root = Path(args.output_dir) if args.output_dir else (
        ROOT / "results" / f"ab-{_dt.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    ensure_dir(out_root)

    pkg_env = this_pkg_env()
    flagtree_version = rt.ENV_INFO.get("flagtree")
    manifest = {
        "created": now(),
        "flagtree_version": flagtree_version,
        "triton_version": rt.ENV_INFO.get("triton", {}).get("version"),
        "package_version": pkg_env.get("version"),
        "vendor": pkg_env.get("vendor"),
        "num_ops": len(ops),
        "npus": npus,
        "configs": {k: dict(v) for k, v in CONFIGS.items()},
        "benchmark": {"level": args.level, "metrics": args.metrics},
        "accuracy": {"ref": args.ref, "quick": args.quick},
        "max_runs": args.max_runs,
        "retry_skipped": args.retry_skipped,
        "generic_ir": args.generic_ir,
    }
    with (out_root / "manifest.json").open("w") as fh:
        json.dump(manifest, fh, indent=2, default=str)

    plan = []
    for i, op in enumerate(ops):
        order = ["off", "on"] if i % 2 == 0 else ["on", "off"]
        plan.append((op, markers[op], order))

    print(f"[ab] flagtree: {flagtree_version}")
    print(f"[ab] flaggems: {manifest['package_version']}  vendor={manifest['vendor']}")
    print(f"[ab] ops={len(ops)}  npus={npus}  output={out_root}")
    print(f"[ab] OFF={CONFIGS['off']}  ON={CONFIGS['on']}")
    print(f"[ab] benchmark level={args.level} metrics={args.metrics}  "
          f"accuracy ref={args.ref}  max_runs={args.max_runs}  "
          f"generic_ir={args.generic_ir}")

    if args.dry_run:
        for op, marker, order in plan[:20]:
            print(f"  {op:40s} marker={marker:40s} order={order}")
        if len(plan) > 20:
            print(f"  ... and {len(plan) - 20} more")
        return 0

    worker_cfg = {
        "timeout": args.timeout,
        "max_runs": args.max_runs,
        "retry_skipped": args.retry_skipped,
        "ref_cpu": args.ref == "cpu",
        "quick": args.quick,
        "level": args.level,
        "metrics": None if args.metrics.lower() == "none" else args.metrics,
        "skip_cpu": skip_cpu,
        "device_vars": device_vars(),
        "generic_ir": args.generic_ir,
    }

    work_q = mp.Queue()
    status_q = mp.Queue()
    for item in plan:
        work_q.put(item)
    for _ in npus:
        work_q.put(None)

    procs = [
        mp.Process(target=worker_proc, args=(npu, work_q, status_q, out_root, worker_cfg))
        for npu in npus
    ]
    for proc in procs:
        proc.start()

    done = 0
    retry_total = 0
    t_start = time.time()
    stop = threading.Event()

    def drain_status():
        nonlocal done, retry_total
        while not stop.is_set():
            try:
                msg = status_q.get(timeout=1.0)
            except queue.Empty:
                continue
            if msg[0] == "done":
                _, npu, op, dur, n_retries = msg
                done += 1
                retry_total += n_retries
                print(f"[ab] [{done}/{len(plan)}] npu{npu} {op:40s} "
                      f"{dur:7.1f}s retries={n_retries} "
                      f"elapsed={time.time() - t_start:7.0f}s", flush=True)

    drainer = threading.Thread(target=drain_status, daemon=True)
    drainer.start()
    for proc in procs:
        proc.join()
    stop.set()
    drainer.join(timeout=2)

    # Aggregate retry logs.
    cases: dict = {}
    infra: dict = defaultdict(int)
    total_records = 0
    for log_path in sorted(out_root.glob("retries.*.jsonl")):
        for line in log_path.read_text(errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            total_records += 1
            if rec.get("unit") == "case":
                key = f"{rec.get('op')}/{rec.get('config')}/{rec.get('case')}"
                state = cases.setdefault(key, {
                    "op": rec.get("op"), "config": rec.get("config"),
                    "case": rec.get("case"), "attempts": 0, "recovered": False,
                })
                state["attempts"] += 1
                if rec.get("eventually") == "passed":
                    state["recovered"] = True
            else:
                infra[f"{rec.get('stage')}/{rec.get('reason')}"] += 1
    recovered_cases = sum(1 for s in cases.values() if s["recovered"])
    still_failing_cases = sum(1 for s in cases.values() if not s["recovered"])
    retry_summary = {
        "total_retry_records": total_records,
        "recovered_cases": recovered_cases,
        "still_failing_cases": still_failing_cases,
        "cases": cases,
        "infra": dict(infra),
    }
    with (out_root / "retry_summary.json").open("w") as fh:
        json.dump(retry_summary, fh, indent=2, default=str)

    print(f"\n[ab] finished {done}/{len(plan)} ops in {time.time() - t_start:.0f}s")
    print(f"[ab] retry records: {retry_summary['total_retry_records']}  "
          f"recovered cases: {recovered_cases}  "
          f"still failing: {still_failing_cases}")

    if not args.no_compare:
        on_dir = out_root / "on" / "results"
        off_dir = out_root / "off" / "results"
        comp_dir = out_root / "comparison"
        print(f"[ab] comparing {off_dir} vs {on_dir} -> {comp_dir}")
        rc = subprocess.call([
            sys.executable, str(TOOLS_DIR / "compare_results.py"),
            "--on", str(on_dir), "--off", str(off_dir),
            "--output-dir", str(comp_dir),
        ])
        if rc != 0:
            print(f"[ab] compare_results.py exited with {rc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

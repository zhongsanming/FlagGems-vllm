#!/usr/bin/env python3
"""Reclaim disk from regenerable Triton/Ascend launcher precompiled headers.

``precompiled.h.gch`` is the big one and it accumulates: its cache key
includes the backend version hash, so every flagtree rebuild leaves a fresh
copy behind and old ones are never removed.

The built launcher shared object does not depend on the ``.gch`` (it is only
read while compiling the launcher), so deleting these never forces a kernel
recompile -- only the next launcher build is slower.

Dry-run by default; pass --delete to remove.

    python tools/clean_triton_cache.py
    python tools/clean_triton_cache.py --delete
    python tools/clean_triton_cache.py ~/.triton/cache results/ab-lv --delete
"""

import argparse
import os
from pathlib import Path

NAMES = ("precompiled.h.gch", "precompiled.h")
DEFAULT_ROOTS = (os.environ.get("TRITON_CACHE_DIR") or "~/.triton/cache",)


def human(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024 or unit == "TiB":
            return f"{n:.1f} {unit}"
        n /= 1024


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("roots", nargs="*", default=None,
                    help="cache roots (default $TRITON_CACHE_DIR or ~/.triton/cache)")
    ap.add_argument("--delete", action="store_true",
                    help="actually remove the files (default: report only)")
    args = ap.parse_args()

    roots = [Path(r).expanduser() for r in (args.roots or DEFAULT_ROOTS)]

    total = 0
    count = 0
    for root in roots:
        if not root.is_dir():
            print(f"[skip] {root} (not a directory)")
            continue
        for p in root.rglob("*"):
            if not p.is_file() or p.name not in NAMES:
                continue
            size = p.stat().st_size
            total += size
            count += 1
            if args.delete:
                p.unlink()
            else:
                print(f"  {human(size):>10s}  {p}")

    verb = "removed" if args.delete else "would remove"
    print(f"{verb} {count} file(s), {human(total)} "
          f"from {', '.join(str(r) for r in roots)}")
    if not args.delete and count:
        print("re-run with --delete to remove them")


if __name__ == "__main__":
    main()

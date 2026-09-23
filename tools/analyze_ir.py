#!/usr/bin/env python3
"""Compare LaneVectorize OFF vs ON dumped TTIR, per kernel.

The FlagTree toggles are part of the JIT cache key, so OFF and ON never share
cache directories and kernels cannot be matched by hash. This matches kernels
by file name and compares the *set* of normalized IR bodies per name, which is
what tells you whether the pass changed codegen:

  * ``on+``  = IR bodies that exist only in ON  -> the pass changed something
  * ``off+`` = IR bodies that exist only in OFF -> extra OFF specializations
               (case-coverage difference, not a codegen change)

If ``on+`` is 0 the pass produced no change: OFF skips the pass entirely, so
every OFF body is un-vectorized, and ON being a subset of OFF proves ON is too.

Normalization: SSA values (``%foo``, ``%0``) -> ``%V``, blocks (``^bb0``) ->
``^bb``, whitespace collapsed. Locations are already absent from the dumps
(generic printer with ``print_debug_info=False``).

Usage:
    python tools/analyze_ir.py <off_cache_root> <on_cache_root>
    python tools/analyze_ir.py results/off/cache results/on/cache --diffs 3
"""

import argparse
import collections
import difflib
import hashlib
import re
from pathlib import Path

SSA = re.compile(r"%[A-Za-z_0-9]+")
BB = re.compile(r"\^bb\d+")
FN_TYPE = re.compile(r"function_type = \(([^)]*)\) -> \(([^)]*)\)")
ARG_ATTRS = re.compile(r"arg_attrs = \[([^\]]*)\]")

VEC_OPS = ("tensor.reshape", "tensor.concat", "tt.broadcast",
           "tt.expand_dims", "tensor.extract_slice", "tt.join", "tt.split")


def norm(text: str) -> str:
    text = SSA.sub("%V", text)
    text = BB.sub("^bb", text)
    lines = [re.sub(r"\s+", " ", ln).strip() for ln in text.splitlines()]
    return "\n".join(ln for ln in lines if ln)


def digest(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()[:12]


def signature(text: str):
    ft = FN_TYPE.search(text)
    aa = ARG_ATTRS.search(text)
    return (ft.group(0) if ft else "", aa.group(0) if aa else "")


def vec_counts(text: str):
    return {op: text.count(op) for op in VEC_OPS if text.count(op)}


def collect(root: Path):
    out = collections.defaultdict(list)
    for p in sorted(root.rglob("*.ttir")):
        raw = p.read_text(encoding="utf-8", errors="replace")
        n = norm(raw)
        out[p.name].append(
            {"path": p, "norm": n, "dig": digest(n), "sig": signature(n)}
        )
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("off_root")
    ap.add_argument("on_root")
    ap.add_argument("--diffs", type=int, default=0,
                    help="print up to N example unified diffs for changed kernels")
    ap.add_argument("--diff-dir", default=None,
                    help="write a .diff file per changed kernel into this dir")
    args = ap.parse_args()

    off = collect(Path(args.off_root))
    on = collect(Path(args.on_root))
    names = sorted(set(off) | set(on))

    rows = []
    tot = collections.Counter()
    for name in names:
        off_v = {v["dig"]: v for v in off.get(name, [])}
        on_v = {v["dig"]: v for v in on.get(name, [])}
        same = len(set(off_v) & set(on_v))
        on_only = len(set(on_v) - set(off_v))
        off_only = len(set(off_v) - set(on_v))
        rows.append((name, len(off_v), len(on_v), same, on_only, off_only))
        tot["same"] += same
        tot["on_only"] += on_only
        tot["off_only"] += off_only
    rows.sort(key=lambda r: (r[4] + r[5]), reverse=True)

    print(f"{'kernel':50s} {'off':>5s} {'on':>5s} {'same':>5s} "
          f"{'on+':>5s} {'off+':>5s}  verdict")
    print("-" * 92)
    for name, no, nn, same, on_only, off_only in rows:
        verdict = "same" if (on_only == 0 and off_only == 0) else "CHANGED"
        print(f"{name:50s} {no:5d} {nn:5d} {same:5d} "
              f"{on_only:5d} {off_only:5d}  {verdict}")

    print()
    print(f"distinct IR bodies: same={tot['same']} "
          f"on_only={tot['on_only']} off_only={tot['off_only']}")
    changed = [r[0] for r in rows if r[4] or r[5]]
    print(f"kernels with any difference: {len(changed)}/{len(names)}")
    if tot["on_only"] == 0:
        print("=> every ON body is byte-identical to an OFF body: "
              "the pass changed NO IR")

    def pair(name):
        off_v = {v["dig"]: v for v in off.get(name, [])}
        on_v = {v["dig"]: v for v in on.get(name, [])}
        on_extra = [v for d, v in on_v.items() if d not in off_v]
        off_extra = [v for d, v in off_v.items() if d not in on_v]
        if not on_extra or not off_extra:
            return None
        ov = min(on_extra, key=lambda v: len(v["norm"]))
        cand = [x for x in off_extra if x["sig"] == ov["sig"]] or off_extra
        of = min(cand, key=lambda x: abs(len(x["norm"]) - len(ov["norm"])))
        return of, ov

    if args.diff_dir:
        out_dir = Path(args.diff_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        written = 0
        for name in changed:
            pr = pair(name)
            if not pr:
                continue
            of, ov = pr
            d = difflib.unified_diff(of["norm"].splitlines(),
                                     ov["norm"].splitlines(),
                                     "off", "on", lineterm="")
            (out_dir / f"{name}.ttir.diff").write_text("\n".join(d) + "\n")
            written += 1
        print(f"wrote {written} diffs to {out_dir}")

    if args.diffs:
        print()
        print("=" * 92)
        shown = 0
        for name in changed:
            if shown >= args.diffs:
                break
            pr = pair(name)
            if not pr:
                continue
            of, ov = pr
            print(f"\n### {name}")
            print(f"vec ops  off={vec_counts(of['norm'])}  "
                  f"on={vec_counts(ov['norm'])}")
            d = difflib.unified_diff(of["norm"].splitlines(),
                                     ov["norm"].splitlines(),
                                     "off", "on", lineterm="")
            for line in list(d)[:120]:
                print("  " + line)
            shown += 1


if __name__ == "__main__":
    main()

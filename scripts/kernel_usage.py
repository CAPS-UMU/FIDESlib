#!/usr/bin/env python3
"""Static CUDA kernel usage + runtime cross-check against gcov coverage.

gcov only instruments the host pass, so `__global__` bodies never appear in the
coverage report and kernel-heavy files look artificially cold.  This tool adds
the orthogonal, kernel-level view in two steps:

1. *Static reachability*: for every kernel defined under `src/` + `api/`, find
   references by live code -- a `<<<>>>` launch, a function-pointer hand-off,
   a launcher / macro argument, ... versus appearing only in declaration,
   definition or explicit-instantiation idioms.
2. *Runtime exercise* (optional): map every library reference site to the gcov
   hit count in a Cobertura `coverage.xml`.  A kernel whose launch sites all
   have 0 hits has a *call site* but is never actually launched -- the static
   analysis cannot see that, coverage can.  This catches kernels that are dead
   at runtime even though the code around them compiles and is reached.

Classification:
    exercised     at least one library reference line has hits > 0
    unexercised   library references exist, but none are executed (all measured
                  lines have hits == 0, or the launch line is not recorded because
                  the compiler removed the unreachable code)
    unknown       library references exist, but their files are not in the report
    dead          no references anywhere
    external      referenced only outside src/ + api/ (tests/benchmarks/examples)

Method notes
------------
- Comments are stripped, so commented-out launches do not count.
- A `__global__` declaration runs (across lines) to the terminating `{` (def) or
  `;` (forward declaration / explicit instantiation).  The kernel name is the
  first `name<template-args>(` after `__global__`.
- A bare `name(...)` call is not a kernel use: a `__global__` function can never
  be called that way, so it is a same-named `__device__`/host overload.
- Only nvcc translation units (`.cu`/`.cuh`) are scanned, so host variables that
  share a kernel's name are not mistaken for references.
- Conservative: an occurrence in a macro body, a string, or a `(void*)Name<...>`
  cast counts as a reference.  "dead" is therefore a lower bound.
- A reference line is looked up at line, line+1 and line-1 (multi-line launch
  statements); the first line present in the report wins.

Usage
-----
    python3 scripts/kernel_usage.py                       # static + default coverage/coverage.xml
    python3 scripts/kernel_usage.py --coverage-xml cov.xml
    python3 scripts/kernel_usage.py --no-coverage
    python3 scripts/kernel_usage.py --output KERNEL_USAGE.md
    python3 scripts/kernel_usage.py --json
    python3 scripts/kernel_usage.py --fail-on-dead --fail-on-unexercised

Exit status: 0 normally; 1 when a requested --fail-on-* condition holds.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

# Where kernels are *defined* / *declared*.
DEF_DIRS = ("src", "api")
# Kernel references only occur in nvcc-compiled translation units.  Scanning
# only these avoids false hits from same-named host variables in .cpp files.
REF_EXTS = (".cu", ".cuh")
EXCLUDE_DIR_PARTS = {
    ".git", "build", "build-coverage", "deps", "coverage",
    "docker", "doxygen", "nvtx", ".cache",
}

RE_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.S)
RE_LINE_COMMENT = re.compile(r"//[^\n]*")
RE_GLOBAL = re.compile(r"\b__global__\b")
# Kernel name (optionally with template arguments) immediately before '('.
RE_NAME = re.compile(r"([A-Za-z_]\w*)\s*(?:<[^;{()]*?>)?\s*\(")
RE_LAUNCH = re.compile(r"\s*(?:<[^;{()]*?>)?\s*<<<")
RE_CALL = re.compile(r"\s*(?:<[^;{()]*?>)?\s*\(")

STATUS_ORDER = ["unexercised", "dead", "external", "unknown", "exercised"]
STATUS_TITLE = {
    "exercised": "Exercised kernels",
    "unexercised": "Referenced but never executed",
    "dead": "Dead kernels — never referenced",
    "external": "Library-dead, referenced only outside `src/`+`api/`",
    "unknown": "Referenced, exercise unknown (not in coverage report)",
}


@dataclass
class Ref:
    file: str
    line: int
    kind: str  # "launch" | "address" | "handoff"
    external: bool
    hits: int | None = None  # filled in when coverage is available
    state: str = "unmeasured"  # covered | uncovered | absent | unmeasured


@dataclass
class Kernel:
    name: str
    file: str
    line: int
    decl_sites: int = 0
    refs: list[Ref] = field(default_factory=list)
    note: str = ""
    status: str = "dead"


def strip_comments(text: str) -> str:
    """Remove comments, preserving the line count so line numbers stay valid."""

    def repl(m: re.Match) -> str:
        return "\n" * m.group(0).count("\n")

    return RE_LINE_COMMENT.sub("", RE_BLOCK_COMMENT.sub(repl, text))


def iter_def_files(root: Path):
    for d in DEF_DIRS:
        base = root / d
        if not base.exists():
            continue
        for p in sorted(base.rglob("*")):
            if p.suffix in (".cu", ".cuh") and p.is_file():
                yield p


def iter_ref_files(root: Path):
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.suffix not in REF_EXTS:
            continue
        if any(part in EXCLUDE_DIR_PARTS for part in p.relative_to(root).parts):
            continue
        yield p


def find_decl_end(text: str, start: int, limit: int = 8000) -> int:
    depth = 0
    i = start
    end = min(len(text), start + limit)
    while i < end:
        ch = text[i]
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth = max(0, depth - 1)
        elif depth == 0 and ch in "{;":
            return i
        i += 1
    return end


def _first_off(text: str, name: str) -> int:
    m = re.search(rf"\b{re.escape(name)}\b", text)
    return m.start() if m else len(text)


def collect(root: Path):
    cleaned: dict[str, str] = {}
    raw: dict[str, str] = {}
    for path in iter_ref_files(root):
        rel = str(path.relative_to(root))
        text = path.read_text(errors="replace")
        raw[rel] = text
        cleaned[rel] = strip_comments(text)

    kernels: list[Kernel] = []
    spans: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for path in iter_def_files(root):
        rel = str(path.relative_to(root))
        text = cleaned[rel]
        for m in RE_GLOBAL.finditer(text):
            end = find_decl_end(text, m.end())
            nm = RE_NAME.search(text[m.end():end])
            if not nm:
                continue
            name = nm.group(1)
            line = text.count("\n", 0, m.start()) + 1
            kernels.append(Kernel(name=name, file=rel, line=line))
            spans.setdefault((rel, name), []).append((m.start(), end))

    decl_sites_by_name: dict[str, int] = {}
    for (_rel, name), found in spans.items():
        decl_sites_by_name[name] = decl_sites_by_name.get(name, 0) + len(found)

    merged: dict[str, Kernel] = {}
    for k in kernels:
        if k.name not in merged:
            merged[k.name] = k
    for name, k in merged.items():
        k.decl_sites = decl_sites_by_name.get(name, 0)

    # Scan each file once per unique kernel name, so every site is recorded once.
    for name, k in merged.items():
        for rel, text in cleaned.items():
            decl = spans.get((rel, name), [])
            for m in re.finditer(rf"\b{re.escape(name)}\b", text):
                off = m.start()
                if any(s <= off < e for s, e in decl):
                    continue
                line_end = text.find("\n", off)
                line_end = len(text) if line_end == -1 else line_end
                rest = text[m.end():line_end]
                addr_of = text[max(0, off - 1):off] == "&"
                external = rel.split("/", 1)[0] not in DEF_DIRS
                if RE_LAUNCH.match(rest):
                    kind = "launch"
                elif addr_of:
                    kind = "address"
                elif external:
                    # Outside src/+api/, only high-confidence kernel uses; a bare
                    # identifier is often a same-named host variable.
                    if not rest.lstrip().startswith("<"):
                        continue
                    kind = "handoff"
                elif RE_CALL.match(rest):
                    continue  # plain call: a different overload, not the kernel
                else:
                    kind = "handoff"
                k.refs.append(
                    Ref(file=rel, line=text.count("\n", 0, off) + 1,
                        kind=kind, external=external)
                )

    for k in merged.values():
        if any(not r.external for r in k.refs):
            continue
        # Note only matters for non-library-used kernels.
        decl_file = raw.get(k.file, "")
        commented_launch = re.search(
            rf"\b{re.escape(k.name)}\b\s*(?:<[^;{{()]*?>)?\s*<<<", decl_file
        )
        decl_text = cleaned.get(k.file, "")
        off = _first_off(decl_text, k.name)
        window = decl_text[max(0, off - 200):off + 200]
        if "[[maybe_unused]]" in window:
            k.note = "annotated `[[maybe_unused]]`"
        elif commented_launch:
            k.note = "only commented-out launch(es)"
        else:
            k.note = "declaration / instantiation only"

    return merged, len(cleaned)


def load_coverage(path: Path) -> dict[str, dict[int, int]]:
    """Parse a Cobertura XML into {filename: {line: hits}}."""
    if not path or not path.exists():
        return {}
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError:
        return {}
    cov: dict[str, dict[int, int]] = {}
    for cls in root.iter("class"):
        fn = cls.get("filename")
        if not fn:
            continue
        d = cov.setdefault(fn, {})
        for ln in cls.findall("lines/line"):
            try:
                d[int(ln.get("number"))] = int(ln.get("hits"))
            except (TypeError, ValueError):
                continue
    return cov


def site_state(cov: dict[str, dict[int, int]], ref: Ref) -> tuple[str, int | None]:
    """Coverage state of a reference line, tolerating multi-line statements.

    - "covered"/"uncovered": an executable line was recorded (hits > 0 / == 0).
    - "absent": the file is measured but the line is not in the report — for an
      executable `<<<>>>` line this means the compiler emitted no counter
      (unreachable, e.g. after an unconditional `return`), i.e. not exercised.
    - "unmeasured": the file is not in the coverage report at all.
    """
    d = cov.get(ref.file)
    if d is None:
        return "unmeasured", None
    for delta in (0, 1, -1):
        h = d.get(ref.line + delta)
        if h is not None:
            return ("covered" if h > 0 else "uncovered"), h
    return "absent", None


def classify(k: Kernel) -> str:
    if not k.refs:
        return "dead"
    lib = [r for r in k.refs if not r.external]
    if not lib:
        return "external"
    if any(r.state == "covered" for r in lib):
        return "exercised"
    if any(r.state in ("uncovered", "absent") for r in lib):
        return "unexercised"
    return "unknown"


def apply_coverage(merged: dict[str, Kernel], cov: dict[str, dict[int, int]]) -> bool:
    """Fill in per-ref state/hits and per-kernel status.  Returns True if cov was used."""
    for k in merged.values():
        for r in k.refs:
            r.state, r.hits = site_state(cov, r)
        k.status = classify(k)
    return bool(cov)


def _site_str(r: Ref) -> str:
    if r.state in ("covered", "uncovered"):
        info = f"hits={r.hits}"
    elif r.state == "absent":
        info = "not-recorded"
    else:
        info = "?"
    return f"`{r.file}:{r.line}` ({r.kind}, {info})"


def _sites_summary(refs: list[Ref], max_n: int = 4) -> str:
    seen: list[str] = []
    for r in refs:
        if r.external:
            continue
        s = _site_str(r)
        if s not in seen:
            seen.append(s)
    if len(seen) > max_n:
        return "; ".join(seen[:max_n]) + f" (+{len(seen) - max_n} more)"
    return "; ".join(seen)


def render_markdown(
    merged: dict[str, Kernel], n_files: int, coverage_used: bool, cov_path: str
) -> str:
    by_status: dict[str, list[Kernel]] = {s: [] for s in STATUS_ORDER}
    for k in merged.values():
        by_status[k.status].append(k)
    for s in by_status:
        by_status[s].sort(key=lambda k: k.name)

    out: list[str] = []
    out.append("# Static CUDA kernel usage")
    out.append("")
    out.append(
        "Generated by [`scripts/kernel_usage.py`](scripts/kernel_usage.py). gcov cannot "
        "instrument `__global__` bodies, so this is a separate static + runtime view of "
        "kernel reachability."
    )
    out.append("")
    counts = " · ".join(f"**{len(by_status[s])}** {s}" for s in STATUS_ORDER)
    out.append(
        f"Scanned {n_files} files; found **{len(merged)}** unique kernels under "
        f"`src/` + `api/`: {counts}."
    )
    if coverage_used:
        out.append(
            f"Reference sites cross-checked against "
            f"[`{cov_path}`]({cov_path}) (gcov hit counts)."
        )
    else:
        out.append(
            "No coverage report was supplied (`--coverage-xml`), so the "
            "**exercised / unexercised** split is unavailable."
        )
    out.append("")

    if by_status["unexercised"]:
        out.append("## Referenced but never executed")
        out.append("")
        out.append(
            "These kernels *have* library call sites, but none are executed in the "
            "coverage run: every call-site line has **0 hits** (or is not recorded, "
            "meaning the compiler removed unreachable code). The launch is never "
            "reached — dead path, disabled feature, or an untested configuration."
        )
        out.append("")
        out.append("| Kernel | Defined in | Library call sites |")
        out.append("|---|---|---|")
        for k in by_status["unexercised"]:
            sites = _sites_summary(k.refs)
            out.append(f"| `{k.name}` | `{k.file}:{k.line}` | {sites} |")
        out.append("")

    if by_status["dead"]:
        out.append("## Dead kernels — never referenced")
        out.append("")
        out.append("| Kernel | Defined in | Declaration sites | Notes |")
        out.append("|---|---|---:|---|")
        for k in by_status["dead"]:
            out.append(
                f"| `{k.name}` | `{k.file}:{k.line}` | {k.decl_sites} | {k.note} |"
            )
        out.append("")

    if by_status["external"]:
        out.append("## Library-dead, referenced only outside `src/`+`api/`")
        out.append("")
        out.append(
            "Not reached by any library code path; the listed references are test / "
            "benchmark / example call sites."
        )
        out.append("")
        out.append("| Kernel | Defined in | Referenced from |")
        out.append("|---|---|---|")
        for k in by_status["external"]:
            sites = sorted({f"`{r.file}:{r.line}`" for r in k.refs if r.external})
            preview = ", ".join(sites[:3])
            if len(sites) > 3:
                preview += f" (+{len(sites) - 3})"
            out.append(f"| `{k.name}` | `{k.file}:{k.line}` | {preview} |")
        out.append("")

    if by_status["unknown"]:
        out.append("## Referenced, exercise unknown")
        out.append("")
        out.append(
            "Library call sites exist but their files are not in the coverage "
            "report at all, so execution cannot be determined."
        )
        out.append("")
        for k in by_status["unknown"]:
            sites = _sites_summary(k.refs)
            out.append(f"- `{k.name}` ({k.file}:{k.line}) — {sites}")
        out.append("")

    out.append(
        "> Conservative by construction: occurrences inside macros, strings or "
        "`(void*)Name<...>` casts count as references, so **dead** is a lower bound. "
        "`unexercised` depends on the coverage run's test scope; measure it against "
        "the full suite to avoid subset artifacts."
    )
    out.append("")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", default=".", help="repository root (default: .)")
    ap.add_argument(
        "--coverage-xml",
        default="coverage/coverage.xml",
        help="Cobertura XML to cross-check call sites against (default: coverage/coverage.xml)",
    )
    ap.add_argument(
        "--no-coverage", action="store_true", help="skip the gcov cross-check"
    )
    ap.add_argument("--output", metavar="PATH", help="write the markdown report to PATH")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    ap.add_argument(
        "--fail-on-dead", action="store_true", help="exit 1 if any kernel is unreferenced"
    )
    ap.add_argument(
        "--fail-on-unexercised",
        action="store_true",
        help="exit 1 if any kernel has call sites that are never executed",
    )
    ap.add_argument(
        "--fail-on-unused",
        action="store_true",
        help="deprecated alias for --fail-on-dead",
    )
    args = ap.parse_args()

    root = Path(args.root).resolve()
    if not any((root / d).exists() for d in DEF_DIRS):
        print(f"no {DEF_DIRS} directories under {root}", file=sys.stderr)
        return 2

    merged, n_files = collect(root)
    coverage_used = False
    if not args.no_coverage:
        cov_path = Path(args.coverage_xml)
        if not cov_path.is_absolute():
            cov_path = root / cov_path
        cov = load_coverage(cov_path)
        coverage_used = apply_coverage(merged, cov)
    else:
        for k in merged.values():
            k.status = classify(k)

    by_status: dict[str, list[Kernel]] = {s: [] for s in STATUS_ORDER}
    for k in merged.values():
        by_status[k.status].append(k)

    if args.json:
        def entry(k: Kernel) -> dict:
            return {
                "name": k.name,
                "file": k.file,
                "line": k.line,
                "status": k.status,
                "declaration_sites": k.decl_sites,
                "note": k.note,
                "refs": [
                    {
                        "file": r.file,
                        "line": r.line,
                        "kind": r.kind,
                        "external": r.external,
                        "state": r.state,
                        "hits": r.hits,
                    }
                    for r in k.refs
                ],
            }

        payload = {
            "scanned_files": n_files,
            "coverage_xml": str(args.coverage_xml) if coverage_used else None,
            "total": len(merged),
            "counts": {s: len(by_status[s]) for s in STATUS_ORDER},
            "kernels": [entry(k) for k in sorted(merged.values(), key=lambda k: (k.status, k.name))],
        }
        text = json.dumps(payload, indent=2)
        if args.output:
            Path(args.output).write_text(text + "\n")
        print(text)
    else:
        text = render_markdown(merged, n_files, coverage_used, str(args.coverage_xml))
        if args.output:
            Path(args.output).write_text(text)
            print(f"wrote {args.output}")
        else:
            print(text)

    fail = False
    if args.fail_on_dead or args.fail_on_unused:
        fail = fail or bool(by_status["dead"])
    if args.fail_on_unexercised:
        fail = fail or bool(by_status["unexercised"])
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())

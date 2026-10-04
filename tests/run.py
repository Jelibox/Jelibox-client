#!/usr/bin/env python3
"""
One command to check everything:

    python tests/run.py              # all groups
    python tests/run.py --fast       # skip the GUI group (no display needed)
    python tests/run.py --only unit,repo
    python tests/run.py -v           # show every test name
    python tests/run.py -k assistant # only tests whose name matches

Groups (each runs in its own Python process, because the real modules freeze
workspace globals at import time):

    repo         repo hygiene: compile, branding, licence, installers, assets, no hard-coded paths
    unit         pure logic: config, theme contrast, merge geometry, validators
    integration  real modules on an isolated temp workspace: classes, VOC/YOLO, Label Assistant
                 flow (fake model), imports, the Streamlit page
    gui          the real Tk windows: header, buttons, dialogs, theme restart, picker

Nothing here touches your real data: every test works in a temp directory, and
at the end the runner verifies that configs/, vocdataset/, YOLOdataset/,
models/ and datasetsInput/ are exactly as they were.

Exit code is 0 only when everything passed.
"""
import argparse
import hashlib
import os
import re
import subprocess
import sys
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
GROUPS = ["repo", "unit", "integration", "gui"]
GUARDED = ["configs", "vocdataset", "YOLOdataset", "models", "datasetsInput"]


NOISE = re.compile(r"^(\[(ClassManager|WorkspaceConfig|GUI|INFO|Config|Assistant|STREAM)\]|\s+(Original|Display|Annotations):)")


def clean_output(text):
    """Drop the app's own progress chatter so only test results/failures remain."""
    return "\n".join(line for line in text.splitlines() if not NOISE.match(line))


def github_annotation(group, text, limit=3500):
    """A GitHub Actions `::error` workflow command: it shows the failure on the run page and in the
    checks API, so a red CI run can be diagnosed without downloading logs. Keeps the END of the
    output, where unittest prints the failures."""
    body = clean_output(text)[-limit:]
    body = body.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
    return f"::error title=Jelibox tests: {group} failed::{body}"


def fingerprint():
    """(file count, hash of names+sizes+mtimes) for each guarded data folder."""
    out = {}
    for name in GUARDED:
        root = os.path.join(REPO, name)
        h, n = hashlib.sha1(), 0
        for dirpath, dirs, files in os.walk(root):
            dirs.sort()
            for f in sorted(files):
                p = os.path.join(dirpath, f)
                try:
                    st = os.stat(p)
                except OSError:
                    continue
                n += 1
                h.update(f"{os.path.relpath(p, root)}|{st.st_size}|{int(st.st_mtime)}".encode())
        out[name] = (n, h.hexdigest())
    return out


def run_group(group, verbose, pattern):
    cmd = [sys.executable, "-m", "unittest", "discover", "-s", f"tests/{group}", "-t", "."]
    if verbose:
        cmd.append("-v")
    if pattern:
        cmd += ["-k", pattern]
    env = dict(os.environ, YOLO_AUTOINSTALL="False", PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    t0 = time.time()
    p = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
    out = (p.stdout or "") + (p.stderr or "")
    ran = re.search(r"Ran (\d+) tests? in", out)
    skipped = re.search(r"skipped=(\d+)", out)
    return {
        "group": group, "ok": p.returncode == 0, "output": out,
        "ran": int(ran.group(1)) if ran else 0,
        "skipped": int(skipped.group(1)) if skipped else 0,
        "seconds": time.time() - t0,
    }


def main():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    ap = argparse.ArgumentParser(description="Run all Jelibox checks.")
    ap.add_argument("--only", help="comma separated groups: " + ",".join(GROUPS))
    ap.add_argument("--fast", action="store_true", help="skip the GUI group")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("-k", dest="pattern", help="only run tests whose name matches this substring")
    args = ap.parse_args()

    groups = [g.strip() for g in args.only.split(",")] if args.only else list(GROUPS)
    if args.fast and "gui" in groups:
        groups.remove("gui")
    unknown = [g for g in groups if g not in GROUPS]
    if unknown:
        ap.error(f"unknown group(s): {unknown}")

    before = fingerprint()
    results = []
    print(f"Jelibox checks  ({sys.executable})\n")
    for g in groups:
        print(f"  {g:<12} ...", end="", flush=True)
        r = run_group(g, args.verbose, args.pattern)
        results.append(r)
        status = "PASS" if r["ok"] else "FAIL"
        extra = f", {r['skipped']} skipped" if r["skipped"] else ""
        print(f"\r  {g:<12} {status}  {r['ran']} tests{extra}  ({r['seconds']:.1f}s)")

    after = fingerprint()
    changed = [k for k in GUARDED if before[k] != after[k]]
    print(f"  {'data guard':<12} {'PASS' if not changed else 'FAIL'}  "
          f"{'your real data was not touched' if not changed else 'CHANGED: ' + ', '.join(changed)}")

    failed = [r for r in results if not r["ok"]]
    for r in failed:
        print("\n" + "=" * 70 + f"\n{r['group']} - output\n" + "=" * 70)
        print(clean_output(r["output"]))
        if os.environ.get("GITHUB_ACTIONS"):
            print(github_annotation(r["group"], r["output"]))
    if args.verbose and not failed:
        for r in results:
            print("\n" + r["output"])

    total = sum(r["ran"] for r in results)
    ok = not failed and not changed
    print(f"\n{'ALL GOOD' if ok else 'PROBLEMS FOUND'}: {total} tests in {sum(r['seconds'] for r in results):.1f}s")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

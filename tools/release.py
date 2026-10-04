#!/usr/bin/env python3
"""
Release helper: turns "what is in [Unreleased]" into a tagged version.

    python tools/release.py patch|minor|major|X.Y.Z [--dry-run] [--skip-tests]
    python tools/release.py --check-tag v0.2.0     # CI: tag matches utils.__version__
    python tools/release.py --notes v0.2.0         # CI: print that version's changelog section

What a release does (see RELEASING.md for the why):
  1. refuses unless you are on `main` with a clean tree and [Unreleased] has entries
  2. runs the test suite
  3. sets utils.__version__, moves [Unreleased] into "## [X.Y.Z] - date" and fixes the compare links
  4. commits "Release vX.Y.Z" and creates the annotated tag vX.Y.Z
It never pushes - it prints the two commands to do that yourself.
"""
import argparse
import datetime
import os
import re
import subprocess
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INIT = os.path.join(REPO, "utils", "__init__.py")
CHANGELOG = os.path.join(REPO, "CHANGELOG.md")
REPO_URL = "https://github.com/Jelibox/Jelibox-client"
SEMVER = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")


# ------------------------------------------------------------ pure helpers
def parse_version(text):
    m = SEMVER.match(text.strip().lstrip("v"))
    if not m:
        raise ValueError(f"'{text}' is not a version like 1.2.3")
    return tuple(int(x) for x in m.groups())


def next_version(current, how):
    major, minor, patch = parse_version(current)
    if how == "major":
        return f"{major + 1}.0.0"
    if how == "minor":
        return f"{major}.{minor + 1}.0"
    if how == "patch":
        return f"{major}.{minor}.{patch + 1}"
    new = ".".join(str(x) for x in parse_version(how))
    if parse_version(new) <= parse_version(current):
        raise ValueError(f"new version {new} must be greater than current {current}")
    return new


def read_version(init_text):
    m = re.search(r'^__version__\s*=\s*"([^"]+)"', init_text, re.M)
    if not m:
        raise ValueError("no __version__ found in utils/__init__.py")
    return m.group(1)


def write_version(init_text, version):
    return re.sub(r'^(__version__\s*=\s*")[^"]+(")', rf"\g<1>{version}\g<2>", init_text, count=1, flags=re.M)


def unreleased_body(changelog):
    m = re.search(r"^## \[Unreleased\][^\n]*\n(.*?)(?=^## \[|^\[Unreleased\]:|\Z)", changelog, re.M | re.S)
    return m.group(1).strip() if m else ""


def release_changelog(changelog, version, date, repo_url=REPO_URL):
    """Move [Unreleased] into a dated version section and keep the compare links right."""
    body = unreleased_body(changelog)
    if not body:
        raise ValueError("[Unreleased] is empty - write down what changed first")
    nl = "\r\n" if "\r\n" in changelog else "\n"
    text = changelog.replace("\r\n", "\n")

    text = re.sub(r"^## \[Unreleased\][^\n]*\n(.*?)(?=^## \[|^\[Unreleased\]:|\Z)",
                  f"## [Unreleased]\n\n## [{version}] - {date}\n\n{body}\n\n", text, count=1, flags=re.M | re.S)

    prev = re.search(r"^\[(\d+\.\d+\.\d+)\]:", text, re.M)
    compare = (f"[{version}]: {repo_url}/compare/v{prev.group(1)}...v{version}" if prev
               else f"[{version}]: {repo_url}/releases/tag/v{version}")
    if re.search(r"^\[Unreleased\]:", text, re.M):
        text = re.sub(r"^\[Unreleased\]:.*$", f"[Unreleased]: {repo_url}/compare/v{version}...HEAD\n{compare}",
                      text, count=1, flags=re.M)
    else:
        text = text.rstrip("\n") + f"\n\n[Unreleased]: {repo_url}/compare/v{version}...HEAD\n{compare}\n"
    return text.replace("\n", nl)


def extract_notes(changelog, version):
    text = changelog.replace("\r\n", "\n")
    m = re.search(rf"^## \[{re.escape(version)}\][^\n]*\n(.*?)(?=^## \[|^\[Unreleased\]:|\Z)", text, re.M | re.S)
    if not m or not m.group(1).strip():
        raise ValueError(f"no changelog section for {version}")
    return m.group(1).strip() + "\n"


# --------------------------------------------------------------- git glue
def git(*args, check=True):
    r = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True)
    if check and r.returncode != 0:
        raise SystemExit(f"git {' '.join(args)} failed:\n{r.stderr.strip()}")
    return r.stdout.strip()


def read(path):
    with open(path, encoding="utf-8", newline="") as f:
        return f.read()


def write(path, text):
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(text)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Create a Jelibox release (commit + tag, no push).")
    ap.add_argument("version", nargs="?", help="patch | minor | major | X.Y.Z")
    ap.add_argument("--dry-run", action="store_true", help="show what would change, touch nothing")
    ap.add_argument("--skip-tests", action="store_true")
    ap.add_argument("--check-tag", metavar="TAG", help="CI: fail unless TAG equals utils.__version__")
    ap.add_argument("--notes", metavar="TAG", help="CI: print the changelog section of TAG")
    args = ap.parse_args(argv)

    current = read_version(read(INIT))

    if args.check_tag:
        if args.check_tag.lstrip("v") != current:
            raise SystemExit(f"tag {args.check_tag} does not match utils.__version__ = {current}")
        print(f"ok: {args.check_tag} == {current}")
        return 0
    if args.notes:
        sys.stdout.write(extract_notes(read(CHANGELOG), args.notes.lstrip("v")))
        return 0
    if not args.version:
        ap.error("give a version: patch | minor | major | X.Y.Z")

    new = next_version(current, args.version)
    date = datetime.date.today().isoformat()
    new_changelog = release_changelog(read(CHANGELOG), new, date)     # raises if [Unreleased] is empty

    problems = []
    branch = git("rev-parse", "--abbrev-ref", "HEAD")
    if branch != "main":
        problems.append(f"you are on '{branch}', releases are cut from 'main'")
    if git("status", "--porcelain"):
        problems.append("the working tree is not clean - commit or stash first")
    if git("tag", "--list", f"v{new}"):
        problems.append(f"tag v{new} already exists")

    print(f"Releasing {current} -> {new}  ({date})")
    print("\n--- changelog section ---\n" + extract_notes(new_changelog, new))
    if problems:
        for p in problems:
            print(f"  ! {p}")
        if not args.dry_run:
            return 1
    if args.dry_run:
        print("(dry run - nothing was changed)")
        return 0

    if not args.skip_tests:
        print("Running the test suite ...")
        if subprocess.run([sys.executable, os.path.join(REPO, "tests", "run.py")], cwd=REPO).returncode != 0:
            print("Tests failed - release aborted.")
            return 1

    write(INIT, write_version(read(INIT), new))
    write(CHANGELOG, new_changelog)
    git("add", "utils/__init__.py", "CHANGELOG.md")
    git("commit", "-m", f"Release v{new}")
    git("tag", "-a", f"v{new}", "-m", f"Jelibox {new}")
    print(f"\nDone: committed and tagged v{new}. Nothing is pushed yet. When you are happy:\n"
          f"    git push <remote> main\n    git push <remote> v{new}\n"
          f"(The tag push triggers the Release workflow, which publishes the GitHub Release.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

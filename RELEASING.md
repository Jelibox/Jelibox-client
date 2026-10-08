# Releasing Jelibox

This is the whole process, start to finish. Follow it as written and a release takes about five minutes.

## 1. Branching model (GitHub Flow)

`main` is always working and always releasable. Everything else is a short-lived branch that ends in a pull request.

| Branch | Used for | Example |
|---|---|---|
| `main` | the released/releasable code; protected | |
| `feature/<topic>` | new functionality | `feature/yolo-world-presets` |
| `fix/<topic>` | bug fixes | `fix/export-progress-hang` |
| `docs/<topic>` | documentation only | `docs/linux-install` |
| `release/<X.Y>` | **only if** you must patch an old version while `main` has moved on | `release/0.2` |

Why not Git Flow (`develop` + `release/*` for every version)? That model suits teams shipping several versions in parallel. With one main line, a `develop` branch only adds merges to forget. Tags mark releases; branches stay few.

**Day-to-day:**

```bash
git switch main && git pull
git switch -c feature/my-change

# ... work, then keep the checks green:
python tests/run.py

git add -A
git commit -m "Add preset prompts to the Label Assistant"
git push -u origin feature/my-change      # then open a Pull Request into main
```

- One topic per branch, small PRs.
- Commit messages: imperative, specific, under ~72 characters ("Fix crash when a class is deleted"), not "update" or "fix bug".
- **Every PR that a user would notice adds a line under `[Unreleased]` in `CHANGELOG.md`.** That is what the release notes are built from.
- CI (`.github/workflows/ci.yml`) runs the repo and unit checks on Linux and Windows for every PR. Merge only when it is green. Run the full `python tests/run.py` locally first: it also covers the GUI and integration groups.
- Merge with **Squash and merge** so `main` reads as one commit per change.

### One-time GitHub settings (Settings of the repo)
1. **Pages → Source: GitHub Actions** (publishes `index.html`).
2. **Branches → Add rule for `main`**: require a pull request, require the `CI` check to pass, block force pushes.
3. **Actions → General → Workflow permissions**: leave "Read repository contents" (the release workflow asks for write access itself).

## 2. Version numbers (Semantic Versioning)

`MAJOR.MINOR.PATCH`, e.g. `0.3.1`.

| Bump | When | Jelibox examples |
|---|---|---|
| **PATCH** `0.3.0 → 0.3.1` | bug fixes only, nothing new | fix a crash, fix a typo in a dialog |
| **MINOR** `0.3.1 → 0.4.0` | new features, still compatible | a new export format, a new dialog |
| **MAJOR** `0.x → 1.0.0`, `1.x → 2.0.0` | something users must adapt to | a changed workspace/config format with no migration, dropping a platform |

While the version is `0.x` the project is "not 1.0 yet": minor releases may still change things. Move to `1.0.0` when you are ready to promise stability.

## 3. Cutting a release

Prerequisites: you are on `main`, up to date, the tree is clean, and `[Unreleased]` in `CHANGELOG.md` lists what changed.

```bash
git switch main && git pull

python tools/release.py minor --dry-run     # preview: new version + the notes. Touches nothing.
python tools/release.py minor               # for real: bump version + changelog -> tests on the result -> commit -> tag
```

`patch`, `minor`, `major`, or an explicit `0.4.0` are accepted. The tool refuses if you are not on `main`, the tree is dirty, `[Unreleased]` is empty, the tag already exists, or any test fails.

It creates the commit `Release v0.4.0` and the annotated tag `v0.4.0` **locally**. Nothing is public yet. Check with `git log -3` and `git show v0.4.0`, then publish:

```bash
git push origin main
git push origin v0.4.0            # pushing the tag is what triggers the release
```

(Replace `origin` with your remote name; for this repo that is currently `jelibox`.)

### What happens next (automatically)
`.github/workflows/release.yml` starts on the tag and:
1. checks the tag equals `utils.__version__`,
2. runs the repo + unit checks again,
3. publishes a **GitHub Release** named "Jelibox v0.4.0" whose notes are the matching section of `CHANGELOG.md`. GitHub attaches the source `.zip`/`.tar.gz` by itself.

Open the **Releases** page and read the result. If the Pages workflow also ran, the site is updated too.

## 4. Pre-releases

For something people should try but not rely on, tag with a suffix: `v0.5.0-rc1`. The release workflow marks any tag containing `-` as a *pre-release*. Create it by hand (`git tag -a v0.5.0-rc1 -m "..." && git push origin v0.5.0-rc1`); the version in `utils/__init__.py` stays at the last real release.

## 5. Hotfix for an old version

Only needed when `main` already contains unreleased work you do not want to ship yet.

```bash
git switch -c release/0.4 v0.4.0         # branch from the released tag
git cherry-pick <commit-with-the-fix>    # fix first lands on main, then is copied here
# add a line under a new "## [0.4.1]" heading in CHANGELOG.md, set __version__ = "0.4.1"
git commit -am "Release v0.4.1" && git tag -a v0.4.1 -m "Jelibox 0.4.1"
git push origin release/0.4 v0.4.1
```

If `main` is still releasable, skip all this: fix on a `fix/*` branch, merge, release a PATCH from `main`.

## 6. Something went wrong

- **Not pushed yet:** `git tag -d v0.4.0` and `git reset --hard HEAD~1` (only for the local release commit).
- **Tag pushed, release is bad:** delete the GitHub Release in the UI, delete the tag (`git push origin :refs/tags/v0.4.0` and `git tag -d v0.4.0`), fix, and release again. If people may have downloaded it, do not reuse the number: ship `0.4.1`.
- **Release workflow failed:** open the run under **Actions**, fix the cause, then **Re-run all jobs**. The version check failing means the tag and `utils/__init__.py` disagree.

## 7. Checklist

- [ ] All intended PRs are merged and CI is green
- [ ] `python tests/run.py` passes locally (including the GUI group)
- [ ] `[Unreleased]` in `CHANGELOG.md` is accurate and readable by a user
- [ ] `python tools/release.py <bump> --dry-run` looks right
- [ ] `python tools/release.py <bump>`, then push `main` and the tag
- [ ] The GitHub Release page shows the right notes

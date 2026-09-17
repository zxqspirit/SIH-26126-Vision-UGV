# SIH 26126 — Git Workflow

This document is the authority for how the repository is branched, reviewed, and merged. It is written so a new team member can orient on day one and so the team does not drift into ad-hoc practices under deadline pressure.

The project repository is `SIH-26126-Vision-UGV` under `zxqspirit` on GitHub, private.

---

## 1. Branch model

### Long-lived shared branch

- **`main`** — integration branch. Reviewed, CI-green where applicable, and the branch the team presents from. Direct pushes to `main` are not allowed.

### Member branches

- **`member-1-perception`** — perception work: segmentation model, inference plumbing, terrain cost, perception confidence.
- **`member-2-localization`** — localization/depth work: depth geometry, visual odometry/SLAM plumbing, localization confidence, tracking diagnostics.
- **`member-3-navigation`** — planning/control work: costmap, planner, safety gate, motion command.
- **`member-4-integration`** — integration, evaluation scripts, docs, launch wiring, benchmark/reporting.

Member branches are short-lived feature branches off `main`, owned by the member for day-to-day work but not treated as exclusive territory. Any teammate may review, comment, or open a fix PR against them.

### Other branch conventions

- Intentional, descriptive, lowercase, hyphen-separated.
- Examples:
  - `fix/perception-confidence-threshold`
  - `docs/validation-strategy`
  - `experiment/EXP-003-fusion-v1`
  - `config/slam-params-initial`
- Branches should not drift forever. If a member branch falls far behind `main`, rebase or merge `main` in before continuing.

---

## 2. Branch rules

1. Work happens on a feature or member branch, not on `main`.
2. `main` is protected from direct pushes.
3. Short-lived branches are deleted after merge when they have served their purpose.
4. No branch is a hidden integration branch unless it is named deliberately and the team knows about it.
5. Do not rewrite shared history. If a branch is only local to you, that is your call; once a branch is shared or has a PR open, do not force-push it without explicit team agreement.

---

## 3. Commit format

Use conventional-style prefixes so the log is scannable without opening every commit.

- `feat/<area>:` new behavior or module
- `fix/<area>:` bug fix
- `docs/<area>:` documentation only
- `config/<area>:` configuration change
- `test/<area>:` tests only
- `chore/<area>:` maintenance, dependency, tooling, or refactor without behavior change

Rules:

- Subject line is one short sentence, imperative mood, no trailing period.
- Body explains what changed and why, especially for safety, configuration, interface, or test changes.
- If the commit changes an interface or safety behavior, say what the old behavior was and what the new behavior is.
- Do not bundle unrelated changes in one commit.
- Do not commit cache artifacts, debug-only files, or local experiment junk. If a file exists only because your machine needed it during development, it belongs in `.gitignore`, not in the repo.

---

## 4. Pull-request process

1. Every change to `main` goes through a PR. No direct pushes to `main`.
2. PR title is the intended merge commit subject.
3. PR description says:
   - what changed
   - why it changed
   - what was tested
   - what reviewers should look at
4. If the PR changes an interface, config schema, safety behavior, or test expectations, call that out explicitly.
5. A PR is ready for review when:
   - it runs in the relevant environment,
   - its tests pass or its test gaps are explicitly noted,
   - its docs are updated if behavior changed,
   - its diff is focused enough to review in one sitting.
6. Self-review first: read your own diff, remove debug prints and commented-out experiments, and make sure the commit history is coherent before requesting review.
7. Reviews are asynchronous and specific. Vague approvals are not a substitute for checking the actual change.

---

## 5. Review checklist

For every PR, at minimum:

- Which area is this? Which module(s) does it touch?
- Does it change an interface or data contract? If yes, is the change intentional and documented?
- Does it change safety behavior, confidence thresholds, stopping logic, or the command path? If yes, that is a hard review requirement.
- Does it change configuration? If yes, is the config still valid with the existing defaults, and is the change backward-compatible or intentionally breaking?
- Are there tests, or is there a documented reason why a test is not practical yet?
- Are there leftover debug logs, print statements, TODO-that-should-not-ship, or commented-out code that should be removed?
- Does the diff include anything that should be in `.gitignore` instead?
- If this is a member-branch PR meant to merge toward `main`, does it rebase/merge cleanly against current `main`?
- For performance-sensitive changes: is there a note about latency, memory, or FPS impact, even if it is just "no measurement yet"?

---

## 6. Merge policy

- Merge only after approval.
- Approval is stricter for:
  - anything in `src/safety/`
  - anything that changes the command path or safety gate
  - anything that changes shared config schemas or interface types
  - anything that changes `main` in a way that could break integration for the other members
- Keep `main` readable. Prefer merge commits or rebased linear history; the important thing is that the history is understandable, not which exact mechanism the team picks.
- Never merge a branch just because it is "almost done."
- Never force-push shared branches.
- After merge, delete short-lived member branches. Long-lived branches, if any, are the exception and must be named deliberately.

---

## 7. Large-file strategy

This repository is a source tree, not an artifact store.

Before adding anything large, ask: is this a source file, a build artifact, or data?

- Source file: commit it.
- Build artifact: do not commit it; regenerate it.
- Data or model artifact: do not commit it raw unless the team has explicitly decided this is a small reference file.

If a file is large enough to hurt clone/fetch for the whole team, treat it as a candidate for external storage, a separate data repository, or a download-and-cache step in the experiment pipeline — not as a normal tracked file.

If the team decides some model or dataset must live in this repo, document that decision here and near the file itself, and make sure `.gitignore` is not silently allowing more of the same than was intended.

Do not use Git as a backup system for raw field data. Keep the code and docs that describe the data in the repo; keep the raw data under a managed path outside the repo or in a separate storage solution.

---

## 8. Dataset and model handling

Default stance for this repo:

- **`datasets/raw/`** — raw sensor captures and bags. Normally ignored in this repo. If the team needs versioned datasets, that is a separate storage problem.
- **`datasets/processed/`** and **`datasets/annotations/`** — derived artifacts. Review for size before tracking. If derived data can be regenerated from raw data plus code, it does not belong in Git.
- **`models/checkpoints/`** — ignored by default. Trained weights are artifacts. Commit metadata and the loading interface; the binary weights are not automatically repo material.
- **`models/metadata/`** — acceptable for small descriptor files: model name, version, training config pointer, license, input/output spec. That is what makes a model replaceable.
- **`results/`** — ignored by default. Experiment outputs belong to a run, not to the source tree. Commit a specific result artifact only when it is deliberately chosen for documentation.

If the team needs to share a model or dataset with a reviewer, use a temporary link, a separate storage location, or a clearly scoped data package — not "I committed it to the repo because it was convenient."

---

## 9. Hard constraints

- No force-push to shared branches.
- No destructive reset that loses history without explicit team agreement.
- No history deletion without explicit team agreement.
- No merge without approval.
- No push to `main` without approval.
- Safety and interface changes are review-first, not merge-first.

---

## 10. Where to record deviations

If the team deliberately deviates from this workflow for a specific reason, record it as a short note in this file or in `docs/architecture/phase-0-decisions.md`. Do not let exceptions become invisible habits.

---
name: feature-spec
description: Starts a phase of work by finding the next unticked phase in specs/roadmap.md, creating a branch, interviewing the user about scope/decisions/context, and writing specs/features/YYYY-MM-DD-<name>/{requirements,plan,validation}.md. Trigger when the user says "feature spec", "next phase", "start the next roadmap phase", or invokes /feature-spec.
---

# Feature Spec

Written for any coding agent. Every step is a manual checklist item — nothing here needs a tool
only one agent has, except the interview in step 3, which a human can hold as a conversation.

## When a spec is required

**Required** — a roadmap phase, a release, an upstream `rfdetr` alignment, or any change touching a
path CI cannot execute: `src/gpu/`, `src/backends/tensorrt_backend.cpp`,
`src/backends/executorch_backend.cpp`, `deploy/export_executorch.py`.

**Not required** — bug fixes, documentation, dependency bumps with no contract change. Those are
recorded in `CHANGELOG.md` only. Do not manufacture a spec directory for a two-line fix.

## Workflow

### 1. Find the next phase

Read `specs/roadmap.md`. The next phase is the **first section whose items are all `[ ]`**. Note
its number and name — they become the branch name and the spec directory name.

If the user named a specific phase instead, use that one and say which.

### 2. Create the branch

Git-flow: branch from `develop`, never from `master`.

```bash
git checkout develop && git pull
git checkout -b feature/phase-<N>-<kebab-name>
```

### 3. Interview the user — BEFORE writing any file

Ask exactly **three** grouped questions, in one exchange. Do not write to disk until all three are
answered.

| Group | What to ask about |
|-------|-------------------|
| **Scope** | What is in, what is explicitly out, which files and targets are touched |
| **Decisions** | The choices that would otherwise be made silently — data format, where a fixture lives, whether a change is allowed to touch `src/`, tolerance values |
| **Context** | Constraints shaping the work — hardware available for verification, CI limits, related open items, anything upstream |

For a phase whose detail is already fully written in `specs/roadmap.md`, say so and confirm rather
than re-asking; the interview exists to surface undecided things, not to re-read the roadmap aloud.

### 4. Read the constitution before drafting

Always: `specs/mission.md`, `specs/tech-stack.md`, and the phase's own roadmap section.
Additionally, when the phase touches `src/gpu/` or `data/dali/`: `specs/gpu-pipeline.md` — its
8-rule model contract is the review checklist for anything GPU.

Also read `AGENTS.md` for the exact build, test, lint, and sanitizer commands. Never invent a
command; quote the one that is written down.

### 5. Write the spec directory

`specs/features/YYYY-MM-DD-<feature-name>/` using today's date.

**`requirements.md`**
- *Scope* — an "In" table of deliverables with real paths, and an explicit "Out" list. Name the
  phase it implements and link back to `specs/roadmap.md`.
- *Decisions* — each choice with the reason. Where a decision is forced by an architectural
  commitment in `specs/mission.md`, cite it.
- *Context* — existing patterns to follow with `path:line` references, constraints carried in from
  the constitution, and any open question to resolve during implementation.

**`plan.md`**
- Numbered task groups, each independently implementable, each stating whether it needs a GPU.
- Sub-tasks numbered continuously across groups, with the file each one creates or edits and the
  `CMakeLists.txt` line where it must be registered.

**`validation.md`**
- *Automated — no GPU*: the commands from `AGENTS.md` that must pass on CI.
- *Automated — with a device*: what only real hardware can prove, with explicit tolerances.
- *Compile-without-device*: for anything GPU, that the target still builds and the tests report
  `SKIPPED` rather than `FAILED`.
- *Manual*: what a human has to look at.
- *Definition of done*: CHANGELOG updated, roadmap items ticked, branch merged and deleted.

Every tolerance must be a number. "Close enough" is not a gate.

## Constraints

- No new dependency without user approval — `specs/tech-stack.md` owns the pins.
- Respect the architectural commitments in `specs/mission.md`; in particular, unit tests need no
  model file, and GPU tests skip rather than fail.
- Keep the phase independently shippable and the tree building at every step.

## Closing a phase

When `validation.md` is fully ticked: update `CHANGELOG.md` under `[Unreleased]` in the house style
(prose plus a per-file table), tick the phase's items in `specs/roadmap.md` and mark the heading
`(Complete)`, then merge into `develop` and delete the branch.

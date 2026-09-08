# OpenCode Workflow — Reasoner, Planner, Implementer

A three-role delegation setup for this repository, following
[AI coding workflows: cloud to local](https://olibartfast.ninja/blog/ai-coding-workflows-cloud-to-local.html).
The premise: spend an expensive model on decisions that are hard to reverse, a
mid-tier model on decomposition and review, and a cheap or local model on the
mechanical production of code — with the cheap model boxed in tightly enough
that its mistakes are contained rather than trusted.

This sits **beside** the workflow in [AGENTS.md](../AGENTS.md), it does not
replace it. The roadmap phases, the spec triple, the `CHANGELOG.md` update and
the git-flow rules are the same regardless of which harness drives them.

## The roles

| Role | Tier | Model | Mode | Can edit |
|------|------|-------|------|----------|
| `reasoner` | frontier | `deepseek/deepseek-v4-pro` | primary | nothing |
| `planner` | midtier | `deepseek/deepseek-v4-flash` | primary | `specs/`, `docs/`, `CHANGELOG.md`, `.opencode/` |
| `implementer` | worker | `nvidia/meta/muse-glimmer-30b` | subagent | only what the packet names |

Definitions live in [`.opencode/agents/`](../.opencode/agents/); shared defaults
in [`.opencode/opencode.jsonc`](../.opencode/opencode.jsonc).

**reasoner** decides architecture, the GPU 8-rule model contract, backend/ABI
boundaries, concurrency, tolerances — and gives accept/reject verdicts. It is
read-only by configuration, not by convention: `edit`, `write` and `patch` are
`deny`, and its `bash` allowlist holds only inspection commands. It can reject a
diff; it cannot produce one.

**planner** turns a decision into delegation packets, dispatches them, and
judges what comes back. It can write specs and changelog entries but not
production code — `src/`, `tests/` and the build files are denied to it. That is
the point: it cannot quietly do the worker's job when the worker disappoints.

**implementer** implements exactly one packet, then runs the scoreboard once and
stops. `edit` is deny-by-default with an allowlist; `webfetch` and `websearch`
are denied outright; `steps: 12` caps the loop; `temperature: 0`. It never
designs and never repairs after a failed scoreboard.

## The packet

Every dispatch to `implementer` carries five parts and nothing else:

1. **Writable paths** — the complete list of files it may create or modify.
2. **Read-only context paths** — headers and existing implementations it must
   read to get signatures and idiom right.
3. **Required final state, in prose** — what must be true when it is done. Not a
   diff, not "insert after line 40". The worker writes each file's complete
   final content; it never patches by anchor.
4. **Exact identifiers** — every type, function, member, target and CMake option
   spelled as it must appear. The worker never invents a name; a missing one is
   reported as a gap, not guessed.
5. **The scoreboard command** — `./scripts/scoreboard.sh`, run exactly once.

A packet that needs a design decision to be answerable is not a packet. It goes
up to `reasoner` first.

## The scoreboard

[`scripts/scoreboard.sh`](../scripts/scoreboard.sh) is the single pass/fail gate.
It runs four stages and prints one verdict line:

```
== format ==     clang-format-18 --dry-run --Werror over src/ tests/
== configure ==  cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
                       -DUSE_ONNX_RUNTIME=ON -DWERROR=ON
== build ==      cmake --build build --parallel
== test ==       ctest --test-dir build --output-on-failure
SCOREBOARD: PASS
```

`BUILD_DIR=<dir> ./scripts/scoreboard.sh` builds elsewhere; the default is
`build`. The stages mirror the `default` CMake preset plus the `-DWERROR=ON`
that CI uses, so a green scoreboard means the change would survive CI's
ONNX Runtime lane.

It deliberately does **not** cover what CI cannot run either: the TensorRT and
ExecuTorch backends, `src/gpu/`, and the DALI/CUDA halves. Work touching those
paths needs a spec directory (per `AGENTS.md`) and the manual
[gpu-verify](../.claude/skills/gpu-verify/SKILL.md) gate on real hardware. Never
let a green scoreboard stand in for that.

A worker runs the scoreboard once, at the end, and reports the result whether it
passed or failed. A failing scoreboard is a valid report. The prohibition on
repairing after it is what keeps a cheap model from thrashing a tree into a
shape nobody planned.

## The loop

1. **Frame.** Start `reasoner` on the next unticked phase in
   [`specs/roadmap.md`](../specs/roadmap.md). It reads the specs and states the
   decision and the required final state. Routine work — a rename, a doc fix, a
   dependency bump with no contract change — skips this step and starts at 2.
2. **Decompose.** `planner` writes the spec triple under
   `specs/features/YYYY-MM-DD-<name>/` when one is required (a roadmap phase, a
   release, an `rfdetr` alignment, or any change touching a path CI cannot
   execute), branches from `develop`, and breaks `plan.md` into packets.
3. **Dispatch.** `planner` sends one packet to `implementer`. Two packets run
   concurrently only if their writable-path sets are disjoint.
4. **Implement.** `implementer` writes the named files, builds and runs targeted
   tests as it goes, runs the scoreboard once, reports, stops.
5. **Judge.** `planner` reads the diff against the required final state item by
   item and checks no path outside the writable list was touched. A green
   scoreboard is necessary, not sufficient — it does not prove the worker built
   the thing that was asked for. Accept, or reject with the specific failing
   item and re-dispatch a corrected packet. `planner` does not fix the output
   itself; that hides an underspecified packet and moves work back up-tier.
6. **Escalate** to `reasoner` when a packet keeps coming back wrong, when the
   plan contradicts [`specs/mission.md`](../specs/mission.md), or when the right
   fix changes a contract rather than an implementation.
7. **Close.** `validation.md` passes, `CHANGELOG.md` updated, merged to
   `develop`, phase ticked in the roadmap.

## Running it

```bash
opencode agent list                 # reasoner, planner, implementer should appear
opencode                            # then switch primary agent in the TUI
opencode run --agent planner "..."  # or drive a role headlessly
```

`planner` and `reasoner` are `mode: primary` — pick one as the session driver.
`implementer` is `mode: subagent`, reachable only through delegation, never
driven directly.

### Credentials

```bash
opencode providers login   # deepseek, nvidia
```

`deepseek` is already authenticated. `nvidia` is not — the `implementer` tier
needs an NVIDIA API key (`NVIDIA_API_KEY`, an OpenAI-compatible endpoint)
before it will run. Muse Glimmer 30B there is tool-calling capable with a 131k
context, which is what the packet format assumes.

## Moving a tier onto local hardware

The provider is the first segment of the model string, so a tier moves to your
own machine by editing one line in one file:

```yaml
# .opencode/agents/implementer.md
model: lmstudio/<model-id>      # was nvidia/meta/muse-glimmer-30b
```

The brief, the packet format, the permission block and the scoreboard are
unchanged. That is the property worth protecting: the guardrails are what make a
weak local model usable, so they must not be entangled with the choice of
provider. Point LM Studio (or any OpenAI-compatible server) at the default port,
run `opencode models` to confirm the id resolves, and the same worker runs on
your hardware instead of someone else's.

Move the worker tier first — it does the most calls and the least thinking, and
a tight packet plus a hard scoreboard is exactly the setup a 30B-class local
model can survive. Moving `reasoner` local is the last step, not the first.

## Hardening a packet further

The checked-in `implementer.md` allows edits under `src/`, `tests/`,
`include/`, `cmake/`, `deploy/`, `scripts/` and the CMake files, and relies on
the packet to narrow that to the specific files. When a packet touches something
delicate, copy it to a packet-specific agent and name the files outright:

```yaml
---
description: Packet 3 — segmentation export for ExecuTorch.
mode: subagent
model: nvidia/meta/muse-glimmer-30b
steps: 12
permission:
  edit:
    "*": deny
    "deploy/export_executorch.py": allow
  bash:
    "*": deny
    "./scripts/scoreboard.sh*": allow
  webfetch: deny
  websearch: deny
---
```

Then the sandbox, not the prose, is what stops an out-of-scope edit.

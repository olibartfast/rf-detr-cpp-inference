# Delegation Workflow — Reasoner, Planner, Implementer

A three-role delegation setup for this repository, following
[AI coding workflows: cloud to local](https://olibartfast.ninja/blog/ai-coding-workflows-cloud-to-local.html).
The premise: spend an expensive model on decisions that are hard to reverse, a
mid-tier model on decomposition and review, and a cheap or local model on the
mechanical production of code — with the cheap model boxed in tightly enough
that its mistakes are contained rather than trusted.

Nothing here is specific to one agent harness. The roles, the packet, the
scoreboard and the loop are the contract; a harness — OpenCode, Claude Code,
Codex CLI, or whatever you drive next — only has to supply four primitives to
host it. ["Hosting it in a harness"](#hosting-it-in-a-harness) maps the
contract onto the ones already in use here and says what to do when a harness
is missing a primitive.

This sits **beside** the workflow in [AGENTS.md](../AGENTS.md), it does not
replace it. The roadmap phases, the spec triple, the `CHANGELOG.md` update and
the git-flow rules are the same regardless of which harness drives them.

## The roles

| Role | Tier | Driven as | May edit |
|------|------|-----------|----------|
| `reasoner` | frontier | session driver | nothing |
| `planner` | midtier | session driver | `specs/`, `docs/`, `CHANGELOG.md`, harness config |
| `implementer` | worker | delegated only | only what the packet names |

The prose below is the role brief. It is the same text in every harness — paste
it into whatever that harness calls an agent, a profile or a mode, and change
only the model id and the permission syntax.

**reasoner** decides architecture, the GPU 8-rule model contract, backend/ABI
boundaries, concurrency, tolerances — and gives accept/reject verdicts. It is
read-only by configuration, not by convention: file writes and patches are
denied, and its shell allowlist holds only inspection commands. It can reject a
diff; it cannot produce one.

**planner** turns a decision into delegation packets, dispatches them, and
judges what comes back. It can write specs and changelog entries but not
production code — `src/`, `tests/` and the build files are denied to it. That is
the point: it cannot quietly do the worker's job when the worker disappoints.

**implementer** implements exactly one packet, then runs the scoreboard once and
stops. Editing is deny-by-default with an allowlist; web fetch and web search
are denied outright; the agent loop is capped at ~12 steps; temperature 0. It
never designs and never repairs after a failed scoreboard.

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

The packet is the part that survives every change of harness, model and
provider. When a harness gives you less enforcement than you want, the packet is
what is left — keep it exact.

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

Being a plain script, the scoreboard is the one piece no harness can
misinterpret — it is the same command whether a role runs it, a human runs it,
or CI does.

## The loop

1. **Frame.** Start `reasoner` on the next unticked phase in
   [`specs/roadmap.md`](roadmap.md). It reads the specs and states the
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
   plan contradicts [`specs/mission.md`](mission.md), or when the right
   fix changes a contract rather than an implementation.
7. **Close.** `validation.md` passes, `CHANGELOG.md` updated, merged to
   `develop`, phase ticked in the roadmap.

The loop does not care whether a step crosses a harness boundary. Framing in one
tool and implementing in another is a normal way to run it — the packet and the
scoreboard travel as text, and step 5 reads a git diff, not a transcript.

## Hosting it in a harness

A harness needs four primitives, and only the fourth is ever truly missing:

| # | Primitive | What the workflow needs it for |
|---|-----------|-------------------------------|
| 1 | **Role brief** — per-role instructions | The three briefs above, one per role |
| 2 | **Model selection** — per role, not per session | The tiering: frontier / midtier / worker |
| 3 | **Write restriction** — deny by default, allowlist paths | `reasoner` reads only; `planner` cannot touch `src/`; the worker is held to the packet |
| 4 | **Delegation** — one role invoking another as a subagent | Steps 3–5 without a human in the middle |

Every harness also needs to read [AGENTS.md](../AGENTS.md); most do so by
convention, and the rest take a pointer to it in their own config.

### OpenCode

The configuration this repository was built against. Role definitions live in
`.opencode/agents/{reasoner,planner,implementer}.md` (not checked in); shared
defaults in [`.opencode/opencode.jsonc`](../.opencode/opencode.jsonc).

| Primitive | Where |
|-----------|-------|
| Role brief | `.opencode/agents/<role>.md` body |
| Model | `model:` in that file's frontmatter, `provider/model-id` |
| Write restriction | `permission:` block — `edit`/`write`/`patch`/`bash` glob → `allow`\|`ask`\|`deny` |
| Delegation | `mode: primary` (session driver) vs `mode: subagent` (delegated only) |

`planner` and `reasoner` are `mode: primary` — pick one as the session driver.
`implementer` is `mode: subagent`, reachable only through delegation, never
driven directly. `steps: 12` caps the worker's loop and `temperature: 0` keeps
it literal.

```bash
opencode agent list                 # reasoner, planner, implementer should appear
opencode                            # then switch primary agent in the TUI
opencode run --agent planner "..."  # or drive a role headlessly
opencode providers login            # per-provider credentials
```

The tiering checked into `opencode.jsonc` is `deepseek/deepseek-v4-pro`
(reasoner), `deepseek/deepseek-v4-flash` (planner) and
`nvidia/meta/muse-glimmer-30b` (implementer). The worker tier needs an
`NVIDIA_API_KEY` against NVIDIA's OpenAI-compatible endpoint; Muse Glimmer 30B
there is tool-calling capable with a 131k context, which is what the packet
format assumes.

### Claude Code

| Primitive | Where |
|-----------|-------|
| Role brief | `.claude/agents/<role>.md` body, with `description:` written so the driver picks it |
| Model | `model:` in that file's frontmatter |
| Write restriction | `tools:` in the frontmatter (omit `Edit`/`Write` for `reasoner`), plus `permissions.deny` globs in `.claude/settings.json` |
| Delegation | the `Agent` tool — a subagent file is only reachable through it |

The roles this repository has been driven with are user-level subagents in
`~/.claude/agents/` (not checked in), and they split `reasoner` in two:

| Contract role | Subagent | Model | Tools |
|---------------|----------|-------|-------|
| `reasoner` (design decisions) | `architect` | `claude-fable-5-1` | full, including `Edit`/`Write` |
| `reasoner` (accept/reject verdicts) | `reviewer` | `claude-opus-5-5` | `Read`, `Grep`, `Glob`, `Bash`; `Edit`/`Write` disallowed |
| `planner` | `planner` | `claude-opus-5-5` | full |
| `implementer` | `implementer` | `claude-sonnet-5` | `WebFetch`/`WebSearch` disallowed, `maxTurns: 12` |

`reviewer` is the read-only half the brief asks for. `architect` is not: it can
write, so its read-only status is by convention, and a diff it produces goes
through `reviewer` like any other. `planner` likewise has no `src/` deny, so the
judge step, not the sandbox, keeps it from doing the worker's job. When the model
line-up changes, update the `model:` lines and this table together.

`AGENTS.md` is already loaded via [`CLAUDE.md`](../CLAUDE.md). Repo procedures
that predate this file are skills under `.claude/skills/` — `feature-spec`,
`release`, `rfdetr-alignment`, `gpu-verify` — and they are the same checklists
steps 2 and 7 of the loop call for; a `planner` running here should invoke them
rather than restate them.

### Codex CLI

Roles are profiles rather than agent files, and there is no in-session
delegation, so `planner` dispatches by shelling out.

| Primitive | Where |
|-----------|-------|
| Role brief | `AGENTS.md` plus the packet text passed as the prompt |
| Model | a `[profiles.<role>]` block in `~/.codex/config.toml`, selected with `--profile` |
| Write restriction | sandbox mode — `read-only` for `reasoner`, `workspace-write` for the others — plus the approval policy |
| Delegation | none in-session: run the worker as `codex exec --profile implementer "<packet>"` and read its diff |

Use `read-only` for `reasoner` and take the worker's path narrowing from the
packet and the judge step, not from the sandbox: `workspace-write` is
repository-wide, so step 5's `git diff --name-only` check is doing the work a
`permission:` block does elsewhere. Run the worker on a scratch branch or a
worktree so a stray edit is cheap to drop.

### Any other harness — MetaMuse, and whatever lands next

The three sections above are the harnesses this repository has actually been
driven with, so they are the only ones whose config paths are stated as fact.
For anything else, answer the four questions in the table above from that
harness's own documentation before adapting anything, then map them one by one
and write the mapping down here as a fourth section. Most harnesses have 1 and
2, a sandbox or permission layer covering 3, and either subagents or a headless
`run`/`exec` command covering 4. Where it lacks one, say which and degrade
explicitly rather than assuming the guardrail is there.

The degraded mode is worth naming because it is what every unfamiliar harness
gives you on day one, and it still works:

- **No per-role write restriction** — the packet's writable-path list is
  advisory, and step 5 enforces it with `git diff --name-only` against a clean
  branch. Reject on an out-of-scope path exactly as if a sandbox had blocked it.
- **No delegation** — run each role as its own session, or its own tool, and be
  the courier yourself. The packet is already a self-contained text artifact for
  precisely this reason; paste it in, paste the report back out.
- **No per-role model selection** — collapse `reasoner` and `planner` onto one
  model before you collapse either into `implementer`. The tier split that
  matters most is design-vs-production, and a worker with no design authority is
  the whole safety argument.

What must never be dropped, whatever the harness: one packet per dispatch, the
worker not designing, the scoreboard run once and reported honestly, and
`planner` judging the diff instead of repairing it.

## Moving a tier onto local hardware

Tiering is a per-role model id, so a tier moves to your own machine by editing
one line in one file — in OpenCode the provider is the first segment of the
model string:

```yaml
# .opencode/agents/implementer.md
model: lmstudio/<model-id>      # was nvidia/meta/muse-glimmer-30b
```

The brief, the packet format, the permission block and the scoreboard are
unchanged. That is the property worth protecting: the guardrails are what make a
weak local model usable, so they must not be entangled with the choice of
provider — or of harness. Point LM Studio (or any OpenAI-compatible server) at
the default port, confirm the id resolves (`opencode models`, or your harness's
equivalent), and the same worker runs on your hardware instead of someone
else's.

Move the worker tier first — it does the most calls and the least thinking, and
a tight packet plus a hard scoreboard is exactly the setup a 30B-class local
model can survive. Moving `reasoner` local is the last step, not the first.

## Hardening a packet further

The role-level allowlist is deliberately broader than any one packet: edits
under `src/`, `tests/`, `include/`, `cmake/`, `deploy/`, `scripts/` and the
CMake files, with the packet narrowing that to specific files. When a packet
touches something delicate, copy the worker definition to a packet-specific one
and name the files outright, so the sandbox rather than the prose is what stops
an out-of-scope edit.

OpenCode:

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

Claude Code — a packet-specific subagent, narrowed by its tool list:

```yaml
---
name: packet-3-executorch-export
description: Packet 3 — segmentation export for ExecuTorch. Use only when dispatched with the packet.
tools: Read, Edit, Bash
model: haiku
---
```

The `tools:` list is the per-subagent lever: dropping `WebFetch` and `WebSearch`
there is equivalent to the `deny` lines above. Path denials are not per-subagent
— `permissions.deny` in `.claude/settings.json` (e.g. `"Edit(src/gpu/**)"`)
applies to every session in the project, so use it only for paths that should be
off limits to all of them, and let the packet plus the judge step hold the rest.

Codex CLI has no per-path allowlist, so narrow it with the filesystem instead:
run the packet in a git worktree containing only what it may touch, or accept
that the judge step is the enforcement and keep the packet's writable list
short.

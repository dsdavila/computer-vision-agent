# Agentic CV Pipeline Harness — Design

**Status:** design sketch, pre-implementation
**Purpose of this document:** enough shared context for an implementing agent (Claude Code) to take a first pass and surface the problems that only appear in code.

---

## 1. Goal

Take (a) a set of images or video(s) and (b) a natural-language specification of what to detect/track, and automatically assemble, configure, and verify a working vision pipeline from a catalog of pre-built components.

**Target user:** a team that needs a deployed vision pipeline and cannot get CV engineering headcount. Not a CV engineer who wants a faster loop. This drives most of the requirements below — in particular the feasibility verdict (§8), the drift monitoring (§11), and the exportable handoff (§11).

**Explicit non-goal:** impressing CV practitioners with catalog breadth. Coverage of SOTA methods is worth less to this user than a pipeline that still works in November.

---

## 2. Core design commitments

| Commitment | Rationale |
|---|---|
| Frozen, versioned component catalog; agent composes and configures, never authors | Generated code has no provenance, can't be regression-tested, and destroys the audit story for accreditation review. Also makes weak on-prem models tolerable. |
| LLM proposes, deterministic code disposes | LLM never in the inner scoring loop. It navigates coarse search space and diagnoses failure modes; parameter sweeps are a `for` loop. Violating this breaks cost, latency, and reproducibility at once. |
| Measurable properties belong to the profiler, not the agent | Scene characteristics are cheap deterministic measurements. The LLM classifies over a handful of regimes rather than reasoning open-endedly about the data. |
| Append-only decision ledger is the primary artifact | It is the leave-behind for humans, the input to future agents, and the case base that makes the system improve without the model improving. |
| Human adjudication at one specific point, on domain questions only | Half a day of "is this the thing, yes/no" converts label-free search into supervised AutoML. Requires domain knowledge, not CV knowledge. |

---

## 3. Architecture

| Layer | Responsibility | LLM involvement |
|---|---|---|
| Spec compiler | NL spec → typed task contract: ontology, spatial/temporal predicates, operating point (recall- vs precision-weighted), hardware envelope, success criteria | High. One-shot, human-confirmed. |
| Data profiler | Cheap deterministic probes → regime vector (§4) | None |
| Feasibility gate | Regime vector + contract → proceed / refuse with physical reason (§8) | Low. Rule-driven. |
| Capability registry | Typed components with declared preconditions, I/O contracts, cost, license, version | None |
| Planner | Regime + contract → 3–6 candidate topologies; retrieves seed configs from case base | Medium. Constrained generation over registry only. |
| Stage specialists | Per-stage optimization: detection, association, ReID/appearance, event logic | Existing VLM tuner drops in here as the detection/association specialist |
| Scorer | Tiered scoring stack (§6) | VLM only at tier 2 |
| Packager | Config freeze, container build, ledger export, replay bundle | None |
| Monitor | Drift detection against frozen eval set, re-tune trigger, regression gate | None |

The task contract is the human-confirmed interface. Everything downstream compiles against it. Without it this is a demo rather than a system.

---

## 4. Profiler: the regime vector

Every axis of the exploration space that can be measured, is measured. Deterministic code, no model involvement.

| Axis | Measurement | Constrains |
|---|---|---|
| Pixels on target | Proposal box height distribution, p10/p50 | Input scale, tiling strategy, detector family |
| Congestion | Detections per frame, mean pairwise IoU, occlusion rate | Tracker family, association gating |
| Motion dynamics | Optical-flow magnitude, per-track displacement variance, camera-motion estimate | Buffer length, Kalman assumptions, camera-motion compensation |
| Lighting | Frame-level intensity variance over time, saturation/clipping fraction | Augmentation; whether appearance ReID is viable at all |
| Appearance separability | Inter-instance embedding distance on sampled crops | ReID on/off, appearance weight in association |
| Target novelty | Open-vocab score distribution against the spec ontology | Off-the-shelf vs. few-shot vs. fine-tune |

**Honest note for implementers:** the regime → topology mapping is hand-authored expertise at the start. It is a lookup table with a language model on the front, not emergent reasoning. That's fine — it's the moat — but it should never be described internally as anything else.

---

## 5. Search space collapse

| Stage | Approximate size | Mechanism |
|---|---|---|
| Raw catalog | ~8 detectors × 4 scale/tile strategies × 5 trackers × 4 ReID options ≈ 640 topologies | — |
| × parameters | ~8 tunable params × 5 values ≈ 3.9×10⁵ per topology → ~2.5×10⁸ | — |
| Tier-0 constraints | ~150 topologies | Latency, VRAM, airgap, license |
| Regime conditioning | 6–10 topologies | §4 mapping |
| Case-base retrieval | 6–10 seeded configs | Nearest prior solution supplies starting params |
| Staged local search | ~25–40 evals per topology | ±2 steps per param, coordinate descent, not grid |
| **Total** | **~200–400 pipeline evaluations** | Hours on a short clip |

**Stage ordering:** detection quality → association → appearance/ReID → event logic. Detection dominates every downstream metric; tuning association against a bad detector wastes budget.

This greedy staging is provably suboptimal — detector operating point and association thresholds interact. Budget one joint refinement pass over a narrow neighborhood at the end. Expect a few points, not a transformation. Mark all entries from that pass as confounded (§9).

---

## 6. Scoring: tiered, because proxies don't transfer

**The central technical risk.** Label-free proxy metrics (`short_track_ratio`, `conf_p50_margin`, `near_threshold_fraction`, track-count stability) rank configurations *within* an equivalence class — same detector family, same class vocabulary. They do **not** rank *across* classes.

Concretely: a fragmented-track signature from an open-vocab detector that half-recognizes the target is indistinguishable from the signature of a strong detector at too high a threshold. The correct response is opposite in each case (swap the detector vs. lower the threshold). Treating proxies as globally comparable will confidently select the wrong architecture.

| Tier | Mechanism | Valid for | Cost |
|---|---|---|---|
| 0 | Hard constraints: latency, VRAM, license, airgap, vocabulary coverage | Filtering, never scoring | Free |
| 1 | Label-free proxies | Parameters within one topology + vocabulary | Cheap, automatic |
| 2 | VLM pairwise preference over sampled crops, ranked via Bradley-Terry | Coarse cross-config ordering | Moderate, noisy |
| 3 | Human-adjudicated micro-eval set | Final cross-config selection, regression gating | One-time human cost |

Tier 2 uses pairwise comparison deliberately: VLM judges are considerably more reliable at "which of these two is better" than at absolute quality scores. Use it to prune, not to decide.

Every ledger entry records which tier produced its score, so trust is inspectable.

---

## 7. The anchor: human adjudication budget

The first thing auto-annotation buys is a **held-out evaluation set**, not training data. With a trustworthy eval slice the problem collapses from label-free search (hard) to supervised AutoML (well understood). A hundred well-chosen labeled frames beats ten thousand noisy pseudo-labels.

| Artifact | Volume | Human action | Est. time |
|---|---|---|---|
| Detection micro-eval | ~200 diversity/uncertainty-sampled frames, ~1,500 boxes | Accept / reject / nudge open-vocab + SAM2 proposals | 1.5–2.5 h |
| Tracking micro-eval | 3 clips × ~300 frames | Review track-level events only (births, deaths, switches), not per-frame | ~1 h |
| Spec confirmation | 1 pass | Confirm compiled task contract | 10 min |

**UI requirement:** the adjudication interface must expose zero CV surface — no mAP, no thresholds, no configs. "Is this the thing, yes or no." A domain expert can do this; that's why it doesn't require the headcount the team can't get.

Position honestly to users: zero to deployed pipeline in a day, four hours of which is clicking yes on boxes.

---

## 8. Feasibility verdict — say no early

An expert's highest-leverage output is frequently a refusal. The profiler must be able to return, *before* the search runs:

> Target averages 11 px tall. No catalog method achieves useful recall below ~20 px. Change the lens or accept the number.

A team without a CV engineer cannot distinguish "the system worked hard and 45% recall is the ceiling for this data" from "the system is broken." Without an early feasibility verdict with a specific physical reason, they get a mediocre pipeline they can't interpret and conclude the product doesn't work.

Related: report a **performance envelope**, not a point estimate. Where the pipeline is reliable, where it degrades, what it will miss.

---

## 9. The decision ledger

Append-only. The winning config is a pointer into the ledger, not the artifact itself. Negative results survive — the eight branches that failed and why *are* the search pruning, and are worth more to the next agent than the final config.

| Field | Purpose |
|---|---|
| `config_delta` | What changed; structurally diffable |
| `evidence` | Triggering metric values — values, not adjectives |
| `hypothesis` | Predicted effect, stated numerically **before** the run |
| `outcome` | Measured effect |
| `verdict` | confirmed / refuted / inconclusive — computed, not narrated |
| `regime_vector` | Profiler output at the time |
| `scorer_tier` | Which tier produced the score |
| `confounded` | True for joint-refinement entries and any multi-variable change |
| `provenance` | Model, model version, seed, catalog version |

`hypothesis` + `verdict` is what makes the ledger worth keeping. It lets a later run ask "in this regime, did raising the spawn threshold historically help?" and answer from evidence rather than from a prior baked into a prompt. It also yields a measurable quality signal on the local endpoint: a high refutation rate means the model is reasoning badly.

**Confirmed ≠ caused.** A proxy improving after a change is not attribution. Prefer single-variable deltas during staged search; flag everything else `confounded` so retrieval can downweight it.

**Two renderings, one store.** Future agents need the full typed record. A human returning in month seven needs a short causal narrative: what the data looked like, why this detector, known weak spots. Generate the narrative *at decision time* — don't depend on whatever model is available later.

---

## 10. Case base

Every solved problem writes `(regime_vector, winning_config, achieved_score, spec_class)` to a store. New problems retrieve nearest neighbours and seed from them.

Retrieval requires almost no model capability, so **the system improves over time without the on-prem model improving**. This decoupling is the strongest argument for the frozen-catalog architecture. Build for it from day one, before there are cases to retrieve.

---

## 11. Operations

**Drift.** Replacing a person means inheriting the job they'd have done later: camera bumped, seasons change, new site with different lighting. Nobody on this team will notice degradation. Drift detection against the frozen eval set, automatic re-tune, regression gate. Core to the value proposition, not phase two.

**Handoff.** When the system fails, the team's only move is to hire a contractor for a week. Export a real repo: configs, eval set, ledger, replay bundle. An opaque service makes the first serious failure terminal for the account.

**Provenance.** For federal/regulated deployment, "why did the system choose this pipeline" will be asked in review. The ledger must answer it without reconstruction.

---

## 12. The glue problem

Component selection is not where real deployments die. Glue is: ROI masks, line-crossing logic, class merging, dwell-time rules, alert debounce, cross-camera dedup, calibration.

With a frozen catalog, all of this must be expressible declaratively. Required:

- A typed pipeline graph spec
- A small predicate language for event logic

This is a substantial design commitment and the piece most likely to be underestimated. Skip it and the agent routinely reaches a pipeline that is 90% right and needs three lines of code it isn't allowed to write.

**Controlled escape hatch (optional, one slot only):** generated code permitted in exactly one place with a fixed signature — a pure function from a track record to a boolean. Sandboxed, no imports, unit-tested against the eval set, auto-rejected on exception or regression. Weak models write a ten-line predicate reliably; they can't maintain a codebase. That distinction is what the architecture encodes.

---

## 13. Versioning

The ledger references catalog components and their parameters. Retire a detector or rename a parameter and every historical entry silently degrades into noise that still reads as authoritative.

- Version the catalog
- Pin ledger entries to catalog versions
- Write migrations on deprecation

Boring, and it determines whether the case base is an asset in two years or stale JSON nobody trusts.

---

## 14. Known risks

| Risk | Severity | Mitigation |
|---|---|---|
| Proxy metrics treated as cross-config comparable | Critical | Tiered scorer (§6), tier recorded per entry |
| "Arbitrary CV problem" scope creep | High | Pick two verticals. Surveillance/security/inspection first — roughly rigid targets, mostly static cameras |
| Open-vocab performance is bimodal, not continuous | High | Early regime-detection check that escalates to few-shot/fine-tune instead of grinding thresholds in the useless regime |
| Agent proliferation | Medium | One orchestrator + 5–6 tools with strict schemas. Multi-agent only where subproblems need isolated context |
| On-prem model degradation | Medium | Differentiation lives in deterministic scaffolding. Tight, heavily validated structured-output contracts. A weaker model must produce a worse pipeline, not a broken run |
| Glue expressiveness underestimated | Medium | §12 — design the predicate language early |
| Greedy staging misses interactions | Low | Bounded joint refinement pass, marked confounded |

---

## 15. v0 scope

Narrow aggressively:

- Fixed topology family: detector → tracker → optional ReID
- Person and vehicle, plus open-vocab targets
- Surveillance video only
- Build for real: spec compiler, capability registry, tiered scorer, ledger schema
- Drop the existing VLM tuner in as the detection/association specialist

That yields an end-to-end spine in weeks. The scoring stack is the piece that gets iterated on for a year.

---

## 16. Research angle

No benchmark exists of `(spec, data, expected pipeline)` triplets. Defining one is a real contribution independent of the system itself.

---

## Implementation notes

Living section. Appended to as implementation proceeds. Notes are proposals until the design agents/human confirm.

### Milestone plan (v0 spine, per §15)

**M1 — Contracts & registry (schemas only, no runtime)**
- `TaskContract` schema: ontology, spatial/temporal predicates, operating point, hardware envelope, success criteria.
- `CapabilityRegistry` schema: component manifest with declared preconditions, I/O types, cost, license, pinned catalog version (§13).
- `LedgerEntry` schema exactly as §9 — `hypothesis`, `verdict`, `scorer_tier`, `confounded`, `provenance` are non-negotiable.
- `RegimeVector` schema for §4 axes.
- Output: language-agnostic JSON Schemas + generated typed bindings. Everything downstream compiles against these.

**M2 — Profiler + feasibility gate (deterministic, no LLM)**
- Each §4 axis implemented as an independent probe returning a scalar or distribution.
- Feasibility gate is a rule table over regime vector → verdict with a physical-reason string (§8). Ship with ~6 hard rules to start (pixels-on-target floor, congestion ceiling, appearance-separability minimum for ReID, etc.).
- Deliverable: `profile(video) → RegimeVector`; `feasibility(contract, regime) → Verdict`.

**M3 — Planner + ledger + tier-0/1 scorer**
- Planner: regime vector + contract → candidate topologies via hand-authored lookup table (§4 honest note). LLM used only for constrained generation over registry entries; never free-form.
- Ledger: append-only store, dual rendering (typed record + narrative generated at decision time per §9).
- Tier-0 (hard-constraint filter) and Tier-1 (label-free proxies). Enforce in the type system that Tier-1 scores are only comparable *within* an equivalence class (§6 central risk).
- Drop existing VLM tuner in here as the detection/association specialist.

**M4 — Packager + minimal glue**
- Config freeze, container build, replay bundle, ledger export.
- Predicate language for event logic (§12) — even a minimal grammar. Flagged as the most underestimated piece; do not defer.
- Explicitly deferred to post-v0: tier-2 VLM Bradley-Terry, tier-3 human adjudication UI, case base retrieval, drift monitor. Designed-for now, built later.

### Interfaces to freeze before writing code
1. `TaskContract` — everything compiles against this; retrofits are expensive.
2. `LedgerEntry` — schema drift silently corrupts the case base (§13).
3. Registry component manifest — same reason.

### Resolved decisions
- **Language:** Python end-to-end. Single runtime for CV, orchestration, and schemas. Pydantic for the typed contracts in M1.
- **LLM endpoint:** local on-prem from day one. Matches §14 on-prem-degradation risk row and the airgap constraint; no hosted-API code path in v0. Structured-output contracts must be tight enough that a weaker model produces a worse pipeline, not a broken run.
- **Ledger store:** files-in-git. One append-only file per ledger entry, catalog-version pinned in the entry (§13). Gives free provenance for regulated review (§11), works airgapped, and diff-review of ledger PRs is a real audit surface. Case base is a directory of resolved cases in the same repo.

### Still open
- **Catalog v0 breadth.** §5 assumes ~8 detectors × 4 scale strategies × 5 trackers × 4 ReID. Ship v0 smaller (say 3×2×2×2) and grow, or invest in full breadth up front? Smaller catalog keeps the regime→topology lookup table (§4) hand-authorable in week one.

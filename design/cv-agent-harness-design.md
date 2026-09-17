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
- `CapabilityRegistry` schema: component manifest with declared preconditions, I/O types, license, pinned catalog version (§13). **Cost is not a scalar field** — it is a function of config, measured on target hardware and cached with the catalog version, not declared. Tiling cost scales roughly as tile-count × overlap-factor (2×2 with overlap ≈ 4–5× single-pass); scalar cost lets the case base learn that tiling is free.
- `LedgerEntry` schema exactly as §9 — `hypothesis`, `verdict`, `scorer_tier`, `confounded`, `provenance` are non-negotiable.
- `RegimeVector` schema for §4 axes, extended with a per-axis confidence field (see §4.1 proposal below).
- `EvalSet` schema: reference to a ground-truth file (COCO for detection, MOT for tracking) plus scoring-function identifier and catalog-version pin.
- `MappingRules` schema (see resolved decision below): per-axis rules + override table for known joint interactions.
- Output: language-agnostic JSON Schemas + generated typed bindings (Pydantic). Everything downstream compiles against these.

**M2 — Profiler + feasibility gate (deterministic, no LLM)**
- Each §4 axis implemented as an independent probe returning a scalar or distribution, plus a **confidence** value where the probe cannot be made detector-independent (§4.1).
- Feasibility gate is a rule table over regime vector → verdict with a physical-reason string (§8). **Rule thresholds are catalog-derived and pinned to the catalog version that produced them (§13), not hardcoded** — a hardcoded threshold silently goes wrong the first time the catalog changes.
- Starting rule set: pixels-on-target floor, congestion ceiling, appearance-separability minimum for ReID, and a **low-profiling-confidence refusal** (§4.1) — no verdict is preferable to a confident-wrong verdict on bad input.
- **Probe-discrimination validation:** each probe must demonstrate that it separates a held-out regime set before entering the vector. No unvalidated probe ships. Right now nothing in the design checks that probes actually discriminate; this closes that gap.
- Deliverable: `profile(video) → RegimeVector`; `feasibility(contract, regime) → Verdict`.

**M3 — Planner + ledger + tiered scorer (tiers 0, 1, 3)**
- Planner: regime vector + contract → candidate topologies via hand-authored lookup table (§4 honest note). LLM used only for constrained generation over registry entries; never free-form.
- Ledger: append-only, one file per entry in the git repo. Dual rendering per §9 (typed record + narrative generated at decision time). Narrative + final config is the audit surface, not the raw ledger.
- **Derived index (cache, never authoritative):** SQLite or parquet index over regime vectors rebuilt from git contents; supports case-base nearest-neighbour retrieval. Git is not an index — retrieval degrades past ~10 cases at the 200–400 entries per problem §5 anticipates. Rebuild-from-git is a one-command operation.
- Tier-0 (hard-constraint filter): free, filtering only.
- Tier-1 (label-free proxies): scoped in the type system to intra-topology, intra-vocabulary comparisons only (§6 central risk). A tier-1 score is not orderable across topologies at the API level.
- **Tier-3 minimal (pulled into v0, not deferred):** ingest a user-supplied ground-truth file (COCO for detection, MOT for tracking) and a scoring function. No annotation tooling, no adjudication UI — the human produces the file however they like. **All cross-topology selection routes through tier-3 exclusively.** Without this, M3 ships the exact §6 failure mode.
- Drop existing VLM tuner in here as detection/association specialist.

**M4 — Packager + minimal glue**
- Config freeze, container build, replay bundle, ledger export.
- Predicate language for event logic (§12), even a minimal grammar. Flagged as the most underestimated piece; do not defer.
- **v0 exits here.** Post-v0 items designed for, not built: tier-2 VLM Bradley-Terry, case-base retrieval logic on top of the derived index, drift monitor.

**M4.5 — Proposal pipeline (out of v0; feeds M5)**
- Open-vocab detector + SAM2 mask/box generation over sampled frames. Produces the proposal stream M5's UI reviews.
- **Load-bearing component is the sampling strategy, not the models.** Uniform sampling of 200 frames returns the modal easy case; the eval set fails to cover regimes where candidate configs actually differ; tier-3 then produces false confidence with a credible number — **worse than no tier-3**.
- Sampling stratifies over: (a) the §4 regime axes, and (b) **disagreement between candidate configs**. (b) is chicken-and-egg with M3's planner — expect a two-phase sampler: regime-stratified pass first, disagreement-augmented pass after initial candidates propose.
- 200-frame budget × ~6 regimes × ~4 disagreement bands ≈ 8 frames/stratum. Tight, especially for rare-class detection recall. Sampler must degrade cleanly under budget pressure (oversample regimes with low profiling confidence per §4.1; skip disagreement pass on unanimous regions).

**M5 — Adjudication UI (out of v0; product boundary)**
- Auto-annotation review over the M4.5 proposal stream — accept / reject / nudge for detection, track-level events only for tracking. **Zero CV surface** — no mAP, no thresholds, no configs. A domain expert produces a ground-truth file without knowing what a ground-truth file is.
- Output is the same COCO/MOT format that M3's tier-3 ingest already consumes; M4.5 + M5 together replace the "user produces this out-of-band" assumption from v0.
- **v0 delivery boundary, stated explicitly:** v0 serves a pilot who can produce a ground-truth file out-of-band. §1's target user — a team that cannot get CV engineering headcount — is **not** served until M4.5 and M5 both ship. Scope claim, not design claim; belongs in v0 launch positioning.

### Interfaces to freeze before writing code
1. `TaskContract` — everything compiles against this; retrofits are expensive.
2. `LedgerEntry` — schema drift silently corrupts the case base (§13).
3. Registry component manifest — same reason. **Cost is a function of config, not a field.** Scalar cost is the schema change that most quietly ruins tier-0 and the case base.
4. `EvalSet` (COCO/MOT reference + scorer id + catalog pin) — cross-topology selection depends on it; retrofitting means re-running search.
5. `MappingRules` — per-axis rules + override table (see resolved decision). Adding a topology or a regime is a single-rule change against this schema, not a cell-grid rewrite.

### Resolved decisions
- **Language:** Python end-to-end. Single runtime for CV, orchestration, and schemas. Pydantic for the typed contracts in M1.
- **LLM endpoint:** interface-agnostic; both a local on-prem endpoint and a hosted endpoint are configured in development. **Local is the ship gate; hosted is a diagnostic reference** for distinguishing under-specified scaffolding from weak-model behaviour — those have opposite fixes and running only local hides the distinction. The §9 refutation-rate signal is the comparison instrument. Deployment airgap constraint (§14) still holds — the hosted path is a dev-time tool, never on the runtime path. **Any change to a structured-output contract (LLM I/O schema) re-triggers local validation before the change lands** — otherwise the ship gate erodes one unaudited edit at a time.
- **Ledger store:** files-in-git as the system of record; one append-only file per entry, catalog-version pinned per §13. **Derived SQLite/parquet index** rebuilt from git contents supports case-base nearest-neighbour retrieval — explicitly a cache, never authoritative. **Audit surface is the §9 narrative plus the final config**, with individual entries pulled on demand as backing evidence.
  - **Cost is non-zero and named:** ~400 entries per problem at ~2 KB per entry × 100 problems ≈ 40K files and <100 MB. Full rebuild is seconds to low minutes. Steady-state ingest is **incremental, keyed on last-seen commit SHA** — O(new entries). Full rebuild remains the migration path for §13 catalog-version transitions.
- **Catalog v0 breadth:** four axes, 24 topologies:
  - detector ∈ {general closed-set, open-vocab, small-object architecture}
  - tile ∈ {off, on}
  - tracker ∈ {motion-only, appearance-assisted}
  - reid ∈ {off, on}

  Components chosen to **span the regime axes**, not to be individually best. Tiling a general detector and the small-object architecture **overlap deliberately on the pixels-on-target axis** — keep both. That axis has the least predictable outcome and resolving it empirically is what the case base is for. Revisit after ~10 cases with evidence, not before.

  Rationale for shipping small: catalog breadth is gated by profiler discriminative power, not the reverse. See mapping-rules decision below — the "hand-authored table" is now per-axis rules, not a cell grid, but the profiler still has to *distinguish* enough regimes to make more axes useful.

- **Regime→topology mapping structure: per-axis rules + override table.** Author one rule per regime axis, each constraining the catalog dimensions it physically implicates. Compose by intersection:

  | Regime axis | Constrains |
  |---|---|
  | pixels-on-target | {detector, tile} |
  | congestion | {tracker} |
  | motion dynamics | {tracker} |
  | appearance separability | {reid} |
  | target novelty | {detector} |

  That is 6–8 rules, O(regimes + topologies), not O(regimes × topologies). Growing either axis costs one rule, not 50 cells.

  Composition misses genuine joint interactions (e.g. congestion says appearance-assisted while motion dynamics says motion-only — same axis, opposite implications). **Layer a short override table on top of the compositional defaults for known joint interactions.** Discovery mechanism for overrides is not the author's intuition — it is §9 `confirmed` verdicts on the bounded joint-refinement pass (§5). Overrides added without ledger evidence are candidates, not decisions.

  **Pending §4 amendment:** §4's "It is a lookup table with a language model on the front" language now describes the wrong structure. The claim should say "per-axis rules composed by intersection, with an evidence-backed override table for known joint interactions." Deferred to design agents per the "no in-place §1–16 edits" standing instruction — flag in chat.

### Superseded
- ~~LLM endpoint: local on-prem from day one, no hosted-API code path in v0.~~ Conflated deployment constraint with development constraint; running only local makes it impossible to distinguish scaffolding weakness from model weakness.
- ~~Ledger audit surface: diff-review of ledger PRs.~~ Nobody reviews 400 JSON files; audit is narrative + config, entries pulled on demand.
- ~~Tier-3 deferred to post-v0; M3 ships tier-0/1 only.~~ Would ship a multi-topology planner scored only by tier-1, reinstating the §6 failure mode. Minimal tier-3 (COCO/MOT file + scorer, no UI) pulled into M3.
- ~~Feasibility rule thresholds hardcoded (e.g. "useful recall below ~20 px").~~ Must be measured against the catalog and versioned with it per §13; hardcoded values go silently wrong on catalog change.
- ~~Catalog v0 is three-axis (3×2×2 = 12 topologies).~~ Missed the scale/tile axis from §5; corrected to four-axis, 24 topologies (see resolved).
- ~~Regime→topology mapping as an enumerated cell grid (~24 topologies × 6–8 regimes ≈ 150 cells).~~ Replaced by per-axis compositional rules + evidence-backed override table (O(regimes + topologies)). Growth costs one rule, not 50 cells.
- ~~Component cost as a scalar field on the registry manifest.~~ Cost is a function of config, measured on target hardware and versioned with the catalog. Scalar cost quietly breaks tier-0 filtering — the case base then learns that tiling is free.

### Risks — amendments to §14

| Risk | Severity | Mitigation |
|---|---|---|
| Uniform-sampled eval set produces false confidence in tier-3 | Critical | Stratified sampling over regime axes + candidate-config disagreement (M4.5). Uniform sampling returns the modal easy case; tier-3 then rank-orders configs that never disagreed on anything hard. |
| Scalar declared cost lets tier-0 gate mis-price tiling | High | Cost is a function of config, measured on target hardware, versioned with catalog. See M1 registry manifest. |
| Compositional mapping misses joint interactions between regime axes | Medium | Override table on top of per-axis rules; overrides added only from §9 `confirmed` ledger verdicts, not intuition. |

### §4.1 proposal — profiler bootstrap

The profiler has a bootstrap problem not addressed in §4. Pixels-on-target and congestion are measured from proposals, so if the open-vocab detector falls in the failed half of the bimodal open-vocab regime (§14 row 3), the regime vector is garbage and every downstream decision is confidently wrong on bad input.

- Probes must be detector-independent where possible (e.g. optical-flow magnitude, frame intensity variance, saturation/clipping — none require object proposals).
- Where a probe is unavoidably proposal-dependent (pixels-on-target, congestion, appearance separability), it **emits a confidence alongside the value**.
- The feasibility gate (§8) **refuses on low profiling confidence** rather than proceeding. No verdict beats a confident-wrong verdict for a team without a CV engineer to interpret it.
- **Probe-discrimination validation** lives in M2: each probe demonstrates on a held-out regime set that it separates regimes it claims to distinguish. Failure to discriminate is a shipping blocker for that probe.

### Still open
- Nothing currently blocking v0 milestone entry.

# Agentic CV Pipeline Harness — Design

**Revision:** v0.3 — implementation notes merged into the body; §§1–16 now editable in place
**Status:** interfaces being frozen; no runtime code yet
**Purpose:** the working design of record. §§1–15 are the design. §§16–18 are the v0 build plan. §19 records what was decided, superseded, and left open.

> **Process change from v0.2:** the "append to Implementation notes, never edit §§1–16" rule existed so reversals stayed legible. It worked, and it is now retired — it was the source of the §5/§15 catalog contradiction. Reversals live in §19 Superseded. Edit the body directly from here on.

---

## 1. Goal and delivery boundary

Take (a) a set of images or video(s) and (b) a natural-language specification of what to detect/track, and automatically assemble, configure, and verify a working vision pipeline from a catalog of pre-built components.

**Target user:** a team that needs a deployed vision pipeline and cannot get CV engineering headcount. Not a CV engineer who wants a faster loop. This drives the feasibility verdict (§8), the drift monitoring (§11), and the exportable handoff (§11).

**Delivery boundary — state this in launch positioning, do not let it stay implicit.** v0 accepts a ground-truth file in a standard format (COCO for detection, MOT for tracking) as its tier-3 anchor. COCO and MOT are CV surface. v0 is therefore deliverable to a friendly pilot who can produce such a file; it is **not** deliverable to the target user above until M4.5 and M5 ship (§17). Showing v0 to the stated target user is showing a developer tool to someone who cannot use it.

**Explicit non-goal:** catalog breadth for its own sake. Coverage of SOTA methods is worth less to this user than a pipeline that still works in November.

---

## 2. Core design commitments

| Commitment | Rationale |
|---|---|
| Frozen, versioned component catalog; agent composes and configures, never authors | Generated code has no provenance, can't be regression-tested, and destroys the audit story for accreditation review. Also makes weak on-prem models tolerable. |
| LLM proposes, deterministic code disposes | LLM never in the inner scoring loop. It navigates coarse search space and diagnoses failure modes; parameter sweeps are a `for` loop. Violating this breaks cost, latency, and reproducibility at once. |
| Measurable properties belong to the profiler, not the agent | Scene characteristics are cheap deterministic measurements. The LLM classifies over a handful of regimes rather than reasoning open-endedly about the data. |
| Append-only decision ledger is the primary artifact | The leave-behind for humans, the input to future agents, and the case base that makes the system improve without the model improving. |
| Human adjudication at one specific point, on domain questions only | Converts label-free search into supervised AutoML. Requires domain knowledge, not CV knowledge. |
| Local model is the ship gate; hosted model is the diagnostic reference | Running only local makes under-specified scaffolding indistinguishable from a weak model, and those have opposite fixes (§18). |

---

## 3. Architecture

| Layer | Responsibility | LLM involvement |
|---|---|---|
| Spec compiler | Natural-language spec → task contract: ontology, spatial/temporal predicates, operating point, hardware envelope, success criteria | High. One-shot, human-confirmed. |
| Data profiler | Cheap deterministic probes → scene properties with per-property confidence (§4, §4.1) | None |
| Feasibility gate | Scene properties + contract → proceed / refuse with physical reason (§8); refuses on low-confidence profiling | Low. Rule-driven. |
| Capability registry | Components with preconditions, I/O contracts, two-tier cost function (§13), license, pinned version | None |
| Planner | Scene properties + contract → candidate topologies via compositional rules (§4.2); seeds from case base when available | Medium. Constrained generation over registry only. |
| Stage specialists | Per-stage optimization: detection, association, ReID/appearance, event logic | Existing VLM tuner drops in as the detection/association specialist |
| Scorer | Tiered scoring stack (§6) | VLM only at tier 2 |
| Packager | Cost calibration, config freeze, container build, ledger export, replay bundle | None |
| Monitor | Drift detection against frozen eval set, re-tune trigger, regression gate | None |

The task contract is the human-confirmed interface. Everything downstream compiles against it. Without it this is a demo rather than a system.

---

## 4. Profiler: scene properties

Every measurable property of the scene is measured. Deterministic code, no model involvement. Terminology: **scene properties** are the measurements below; **regime** is a discrete class of scene (§2) that scene properties place us in.

| Property | Measurement | Proposal-dependent? | Constrains |
|---|---|---|---|
| Pixels on target | Proposal box height distribution, p10/p50 | Yes | Detector, tile strategy |
| Congestion | Detections per frame, mean pairwise IoU, occlusion rate | Yes | Tracker, association gating |
| Motion dynamics | Optical-flow magnitude, per-track displacement variance, camera-motion estimate | No | Tracker, buffer length, Kalman assumptions, camera-motion compensation |
| Lighting | Frame-level intensity variance over time, saturation/clipping fraction | No | Augmentation; whether appearance ReID is viable at all |
| Appearance separability | Inter-instance embedding distance on sampled crops | Yes | ReID on/off, appearance weight in association |
| Target novelty | Open-vocab score distribution against the spec ontology | Yes | Detector |

### 4.1 Probe confidence and the bootstrap problem

Four of six probes are measured **from proposals**, which means they depend on a detector that may itself be failing. If the open-vocab detector lands in the failed half of the bimodal performance range (§14), the scene properties are garbage and every downstream decision is confidently wrong on bad input.

- Probes are detector-independent wherever possible. Optical flow, intensity variance, and saturation/clipping need no proposals.
- Where a probe is unavoidably proposal-dependent, it **emits a confidence alongside the value**, not a bare scalar.
- The feasibility gate **refuses on low profiling confidence** rather than proceeding. For a team with no CV engineer, no verdict beats a confident-wrong verdict.
- **Probe-discrimination validation** (M2): each probe must demonstrate on a held-out set that it distinguishes the scene classes it claims to. Failure to discriminate blocks that probe from shipping. A probe returning a plausible number on every scene is worse than a missing probe.

### 4.2 Scene properties → topology mapping: compositional rules, not a cell table

Terminology: a **topology** is a specific combination of catalog components — one detector + tile setting + tracker + ReID choice. There are 24 in the v0 catalog (§15). The mapping selects a handful of promising topologies for a given scene.

Different scene properties constrain **different** catalog dimensions, so the mapping is authored as per-property rules composed by intersection rather than enumerated over the cross-product:

| Scene property | Constrains |
|---|---|
| Pixels on target | {detector, tile} |
| Target novelty | {detector} |
| Congestion | {tracker} |
| Motion dynamics | {tracker} |
| Appearance separability | {reid} |

6–8 rules, O(properties + topologies). Growing either axis costs one rule, not fifty cells.

Composition misses genuine joint interactions — congestion may argue for appearance-assisted tracking while motion dynamics argues for motion-only. A short **override table** for known joint cases sits on top of the compositional defaults.

**Override promotion criterion.** An override is promoted only when **two single-variable deltas along the axes in question each fail — refuted or inconclusive — to explain the observed improvement**. That isolates the interaction. Joint-refinement-pass `confirmed` verdicts are explicitly *not* sufficient: §9 marks those entries `confounded` precisely because they cannot be attributed to any single axis pair. Anything weaker than the two-delta test is a candidate for further probing, not a decision. Overrides added on correlation become permanent noise in a hand-authored rule set.

**Honest note for implementers:** these rules are hand-authored expertise at the start, not emergent reasoning. That is fine — it is the moat — but it should never be described internally as anything else.

---

## 5. Search space collapse

Worked for the v0 catalog (§15). The general argument holds at any catalog size; the numbers move.

| Stage | v0 | Full-catalog aspiration | Mechanism |
|---|---|---|---|
| Raw catalog | 3 detectors × 2 tile × 2 trackers × 2 ReID = 24 topologies | ~640 topologies | — |
| × parameters | ~8 params × 5 values ≈ 3.9×10⁵ per topology → ~9.4×10⁶ | ~2.5×10⁸ | — |
| Tier-0 constraints | ~14 topologies | ~150 | Latency, VRAM, airgap, license — against the reference cost function with declared hardware scaling (§13) |
| Scene conditioning | 3–5 topologies | 6–10 | §4.2 compositional rules |
| Case-base retrieval | no-op in v0 | 6–10 seeded configs | Post-v0. Registry defaults supply seeds until then |
| Staged local search | ~25–40 evals per topology | ~25–40 | ±2 steps per param, coordinate descent, not grid |
| **Total** | **~100–200 evaluations** | **~200–400 evaluations** | Hours on a short clip |

Scene conditioning is the dominant reduction and it is bounded by profiler discriminative power, not catalog size. That is why catalog breadth is gated by the profiler and not the reverse.

**Stage ordering:** detection quality → association → appearance/ReID → event logic. Detection dominates every downstream metric; tuning association against a bad detector wastes budget.

Greedy staging is provably suboptimal — detector operating point and association thresholds interact. Budget one joint refinement pass over a narrow neighborhood at the end. Expect a few points, not a transformation. Mark all entries from that pass `confounded` (§9).

---

## 6. Scoring: tiered, because proxies don't transfer

**The central technical risk.** Label-free proxy metrics (`short_track_ratio`, `conf_p50_margin`, `near_threshold_fraction`, track-count stability) rank configurations *within* an equivalence class — same detector family, same class vocabulary. They do **not** rank *across* classes.

A fragmented-track signature from an open-vocab detector that half-recognizes the target is indistinguishable from the signature of a strong detector at too high a threshold. The correct response is opposite in each case. Treating proxies as globally comparable will confidently select the wrong architecture.

| Tier | Mechanism | Valid for | v0 |
|---|---|---|---|
| 0 | Hard constraints: latency, VRAM, license, airgap, vocabulary coverage | Filtering, never scoring | Yes |
| 1 | Label-free proxies | Parameters within one topology + vocabulary | Yes |
| 2 | VLM pairwise preference, ranked via Bradley-Terry | Coarse cross-config ordering | No — post-v0 |
| 3 | Ground-truth eval set + scoring function | Cross-config selection, regression gating | Yes, minimal form (§7) |

**All cross-topology selection routes through tier 3 exclusively.** Tier-1 scores are scoped in the type system to intra-topology, intra-vocabulary comparison and are not orderable across topologies at the API level. A multi-topology planner scored by tier 1 is the exact bug this section exists to prevent; where tier 3 is unavailable for a run, the planner is restricted to a single topology.

Tier 2 uses pairwise comparison deliberately: VLM judges are considerably more reliable at "which of these two is better" than at absolute quality scores. Use it to prune, not to decide.

Every ledger entry records which tier produced its score.

---

## 7. The eval anchor

The first thing auto-annotation buys is a **held-out evaluation set**, not training data. With a trustworthy eval slice the problem collapses from label-free search (hard) to supervised AutoML (well understood). A hundred well-chosen labeled frames beats ten thousand noisy pseudo-labels.

| | v0 (minimal tier 3) | Target-user form (M4.5 + M5) |
|---|---|---|
| Input | COCO / MOT file, produced however the pilot likes | Review over auto-generated proposals |
| Tooling | Scoring function only | Proposal pipeline + review UI |
| Audience | Friendly pilot with CV surface tolerance | §1 target user |

### 7.1 Two sets, not one

Sampling on **config disagreement** and sampling for an **unbiased per-scene-class estimate** are different jobs, and merging them breaks the budget (§7.2). Split them:

| Set | Sampled on | Used for | Never used for |
|---|---|---|---|
| Eval set | Scene strata only (~6) | Tier-3 scores, cross-config selection, regression gating, drift baseline | — |
| Diagnostic set | Candidate-config disagreement | Tier-2 pruning, human spot-checks, failure-mode diagnosis | Any scored number |

The diagnostic set is where configs actually differ, which makes it maximally informative and maximally biased. Keeping it out of the headline number is what lets it be sampled aggressively.

### 7.2 Budget derivation

Uniform sampling returns the modal easy case; the eval set then fails to cover regimes where candidates differ, and tier 3 produces false confidence with a credible-looking number. Worse than no tier 3. But stratification has an arithmetic cost that the original half-day budget did not survive:

| Quantity | Merged-set design | Split-set design (§7.1) |
|---|---|---|
| Strata | 6 scene classes × 4 disagreement bands = 24 | 6 scene classes |
| Boxes per stratum at ~1,500 total | ~62 | ~250 |
| CI half-width at p≈0.9 | ±7.5 pts | ±3.7 pts |
| Boxes needed to resolve a 3-pt difference | ~385/stratum → ~9,200 total | ~385/stratum → ~2,300 total |
| Adjudication time | ~9–15 h | ~2.5–4 h |

So: per-stratum CIs at a merged 24-stratum design are an **exclusion** instrument only, useless for ranking close configs. The split design keeps stratum-level rigor within a plausible budget. Revised target-user budget:

| Artifact | Volume | Human action | Est. time |
|---|---|---|---|
| Detection eval set | ~6 scene strata, ~2,300 boxes | Accept / reject / nudge proposals | 2.5–4 h |
| Tracking eval set | 3 clips × ~300 frames | Review track-level events only (births, deaths, switches) | ~1 h |
| Diagnostic set | ~300 boxes on disagreement | Same action, separate store | ~30 min |
| Spec confirmation | 1 pass | Confirm compiled task contract | 10 min |

Roughly half a day still, but say four to six hours rather than "four hours of clicking yes," and stop describing the eval set as ~200 frames.

**Tier-3 API returns per-stratum confidence intervals, never a scalar pooled over strata.** A pooled scalar reintroduces the false-confidence trap that stratification exists to eliminate. Where a pooled number is genuinely wanted (a headline figure for a report), it carries the per-stratum spread alongside it.

**Sampler degradation under budget pressure:** oversample scenes with low profiling confidence (§4.1); never silently drop a stratum — an empty stratum is reported as no-estimate, not as a missing row.

**UI requirement (M5):** the adjudication interface exposes zero CV surface — no mAP, no thresholds, no configs. "Is this the thing, yes or no." A domain expert can do this; that is why it does not require the headcount the team cannot get.

---

## 8. Feasibility verdict — say no early

An expert's highest-leverage output is frequently a refusal. The gate must be able to return, *before* the search runs:

> Target averages 11 px tall. No catalog method achieves useful recall below ~20 px. Change the lens or accept the number.

A team without a CV engineer cannot distinguish "the system worked hard and 45% recall is the ceiling for this data" from "the system is broken." Without an early verdict carrying a specific physical reason, they get a mediocre pipeline they cannot interpret and conclude the product does not work.

**Thresholds are catalog-derived and version-pinned.** "Useful recall below ~20 px" is a claim about the catalog. Derive from catalog benchmark data, pin to the catalog version that produced it (§13), re-derive on catalog change. A hardcoded threshold goes silently wrong the first time a component is added or retired.

Starting rule set: pixels-on-target floor, congestion ceiling, appearance-separability minimum for ReID, low-profiling-confidence refusal (§4.1).

Report a **performance envelope**, not a point estimate: where the pipeline is reliable, where it degrades, what it will miss.

---

## 9. The decision ledger

Append-only. The winning config is a pointer into the ledger, not the artifact itself. Negative results survive — the branches that failed and why *are* the search pruning, and are worth more to the next agent than the final config.

| Field | Purpose |
|---|---|
| `config_delta` | What changed; structurally diffable |
| `evidence` | Triggering metric values — values, not adjectives |
| `hypothesis` | Predicted effect, stated numerically **before** the run |
| `outcome` | Measured effect |
| `verdict` | confirmed / refuted / inconclusive — computed, not narrated |
| `scene_properties` | Profiler output at the time, with per-property confidence (§4.1) |
| `scorer_tier` | Which tier produced the score |
| `confounded` | True for joint-refinement entries and any multi-variable change |
| `cost_estimated` | Planner's reference-cost estimate at decision time (§13) |
| `cost_measured` | Target-measured cost where available |
| `provenance` | Model, model version, seed, catalog version |

`hypothesis` + `verdict` is what makes the ledger worth keeping. A later run can ask "in this regime, did raising the spawn threshold historically help?" and answer from evidence rather than a prior baked into a prompt. It also yields a measurable quality signal on the local endpoint: a high refutation rate means the model is reasoning badly.

`cost_estimated` vs `cost_measured` divergence is its own signal: sustained drift means the reference SKU no longer represents the fleet.

**Confirmed ≠ caused.** A proxy improving after a change is not attribution. Prefer single-variable deltas during staged search; flag everything else `confounded` so retrieval can downweight it.

**Two renderings, one store.** Future agents need the full typed record. A human returning in month seven needs a short causal narrative: what the data looked like, why this detector, known weak spots. Generate the narrative *at decision time* — do not depend on whatever model is available later.

### 9.1 Storage

Files-in-git is the system of record: one append-only file per entry, catalog version pinned in the entry (§13). Free provenance for regulated review (§11), works airgapped.

A **derived index** (SQLite or parquet) over scene properties supports case-base nearest-neighbour retrieval. Explicitly a cache, never authoritative, rebuildable from git by one command. Git is not an index; retrieval degrades badly past ~10 cases at 100–200 entries each.

Rebuild cost is real and small: ~400 entries/problem at ~2 KB is roughly 40K files and under 100 MB at 100 problems, so a full rebuild is seconds to low minutes. Steady state uses **incremental ingest keyed on last-seen commit SHA**, O(new entries). Full rebuild remains the migration path for §13.

PR diff-review of ledger files is **not** an audit surface — nobody reviews 400 JSON files. The audit surface is the narrative plus the final config, with entries pulled on demand as backing evidence.

---

## 10. Case base

Every solved problem writes `(scene_properties, winning_config, achieved_score, spec_class)` to the store. New problems retrieve nearest neighbours and seed from them.

Retrieval requires almost no model capability, so **the system improves over time without the on-prem model improving**. This decoupling is the strongest argument for the frozen-catalog architecture. Schema and storage are built in v0 (§9.1); retrieval logic is post-v0.

---

## 11. Operations

**Drift.** Replacing a person means inheriting the job they would have done later: camera bumped, seasons change, new site with different lighting. Nobody on this team will notice degradation. Drift detection against the frozen eval set, automatic re-tune, regression gate. Core to the value proposition, though not in v0 (§17).

**Handoff.** When the system fails, the team's only move is to hire a contractor for a week. Export a real repo: configs, eval set, ledger, replay bundle. An opaque service makes the first serious failure terminal for the account.

**Provenance.** For federal/regulated deployment, "why did the system choose this pipeline" will be asked in review. The ledger must answer it without reconstruction.

---

## 12. The glue problem

Component selection is not where real deployments die. Glue is: ROI masks, line-crossing logic, class merging, dwell-time rules, alert debounce, cross-camera dedup, calibration.

With a frozen catalog, all of this must be expressible declaratively. Required:

- A typed pipeline graph spec
- A small predicate language for event logic

The most underestimated piece in the design. Skip it and the agent routinely reaches a pipeline that is 90% right and needs three lines of code it is not allowed to write.

**Controlled escape hatch (optional, one slot only):** generated code permitted in exactly one place with a fixed signature — a pure function from a track record to a boolean. Sandboxed, no imports, unit-tested against the eval set, auto-rejected on exception or regression. Weak models write a ten-line predicate reliably; they cannot maintain a codebase. That distinction is what the architecture encodes.

---

## 13. Versioning and the cost model

The ledger references catalog components and their parameters. Retire a detector or rename a parameter and every historical entry silently degrades into noise that still reads as authoritative.

- Version the catalog
- Pin ledger entries to catalog versions
- Pin feasibility thresholds (§8) to the catalog version that produced them
- Write migrations on deprecation

### 13.1 Cost is a two-tier function of config

Cost is never a scalar manifest field. Tiling cost scales with tile count — 2×2 with overlap is roughly 4–5× a single-pass detector — so a flat field teaches the case base that tiling is free, and the wrong lesson becomes durable.

| Tier | Measured | Used by |
|---|---|---|
| Reference cost | On a canonical reference SKU at registry-register time, as a function of config parameters (tile count, input resolution, batch size, tracker window). Cached with catalog version | Planner during search, with a declared hardware scaling function |
| Target cost | Re-measured on the hardware named in the `TaskContract` envelope, during a mandatory calibration pass before config freeze | **Tier-0 gate. Never reference-scaled cost** |

The planner needs an estimate before target hardware is in the loop, which is why reference cost exists; the tier-0 gate must not ship a latency claim derived from a scaling function, which is why target cost exists. The ledger records both (§9), and sustained divergence flags reference-SKU drift.

---

## 14. Known risks

| Risk | Severity | Mitigation |
|---|---|---|
| Proxy metrics treated as cross-config comparable | Critical | Tiered scorer (§6); tier-1 not orderable across topologies at the API level; planner restricted to one topology absent tier 3 |
| Eval-set sampling produces false confidence | Critical | §7.1 split sets; scene strata only for scored numbers; per-stratum CIs, never a pooled scalar |
| Profiler bootstrap: probes measured from failing proposals | High | §4.1 — per-probe confidence, feasibility refuses on low confidence, probe-discrimination validation in M2 |
| "Arbitrary CV problem" scope creep | High | Two verticals. Surveillance/security/inspection first — roughly rigid targets, mostly static cameras |
| Open-vocab performance is bimodal, not continuous | High | Early regime check that escalates to few-shot/fine-tune instead of grinding thresholds in the useless regime |
| Cost modelled as a scalar, or tier-0 gating on scaled cost | High | §13.1 two-tier model; tier-0 gates on target-measured cost only |
| Compositional mapping misses joint interactions | Medium | §4.2 override table, promoted only on the two-failed-single-variable-delta criterion |
| Agent proliferation | Medium | One orchestrator + 5–6 tools with strict schemas |
| On-prem model degradation | Medium | §18 — contracts authored and validated against local; local is the ship gate |
| Glue expressiveness underestimated | Medium | §12 — predicate language in M4, not deferred |
| Greedy staging misses interactions | Low | Bounded joint refinement pass, marked confounded |

---

## 15. v0 catalog

Four axes, 24 topologies. Chosen to span the scene properties rather than to be individually best.

| Catalog axis | v0 values | Scene property covered |
|---|---|---|
| Detector | general closed-set, open-vocab, small-object architecture | Target novelty; pixels on target |
| Tile strategy | off, on | Pixels on target |
| Tracker | motion-only, appearance-assisted | Congestion, motion dynamics |
| ReID | off, on | Appearance separability |

**Deliberate overlap.** Tiling a general detector and using a small-object architecture attack pixels-on-target by different means. Both stay in. That axis has the least predictable answer in advance, and learning which wins per regime is what the case base is for. Revisit after ~10 solved cases with evidence, not before. This only works if §13.1's cost model is honest — the two have materially different latency and VRAM.

Also in v0 scope: person and vehicle plus open-vocab targets; surveillance video only; existing VLM tuner as the detection/association specialist.

---

## 16. Frozen interfaces

Freeze before writing runtime code:

| # | Interface | Why it cannot be retrofitted |
|---|---|---|
| 1 | `TaskContract` | Everything compiles against it |
| 2 | `LedgerEntry` | Schema drift silently corrupts the case base (§13) |
| 3 | Registry component manifest, including the two-tier cost function signature (§13.1) | Scalar cost is the schema change that most quietly ruins tier-0 and the case base |
| 4 | `EvalSet` — GT file reference + scorer id + catalog pin + stratum labels | Cross-topology selection depends on it; retrofitting means re-running search |
| 5 | `MappingRules` — per-property rules + override table (§4.2) | Adding a topology or scene class is a one-rule change against this schema, not a grid rewrite |
| 6 | `SceneProperties` — §4 properties with per-property confidence (§4.1) | Confidence added later means every historical entry is unlabelled |

---

## 17. Milestones

| M | Contents | Notes |
|---|---|---|
| M1 | Contracts & registry, schemas only, no runtime | The six interfaces in §16, as JSON Schemas + Pydantic bindings |
| M2 | Profiler + feasibility gate, deterministic | Each §4 scene property as an independent probe with confidence where proposal-dependent. Catalog-derived, version-pinned thresholds (§8). Probe-discrimination validation gates each probe |
| M3 | Planner + ledger + tiers 0/1/3-minimal | Planner via §4.2 compositional rules; LLM only for constrained generation over registry entries. Ledger with dual rendering and derived index (§9.1). Tier 3 ingests COCO/MOT + scoring function — **this is what makes a multi-topology planner legitimate**. VLM tuner drops in here |
| M4 | Packager + glue | Cost calibration pass (§13.1), config freeze, container build, replay bundle, ledger export. Predicate language for event logic (§12) — minimal grammar, not deferred. **v0 exits here** |
| M4.5 | Proposal pipeline | Open-vocab + SAM2 proposals. **Load-bearing component is the sampling strategy, not the models** (§7). Two-phase: scene-stratified pass first, disagreement pass after initial candidates propose |
| M5 | Adjudication UI | Review over M4.5 proposals, zero CV surface (§7). Emits the same COCO/MOT format M3 already ingests. **This milestone is the boundary between a developer tool and the product §1 describes** |
| post-v0 | Tier-2 Bradley-Terry, case-base retrieval, drift monitor | Designed for now, built later |

---

## 18. The validation gate

**Endpoint policy.** The LLM endpoint interface is provider-agnostic. Local and hosted are both configured in development. **Local is the ship gate; hosted is the diagnostic reference** for separating under-specified scaffolding from weak-model behaviour, which have opposite fixes. The §9 refutation-rate signal is the comparison instrument. Contracts are authored and validated against local first — if they drift toward what the hosted model can do, the gate silently stops constraining anything. Hosted-only failures are logged, not gating. The deployment airgap constraint stands: hosted is a dev-time tool, never on the runtime path.

**Change detection is mechanical, not intentional.** Hash each contract's JSON Schema, pin the hashes in a manifest, CI fails on mismatch and re-runs the fixture suite against the local endpoint. A convention that developers remember to re-validate is not a gate.

| Check | Threshold | Gating |
|---|---|---|
| Conformance | ≥98% of ~100 fixture inputs spanning regimes return schema-valid parseable output | Yes |
| Referential integrity | 100% of referenced component/param names resolve to real registry entries | Yes, hard fail. Constrained generation should make this unreachable |
| Refutation rate (§9) | None yet | **No.** No baseline exists, so any floor set now is invented. Log from day one; set after ~20 solved problems |

---

## 19. Decision log

### Resolved

- **Language:** Python end-to-end, Pydantic for typed contracts.
- **Endpoint:** provider-agnostic; local gates, hosted diagnoses (§18).
- **Ledger store:** files-in-git authoritative, derived index as rebuildable cache, incremental ingest on commit SHA (§9.1).
- **v0 catalog:** four axes, 24 topologies (§15).
- **Scene properties → topology mapping:** compositional per-property rules plus override table with the two-delta promotion criterion (§4.2).
- **Tier 3 in v0:** minimal form, GT file + scoring function; UI deferred to M5.
- **Cost:** two-tier, reference for search and target-measured for the tier-0 gate (§13.1).
- **Eval sampling:** split eval set (scene strata, scored) from diagnostic set (disagreement, never scored) (§7.1).

### Superseded

| Was | Now | Why |
|---|---|---|
| Local on-prem only, no hosted code path in v0 | Both configured; local gates, hosted diagnoses | Local-only makes under-specified scaffolding indistinguishable from a weak model |
| Tier 3 deferred to post-v0 | Minimal tier 3 in M3 | M3 ships a multi-topology planner; tier-1-only selection is the §6 failure mode |
| Ledger PR diff-review is a real audit surface | It is not | Nobody reviews 400 JSON files |
| M5 contains "nothing else" | Split M4.5 / M5 | The UI sits on a non-trivial proposal + sampling pipeline |
| Component cost as a scalar manifest field | Two-tier measured cost function (§13.1) | A flat field teaches the case base that tiling is free |
| Cost measured on target hardware, full stop | Reference cost for search, target cost for the gate | Unimplementable as stated — the planner needs an estimate before target hardware is in the loop |
| Scene → topology as a dense cell table | Compositional per-property rules | 24 topologies × 6–10 scene classes crosses 200 cells; composition is O(properties + topologies) |
| Overrides promoted from `confirmed` ledger verdicts | Two failed single-variable deltas required | Joint-pass entries are `confounded` by §9 and cannot be attributed to an axis pair |
| Catalog v0 is three-axis (12 topologies) | Four-axis, 24 | Scale/tile axis was dropped from the enumeration by mistake |
| One eval set, sampled on scene × disagreement | Two sets (§7.1) | 24 strata at ~62 boxes gives ±7.5-pt CIs — exclusion only, useless for ranking |
| Eval budget ~200 frames / 1.5–2.5 h | ~2,300 boxes / 2.5–4 h detection, plus tracking and diagnostic | Original figure was derived for unstratified review |
| Feasibility thresholds hardcoded | Catalog-derived, version-pinned | Hardcoded values go silently wrong on catalog change |
| §§1–16 append-only, changes go to Implementation notes | Body is editable; reversals recorded here | The rule produced the §5/§15 catalog contradiction |

### Open

- Second vertical, after surveillance/security/inspection.
- Whether the deliberate §15 overlap survives ~10 solved cases.
- Whether 6 scene strata is the right granularity, or whether some scene properties need splitting once §4.1 confidence data exists — each split multiplies the §7.2 budget.
- Tracking eval budget has not had the §7.2 treatment. Track-level event review is a different unit than boxes and the 3-clip figure is inherited, not derived.

---

## 20. Research angle

No benchmark exists of `(spec, data, expected pipeline)` triplets. Defining one is a real contribution independent of the system itself.

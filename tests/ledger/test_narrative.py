"""Narrative rendering tests. Deterministic template output — assertions
check that specific fields land in the string, not the exact prose."""

from __future__ import annotations

from cv_agent.ledger import render_narrative
from cv_agent.schemas import (
    AppearanceSeparability,
    Congestion,
    Lighting,
    MetricReading,
    MotionDynamics,
    PixelsOnTarget,
    Prediction,
    ProbeConfidence,
    SceneProperties,
    ScorerTier,
    TargetNovelty,
    Verdict,
)


# --- helpers ---------------------------------------------------------------


def _scene(**overrides) -> SceneProperties:
    defaults = dict(
        pixels_on_target=PixelsOnTarget(p10_px=32.0, p50_px=42.0, confidence=ProbeConfidence.HIGH),
        congestion=Congestion(
            detections_per_frame_mean=6.2,
            mean_pairwise_iou=0.08,
            occlusion_rate=0.15,
            confidence=ProbeConfidence.MEDIUM,
        ),
        motion_dynamics=MotionDynamics(
            optical_flow_mag_p50=1.4,
            per_track_displacement_variance=0.9,
            camera_motion_estimate=0.02,
            confidence=ProbeConfidence.HIGH,
        ),
        lighting=Lighting(
            intensity_variance_over_time=310.5,
            saturation_clipping_fraction=0.01,
            confidence=ProbeConfidence.HIGH,
        ),
        appearance_separability=AppearanceSeparability(
            inter_instance_distance_p50=0.62, confidence=ProbeConfidence.MEDIUM
        ),
        target_novelty=TargetNovelty(
            open_vocab_score_p10=0.31, open_vocab_score_p50=0.55, confidence=ProbeConfidence.HIGH
        ),
        catalog_version="v0-2026-09-17",
    )
    defaults.update(overrides)
    return SceneProperties(**defaults)


def _base_kwargs(**overrides):
    defaults = dict(
        changed_parameters=("detector.confidence_threshold",),
        full_config={"detector.confidence_threshold": 0.4},
        hypothesis=(
            Prediction(
                metric_name="short_track_ratio",
                baseline_value=0.42,
                predicted_delta=-0.10,
                tolerance=0.03,
            ),
        ),
        outcome=(MetricReading(name="short_track_ratio", value=0.33),),
        verdict=Verdict.CONFIRMED,
        scene_properties=_scene(),
        scorer_tier=ScorerTier.LABEL_FREE_PROXY,
        confounded=False,
    )
    defaults.update(overrides)
    return defaults


# --- section: change --------------------------------------------------------


def test_single_change_names_parameter_and_value():
    text = render_narrative(**_base_kwargs())
    assert "Changed detector.confidence_threshold=0.4" in text


def test_multi_change_lists_all_parameters():
    text = render_narrative(
        **_base_kwargs(
            changed_parameters=("detector.confidence_threshold", "tile.tile_count"),
            full_config={"detector.confidence_threshold": 0.4, "tile.tile_count": 4},
            confounded=True,  # required for multi-variable per §9
        )
    )
    assert "2 parameters" in text
    assert "detector.confidence_threshold=0.4" in text
    assert "tile.tile_count=4" in text


def test_root_entry_says_new_root_config():
    text = render_narrative(**_base_kwargs(changed_parameters=()))
    assert "New root config" in text


# --- section: hypothesis + outcome + verdict -------------------------------


def test_confirmed_verdict_appears():
    text = render_narrative(**_base_kwargs(verdict=Verdict.CONFIRMED))
    assert "Verdict: confirmed" in text


def test_refuted_verdict_appears():
    text = render_narrative(**_base_kwargs(verdict=Verdict.REFUTED))
    assert "Verdict: refuted" in text


def test_hypothesis_names_metric_and_predicted_delta():
    text = render_narrative(**_base_kwargs())
    assert "short_track_ratio" in text
    assert "-0.100" in text  # predicted_delta with :+.3f
    assert "±0.030" in text


def test_measured_delta_included_when_outcome_matches():
    text = render_narrative(**_base_kwargs())
    # baseline 0.42, outcome 0.33 → measured Δ = -0.090
    assert "measured Δ=-0.090" in text


def test_stratum_named_when_present():
    hyp = (
        Prediction(
            metric_name="recall",
            baseline_value=0.5,
            predicted_delta=0.1,
            tolerance=0.05,
            stratum="small_targets",
        ),
    )
    out = (
        MetricReading(
            name="recall",
            value=0.62,
            stratum="small_targets",
            ci_low=0.55,
            ci_high=0.68,
        ),
    )
    text = render_narrative(**_base_kwargs(hypothesis=hyp, outcome=out))
    assert "[small_targets]" in text


def test_outcome_without_matching_prediction_shown_as_pred_only():
    # Hypothesis mentions metric X, outcome has X and Y — narrative should
    # still show X with measured delta, and not crash on Y.
    hyp = (
        Prediction(
            metric_name="short_track_ratio",
            baseline_value=0.42,
            predicted_delta=-0.10,
            tolerance=0.03,
        ),
    )
    out = (
        MetricReading(name="short_track_ratio", value=0.33),
        MetricReading(name="conf_p50_margin", value=0.12),
    )
    text = render_narrative(**_base_kwargs(hypothesis=hyp, outcome=out))
    assert "short_track_ratio" in text
    assert "measured Δ=-0.090" in text


# --- section: scene --------------------------------------------------------


def test_scene_summary_present():
    text = render_narrative(**_base_kwargs())
    assert "target p50 42.0px" in text
    assert "6.2/frame" in text
    assert "0.15 occlusion" in text
    assert "motion flow p50 1.40" in text
    assert "lighting variance 310.5" in text
    assert "separability 0.62" in text
    assert "target-novelty p50 0.55" in text


# --- section: caveats ------------------------------------------------------


def test_confounded_caveat():
    text = render_narrative(**_base_kwargs(confounded=True))
    assert "Confounded" in text


def test_no_confounded_caveat_when_not_flagged():
    text = render_narrative(**_base_kwargs(confounded=False))
    assert "Confounded" not in text


def test_low_confidence_axes_named():
    scene = _scene(
        target_novelty=TargetNovelty(
            open_vocab_score_p10=0.05, open_vocab_score_p50=0.08, confidence=ProbeConfidence.LOW
        ),
    )
    text = render_narrative(**_base_kwargs(scene_properties=scene))
    assert "LOW confidence" in text
    assert "target_novelty" in text


def test_multiple_low_axes_named():
    scene = _scene(
        lighting=Lighting(
            intensity_variance_over_time=310.5,
            saturation_clipping_fraction=0.01,
            confidence=ProbeConfidence.LOW,
        ),
        target_novelty=TargetNovelty(
            open_vocab_score_p10=0.05, open_vocab_score_p50=0.08, confidence=ProbeConfidence.LOW
        ),
    )
    text = render_narrative(**_base_kwargs(scene_properties=scene))
    assert "lighting" in text
    assert "target_novelty" in text


def test_no_low_confidence_caveat_when_none_low():
    text = render_narrative(**_base_kwargs())
    assert "LOW confidence" not in text


# --- section: tier ---------------------------------------------------------


def test_tier_0_caveat():
    text = render_narrative(**_base_kwargs(scorer_tier=ScorerTier.HARD_CONSTRAINT))
    assert "Tier 0" in text
    assert "hard-constraint" in text


def test_tier_1_caveat():
    text = render_narrative(**_base_kwargs(scorer_tier=ScorerTier.LABEL_FREE_PROXY))
    assert "Tier 1" in text
    assert "not orderable across topologies" in text


def test_tier_3_caveat():
    text = render_narrative(**_base_kwargs(scorer_tier=ScorerTier.GROUND_TRUTH))
    assert "Tier 3" in text
    assert "per-stratum CIs" in text


# --- determinism -----------------------------------------------------------


def test_same_input_produces_same_output():
    kwargs = _base_kwargs()
    a = render_narrative(**kwargs)
    b = render_narrative(**kwargs)
    assert a == b


# --- integration: renderer produces a valid LedgerEntry.narrative ----------


def test_output_satisfies_ledger_entry_narrative_field():
    from datetime import datetime, timezone

    from cv_agent.schemas import (
        CostMeasurement,
        LedgerEntry,
        Provenance,
    )

    text = render_narrative(**_base_kwargs())
    assert text  # non-empty (LedgerEntry.narrative has min_length=1)

    # Round-trip: build a LedgerEntry with this narrative and it validates.
    entry = LedgerEntry(
        id="entry-0001",
        created_at=datetime(2026, 9, 18, tzinfo=timezone.utc),
        task_contract_name="dock",
        full_config={"detector.confidence_threshold": 0.4},
        changed_parameters=("detector.confidence_threshold",),
        scene_properties=_scene(),
        hypothesis=(
            Prediction(
                metric_name="short_track_ratio",
                baseline_value=0.42,
                predicted_delta=-0.10,
                tolerance=0.03,
            ),
        ),
        evidence=(MetricReading(name="short_track_ratio", value=0.42),),
        outcome=(MetricReading(name="short_track_ratio", value=0.33),),
        verdict=Verdict.CONFIRMED,
        scorer_tier=ScorerTier.LABEL_FREE_PROXY,
        confounded=False,
        cost_estimated=CostMeasurement(
            config={"input_res": 640}, latency_ms=18.0, vram_mb=1400.0
        ),
        provenance=Provenance(
            model_id="claude-opus-4-7",
            model_version="4.7.0",
            seed=42,
            catalog_version="v0-2026-09-17",
        ),
        narrative=text,
    )
    assert entry.narrative == text

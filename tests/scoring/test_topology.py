"""Topology runtime type — kind validation, equality, hashing."""

from __future__ import annotations

import pytest

from cv_agent.scoring import Topology

from .conftest import make_detector, make_reid, make_tile, make_topology, make_tracker


def test_topology_constructs():
    t = make_topology()
    assert t.detector.name == "general_v1"
    assert len(t.components()) == 4
    assert t.component_names() == (
        "general_v1",
        "tile_off_v1",
        "motion_only_v1",
        "reid_off_v1",
    )


def test_topology_rejects_wrong_kind_on_detector_slot():
    with pytest.raises(ValueError):
        Topology(
            detector=make_tracker(),  # wrong kind
            tile=make_tile(),
            tracker=make_tracker(),
            reid=make_reid(),
        )


def test_topology_rejects_wrong_kind_on_tracker_slot():
    with pytest.raises(ValueError):
        Topology(
            detector=make_detector(),
            tile=make_tile(),
            tracker=make_detector(),  # wrong kind
            reid=make_reid(),
        )


def test_topology_hashable_and_setable():
    t1 = make_topology()
    t2 = make_topology()
    # Same shape → equal + same hash.
    assert t1 == t2
    assert hash(t1) == hash(t2)
    assert {t1, t2} == {t1}


def test_topology_hash_reflects_config():
    t1 = Topology(
        detector=make_detector(),
        tile=make_tile(),
        tracker=make_tracker(),
        reid=make_reid(),
        config={"detector.confidence_threshold": 0.4},
    )
    t2 = Topology(
        detector=make_detector(),
        tile=make_tile(),
        tracker=make_tracker(),
        reid=make_reid(),
        config={"detector.confidence_threshold": 0.5},
    )
    assert t1 != t2
    assert hash(t1) != hash(t2)


def test_describe_names_all_components():
    t = make_topology()
    desc = t.describe()
    assert "general_v1" in desc
    assert "tile_off_v1" in desc
    assert "motion_only_v1" in desc
    assert "reid_off_v1" in desc

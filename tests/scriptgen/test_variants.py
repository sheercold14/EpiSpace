"""Intervention operator tests.

Unit tests exercise the operators and the MISMATCH blocking on the fake
render backend; integration tests build all four variants on each of the
three rendered trajectories and assert the recompiled golds.
"""

from __future__ import annotations

import pytest

from spatial_episode.scriptgen.compiler import CapabilityCompiler
from spatial_episode.scriptgen.library import SELF_MOTION
from spatial_episode.scriptgen.standards import STD_V1
from spatial_episode.scriptgen.variants import (
    INTERVENTION_KINDS,
    FamilyMismatch,
    VariantBuilder,
)

from _batchdata import load_plan_record, load_render_view, needs_batch
from test_compiler import BINDING, qualifying_fake

ABSTAIN = SELF_MOTION.templates[0].abstain_option


@pytest.fixture(scope="module")
def fake_builder() -> VariantBuilder:
    compiler = CapabilityCompiler(script=SELF_MOTION, std=STD_V1)
    view = qualifying_fake()
    canonical = compiler.compile(view, BINDING, with_essential=True)
    return VariantBuilder(compiler=compiler, view=view, binding=BINDING, canonical=canonical)


def test_all_four_kinds_built(fake_builder: VariantBuilder) -> None:
    variants = fake_builder.build_all(seed=3)
    assert tuple(v.kind for v in variants) == INTERVENTION_KINDS


def test_only_declared_kinds_are_built() -> None:
    script = SELF_MOTION.model_copy(
        update={"variant_expectations": {"drop_filler": "same", "delay": "same"}}
    )
    compiler = CapabilityCompiler(script=script, std=STD_V1)
    view = qualifying_fake()
    canonical = compiler.compile(view, BINDING, with_essential=True)
    builder = VariantBuilder(
        compiler=compiler, view=view, binding=BINDING, canonical=canonical
    )
    assert tuple(v.kind for v in builder.build_all(seed=3)) == (
        "drop_filler",
        "delay",
    )


def test_permute_destroys_tracking(fake_builder: VariantBuilder) -> None:
    permute = fake_builder.build_all(seed=3)[0]
    assert permute.gold == ABSTAIN
    assert permute.certificate.status == "abstain"
    assert permute.certificate.reason == "clause:trackable"
    # Same multiset of frames, different order, question frame kept last.
    assert sorted(permute.frame_sequence) == list(range(13))
    assert permute.frame_sequence[-1] == 12
    assert permute.frame_sequence != tuple(range(13))


def test_drop_key_removes_every_sighting(fake_builder: VariantBuilder) -> None:
    drop_key = fake_builder.build_all(seed=3)[1]
    assert drop_key.gold == ABSTAIN
    assert drop_key.certificate.reason == "frame_var_unresolvable:t_seen"
    # Frames 0 and 1 are the sightings; nothing visible may remain.
    assert set(drop_key.frame_sequence) == set(range(2, 13))


def test_drop_filler_keeps_answer(fake_builder: VariantBuilder) -> None:
    drop_filler = fake_builder.build_all(seed=3)[2]
    assert drop_filler.gold == "left"
    assert drop_filler.certificate.status == "answerable"
    assert len(drop_filler.frame_sequence) < 13
    # Anchors survive: the sighting frame and the question frame.
    assert 1 in drop_filler.frame_sequence and 12 in drop_filler.frame_sequence
    # Essential frames survive by construction.
    assert set(fake_builder.canonical.essential_frames or ()) <= set(
        drop_filler.frame_sequence
    )


def test_delay_grows_knob_only(fake_builder: VariantBuilder) -> None:
    delay = fake_builder.build_all(seed=3, delay_extra=4)[3]
    assert delay.gold == "left"
    canonical_delay = fake_builder.canonical.knob_levels["delay"]
    assert delay.certificate.knob_levels["delay"] == canonical_delay + 4
    assert len(delay.frame_sequence) == 13 + 4
    # The repeated frame is a standstill: consecutive duplicates only.
    repeated = [t for t in set(delay.frame_sequence) if delay.frame_sequence.count(t) > 1]
    assert len(repeated) == 1


def test_variants_never_hand_assign_gold(fake_builder: VariantBuilder) -> None:
    """Every gold equals what the certificate itself concluded."""
    for variant in fake_builder.build_all(seed=3):
        if variant.expectation == "same":
            assert variant.certificate.answer is not None
            assert variant.gold == variant.certificate.answer.label
        else:
            assert variant.certificate.status == "abstain"
            assert variant.gold == ABSTAIN


def test_expectation_disagreement_blocks() -> None:
    """A spec that mis-declares an expectation must MISMATCH, not ship."""
    wrong = SELF_MOTION.model_copy(
        update={"variant_expectations": {**SELF_MOTION.variant_expectations, "permute": "same"}}
    )
    compiler = CapabilityCompiler(script=wrong, std=STD_V1)
    view = qualifying_fake()
    canonical = compiler.compile(view, BINDING, with_essential=True)
    builder = VariantBuilder(compiler=compiler, view=view, binding=BINDING, canonical=canonical)
    with pytest.raises(FamilyMismatch, match="permute"):
        builder.build_all(seed=3)


def test_unanswerable_canonical_rejected() -> None:
    compiler = CapabilityCompiler(script=SELF_MOTION, std=STD_V1)
    view = qualifying_fake()
    bad = compiler.compile(view, BINDING, frame_sequence=tuple(range(2, 13)))
    assert bad.status != "answerable"
    with pytest.raises(ValueError, match="answerable"):
        VariantBuilder(compiler=compiler, view=view, binding=BINDING, canonical=bad)


# --- integration: the three rendered trajectories ---

EXPECTED_SECTORS = {0: "left", 1: "left", 2: "right"}


@needs_batch
@pytest.mark.parametrize("index", [0, 1, 2])
def test_rendered_bundle_variants(index: int, self_motion_compiler: CapabilityCompiler) -> None:
    plan = load_plan_record(index)
    view = load_render_view(index)
    canonical = self_motion_compiler.compile(
        view, plan["binding"], geometry_plan=plan, with_essential=True
    )
    builder = VariantBuilder(
        compiler=self_motion_compiler, view=view, binding=plan["binding"], canonical=canonical
    )
    permute, drop_key, drop_filler, delay = builder.build_all(seed=17)

    sector = EXPECTED_SECTORS[index]
    assert canonical.answer is not None and canonical.answer.label == sector
    assert permute.gold == ABSTAIN and permute.certificate.reason == "clause:trackable"
    assert drop_key.gold == ABSTAIN
    assert drop_filler.gold == sector
    assert delay.gold == sector
    assert (
        delay.certificate.knob_levels["delay"] == canonical.knob_levels["delay"] + 4
    )
    # drop_key kept only definitely-invisible frames.
    assert all(
        row.tristate == "invisible" for row in drop_key.certificate.target_visibility
    )

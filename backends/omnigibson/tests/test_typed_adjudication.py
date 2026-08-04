from omnigibson_episode.typed_adjudication import adjudicate_quality


def _quality(*, gate_pass: bool = True, rgb_std: float = 30.0) -> dict:
    return {
        "trajectory_class": "T10",
        "integrity_status": "pass",
        "visual_status": "pass" if rgb_std >= 5.0 else "warning",
        "gates": {
            "T10": {
                "status": "pass" if gate_pass else "fail",
                "checks": {"facing_anchor_visible": gate_pass},
            }
        },
        "views": [
            {
                "view_id": "view-000",
                "rgb_std": rgb_std,
                "rgb_p99": 220.0,
                "rgb_p01": 10.0,
                "sharpness_laplacian_variance": 20.0,
                "visible_instance_count": 4,
            }
        ],
    }


def test_typed_evidence_failure_cannot_be_promoted_by_good_rgb() -> None:
    result = adjudicate_quality(_quality(gate_pass=False))

    assert result["decision"] == "reject_typed_evidence"
    assert result["failed_typed_checks"] == ["facing_anchor_visible"]


def test_near_uniform_rgb_is_an_automatic_visual_rejection() -> None:
    result = adjudicate_quality(_quality(rgb_std=2.0))

    assert result["decision"] == "reject_visual_information"
    assert result["visual_blockers"][0]["reasons"] == ["near_uniform_rgb"]

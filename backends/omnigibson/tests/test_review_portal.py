from omnigibson_episode.review_portal import _page


def test_review_portal_summarizes_each_evidence_unit() -> None:
    page = _page(
        [
            {
                "kind": "static",
                "title": "静态多视图基线",
                "subtitle": "检查任务",
                "sweep_id": "static-v1",
                "count": 46,
                "status_counts": {"passed": 39, "failed": 7},
                "link": "../static/review/index.html",
            }
        ]
    )

    assert "Episode3D Review Hub" in page
    assert "46" in page and "static-v1" in page
    assert "../static/review/index.html" in page
    assert "PRODUCTION · static" in page


def test_review_portal_visually_demotes_historical_sweeps() -> None:
    page = _page(
        [
            {
                "kind": "model_swap",
                "stage": "historical_audit",
                "title": "同类模型替换",
                "subtitle": "旧协议",
                "sweep_id": "model-swap-v4",
                "count": 3,
                "status_counts": {"failed": 3},
                "link": "review.html",
            }
        ]
    )

    assert "HISTORICAL AUDIT · model_swap" in page
    assert "查看失败模式" in page

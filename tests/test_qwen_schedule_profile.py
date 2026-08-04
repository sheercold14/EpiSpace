from __future__ import annotations

from pathlib import Path

import pytest
from scripts.profile_qwen_schedules import (
    require_schedule_unchanged,
    snapshot_schedule,
)


def test_token_profile_schedule_snapshot_rejects_later_drift(tmp_path: Path) -> None:
    schedule = tmp_path / "schedule.jsonl"
    schedule.write_bytes(b'{"record_id":"before"}\n')
    payload, digest = snapshot_schedule(schedule)
    assert payload == b'{"record_id":"before"}\n'

    schedule.write_bytes(b'{"record_id":"after"}\n')
    with pytest.raises(RuntimeError, match="changed after its token-profile snapshot"):
        require_schedule_unchanged(schedule, digest, arm="episode")

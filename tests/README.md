# Test tiers

The copied research test suite contains two intentionally separate tiers:

- `make test` runs portable compiler and acquisition tests without simulator assets.
- `make test-artifact` replays the complete historical suite after trajectory bundles and
  generated datasets have been connected through `workspace.env`.

A failure caused by a missing external bundle is an environment/precondition failure, not a
portable unit-test failure. New deterministic geometry or schema logic should always add a
simulator-free test and be included in `make test` once stable.

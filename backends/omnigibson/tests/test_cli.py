from omnigibson_episode.cli import _parser


def test_intervention_runner_accepts_exact_proposal_filter() -> None:
    args = _parser().parse_args(
        [
            "run-intervention-sweep",
            "--plan",
            "plan.json",
            "--proposal",
            "proposal-a",
            "--proposal",
            "proposal-b",
        ]
    )

    assert args.proposal == ["proposal-a", "proposal-b"]


def test_static_runner_does_not_expose_intervention_filter() -> None:
    args = _parser().parse_args(["run-sweep", "--plan", "plan.json"])

    assert not hasattr(args, "proposal")

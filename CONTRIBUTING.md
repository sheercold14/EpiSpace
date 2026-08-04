# Contributing

## Workflow

1. Create or claim a GitHub issue.
2. Create a short-lived branch: `feature/...`, `fix/...`, `data/...` or `experiment/...`.
3. Keep one conceptual change per pull request.
4. Run `make check` and the relevant tests.
5. Describe code, configuration, data-schema and expected-result changes in the PR.

## Research configuration

- Never edit a frozen recipe in place; create a new version.
- Record seed, model, dataset manifest and evaluator version for every reported experiment.
- Do not commit generated datasets, model weights or machine-specific paths.
- If a schema changes, update producer, consumer, tests and documentation in the same PR.

## Commit style

Use concise imperative commits with a scope when useful:

```text
feat(compiler): add branch-merge episode family
fix(transform): enforce inverse rotation semantics
test(audit): cover missing counterfactual sibling
docs(status): update frozen zero-shot results
```


# trace-core

`trace-core` is the scorer-independent TRACE orchestration module.

Stages:

1. select source rows from a dataset;
2. collect multiple normal-simulator branches from exact snapshots;
3. summarize behavioral tracking and motion stability;
4. apply reset/finite/action/response eligibility gates;
5. call the separately installed `trace-scorer`;
6. apply command-distribution selection;
7. export an immutable selection manifest;
8. rebuild replay using the baseline adapter's reward;
9. optionally pass the replay manifest to a trainer.

Only two JSON files change when moving TRACE to another baseline:

- `baseline_adapter.example.json`: dataset, rollout actor and downstream
  reward/trainer boundary;
- `pipeline.example.json`: target repository's simulator/tool entrypoints.

Run a non-mutating integration check first:

```bash
trace-run \
  --pipeline pipeline.example.json \
  --baseline baseline_adapter.example.json \
  --dry-run --skip-preflight
```

The package contains orchestration and contracts, not a robot implementation.
This is intentional: simulator reset, candidate collection and reward
materialization stay behind configured adapter tools.

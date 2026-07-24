# trace-core

`trace-core` is the scorer-independent TRACE orchestration module.

Stages:

1. select source rows from a dataset;
2. collect multiple normal-simulator branches from exact snapshots;
3. summarize behavioral tracking and motion stability;
4. apply reset/finite/action/response eligibility gates;
5. rank with either the built-in motion-first rule selector or the separately
   installed `trace-scorer`;
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
The exact collector/summary/replay/trainer requirements are documented in
`ADAPTER_CONTRACT.md`; the staged five-group experiment is in `VALIDATION.md`.
The audited current-V12 bridge points are listed in `V12_ADAPTATION.md`.

The first validation should use `"selection": {"backend": "rule"}`.  The hard
eligibility tier first requires every active command component to respond.
Within eligible candidates, weakest-component tracking/response contributes
70% and command-conditional stability contributes 30%.  After this produces a
downstream gain over a same-size random replay, switch only the backend to
`scorer` and provide a scorer checkpoint.

For the current 42-to-45 V12 baseline, start from
`v12_current_baseline_adapter.template.json`. Its actor excludes base linear
velocity but its critic and reward do not. The template therefore preserves
native 48-dimensional replay observations and requires a source-leakage audit
instead of incorrectly zeroing only TRACE rows.

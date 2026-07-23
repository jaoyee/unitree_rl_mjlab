# Integrating trace-scorer

The scorer consumes trajectory-level JSONL summaries and produces an immutable
score artifact. It does not read the source dataset, simulator, policy,
transition rewards, replay buffer, actor or critic.

## Required boundary

The candidate summarizer must provide fields declared by the checkpoint's
`base_feature_names`. Missing fields are represented by the scorer's explicit
missingness indicators; they must not be silently replaced by baseline rewards
or returns.

```python
from trace_scorer import load_scorer

scorer = load_scorer("/path/to/scorer.pt")
scores, missing_counts = scorer.score(summary_rows)
```

For command-distribution or per-start selection, join scores to summaries using
`trace-attach-scores`, then run selection outside this package. Export the
chosen immutable row indices with `trace-export-selection`.

Replay construction happens after selection and is owned by the target
baseline adapter. The scorer score must never be inserted into the transition
reward.

## Checkpoint compatibility

The loader validates:

- checkpoint format and feature schema;
- input and normalization widths;
- positive finite standard deviations;
- strict network state-dict loading;
- absence of reward/return shortcut features.

To change the scorer architecture or feature schema, publish a new format
version rather than weakening these checks.

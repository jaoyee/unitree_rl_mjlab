# trace-scorer

Standalone learned scorer for TRACE candidate summaries.

It has no imports from RWM, actor/critic code, simulator code, replay code, or
reward code. A checkpoint declares its complete feature schema and
normalization statistics. Loading rejects reward-derived shortcut features by
default.

Scoring also fails closed when command context, response, survival, posture or
action-safety semantics are absent, or when more than 20% of one row's
declared features are missing. This prevents a new baseline adapter from
silently producing plausible scores from mostly missing inputs.

Public Python API:

```python
from trace_scorer import load_scorer

scorer = load_scorer("checkpoints/scorer.pt", device="cpu")
scores, missing_counts = scorer.score(summary_rows)
```

Public CLI:

```bash
trace-score \
  --summaries summaries.jsonl \
  --scorer checkpoints/scorer.pt \
  --scores-output scores.jsonl \
  --manifest-output score_manifest.json
```

The bundled checkpoint is the portable V11.1 seed-44 scorer that passed the
existing portability/equivalence audit. Its historical name is retained only
in checkpoint metadata; the runtime module itself is baseline-independent.
That audit establishes software portability, not downstream effectiveness on a
new RWM baseline. The checkpoint must still beat a same-size random replay and
the rule-selected arm under the new baseline's frozen reward and evaluation.

The scorer ranks behavior. It never changes transition rewards and does not
decide how actor/critic training is initialized.

For the current from-zero V12 baseline, use the simulator-summary boundary in
`V12_SUMMARY_ADAPTER.md`. The historical scorer checkpoint is only a candidate
until it beats the rule-selected and same-size random replay arms on V12.

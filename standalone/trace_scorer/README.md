# trace-scorer

Standalone learned scorer for TRACE candidate summaries.

It has no imports from RWM, actor/critic code, simulator code, replay code, or
reward code. A checkpoint declares its complete feature schema and
normalization statistics. Loading rejects reward-derived shortcut features by
default.

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

The scorer ranks behavior. It never changes transition rewards and does not
decide how actor/critic training is initialized.

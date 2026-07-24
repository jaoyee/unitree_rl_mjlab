# TRACE validation order

Keep the selected RWM baseline, reward, training initialization and evaluation
protocol frozen throughout this sequence.

1. Recollect five independent candidate groups from dataset snapshots in the
   normal simulator, using only the selected baseline actor.
2. Materialize two replay arms from each candidate group: motion-first rule
   selection and a same-size, command-distribution-matched random selection.
3. Continue both arms from the same frozen baseline checkpoint. Match replay
   ratio, transition count, optimizer state, seed set and training budget.
4. Evaluate success, velocity error and survival steps with the same horizon.
   The rule selector is accepted only if the aggregate result improves all
   three target metrics without a command-mode regression hidden by averaging.
5. Only after the rule arm passes, run the learned scorer on the same summaries.
   Compare scorer, rule and random arms. A scorer checkpoint is not accepted
   merely because it reproduces historical scores; it must improve downstream
   policy behavior on the new baseline.

For active commands, report each command mode and sign separately, including
pure x, pure y, pure yaw and combined modes. Stability is evaluated only after
the active-component response gate passes. Stand commands use a separate
stationary-stability profile.

For a No-BLV baseline, true simulator base velocity is permitted only in
selection summaries and evaluation. It must be zeroed from selected replay
observations exactly as it is absent from RWM imagination.

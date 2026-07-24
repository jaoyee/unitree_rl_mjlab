# V12 summary adapter

This module remains independent of the V12 RWM implementation. The V12-side
summarizer is responsible only for converting normal-simulator candidate
rollouts into the scorer's trajectory-level behavior fields.

## Data boundary

- Candidates reset from exact snapshots in the selected V12 source dataset.
- Candidate actions come from the selected V12 baseline actor, never an expert
  or E0 actor.
- Commands and physical states come from the normal simulator.
- Do not copy RWM rewards, returns, critic values, world-model losses or source
  labels into summaries.
- Keep `candidate_index`, source-start identity and command-mode identity stable
  so scores can be joined back without reordering.

## Response before stability

For `stand`, stability means small velocity, tilt, body-rate and action
variation around the zero command.

For every nonzero command, first compute whether each active command component
responded in the requested direction. Only then interpret tilt, body rate and
action smoothness as motion stability. A stationary trajectory must not receive
a good moving-command score merely because it has small tilt and smooth
actions.

The required summary fields include:

- active-command masks and the eight command-mode indicators;
- linear and yaw realization ratios in full and steady windows;
- direction correctness and displacement realization;
- linear and yaw tracking errors;
- survival and terminal state;
- tilt, roll/pitch-rate RMS, action saturation and action delta.

Combined commands retain component-level measurements. The selector or scorer
must be able to penalize the weaker active component, so forward motion cannot
hide a missing lateral or yaw response.

## Validation order

1. Generate summaries and run the scorer's fail-closed schema validation.
2. Verify rule selection first against same-size command-matched random replay.
3. Apply the learned scorer to the same immutable candidate pools.
4. Train all comparison arms from zero with the same V12 reward, replay ratio,
   update budget and seeds.
5. Promote the scorer only if success increases, velocity error decreases and
   mean survival steps increases without a major command-mode regression.

The shipped checkpoint passed a portability/equivalence audit on its original
route. That does not establish effectiveness on V12; the matched experiment
above is the required evidence.

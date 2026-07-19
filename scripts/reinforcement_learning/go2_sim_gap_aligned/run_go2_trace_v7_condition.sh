#!/usr/bin/env bash
set -euo pipefail

# V7 canonical entry point. The condition runner remains shared with the V5
# artifact schema so existing immutable checkpoints stay loadable.
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
export REPO
exec bash "${REPO}/scripts/reinforcement_learning/go2_sim_gap_aligned/run_go2_trace_v5_condition_full.sh" "$@"

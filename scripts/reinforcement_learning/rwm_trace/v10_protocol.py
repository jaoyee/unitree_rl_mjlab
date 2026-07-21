"""Single source of truth and shared command binning for Go2 TRACE V10."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


PROTOCOL_SCHEMA = "go2_trace_v10_protocol_v1"
PROTOCOL_VERSION = "go2_trace_v10"
COMMAND_MODES = ("stand", "pure_x", "pure_y", "pure_yaw", "xy", "x_yaw", "y_yaw", "xy_yaw")
MODE_TO_ID = {name: index for index, name in enumerate(COMMAND_MODES)}
AXES = ("x", "y", "yaw")


def sha256_path(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def default_protocol_path() -> Path:
    return Path(__file__).with_name("trace_v10_protocol.json")


def load_protocol(path: str | Path | None = None) -> tuple[dict[str, Any], Path, str]:
    resolved = Path(path or default_protocol_path()).expanduser().resolve()
    data = json.loads(resolved.read_text(encoding="utf-8"))
    if data.get("schema") != PROTOCOL_SCHEMA or data.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError(f"Unsupported TRACE protocol file: {resolved}")
    _validate_protocol(data)
    return data, resolved, sha256_path(resolved)


def _validate_protocol(protocol: Mapping[str, Any]) -> None:
    candidate = protocol["candidate"]
    if int(candidate["source_start_count"]) * int(candidate["trajectories_per_start"]) != int(
        candidate["candidate_trajectory_count"]
    ):
        raise ValueError("Candidate count is inconsistent with starts x branches.")
    if sum(map(int, candidate["source_mode_counts"].values())) != int(candidate["source_start_count"]):
        raise ValueError("Source command-mode counts do not sum to source_start_count.")
    if sum(map(int, candidate["selected_mode_counts"].values())) != int(candidate["selected_trajectory_count"]):
        raise ValueError("Selected command-mode counts do not sum to selected_trajectory_count.")
    if tuple(candidate["source_mode_counts"]) != COMMAND_MODES:
        raise ValueError("Source command modes must use the canonical V10 order.")
    if tuple(candidate["selected_mode_counts"]) != COMMAND_MODES:
        raise ValueError("Selected command modes must use the canonical V10 order.")
    replay = protocol["replay"]
    expected = int(round(int(replay["primary_batch_size"]) * float(replay["ratio"])))
    if int(replay["trace_rows_per_batch"]) != expected:
        raise ValueError("trace_rows_per_batch is inconsistent with batch size and ratio.")
    if float(candidate["action_temperature"]) != 1.0:
        raise ValueError("TRACE V10 freezes candidate action temperature at T=1.")


def classify_command(command: Sequence[float], *, eps: float = 1.0e-3) -> str:
    active = tuple(abs(float(value)) > eps for value in np.asarray(command, dtype=np.float64).reshape(-1)[:3])
    lookup = {
        (False, False, False): "stand",
        (True, False, False): "pure_x",
        (False, True, False): "pure_y",
        (False, False, True): "pure_yaw",
        (True, True, False): "xy",
        (True, False, True): "x_yaw",
        (False, True, True): "y_yaw",
        (True, True, True): "xy_yaw",
    }
    return lookup[active]


def signed_magnitude_buckets(
    command: Sequence[float], protocol: Mapping[str, Any], *, eps: float = 1.0e-3
) -> tuple[int, int, int]:
    """Return one signed 3-bin id per axis, or -1 for an inactive axis."""

    values = np.asarray(command, dtype=np.float64).reshape(-1)[:3]
    edges_by_axis = protocol["candidate"]["signed_magnitude_edges"]
    buckets: list[int] = []
    for axis, value in zip(AXES, values, strict=True):
        if abs(float(value)) <= eps:
            buckets.append(-1)
            continue
        edges = np.asarray(edges_by_axis[axis], dtype=np.float64)
        magnitude = abs(float(value))
        if magnitude < edges[0] - 1.0e-6 or magnitude > edges[-1] + 1.0e-6:
            raise ValueError(f"Command {axis} magnitude {magnitude} is outside V10 edges {edges.tolist()}.")
        magnitude_bin = int(np.digitize(magnitude, edges[1:-1], right=False))
        buckets.append(magnitude_bin + (3 if value > 0.0 else 0))
    return tuple(buckets)  # type: ignore[return-value]


def marginal_vector(command: Sequence[float], protocol: Mapping[str, Any]) -> np.ndarray:
    result = np.zeros(18, dtype=np.int8)
    for axis, bucket in enumerate(signed_magnitude_buckets(command, protocol)):
        if bucket >= 0:
            result[axis * 6 + bucket] = 1
    return result

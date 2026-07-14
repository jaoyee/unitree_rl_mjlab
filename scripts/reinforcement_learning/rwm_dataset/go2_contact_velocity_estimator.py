"""Contact-kinematics base velocity estimator for Unitree Go2.

The estimator only uses signals available in the deployment log: joint
position/velocity, body angular velocity, and foot contact flags.  All inputs
and outputs use the Go2 base frame.  Joint and contact ordering is
``FL, FR, RL, RR`` with three joints per leg.
"""

from __future__ import annotations

import torch


GO2_LEG_ORDER = ("FL", "FR", "RL", "RR")
GO2_HIP_X = (0.1934, 0.1934, -0.1934, -0.1934)
GO2_HIP_Y = (0.0465, -0.0465, 0.0465, -0.0465)
GO2_HIP_OFFSET_Y = (0.0955, -0.0955, 0.0955, -0.0955)
GO2_THIGH_LENGTH = 0.213
GO2_CALF_LENGTH = 0.213

# Fitted on clean MJLab trajectories and validated on independent half-DR
# trajectories.  The raw estimator remains available for experiments that
# must avoid simulator-derived calibration.
GO2_CONTACT_VELOCITY_AFFINE_MATRIX = (
    (1.0805405378341675, 0.0022190690506249666, -0.000533181126229465),
    (0.00948429573327303, 1.0661683082580566, -0.0012887432239949703),
    (0.6503420472145081, 0.29788684844970703, 0.3322185277938843),
)
GO2_CONTACT_VELOCITY_AFFINE_BIAS = (
    -0.00243070675060153,
    0.0003326887381263077,
    0.001981385750696063,
)


def _rotation_x(angle: torch.Tensor) -> torch.Tensor:
    zeros = torch.zeros_like(angle)
    ones = torch.ones_like(angle)
    c = torch.cos(angle)
    s = torch.sin(angle)
    return torch.stack(
        (ones, zeros, zeros, zeros, c, -s, zeros, s, c), dim=-1
    ).reshape(*angle.shape, 3, 3)


def _rotation_y(angle: torch.Tensor) -> torch.Tensor:
    zeros = torch.zeros_like(angle)
    ones = torch.ones_like(angle)
    c = torch.cos(angle)
    s = torch.sin(angle)
    return torch.stack(
        (c, zeros, s, zeros, ones, zeros, -s, zeros, c), dim=-1
    ).reshape(*angle.shape, 3, 3)


def _matvec(matrix: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    return torch.matmul(matrix, vector.unsqueeze(-1)).squeeze(-1)


def go2_foot_positions_and_relative_velocities(
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return foot position and joint-induced velocity in the base frame."""
    if joint_pos.shape[-1] != 12 or joint_vel.shape != joint_pos.shape:
        raise ValueError(
            f"Expected matching (..., 12) joint tensors, got {joint_pos.shape} and {joint_vel.shape}"
        )

    q = joint_pos.reshape(*joint_pos.shape[:-1], 4, 3)
    dq = joint_vel.reshape(*joint_vel.shape[:-1], 4, 3)
    dtype = joint_pos.dtype
    device = joint_pos.device
    prefix = (1,) * (joint_pos.ndim - 1)

    hip_origin = torch.stack(
        (
            torch.tensor(GO2_HIP_X, dtype=dtype, device=device),
            torch.tensor(GO2_HIP_Y, dtype=dtype, device=device),
            torch.zeros(4, dtype=dtype, device=device),
        ),
        dim=-1,
    ).reshape(*prefix, 4, 3)
    lateral_offset = torch.stack(
        (
            torch.zeros(4, dtype=dtype, device=device),
            torch.tensor(GO2_HIP_OFFSET_Y, dtype=dtype, device=device),
            torch.zeros(4, dtype=dtype, device=device),
        ),
        dim=-1,
    ).reshape(*prefix, 4, 3)
    thigh_link = torch.tensor(
        (0.0, 0.0, -GO2_THIGH_LENGTH), dtype=dtype, device=device
    ).reshape(*prefix, 1, 3)
    calf_link = torch.tensor(
        (0.0, 0.0, -GO2_CALF_LENGTH), dtype=dtype, device=device
    ).reshape(*prefix, 1, 3)

    rot_hip = _rotation_x(q[..., 0])
    rot_thigh = torch.matmul(rot_hip, _rotation_y(q[..., 1]))
    rot_calf = torch.matmul(rot_thigh, _rotation_y(q[..., 2]))

    thigh_origin = hip_origin + _matvec(rot_hip, lateral_offset)
    calf_origin = thigh_origin + _matvec(rot_thigh, thigh_link)
    foot_position = calf_origin + _matvec(rot_calf, calf_link)

    axis_x = torch.tensor((1.0, 0.0, 0.0), dtype=dtype, device=device).reshape(
        *prefix, 1, 3
    )
    axis_y = torch.tensor((0.0, 1.0, 0.0), dtype=dtype, device=device).reshape(
        *prefix, 1, 3
    )
    hip_axis = axis_x.expand_as(foot_position)
    thigh_axis = _matvec(rot_hip, axis_y.expand_as(foot_position))
    # The calf rotates around the same base-frame axis as the thigh because
    # both joints are rotations around their local y axes.
    calf_axis = thigh_axis

    relative_velocity = (
        torch.cross(hip_axis, foot_position - hip_origin, dim=-1) * dq[..., 0:1]
        + torch.cross(thigh_axis, foot_position - thigh_origin, dim=-1) * dq[..., 1:2]
        + torch.cross(calf_axis, foot_position - calf_origin, dim=-1) * dq[..., 2:3]
    )
    return foot_position, relative_velocity


def estimate_go2_base_lin_vel_b(
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
    base_ang_vel_b: torch.Tensor,
    foot_contact: torch.Tensor,
    foot_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Estimate base linear velocity from stationary stance-foot constraints.

    Returns ``(velocity, confidence, per_foot_velocity)``.  Confidence is the
    normalized amount of valid support in ``[0, 1]``.  A zero-contact sample
    returns zero velocity and zero confidence; callers should fill those gaps
    with an IMU propagation/filter rather than treating zero as a label.
    """
    if base_ang_vel_b.shape != joint_pos.shape[:-1] + (3,):
        raise ValueError("base_ang_vel_b must have shape (..., 3)")
    if foot_contact.shape != joint_pos.shape[:-1] + (4,):
        raise ValueError("foot_contact must have shape (..., 4)")

    foot_position, relative_velocity = go2_foot_positions_and_relative_velocities(
        joint_pos, joint_vel
    )
    rotational_velocity = torch.cross(
        base_ang_vel_b.unsqueeze(-2).expand_as(foot_position), foot_position, dim=-1
    )
    per_foot_velocity = -(relative_velocity + rotational_velocity)

    weights = foot_contact.to(dtype=joint_pos.dtype).clamp(min=0.0)
    if foot_weight is not None:
        if foot_weight.shape != weights.shape:
            raise ValueError("foot_weight must match foot_contact shape")
        weights = weights * foot_weight.to(dtype=joint_pos.dtype).clamp(min=0.0)
    weight_sum = weights.sum(dim=-1, keepdim=True)
    velocity = (per_foot_velocity * weights.unsqueeze(-1)).sum(dim=-2)
    velocity = velocity / weight_sum.clamp(min=1.0)
    confidence = (weight_sum.squeeze(-1) / 2.0).clamp(max=1.0)
    velocity = torch.where(weight_sum > 0.0, velocity, torch.zeros_like(velocity))
    return velocity, confidence, per_foot_velocity


def robust_fuse_go2_foot_velocities(
    per_foot_velocity: torch.Tensor,
    foot_contact: torch.Tensor,
    consensus_threshold: float = 0.12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reject slipping/inconsistent stance feet and fuse the consensus set."""
    if per_foot_velocity.shape[-2:] != (4, 3):
        raise ValueError("per_foot_velocity must have shape (..., 4, 3)")
    if foot_contact.shape != per_foot_velocity.shape[:-1]:
        raise ValueError("foot_contact must have shape (..., 4)")

    contact = foot_contact > 0.5
    nan = torch.full_like(per_foot_velocity, float("nan"))
    contacted_velocity = torch.where(contact.unsqueeze(-1), per_foot_velocity, nan)
    median = torch.nanmedian(contacted_velocity, dim=-2).values
    residual = torch.linalg.vector_norm(per_foot_velocity - median.unsqueeze(-2), dim=-1)
    residual = torch.where(contact, residual, torch.full_like(residual, float("inf")))
    keep = contact & (residual <= float(consensus_threshold))

    has_contact = contact.any(dim=-1)
    needs_fallback = has_contact & ~keep.any(dim=-1)
    closest = residual.argmin(dim=-1)
    fallback = torch.nn.functional.one_hot(closest, num_classes=4).bool()
    keep = torch.where(needs_fallback.unsqueeze(-1), fallback, keep)

    weights = keep.to(dtype=per_foot_velocity.dtype)
    weight_sum = weights.sum(dim=-1, keepdim=True)
    velocity = (per_foot_velocity * weights.unsqueeze(-1)).sum(dim=-2)
    velocity = velocity / weight_sum.clamp(min=1.0)
    velocity = torch.where(has_contact.unsqueeze(-1), velocity, torch.zeros_like(velocity))
    confidence = (weight_sum.squeeze(-1) / 2.0).clamp(max=1.0)
    return velocity, confidence


class Go2ContactVelocityEstimator:
    """Stateful robust estimator with calibration and temporal filtering."""

    def __init__(
        self,
        *,
        ema_alpha: float = 0.35,
        consensus_threshold: float = 0.12,
        use_sim_calibration: bool = True,
    ) -> None:
        if not 0.0 < ema_alpha <= 1.0:
            raise ValueError("ema_alpha must be in (0, 1]")
        self.ema_alpha = float(ema_alpha)
        self.consensus_threshold = float(consensus_threshold)
        self.use_sim_calibration = bool(use_sim_calibration)
        self._filtered: torch.Tensor | None = None

    def reset(self) -> None:
        self._filtered = None

    def update(
        self,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
        base_ang_vel_b: torch.Tensor,
        foot_contact: torch.Tensor,
        reset_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Estimate one batched time step.

        ``reset_mask`` marks environments whose temporal state must restart at
        this sample.  Samples without a contacted foot carry the previous
        estimate but return zero confidence so they can be excluded from loss.
        """
        _, _, per_foot = estimate_go2_base_lin_vel_b(
            joint_pos, joint_vel, base_ang_vel_b, foot_contact
        )
        raw, confidence = robust_fuse_go2_foot_velocities(
            per_foot, foot_contact, self.consensus_threshold
        )
        calibrated = raw
        if self.use_sim_calibration:
            matrix = raw.new_tensor(GO2_CONTACT_VELOCITY_AFFINE_MATRIX)
            bias = raw.new_tensor(GO2_CONTACT_VELOCITY_AFFINE_BIAS)
            calibrated = torch.matmul(raw, matrix) + bias

        if self._filtered is None or self._filtered.shape != calibrated.shape:
            self._filtered = calibrated.clone()
        else:
            if reset_mask is not None:
                if reset_mask.shape != calibrated.shape[:-1]:
                    raise ValueError("reset_mask must match the estimator batch shape")
                self._filtered = torch.where(
                    reset_mask.unsqueeze(-1), calibrated, self._filtered
                )
            updated = self.ema_alpha * calibrated + (1.0 - self.ema_alpha) * self._filtered
            self._filtered = torch.where(
                (confidence > 0.0).unsqueeze(-1), updated, self._filtered
            )
        return self._filtered.clone(), confidence, raw

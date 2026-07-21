"""Update scheduling helpers shared by FlashSAC agent variants."""

from __future__ import annotations


def should_update_actor(
    update_step: int,
    actor_learning_starts_updates: int,
    actor_update_period: int,
) -> bool:
    """Return whether the actor may update after the critic-only warmup."""

    if actor_learning_starts_updates < 0:
        raise ValueError("actor_learning_starts_updates must be non-negative.")
    if actor_update_period <= 0:
        raise ValueError("actor_update_period must be positive.")
    if update_step < actor_learning_starts_updates:
        return False
    return (update_step - actor_learning_starts_updates) % actor_update_period == 0

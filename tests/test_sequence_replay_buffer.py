import torch

from scripts.reinforcement_learning.rwm.dynamics import SequenceReplayBuffer


def _make_buffer(episode_ids: list[int], timesteps: list[int]) -> SequenceReplayBuffer:
    buffer = SequenceReplayBuffer(
        state_dim=1,
        action_dim=1,
        contact_dim=1,
        termination_dim=1,
        num_envs=1,
        capacity=len(episode_ids),
        device="cpu",
    )
    for episode_id, timestep in zip(episode_ids, timesteps):
        value = float(episode_id * 100 + timestep)
        buffer.states.append(torch.tensor([[value]]))
        buffer.actions.append(torch.tensor([[value]]))
        buffer.next_states.append(torch.tensor([[value + 1.0]]))
        buffer.contacts.append(torch.zeros(1, 1))
        buffer.terminations.append(torch.zeros(1, 1))
    buffer.episode_ids = [torch.tensor([value]) for value in episode_ids]
    buffer.timesteps = [torch.tensor([value]) for value in timesteps]
    return buffer


def test_sample_rejects_episode_and_timestep_boundaries() -> None:
    buffer = _make_buffer(
        episode_ids=[0, 0, 0, 1, 1, 1, 1, 1],
        timesteps=[0, 1, 2, 0, 1, 2, 3, 4],
    )

    states, _, _, _, _ = buffer.sample(
        batch_size=128,
        history_horizon=2,
        forecast_horizon=1,
        device="cpu",
    )

    assert torch.all(states[:, 1:, 0] - states[:, :-1, 0] == 1)
    episode_markers = torch.div(states[:, :, 0], 100, rounding_mode="floor")
    assert torch.all(episode_markers == episode_markers[:, :1])


def test_state_dict_roundtrip_preserves_sequence_metadata() -> None:
    buffer = _make_buffer(
        episode_ids=[4, 4, 4, 5],
        timesteps=[7, 8, 9, 0],
    )

    restored = SequenceReplayBuffer.from_state_dict(buffer.state_dict(), device="cpu")

    assert restored.episode_ids is not None
    assert restored.timesteps is not None
    assert torch.equal(torch.stack(restored.episode_ids), torch.tensor([[4], [4], [4], [5]]))
    assert torch.equal(torch.stack(restored.timesteps), torch.tensor([[7], [8], [9], [0]]))

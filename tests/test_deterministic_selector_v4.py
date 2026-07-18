from scripts.reinforcement_learning.rwm_dataset.select_go2_deterministic_segments_v4 import (
    capacity_adjusted_targets,
    select_exact,
    split_run,
)


def test_split_run_never_leaves_short_tail() -> None:
    chunks = split_run(0, 430, 0, minimum=40, maximum=400)
    assert [len(chunk) for chunk in chunks] == [390, 40]
    assert chunks[0][-1][0] + 1 == chunks[1][0][0]


def test_select_exact_preserves_minimum_and_quota() -> None:
    candidates = []
    for mode in range(8):
        candidates.append([
            [(t, mode) for t in range(400)],
            [(t, mode) for t in range(400, 800)],
            [(t, mode) for t in range(800, 1200)],
        ])
    selected, report = select_exact(candidates, targets=[450] * 8, minimum=40, seed=7)
    assert sum(map(len, selected)) == 3600
    assert min(map(len, selected)) >= 40
    assert all(report["per_mode"][name]["selected"] == 450 for name in report["per_mode"])


def test_capacity_adjustment_redistributes_small_shortfall() -> None:
    candidates = [[[(t, mode) for t in range(102)]] for mode in range(8)]
    candidates[4] = [[(t, 4) for t in range(98)]]
    actual, capacities = capacity_adjusted_targets(candidates, desired=[100] * 8, tolerance=0.02)
    assert actual.sum() == 800
    assert actual[4] == 98
    assert max(actual) <= 102
    assert capacities[4] == 98


if __name__ == "__main__":
    test_split_run_never_leaves_short_tail()
    test_select_exact_preserves_minimum_and_quota()
    test_capacity_adjustment_redistributes_small_shortfall()

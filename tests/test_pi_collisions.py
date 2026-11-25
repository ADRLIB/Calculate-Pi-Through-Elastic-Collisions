import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pi_collisions import CollisionSimulation, count_collisions  # noqa: E402

PI_DIGITS = "314159265"


@pytest.mark.parametrize("n", range(1, 10))
def test_count_matches_digits_of_pi(n):
    assert count_collisions(n) == int(PI_DIGITS[:n])


@pytest.mark.parametrize("n", range(1, 8))
def test_positional_simulation_matches_digits_of_pi(n):
    # This is the engine the pygame app uses; n=6 was where the old
    # fixed-time-step version tunneled and reported 102,883.
    sim = CollisionSimulation(n)
    assert sim.run_to_completion() == int(PI_DIGITS[:n])


@pytest.mark.parametrize("n", [3, 6])
def test_energy_and_ordering_are_preserved(n):
    sim = CollisionSimulation(n)
    energy0 = 0.5 * sim.m_small * sim.v_small**2 + 0.5 * sim.m_big * sim.v_big**2
    while not sim.finished:
        sim.advance(0.05)
        # The small ball must never pass through the wall or the big ball
        assert sim.x_small - sim.r_small >= sim.wall_x - 1e-6
        assert sim.x_big - sim.x_small >= sim.r_small + sim.r_big - 1e-6
    energy = 0.5 * sim.m_small * sim.v_small**2 + 0.5 * sim.m_big * sim.v_big**2
    assert energy == pytest.approx(energy0, rel=1e-9)


def test_advance_respects_event_budget():
    sim = CollisionSimulation(7)
    sim.advance(1e6, max_events=1000)
    assert sim.collisions == 1000
    assert not sim.finished

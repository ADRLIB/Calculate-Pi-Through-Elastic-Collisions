"""Event-driven physics engine for the pi-from-elastic-collisions puzzle.

Setup (1D): a wall on the left, a small ball (mass m) at rest, and a big ball
(mass M = m * 100**(n-1)) sliding left toward it. Every collision is perfectly
elastic. The total number of collisions (ball-ball + ball-wall) spells out the
first n digits of pi.

Why event-driven instead of fixed time steps?
    A fixed-step simulation moves the balls forward by a constant dt and then
    checks for overlap. As M/m grows, the small ball ends up bouncing between
    the wall and the big ball at speeds ~ sqrt(M/m) * v0, so within one step it
    can travel farther than the gap and "tunnel" through the big ball, losing
    collisions. No dt is small enough for every n.

    Instead, between collisions everything moves at constant velocity, so we can
    solve for *exactly when* the next collision happens, jump straight to it,
    and apply the elastic-collision formulas. No overlap checks, no tunneling,
    and the cost is proportional to the number of collisions, not to time.

The hot loops are compiled with numba so millions of collisions take
milliseconds.
"""

import math

import numpy as np
from numba import njit


def mass_ratio(n_digits):
    """Big-ball mass relative to the small ball for n digits of pi."""
    return 100.0 ** (n_digits - 1)


@njit(cache=True)
def _collide(v1, v2, m1, m2):
    """1D elastic collision (conservation of momentum + kinetic energy)."""
    total = m1 + m2
    new_v1 = ((m1 - m2) * v1 + 2.0 * m2 * v2) / total
    new_v2 = ((m2 - m1) * v2 + 2.0 * m1 * v1) / total
    return new_v1, new_v2


@njit(cache=True)
def count_collisions(n_digits):
    """Count collisions without positions or time (velocity-only).

    In 1D with one wall the events must alternate (ball-ball, wall, ball-ball,
    ...), so only the velocities matter for the count.
    """
    m1 = 1.0
    m2 = 100.0 ** (n_digits - 1)
    v1 = 0.0
    v2 = -1.0
    count = 0
    while True:
        if v2 < v1:  # big ball is catching up to the small ball
            v1, v2 = _collide(v1, v2, m1, m2)
            count += 1
        elif v1 < 0.0:  # small ball heading into the wall
            v1 = -v1
            count += 1
        else:  # both moving right and separating: no more collisions
            return count


# Indices into the simulation state array
T, X1, V1, X2, V2 = 0, 1, 2, 3, 4


@njit(cache=True)
def _advance(state, t_end, m1, m2, r1, r2, wall_x, max_events,
             rec_buf, rec_stride, rec_dense, index_offset):
    """Advance `state` in place to time `t_end`, processing every collision.

    state = [t, x1, v1, x2, v2] (x = ball centers, small ball is 1).
    Stops early after `max_events` collisions so a single call can't freeze the
    caller; in that case state[T] < t_end.

    Recording (for plotting): if rec_stride > 0, the velocities right after
    collision number k (1-based, counted across calls via index_offset) are
    written to rec_buf as rows [k, v1, v2] when k <= rec_dense or
    k % rec_stride == 0, until rec_buf is full.

    Returns (collisions processed, rows recorded).
    """
    t, x1, v1, x2, v2 = state[0], state[1], state[2], state[3], state[4]
    events = 0
    n_rec = 0
    cap = rec_buf.shape[0]
    while events < max_events:
        # Time until the balls touch (only if the small ball is closing in)
        t_ball = math.inf
        if v1 > v2:
            gap = (x2 - r2) - (x1 + r1)
            t_ball = max(gap, 0.0) / (v1 - v2)
        # Time until the small ball touches the wall
        t_wall = math.inf
        if v1 < 0.0:
            gap = (x1 - r1) - wall_x
            t_wall = max(gap, 0.0) / -v1

        dt = min(t_ball, t_wall)
        if t + dt > t_end:  # next event is beyond this call's window
            break

        # Jump straight to the collision
        t += dt
        x1 += v1 * dt
        x2 += v2 * dt
        if t_ball <= t_wall:
            v1, v2 = _collide(v1, v2, m1, m2)
        else:
            v1 = -v1
        events += 1

        if rec_stride > 0 and n_rec < cap:
            k = index_offset + events
            if k <= rec_dense or k % rec_stride == 0:
                rec_buf[n_rec, 0] = k
                rec_buf[n_rec, 1] = v1
                rec_buf[n_rec, 2] = v2
                n_rec += 1

    if events < max_events:  # no more events before t_end: coast to it
        dt = t_end - t
        x1 += v1 * dt
        x2 += v2 * dt
        t = t_end

    state[0], state[1], state[2], state[3], state[4] = t, x1, v1, x2, v2
    return events, n_rec


class CollisionSimulation:
    """Wall + small ball + big ball, advanced exactly from collision to collision.

    Units are whatever the caller uses (the pygame app uses pixels and seconds).
    """

    def __init__(
        self,
        n_digits,
        x_small=200.0,
        x_big=380.0,
        v_big=-50.0,
        r_small=45.0,
        r_big=45.0,
        wall_x=55.0,
    ):
        self.n_digits = n_digits
        self.m_small = 1.0
        self.m_big = mass_ratio(n_digits)
        self.r_small = float(r_small)
        self.r_big = float(r_big)
        self.wall_x = float(wall_x)
        self.v0 = float(v_big)
        self.state = np.array([0.0, x_small, 0.0, x_big, v_big], dtype=np.float64)
        self.collisions = 0
        # Phase-space recording (off by default; see enable_recording)
        self._rec_buf = np.zeros((0, 3))
        self._rec_stride = 0
        self._rec_dense = 0
        self.recorded = self._rec_buf  # rows [k, v_small, v_big] from the last advance()

    # Read-only views of the state
    @property
    def time(self):
        return self.state[T]

    @property
    def x_small(self):
        return self.state[X1]

    @property
    def x_big(self):
        return self.state[X2]

    @property
    def v_small(self):
        return self.state[V1]

    @property
    def v_big(self):
        return self.state[V2]

    @property
    def finished(self):
        """True once the balls are separating and no collision can happen again."""
        v1, v2 = self.v_small, self.v_big
        return v1 >= 0.0 and v2 >= v1

    # ~12 ms of numba work: keeps a 60 fps frame responsive even at the densest moment
    MAX_EVENTS_PER_CALL = 750_000

    def enable_recording(self, stride=1, dense=0):
        """Record the state after every `stride`-th collision (plus the first
        `dense` collisions) so callers can plot phase space without storing
        millions of points. Results appear in `self.recorded` after advance()."""
        self._rec_stride = max(1, int(stride))
        self._rec_dense = int(dense)
        capacity = self.MAX_EVENTS_PER_CALL // self._rec_stride + self._rec_dense + 2
        self._rec_buf = np.zeros((capacity, 3))

    def phase_point(self, v_small, v_big):
        """Map velocities to phase space, scaled so the energy circle has radius 1.

        x = sqrt(M) v_big, y = sqrt(m) v_small, divided by sqrt(2E) = sqrt(M)|v0|.
        Kinetic energy conservation means every state lies on this unit circle.
        """
        scale = abs(self.v0)
        return v_big / scale, v_small * math.sqrt(self.m_small / self.m_big) / scale

    @property
    def theta(self):
        """Angle each pair of collisions advances around the phase circle (/2)."""
        return math.atan(math.sqrt(self.m_small / self.m_big))

    def advance(self, dt, max_events=None):
        """Advance the simulation by `dt`. Returns the number of new collisions.

        If more than `max_events` collisions fall inside `dt`, the simulation
        stops early (its clock lags behind real time), which keeps every frame
        responsive and naturally slows down the densest part of the motion.
        """
        if max_events is None:
            max_events = self.MAX_EVENTS_PER_CALL
        new, n_rec = _advance(
            self.state,
            self.time + dt,
            self.m_small,
            self.m_big,
            self.r_small,
            self.r_big,
            self.wall_x,
            min(max_events, self.MAX_EVENTS_PER_CALL),
            self._rec_buf,
            self._rec_stride,
            self._rec_dense,
            self.collisions,
        )
        self.recorded = self._rec_buf[:n_rec]
        self.collisions += new
        return new

    def run_to_completion(self, chunk=1.0):
        """Run until no further collisions are possible. Returns the total count."""
        while not self.finished:
            self.advance(chunk)
        return self.collisions


if __name__ == "__main__":
    import time

    for n in range(1, 10):
        start = time.perf_counter()
        total = count_collisions(n)
        print(f"n={n}: {total:>11,} collisions  ({time.perf_counter() - start:.3f}s)")

"""Live phase-space plot for the pygame app.

Axes are x = sqrt(M) v_big and y = sqrt(m) v_small, scaled so that kinetic
energy conservation puts every state on the unit circle. In these coordinates:

  * a wall bounce flips v_small, so it is a vertical jump (y -> -y);
  * a ball-ball collision conserves momentum, so it moves along a line of
    slope -sqrt(M/m);
  * each collision advances the point by the same angle around the circle
    (2*theta per ball-ball + wall pair, theta = arctan sqrt(m/M)), so the
    number of collisions that fit in the half-turn is about pi / theta.

The panel draws each collision as a dot the moment it happens (colored by
collision order), the first few collisions as connected lines, the shaded
"no more collisions" wedge, and a zoomed inset of the first collisions, where
the zig-zag is otherwise far too small to see.
"""

import math

import numpy as np
import pygame

MAX_POINTS = 20_000     # sample down to about this many dots for large n
ALL_LINES_UP_TO = 60    # draw every segment when there are this few collisions
FIRST_LINES = 40        # otherwise connect only the first this many collisions
INSET_COLLISIONS = 12   # collisions shown in the zoomed inset

PANEL_BG = (255, 255, 255)
BORDER = (225, 220, 216)
GRID = (232, 230, 228)
CIRCLE = (190, 190, 190)
INK = (30, 30, 30)
MUTED = (120, 120, 120)
DONE_FILL = (46, 139, 87, 70)
DONE_EDGE = (46, 139, 87)

# Viridis colormap anchors (dark purple -> yellow) for "collision order"
_VIRIDIS = np.array(
    [(68, 1, 84), (59, 82, 139), (33, 145, 140), (94, 201, 98), (253, 231, 37)], dtype=float
)


def viridis(fractions):
    """Map values in [0, 1] to RGB rows."""
    f = np.clip(np.asarray(fractions, dtype=float), 0.0, 1.0) * (len(_VIRIDIS) - 1)
    i = np.minimum(f.astype(int), len(_VIRIDIS) - 2)
    t = (f - i)[:, None]
    return (_VIRIDIS[i] * (1 - t) + _VIRIDIS[i + 1] * t).astype(int)


def first_velocities(m1, m2, count):
    """(v_small, v_big) after each of the first `count` collisions, starting at (0, -1)."""
    v1, v2 = 0.0, -1.0
    out = [(v1, v2)]
    total = m1 + m2
    for _ in range(count):
        if v2 < v1:
            v1, v2 = ((m1 - m2) * v1 + 2 * m2 * v2) / total, ((m2 - m1) * v2 + 2 * m1 * v1) / total
        elif v1 < 0:
            v1 = -v1
        else:
            break
        out.append((v1, v2))
    return out


class PhasePanel:
    def __init__(self, rect, sim, total_collisions, font, small_font):
        self.rect = pygame.Rect(rect)
        self.sim = sim
        self.total = max(1, total_collisions)
        self.font = font
        self.small_font = small_font

        # Which collisions to plot (the engine samples them for us)
        self.line_limit = self.total if self.total <= ALL_LINES_UP_TO else FIRST_LINES
        stride = max(1, self.total // MAX_POINTS)
        # Collisions alternate ball-ball / wall, so an even stride would only ever
        # sample wall bounces (all in the upper half). Keep it odd.
        stride |= 1
        sim.enable_recording(stride=stride, dense=max(self.line_limit, INSET_COLLISIONS))
        self.dot_radius = 3 if self.total < 100 else 1

        # Geometry (panel-local pixels)
        header = 52
        self.radius = (min(self.rect.w, self.rect.h - header) - 34) / 2
        self.center = (self.rect.w / 2, header + (self.rect.h - header) / 2)

        self.background = self._render_background()
        self.trail = pygame.Surface(self.rect.size, pygame.SRCALPHA)

        start = sim.phase_point(sim.v_small, sim.v_big)
        self.prev = (0, start)  # (collision index, phase point)
        self.inset_points = [start]
        self.inset = self._setup_inset() if self.total > ALL_LINES_UP_TO else None

    # ----- coordinates -----
    def to_px(self, x, y):
        cx, cy = self.center
        return (cx + x * self.radius, cy - y * self.radius)

    # ----- static layer -----
    def _render_background(self):
        surf = pygame.Surface(self.rect.size, pygame.SRCALPHA)
        surf.fill(PANEL_BG)
        pygame.draw.rect(surf, BORDER, surf.get_rect(), 1)
        cx, cy = self.center
        r = self.radius

        # axes
        pygame.draw.line(surf, GRID, (cx - r - 8, cy), (cx + r + 8, cy))
        pygame.draw.line(surf, GRID, (cx, cy - r - 8), (cx, cy + r + 8))

        # "no more collisions" wedge: 0 <= v_small <= v_big, i.e. angle in [0, theta]
        theta = self.sim.theta
        arc = [self.to_px(math.cos(a), math.sin(a)) for a in np.linspace(0, theta, 24)]
        wedge = pygame.Surface(self.rect.size, pygame.SRCALPHA)
        pygame.draw.polygon(wedge, DONE_FILL, [self.center] + arc)
        surf.blit(wedge, (0, 0))
        pygame.draw.line(surf, DONE_EDGE, self.center, arc[-1], 1)

        # energy circle: every state must lie on it
        pygame.draw.circle(surf, CIRCLE, (round(cx), round(cy)), round(r), 1)

        # labels
        surf.blit(self.font.render("Phase space", True, INK), (10, 6))
        subtitle = f"θ = arctan√(m/M) = {theta:.3g} rad    π/θ = {math.pi / theta:,.2f}"
        surf.blit(self.small_font.render(subtitle, True, MUTED), (10, 30))
        x_label = self.small_font.render("√M·v_big", True, MUTED)
        surf.blit(x_label, x_label.get_rect(topright=(self.rect.w - 6, cy + 4)))
        y_label = self.small_font.render("√m·v_small", True, MUTED)
        surf.blit(y_label, y_label.get_rect(bottomleft=(cx + 5, cy - r - 2)))
        done = self.small_font.render("done", True, DONE_EDGE)
        surf.blit(done, done.get_rect(bottomright=(self.rect.w - 6, cy - 18)))
        return surf

    # ----- zoom inset -----
    def _setup_inset(self):
        """Fix the inset's view around the first INSET_COLLISIONS states.

        Near the start the zig-zag is tiny and nearly vertical, so the inset
        stretches x (and says by how much) to make its shape readable.
        """
        sim = self.sim
        pts = [sim.phase_point(v1 * abs(sim.v0), v2 * abs(sim.v0))
               for v1, v2 in first_velocities(sim.m_small, sim.m_big, INSET_COLLISIONS)]
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
        x_span = max(x1 - x0, 1e-300)
        y_span = max(y1 - y0, 1e-300)
        size = round(self.radius * 0.8)
        box = pygame.Rect(0, 0, size, size)
        box.center = (round(self.center[0] + self.radius * 0.28), round(self.center[1]))
        pad = 0.12
        return {
            "box": box,
            "x0": x0 - pad * x_span, "x_span": x_span * (1 + 2 * pad),
            "y0": y0 - pad * y_span, "y_span": y_span * (1 + 2 * pad),
            "stretch": y_span / x_span,
            "region": (x0, x1, y0, y1),
        }

    def _inset_px(self, x, y):
        v = self.inset
        b = v["box"]
        return (b.left + (x - v["x0"]) / v["x_span"] * b.w,
                b.bottom - (y - v["y0"]) / v["y_span"] * b.h)

    def _draw_inset(self, surf):
        v = self.inset
        box = v["box"]

        # mark the zoomed region on the main plot and connect it to the inset
        x0, x1, y0, y1 = v["region"]
        (ax, ay), (bx, by) = self.to_px(x0, y1), self.to_px(x1, y0)
        marker = pygame.Rect(0, 0, max(bx - ax, 8), max(by - ay, 8))
        marker.center = ((ax + bx) / 2, (ay + by) / 2)
        pygame.draw.rect(surf, MUTED, marker, 1)
        pygame.draw.line(surf, CIRCLE, marker.topright, box.topleft)
        pygame.draw.line(surf, CIRCLE, marker.bottomright, box.bottomleft)

        pygame.draw.rect(surf, PANEL_BG, box)
        pygame.draw.rect(surf, MUTED, box, 1)
        clip = surf.get_clip()
        surf.set_clip(box.inflate(-2, -2))
        pts = self.inset_points[: INSET_COLLISIONS + 1]
        colors = viridis(np.arange(len(pts)) / self.total)  # same color scale as the main plot
        px = [self._inset_px(x, y) for x, y in pts]
        for i in range(1, len(px)):
            pygame.draw.aaline(surf, colors[i], px[i - 1], px[i])
        for p, c in zip(px, colors):
            pygame.draw.circle(surf, c, (round(p[0]), round(p[1])), 3)
        if px:
            pygame.draw.circle(surf, INK, (round(px[0][0]), round(px[0][1])), 5, 1)
        surf.set_clip(clip)

        label = self.small_font.render(f"first {INSET_COLLISIONS} collisions", True, MUTED)
        surf.blit(label, label.get_rect(midbottom=(box.centerx, box.top - 2)))
        if v["stretch"] > 1.5:
            note = self.small_font.render(f"x stretched ×{v['stretch']:,.0f}", True, MUTED)
            surf.blit(note, note.get_rect(midtop=(box.centerx, box.bottom + 2)))

    # ----- per frame -----
    def consume(self):
        """Draw the collisions the simulation recorded during its last advance()."""
        rows = self.sim.recorded
        if len(rows) == 0:
            return
        ks = rows[:, 0].astype(np.int64)
        xs, ys = self.sim.phase_point(rows[:, 1], rows[:, 2])
        colors = viridis(ks / self.total)
        r = self.dot_radius
        for k, x, y, color in zip(ks, xs, ys, colors):
            point = (float(x), float(y))
            px = self.to_px(*point)
            prev_k, prev_point = self.prev
            if k <= self.line_limit and k == prev_k + 1:
                pygame.draw.aaline(self.trail, color, self.to_px(*prev_point), px)
            if k <= INSET_COLLISIONS:
                self.inset_points.append(point)
            pygame.draw.circle(self.trail, color, (round(px[0]), round(px[1])), r)
            self.prev = (k, point)

    def draw(self, window):
        surf = self.background.copy()
        surf.blit(self.trail, (0, 0))
        if self.inset is not None:
            self._draw_inset(surf)
        # current state
        x, y = self.sim.phase_point(self.sim.v_small, self.sim.v_big)
        cx, cy = self.to_px(x, y)
        pygame.draw.circle(surf, INK, (round(cx), round(cy)), 6, 2)
        window.blit(surf, self.rect)

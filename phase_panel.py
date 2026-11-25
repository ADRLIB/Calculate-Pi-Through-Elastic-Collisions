"""Live phase-space plot for the pygame app, in the style of the original
matplotlib images ("Pi Collisions for n_digits = N").

Each point is (x, y) = (sqrt(M) * v_big, sqrt(m) * v_small) right after a
collision, with the big ball starting at speed 1, exactly like the notebook.
Energy conservation puts every point on a circle of radius sqrt(M).

How it stays fast enough to update every frame: matplotlib draws the static
parts (axes, ticks, grid, title, labels) once per run into a pygame surface,
and we reuse matplotlib's own data->pixel transform to draw the line and
markers ourselves as each collision happens. The legend is redrawn every
frame so its collision count is live.
"""

import os

import matplotlib

matplotlib.use("Agg")  # render off-screen; pygame owns the window
from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
import numpy as np  # noqa: E402
import pygame  # noqa: E402

MAX_POINTS = 20_000  # for large n, plot about this many collisions (every k-th)

# Original images were 6x6 inch figures; render the same layout scaled to the panel
FIG_INCHES = 6.0
LINE_BLUE = (31, 119, 180)  # matplotlib's default "C0" blue
LEGEND_EDGE = (204, 204, 204)
TEXT = (0, 0, 0)
FONT_PATH = os.path.join(matplotlib.get_data_path(), "fonts", "ttf", "DejaVuSans.ttf")


class PhasePanel:
    def __init__(self, rect, sim, total_collisions):
        self.rect = pygame.Rect(rect)
        self.sim = sim
        self.total = max(1, total_collisions)
        self.n_digits = sim.n_digits
        self.radius = float(np.sqrt(sim.m_big))  # circle radius in plot units

        # Sample down for big n. Collisions alternate ball-ball / wall, so keep
        # the stride odd or we'd only ever see one kind (half the circle).
        stride = max(1, self.total // MAX_POINTS) | 1
        sim.enable_recording(stride=stride, dense=0)

        dpi = self.rect.w / FIG_INCHES
        self.scale = dpi / 72.0  # points -> pixels, for line widths/marker sizes
        # The plot is drawn incrementally onto one persistent surface: matplotlib's
        # axes underneath, then each frame's new line segments and markers on top.
        self.canvas, self._to_px, self.axes_rect = self._render_axes(dpi)
        self.canvas.set_clip(self.axes_rect)

        # matplotlib defaults: linewidth 1.5 pt, marker 'o' size 6 pt
        self.line_width = max(1, round(1.5 * self.scale))
        self.marker_radius = max(1, round(3 * self.scale))
        self.legend_font = pygame.font.Font(FONT_PATH, max(8, round(10 * self.scale)))
        # One pre-drawn marker, stamped with a single batched blits() call per frame
        d = 2 * self.marker_radius + 1
        self.marker = pygame.Surface((d, d), pygame.SRCALPHA)
        pygame.draw.circle(self.marker, LINE_BLUE, (self.marker_radius, self.marker_radius), self.marker_radius)
        self.last_px = None

    # ----- static layer (matplotlib) -----
    def _render_axes(self, dpi):
        size = self.rect.w / dpi, self.rect.h / dpi
        fig = Figure(figsize=size, dpi=dpi)
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_subplot()
        # Fixed limits (matplotlib would autoscale to the finished data; we draw live)
        lim = 1.1 * self.radius
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"Pi Collisions for n_digits = {self.n_digits}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(True)
        fig.tight_layout()
        canvas.draw()

        width, height = canvas.get_width_height()
        surface = pygame.image.frombytes(bytes(canvas.buffer_rgba()), (width, height), "RGBA")

        transform = ax.transData  # data -> display pixels (origin bottom-left)

        def to_px(xs, ys):
            pts = transform.transform(np.column_stack([xs, ys]))
            return pts[:, 0], height - pts[:, 1]

        bbox = ax.get_window_extent()
        axes_rect = pygame.Rect(round(bbox.x0), round(height - bbox.y1), round(bbox.width), round(bbox.height))
        return surface, to_px, axes_rect

    # ----- per frame -----
    def consume(self):
        """Draw the collisions the simulation recorded during its last advance()."""
        rows = self.sim.recorded
        if len(rows) == 0:
            return
        x, y = self.sim.phase_point(rows[:, 1], rows[:, 2])
        px, py = self._to_px(x * self.radius, y * self.radius)
        points = list(zip(px.tolist(), py.tolist()))

        line = ([self.last_px] if self.last_px else []) + points
        if len(line) > 1:
            pygame.draw.lines(self.canvas, LINE_BLUE, False, line, self.line_width)
        r = self.marker_radius
        self.canvas.blits([(self.marker, (round(x) - r, round(y) - r)) for x, y in points], doreturn=False)
        self.last_px = points[-1]

    def _draw_legend(self, window):
        """Upper-right legend like matplotlib's, with a live collision count."""
        s = self.scale
        label = self.legend_font.render(f"Collisions: {self.sim.collisions}", True, TEXT)
        handle_w = round(20 * s)
        pad = round(4 * s)
        w = pad + handle_w + round(8 * s) + label.get_width() + pad * 2
        h = label.get_height() + pad * 2
        box = pygame.Rect(0, 0, w, h)
        box.topright = (self.rect.x + self.axes_rect.right - round(5 * s),
                        self.rect.y + self.axes_rect.top + round(5 * s))

        legend = pygame.Surface(box.size, pygame.SRCALPHA)
        pygame.draw.rect(legend, (255, 255, 255, 205), legend.get_rect(), border_radius=round(3 * s))
        pygame.draw.rect(legend, LEGEND_EDGE, legend.get_rect(), 1, border_radius=round(3 * s))
        mid = h // 2
        pygame.draw.line(legend, LINE_BLUE, (pad, mid), (pad + handle_w, mid), self.line_width)
        pygame.draw.circle(legend, LINE_BLUE, (pad + handle_w // 2, mid), self.marker_radius)
        legend.blit(label, (pad + handle_w + round(8 * s), pad))
        window.blit(legend, box)

    def draw(self, window):
        window.blit(self.canvas, self.rect)
        self._draw_legend(window)

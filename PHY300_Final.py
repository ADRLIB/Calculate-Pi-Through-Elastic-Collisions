"""Pygame visualization of computing pi by counting elastic collisions.

Run:  python PHY300_Final.py [n_digits]

Controls:
    Up / Down     more / fewer digits of pi (restarts)
    Left / Right  slow down / speed up time
    Space         pause / resume
    R             restart
    M             mute / unmute
    Esc           quit
"""

import sys
from pathlib import Path

import pygame
from PIL import Image, ImageFilter

from pi_collisions import CollisionSimulation

# Resolve assets relative to this file so the script runs from any directory
ASSETS = Path(__file__).resolve().parent / "assets"

WIDTH, HEIGHT = 1100, 400
FPS = 60
MIN_DIGITS, MAX_DIGITS = 1, 8  # n = 9 works, but takes ~30 s of wall time to play
SPEEDS = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0]
PI_STRING = "3.14159265358979"
SOUND_MIN_INTERVAL_MS = 40  # don't let thousands of boops pile up per second

# Colors
BACKGROUND = (251, 247, 245)
INK = (0, 0, 0)
MUTED = (110, 110, 110)
SMALL_COLOR = (212, 97, 94)
BIG_COLOR = (112, 134, 168)
MATCH_COLOR = (46, 139, 87)
MISMATCH_COLOR = (200, 60, 50)

RADIUS = 45
FLOOR = pygame.Rect(50, 300, 670, 10)
WALL = pygame.Rect(50, 150, 5, 150)
BALL_Y = FLOOR.y - RADIUS  # balls rest on top of the floor
START_X_SMALL, START_X_BIG, START_V_BIG = 200.0, 380.0, -50.0


def load_phase_image(n_digits):
    """Pre-rendered phase-space plot for n = 1..5, or None if there isn't one."""
    path = ASSETS / "phase_images" / f"n_{n_digits}.png"
    if not path.exists():
        return None
    image = Image.open(path).resize((350, 350)).filter(ImageFilter.SHARPEN)
    return pygame.image.fromstring(image.tobytes(), image.size, image.mode)


def new_simulation(n_digits):
    # Event-driven: computes the exact time of every collision, so fast-moving
    # balls can never tunnel through each other.
    return CollisionSimulation(
        n_digits,
        x_small=START_X_SMALL,
        x_big=START_X_BIG,
        v_big=START_V_BIG,
        r_small=RADIUS,
        r_big=RADIUS,
        wall_x=WALL.right,  # right face of the wall
    )


def pi_from_count(count):
    """31415 -> '3.1415' (the collision count read as digits of pi)."""
    digits = str(count)
    return digits if len(digits) == 1 else f"{digits[0]}.{digits[1:]}"


def mass_label(n_digits):
    return "m" if n_digits == 1 else f"100^{n_digits - 1} m"


class App:
    def __init__(self, n_digits):
        pygame.mixer.init()
        pygame.init()
        self.window = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("PI COLLISION SIMULATOR")
        pygame.display.set_icon(pygame.image.load(ASSETS / "pieImage.jpg"))

        self.sound = pygame.mixer.Sound(ASSETS / "boop.mp3")
        self.sound.set_volume(0.2)
        self.last_sound_ms = 0
        self.muted = False

        # Fonts are created once: SysFont is slow and was being rebuilt every frame
        self.font = pygame.font.SysFont("Cascadia", 28, bold=True)
        self.small_font = pygame.font.SysFont("Cascadia", 20)
        self.pi_font = pygame.font.SysFont("Cascadia Mono, Menlo, Courier New", 34, bold=True)

        self.clock = pygame.time.Clock()
        self.speed_index = SPEEDS.index(1.0)
        self.paused = False
        self.restart(n_digits)

    # ----- state -----
    def restart(self, n_digits=None):
        if n_digits is not None:
            self.n_digits = max(MIN_DIGITS, min(MAX_DIGITS, n_digits))
        self.sim = new_simulation(self.n_digits)
        self.phase_image = load_phase_image(self.n_digits)

    @property
    def speed(self):
        return SPEEDS[self.speed_index]

    # ----- input -----
    def handle_event(self, event):
        if event.type == pygame.QUIT:
            return False
        if event.type != pygame.KEYDOWN:
            return True
        if event.key == pygame.K_ESCAPE:
            return False
        if event.key == pygame.K_UP:
            self.restart(self.n_digits + 1)
        elif event.key == pygame.K_DOWN:
            self.restart(self.n_digits - 1)
        elif event.key == pygame.K_r:
            self.restart()
        elif event.key == pygame.K_SPACE:
            self.paused = not self.paused
        elif event.key == pygame.K_RIGHT:
            self.speed_index = min(self.speed_index + 1, len(SPEEDS) - 1)
        elif event.key == pygame.K_LEFT:
            self.speed_index = max(self.speed_index - 1, 0)
        elif event.key == pygame.K_m:
            self.muted = not self.muted
        return True

    # ----- update -----
    def update(self, frame_time):
        if self.paused:
            return
        # Jump through every collision that happens during this frame
        # (after the last one the balls just keep coasting apart)
        new_collisions = self.sim.advance(frame_time * self.speed)
        now = pygame.time.get_ticks()
        if new_collisions and not self.muted and now - self.last_sound_ms >= SOUND_MIN_INTERVAL_MS:
            self.sound.play()
            self.last_sound_ms = now

    # ----- drawing -----
    def text(self, string, pos, font=None, color=INK):
        self.window.blit((font or self.font).render(string, True, color), pos)

    def draw_ball(self, x, color, label):
        center = (int(x), BALL_Y)
        pygame.draw.circle(self.window, color, center, RADIUS)
        surface = self.small_font.render(label, True, INK)
        self.window.blit(surface, surface.get_rect(midtop=(center[0], FLOOR.bottom + 4)))

    def draw_pi(self, pos):
        """Collision count read as pi; digits turn green as they lock in."""
        estimate = pi_from_count(self.sim.collisions)
        x, y = pos
        label = self.pi_font.render("π ≈ ", True, INK)
        self.window.blit(label, (x, y))
        x += label.get_width()
        for i, char in enumerate(estimate):
            if not self.sim.finished:
                color = INK
            elif i < len(PI_STRING) and char == PI_STRING[i]:
                color = MATCH_COLOR
            else:
                color = MISMATCH_COLOR
            glyph = self.pi_font.render(char, True, color)
            self.window.blit(glyph, (x, y))
            x += glyph.get_width()
        actual = self.small_font.render(f"actual  {PI_STRING[: self.n_digits + 4]}...", True, MUTED)
        self.window.blit(actual, (pos[0], y + 38))

    def draw(self):
        self.window.fill(BACKGROUND)

        if self.phase_image is not None:
            self.window.blit(self.phase_image, (WIDTH - 360, 10))

        pygame.draw.rect(self.window, INK, FLOOR)
        pygame.draw.rect(self.window, INK, WALL)
        self.draw_ball(self.sim.x_small, SMALL_COLOR, "m")
        self.draw_ball(self.sim.x_big, BIG_COLOR, mass_label(self.n_digits))

        self.text(f"Collisions: {self.sim.collisions:,}", (10, 10))
        self.text(f"Big ball velocity: {self.sim.v_big:.2f}", (10, 40))
        self.text(f"n_digits: {self.n_digits}", (10, 70))
        self.draw_pi((470, 150))

        status = "PAUSED" if self.paused else ("DONE" if self.sim.finished else f"speed {self.speed:g}x")
        self.text(status, (470, 240), self.small_font, MUTED)
        self.text(
            "Up/Down digits   Left/Right speed   Space pause   R restart   M mute",
            (10, HEIGHT - 26),
            self.small_font,
            MUTED,
        )
        pygame.display.flip()

    # ----- main loop -----
    def run(self):
        running = True
        while running:
            # Real time since the last frame; capped so dragging the window
            # (which stalls the loop) doesn't cause a huge jump.
            frame_time = min(self.clock.tick(FPS) / 1000.0, 0.1)
            for event in pygame.event.get():
                running = self.handle_event(event) and running
            self.update(frame_time)
            self.draw()
        pygame.quit()


def main():
    n_digits = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    App(n_digits).run()


if __name__ == "__main__":
    main()

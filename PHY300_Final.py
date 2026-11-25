from pathlib import Path

import pygame
from PIL import Image, ImageFilter

from pi_collisions import CollisionSimulation

# Resolve assets relative to this file so the script runs from any directory
ASSETS = Path(__file__).resolve().parent / "assets"

pygame.mixer.init() #strictly for sound
pygame.init()

WIDTH, HEIGHT = 1000, 400
window = pygame.display.set_mode((WIDTH,HEIGHT))

## customizing
pygame.display.set_caption("PI COLLISION SIMULATOR")
icon = pygame.image.load(ASSETS / "pieImage.jpg")
pygame.display.set_icon(icon)

##addition of sound lol
collision_sound = pygame.mixer.Sound(ASSETS / "boop.mp3")
collision_sound.set_volume(0.2)

##creating a definition for object (just for drawing; the physics lives in pi_collisions.py)
class CircularObject:
    ### creating the objects properties
    def __init__(self, xpos, ypos, radius, color):
        self.xpos = xpos
        self.ypos = ypos
        self.radius = radius
        self.color = color

    ### drawing the circle
    def draw(self, window):
        pygame.draw.circle(window, self.color, (int(self.xpos), int(self.ypos)), self.radius)

##creating a definition for the boundaries
class Line:
    ## properties of the line and boundaries
    def __init__(self, x, y, width, height, color):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.color = color

    def draw(self, screen):
        pygame.draw.rect(screen, self.color, (self.x, self.y, self.width, self.height))

## setting the requirements for the being able to calculate PI n_digits is the number of pi digits you get back
n_digits = 5
RADIUS = 45

#creating the boundaries and circles
horizontal_line = Line(50,300, 400, 10, (0,0,0))
verticle_line = Line(50,150,5,150,(0,0,0))
ball_y = horizontal_line.y - RADIUS  # balls rest on top of the floor
stationary_circle = CircularObject(200, ball_y, RADIUS, (212, 97, 94))
moving_circle = CircularObject(380, ball_y, RADIUS, (112, 134, 168))

# Event-driven simulation: computes the exact time of every collision, so
# fast-moving balls can never tunnel through each other (the old fixed-step
# version lost collisions from n = 4 upward).
sim = CollisionSimulation(
    n_digits,
    x_small=stationary_circle.xpos,
    x_big=moving_circle.xpos,
    v_big=-50.0,
    r_small=RADIUS,
    r_big=RADIUS,
    wall_x=verticle_line.x + verticle_line.width,  # right face of the wall
)


def run():
    run = True
    fps = 60
    clock = pygame.time.Clock()

    # Select the phase image for n_digits (pre-rendered images exist for n = 1..5; fall back to n_1)
    image_path = ASSETS / "phase_images" / f"n_{n_digits}.png"
    if not image_path.exists():
        image_path = ASSETS / "phase_images" / "n_1.png"
    image = Image.open(image_path)
    image = image.resize((350, 350))
    image = image.filter(ImageFilter.SHARPEN)

    # Convert back to a format pygame can use
    image = pygame.image.fromstring(image.tobytes(), image.size, image.mode)


    #GAME LOOP
    while run:
        # real time elapsed since the last frame (ticking once per frame keeps the sim in real time)
        frame_time = clock.tick(fps) / 1000.0  # convert milliseconds to seconds
        window.fill((251,247,245))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        # Jump through every collision that happens during this frame
        new_collisions = sim.advance(frame_time)
        if new_collisions:
            collision_sound.play()  # once per frame at most, not once per collision
        stationary_circle.xpos = sim.x_small
        moving_circle.xpos = sim.x_big

        # Drawing all the figures I previously created
        stationary_circle.draw(window)
        moving_circle.draw(window)
        horizontal_line.draw(window)
        verticle_line.draw(window)


        # Setting font
        font = pygame.font.SysFont("Cascadia", 28, bold=True)


        ## displaying the collisions
        text = font.render(f'Ball-to-Ball Collisions: {sim.collisions}', True, (0, 0, 0))
        window.blit(text, (10, 10))

        ## displaying the n_digits that occur which is equivlant the number of PI
        text = font.render(f'n_digits: {n_digits}', True, (0, 0, 0))
        window.blit(text, (10, 70))

        ## velocity
        velocity_text = font.render(f'Velocity: {sim.v_big:.2f}', True, (0, 0, 0))
        window.blit(velocity_text, (10, 40))

        ## PI
        velocity_text = font.render('3.14159265359', True, (0, 0, 0))
        window.blit(velocity_text, (10, 380))

        ## ADDING PHASE IMAGE
        window.blit(image, (625, 10))
        pygame.display.update()
    pygame.quit()

if __name__ == "__main__":
    run()

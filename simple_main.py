import pygame
import random
from pygame.math import Vector2
import math

# Initialize Pygame
pygame.init()

# =========================
# Constants and Configuration
# =========================

# Screen dimensions
WIDTH, HEIGHT = 1920, 1080
fullscreen = False

# Number of stars
NUM_STARS = 500

# Player movement parameters
PLAYER_MOVEMENT_SPEED = 2000  # Pixels per second
DEPTH_CHANGE_RATE = 1  # Adjust this value to control depth change speed

# Depth range
MIN_DEPTH = 0.1
MAX_DEPTH = 10.0

# Colors
STAR_COLOR = (255, 255, 255)
TARGET_COLOR = (255, 0, 0)  # Red for targeting box

# =========================
# Utility Functions
# =========================

def wrap_depth(depth: float) -> float:
    """
    Wrap depth within [MIN_DEPTH, MAX_DEPTH] toroidally.
    """
    depth_range = MAX_DEPTH - MIN_DEPTH
    return MIN_DEPTH + (depth - MIN_DEPTH) % depth_range

def draw_target_box(surface, pos, size, color, thickness=2):
    """
    Draw a retro-style targeting box around a position.

    The box consists of lines on the corners only, giving a retro feel.
    """
    half_size = size // 2

    # Define the rectangle coordinates
    left = pos.x - half_size
    right = pos.x + half_size
    top = pos.y - half_size
    bottom = pos.y + half_size

    # Define the length of corner lines
    corner_length = size // 4

    # Top-left corner
    pygame.draw.line(surface, color, (left, top), (left + corner_length, top), thickness)
    pygame.draw.line(surface, color, (left, top), (left, top + corner_length), thickness)

    # Top-right corner
    pygame.draw.line(surface, color, (right, top), (right - corner_length, top), thickness)
    pygame.draw.line(surface, color, (right, top), (right, top + corner_length), thickness)

    # Bottom-left corner
    pygame.draw.line(surface, color, (left, bottom), (left + corner_length, bottom), thickness)
    pygame.draw.line(surface, color, (left, bottom), (left, bottom - corner_length), thickness)

    # Bottom-right corner
    pygame.draw.line(surface, color, (right, bottom), (right - corner_length, bottom), thickness)
    pygame.draw.line(surface, color, (right, bottom), (right, bottom - corner_length), thickness)

# =========================
# Star Class
# =========================

class Star:
    def __init__(self, x: float, y: float, depth: float):
        self.pos = Vector2(x, y)
        self.depth = depth
        self.base_size = random.randint(1, 3)

    def update(self, player_velocity: Vector2, depth_change: float, dt: float):
        # Update star depth and wrap if necessary
        old_depth = self.depth
        self.depth += depth_change
        wrapped_depth = False

        # Handle depth wrapping
        if self.depth > MAX_DEPTH:
            self.depth = MIN_DEPTH
            wrapped_depth = True
        elif self.depth < MIN_DEPTH:
            self.depth = MAX_DEPTH
            wrapped_depth = True

        # Handle depth-based position inversion when wrapping
        if wrapped_depth:
            self.pos.x = WIDTH - self.pos.x
            self.pos.y = HEIGHT - self.pos.y

        # Parallax effect based on depth
        parallax_factor = 1.0 / self.depth
        self.pos.x -= player_velocity.x * parallax_factor * dt
        self.pos.y -= player_velocity.y * parallax_factor * dt

        # Handle screen wrapping with inversion
        if self.pos.x < 0:
            self.pos.x = WIDTH + self.pos.x
            self.pos.y = HEIGHT - self.pos.y
        elif self.pos.x > WIDTH:
            self.pos.x = self.pos.x - WIDTH
            self.pos.y = HEIGHT - self.pos.y

        if self.pos.y < 0:
            self.pos.y = HEIGHT + self.pos.y
            self.pos.x = WIDTH - self.pos.x
        elif self.pos.y > HEIGHT:
            self.pos.y = self.pos.y - HEIGHT
            self.pos.x = WIDTH - self.pos.x

    def draw(self, surface: pygame.Surface):
        size = max(1, int(self.base_size / self.depth))
        pygame.draw.circle(surface, STAR_COLOR, (int(self.pos.x), int(self.pos.y)), size)

# =========================
# Player Class
# =========================

class Player:
    def __init__(self):
        self.velocity = Vector2(0, 0)
        self.depth = 1.0

    def handle_input(self, dt: float) -> float:
        keys = pygame.key.get_pressed()
        acceleration = Vector2(0, 0)
        depth_change = 0.0

        if keys[pygame.K_w]:
            acceleration.y = -PLAYER_MOVEMENT_SPEED
        if keys[pygame.K_s]:
            acceleration.y = PLAYER_MOVEMENT_SPEED
        if keys[pygame.K_a]:
            acceleration.x = -PLAYER_MOVEMENT_SPEED
        if keys[pygame.K_d]:
            acceleration.x = PLAYER_MOVEMENT_SPEED
        if keys[pygame.K_q]:
            depth_change += DEPTH_CHANGE_RATE * dt
        if keys[pygame.K_e]:
            depth_change -= DEPTH_CHANGE_RATE * dt

        # Update velocity
        self.velocity = acceleration * dt
        self.depth = max(MIN_DEPTH, min(MAX_DEPTH, self.depth + depth_change))
        return depth_change

# =========================
# Game Class
# =========================

class Game:
    def __init__(self):
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT)) if not fullscreen else pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN)
        pygame.display.set_caption("Video Game with Star Targeting")
        self.clock = pygame.time.Clock()
        self.FPS = 60
        self.running = True

        # Initialize player
        self.player = Player()

        # Initialize stars
        self.stars = [Star(random.uniform(0, WIDTH), random.uniform(0, HEIGHT), random.uniform(MIN_DEPTH, MAX_DEPTH)) for _ in range(NUM_STARS)]

        # Initialize targeting system
        self.targeted_star = None

    def run(self):
        while self.running:
            dt = self.clock.tick(self.FPS) / 1000  # Delta time in seconds

            # Event handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self.handle_mouse_click(event.pos)

            # Handle player input
            depth_change = self.player.handle_input(dt)

            # Update stars
            for star in self.stars:
                star.update(self.player.velocity, depth_change, dt)

            # Draw everything
            self.screen.fill((0, 0, 0))  # Black background
            for star in self.stars:
                star.draw(self.screen)

            # Draw targeting box if a star is targeted
            if self.targeted_star is not None:
                pos = self.targeted_star.pos
                size = max(1, int(self.targeted_star.base_size / self.targeted_star.depth))
                box_size = size * 8  # Adjust multiplier for box size
                draw_target_box(self.screen, pos, box_size, TARGET_COLOR, thickness=2)

            # Update display
            pygame.display.flip()

        pygame.quit()

    def handle_mouse_click(self, mouse_pos):
        """
        Handle mouse click events to target stars.
        """
        mouse_vector = Vector2(mouse_pos)
        clicked_star = None

        # Iterate in reverse to prioritize stars drawn on top
        for star in reversed(self.stars):
            size = max(1, int(star.base_size / star.depth))
            if (star.pos - mouse_vector).length() <= size:
                clicked_star = star
                break

        if clicked_star:
            self.targeted_star = clicked_star
        else:
            self.targeted_star = None

# =========================
# Main Execution
# =========================

if __name__ == "__main__":
    game = Game()
    game.run()

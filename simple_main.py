import pygame
import random
from pygame.math import Vector2
from spaceship import *

WIDTH, HEIGHT = 1920, 1080
FULLSCREEN = False
NUM_STARS = 100
PLAYER_SPEED = 2000
DEPTH_RATE = 0.1
MIN_DEPTH = 0.1
MAX_DEPTH = 1.0
BULLET_MAX_DEPTH = 5.0
STAR_COLOR = (255, 255, 255)
TARGET_COLOR = (255, 0, 0)

BASE_DIRECTION_MAP = {
    (0, -1): "up",
    (0, 1): "down",
    (-1, 0): "left",
    (1, 0): "right",
    (1, -1): "up-right",
    (-1, -1): "up-left",
    (1, 1): "down-right",
    (-1, 1): "down-left"
}

DIRECTION_VECTORS = {
    "up": (0, -1),
    "down": (0, 1),
    "left": (-1, 0),
    "right": (1, 0),
    "up-right": (1, -1),
    "up-left": (-1, -1),
    "down-right": (1, 1),
    "down-left": (-1, 1),
}

def get_direction(keys_pressed):
    """
    Returns the direction based on the keys pressed.

    Args:
        keys_pressed (pygame.key.get_pressed): The keys pressed by the user.

    Returns:
        str or None: The direction string based on key presses or None if no direction.
    """
    dx = keys_pressed[pygame.K_d] - keys_pressed[pygame.K_a]
    dy = keys_pressed[pygame.K_s] - keys_pressed[pygame.K_w]
    return BASE_DIRECTION_MAP.get((dx, dy)) if (dx, dy) != (0, 0) else None

def wrap_depth(depth):
    """
    Wraps the depth value to ensure it stays within bounds.

    Args:
        depth (float): The current depth value.

    Returns:
        float: The wrapped depth value.
    """
    return MIN_DEPTH + (depth - MIN_DEPTH) % (MAX_DEPTH - MIN_DEPTH)

def draw_box(surface, position, size, color, thickness=2):
    """
    Draws a box around the given position.

    Args:
        surface (pygame.Surface): The surface to draw on.
        position (pygame.Vector2): The position of the box.
        size (int): The size of the box.
        color (tuple): The color of the box.
        thickness (int): The thickness of the box lines.
    """
    half_size = size // 2
    quarter_size = size // 4
    x, y = position.x, position.y
    lines = [
        ((x - half_size, y - half_size), (x - half_size + quarter_size, y - half_size)),
        ((x - half_size, y - half_size), (x - half_size, y - half_size + quarter_size)),
        ((x + half_size, y - half_size), (x + half_size - quarter_size, y - half_size)),
        ((x + half_size, y - half_size), (x + half_size, y - half_size + quarter_size)),
        ((x - half_size, y + half_size), (x - half_size + quarter_size, y + half_size)),
        ((x - half_size, y + half_size), (x - half_size, y + half_size - quarter_size)),
        ((x + half_size, y + half_size), (x + half_size - quarter_size, y + half_size)),
        ((x + half_size, y + half_size), (x + half_size, y + half_size - quarter_size)),
    ]
    for start_pos, end_pos in lines:
        pygame.draw.line(surface, color, start_pos, end_pos, thickness)

class Star:
    """
    Represents a star in the simulation.
    """
    def __init__(self, x, y, depth):
        """
        Initializes the Star object.

        Args:
            x (float): The x-coordinate of the star.
            y (float): The y-coordinate of the star.
            depth (float): The depth of the star.
        """
        self.position = Vector2(x, y)
        self.depth = depth
        self.size = random.randint(1, 3)

    def update(self, velocity, depth_change, delta_time, is_target=False):
        """
        Updates the position and depth of the star.

        Args:
            velocity (pygame.Vector2): The velocity of the player.
            depth_change (float): The change in depth for the star.
            delta_time (float): The time elapsed since the last update.
            is_target (bool): Whether the star is the target.
        """
        if is_target:
            self.depth = max(MIN_DEPTH, min(MAX_DEPTH, self.depth + depth_change))
            parallax_factor = 1.0 / self.depth
            new_x = max(0, min(WIDTH, self.position.x - velocity.x * parallax_factor * delta_time))
            new_y = max(0, min(HEIGHT, self.position.y - velocity.y * parallax_factor * delta_time))
            self.position.update(new_x, new_y)
        else:
            self.depth = wrap_depth(self.depth + depth_change)
            parallax_factor = 1.0 / self.depth
            self.position.x = (self.position.x - velocity.x * parallax_factor * delta_time) % WIDTH
            self.position.y = (self.position.y - velocity.y * parallax_factor * delta_time) % HEIGHT

    def draw(self, surface):
        """
        Draws the star on the surface.

        Args:
            surface (pygame.Surface): The surface to draw on.
        """
        radius = max(1, int(self.size / self.depth))
        pygame.draw.circle(surface, STAR_COLOR, (int(self.position.x), int(self.position.y)), radius)

class Player:
    """
    Represents the player in the simulation.
    """
    def __init__(self):
        """
        Initializes the Player object.
        """
        self.velocity = Vector2()
        self.depth = 1.0
        self.direction = "up"
        self.last_direction = "up"
        self.scroll_mode = "middle"
        self.scroll_states = ["outward", "middle", "inward"]
        self.wheel = 0 

    def handle_input(self, delta_time):
        """
        Handles player input for movement and depth change.

        Args:
            delta_time (float): The time elapsed since the last update.

        Returns:
            float: The change in depth due to input.
        """
        keys_pressed = pygame.key.get_pressed()
        depth_change = 0.0
        current_direction = get_direction(keys_pressed) or self.last_direction
        self.last_direction = current_direction if current_direction else self.last_direction

        acceleration = Vector2(
            (keys_pressed[pygame.K_d] - keys_pressed[pygame.K_a]) * PLAYER_SPEED,
            (keys_pressed[pygame.K_s] - keys_pressed[pygame.K_w]) * PLAYER_SPEED
        )

        depth_change += (keys_pressed[pygame.K_q] - keys_pressed[pygame.K_e]) * DEPTH_RATE * delta_time
        depth_change += -self.wheel * DEPTH_RATE * delta_time
        self.wheel = 0

        self.velocity = acceleration * delta_time
        self.depth = max(MIN_DEPTH, min(MAX_DEPTH, self.depth + depth_change))

        if self.scroll_mode:
            self.direction = f"{current_direction}_{self.scroll_mode}" if current_direction else f"{self.last_direction}_{self.scroll_mode}"
        else:
            self.direction = current_direction

        if not any([keys_pressed[pygame.K_w], keys_pressed[pygame.K_a], keys_pressed[pygame.K_s], keys_pressed[pygame.K_d]]):
            self.direction = self.last_direction

        return depth_change

    def handle_wheel(self, y):
        """
        Handles the scroll wheel input to change the scroll position.

        Args:
            y (int): The vertical scroll direction (positive or negative).
        """
        current_index = self.scroll_states.index(self.scroll_mode)

        if y > 0 and current_index < len(self.scroll_states) - 1:
            new_index = current_index + 1
            self.scroll_mode = self.scroll_states[new_index]
        elif y < 0 and current_index > 0:
            new_index = current_index - 1
            self.scroll_mode = self.scroll_states[new_index]

        if self.scroll_mode == "outward":
            self.wheel = -1
        elif self.scroll_mode == "inward":
            self.wheel = 1
        else:
            self.wheel = 0

class Bullet:
    """
    Represents a bullet fired by the player.
    """
    def __init__(self, position, direction, initial_depth, spaceship_width, spaceship_height):
        """
        Initializes the Bullet object.

        Args:
            position (pygame.Vector2): The initial position of the bullet.
            direction (str): The direction of the bullet.
            initial_depth (float): The initial depth of the bullet.
            spaceship_width (int): The width of the spaceship.
            spaceship_height (int): The height of the spaceship.
        """
        self.position = Vector2(position)
        self.direction = direction
        self.initial_depth = initial_depth
        self.depth = initial_depth

        base_speed = 200
        base_direction = direction.split("_")[0]
        self.direction_vector = Vector2(DIRECTION_VECTORS.get(base_direction, (0, -1)))

        if self.direction_vector == Vector2(0, 0):
            self.direction_vector = Vector2(0, -1)

        self.velocity = self.direction_vector * base_speed

        if "inward" in direction:
            self.target_depth = MIN_DEPTH
            self.depth_change = 0.25
            self.velocity *= 0.25
        elif "outward" in direction:
            self.target_depth = BULLET_MAX_DEPTH
            self.depth_change = -0.25
            self.velocity *= 0.25
        else:
            self.target_depth = initial_depth
            self.depth_change = 0.0

        ship_size = min(spaceship_width, spaceship_height)
        self.base_size = max(1, int(ship_size / (2 * self.initial_depth)))

        self.lifespan = 2000 if self.depth_change == 0 else 1000
        self.creation_time = pygame.time.get_ticks()
        self.alive = True

    def update(self, delta_time):
        """
        Updates the bullet's position and checks its lifespan.

        Args:
            delta_time (float): The time elapsed since the last update.
        """
        current_time = pygame.time.get_ticks()
        if (current_time - self.creation_time) > self.lifespan:
            self.alive = False
            return

        self.depth += self.depth_change * delta_time

        if self.depth < MIN_DEPTH or self.depth > BULLET_MAX_DEPTH:
            self.alive = False
            return

        parallax_factor = max(1e-6, 2.0 / self.depth)
        self.position += self.velocity * parallax_factor * delta_time

        if (self.position.x < 0 or self.position.x > WIDTH or
            self.position.y < 0 or self.position.y > HEIGHT):
            self.alive = False

    def draw(self, surface):
        """
        Draws the bullet on the surface.

        Args:
            surface (pygame.Surface): The surface to draw on.
        """
        dynamic_size = max(1, int(self.base_size / (self.depth ** 3.14))) // 2

        color_factor = (self.depth - MIN_DEPTH) / (BULLET_MAX_DEPTH - MIN_DEPTH) * 3
        red_value = int(255 - 127 * color_factor)
        color = (red_value, 0, 0)

        pygame.draw.circle(surface, color, (int(self.position.x), int(self.position.y)), dynamic_size)

class Game:
    """
    Represents the game simulation.
    """
    def __init__(self):
        """
        Initializes the Game object.
        """
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN if FULLSCREEN else 0)
        pygame.display.set_caption("Parallax Universe Simulator")
        self.clock = pygame.time.Clock()
        self.running = True
        self.player = Player()
        self.stars = [
            Star(
                random.uniform(0, WIDTH),
                random.uniform(0, HEIGHT),
                random.uniform(MIN_DEPTH, MAX_DEPTH)
            ) for _ in range(NUM_STARS)
        ]
        self.target_star = None
        self.bullets = []
        pygame.event.set_allowed([
            pygame.QUIT,
            pygame.KEYDOWN,
            pygame.KEYUP,
            pygame.MOUSEBUTTONDOWN,
            pygame.MOUSEWHEEL
        ])

    def run(self):
        """
        Runs the game loop.
        """
        while self.running:
            delta_time = self.clock.tick(60) / 1000.0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self.target_star = next(
                        (
                            star for star in reversed(self.stars)
                            if (star.position - Vector2(event.pos)).length() <= max(1, int(star.size / star.depth))
                        ), None
                    )
                elif event.type == pygame.MOUSEWHEEL:
                    self.player.handle_wheel(event.y)
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        direction = self.player.direction
                        if self.player.scroll_mode == "outward":
                            direction = f"{self.player.direction}_outward"
                        bullet_position = Vector2(WIDTH // 2, HEIGHT // 2)
                        spaceship_shape = SPACESHIP_SHAPES.get(self.player.direction, SPACESHIP_SHAPES["up"])
                        spaceship_width = len(spaceship_shape[0]) * PIXEL_SIZE
                        spaceship_height = len(spaceship_shape) * PIXEL_SIZE
                        self.bullets.append(Bullet(bullet_position, direction, self.player.depth, spaceship_width, spaceship_height))

            depth_change = self.player.handle_input(delta_time)
            if self.target_star:
                depth_change += self.center_zoom(delta_time)
            else:
                self.center_zoom(delta_time)

            for star in self.stars:
                star.update(self.player.velocity, depth_change, delta_time, star is self.target_star)

            self.bullets.sort(key=lambda b: b.depth, reverse=True)
            
            for bullet in self.bullets:
                bullet.update(delta_time)
                
            self.bullets = [bullet for bullet in self.bullets if bullet.alive]

            self.screen.fill((0, 0, 0))

            for star in self.stars:
                star.draw(self.screen)

            far_bullets = [b for b in self.bullets if "inward" in b.direction or "outward" not in b.direction]
            near_bullets = [b for b in self.bullets if "outward" in b.direction]

            for bullet in far_bullets:
                bullet.draw(self.screen)

            spaceship_shape = SPACESHIP_SHAPES.get(self.player.direction, SPACESHIP_SHAPES["up"])
            spaceship_width = len(spaceship_shape[0]) * PIXEL_SIZE
            spaceship_height = len(spaceship_shape) * PIXEL_SIZE
            spaceship_position = ((WIDTH - spaceship_width) // 2, (HEIGHT - spaceship_height) // 2)
            draw_spaceship(self.screen, spaceship_shape, spaceship_position)

            for bullet in near_bullets:
                bullet.draw(self.screen)

            if self.target_star:
                box_size = max(1, int(self.target_star.size / self.target_star.depth)) * 8
                draw_box(self.screen, self.target_star.position, box_size, TARGET_COLOR)

            pygame.display.flip()

    def center_zoom(self, delta_time):
        """
        Adjusts the zoom based on the target star's depth.

        Args:
            delta_time (float): The time elapsed since the last update.

        Returns:
            float: The depth change.
        """
        if not self.target_star:
            return 0.0
        center = Vector2(WIDTH / 2, HEIGHT / 2)
        displacement = (center - self.target_star.position) * delta_time
        for star in self.stars:
            star.position += displacement
        for bullet in self.bullets:
            bullet.position += displacement
        depth_delta = (MIN_DEPTH - self.target_star.depth) * delta_time
        if (self.target_star.depth <= MIN_DEPTH and depth_delta < 0) or \
           (self.target_star.depth >= MAX_DEPTH and depth_delta > 0):
            return 0.0
        return depth_delta

if __name__ == "__main__":
    pygame.init()
    Game().run()

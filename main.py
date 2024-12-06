import pygame
import random
import math
from pygame.math import Vector2
import pygame.freetype
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Tuple, Callable, Union, Optional
import textwrap
from constants import *
from celestial_objects import *

pygame.init()

@dataclass
class Player:
    """Represents the player in the universe."""

    depth: float = 1.0
    velocity: Vector2 = field(default_factory=lambda: Vector2(0, 0))
    acceleration: Vector2 = field(default_factory=lambda: Vector2(0, 0))

    def handle_input(self, dt: float) -> float:
        """
        Handle player input and update velocity and depth.

        Args:
            dt (float): Delta time in seconds.

        Returns:
            float: Change in depth.
        """
        keys = pygame.key.get_pressed()
        self.acceleration = Vector2(0, 0)
        depth_change = 0.0

        if keys[pygame.K_w]:
            self.acceleration.y = -PLAYER_MOVEMENT_SPEED
        if keys[pygame.K_s]:
            self.acceleration.y = PLAYER_MOVEMENT_SPEED
        if keys[pygame.K_a]:
            self.acceleration.x = -PLAYER_MOVEMENT_SPEED
        if keys[pygame.K_d]:
            self.acceleration.x = PLAYER_MOVEMENT_SPEED
        if keys[pygame.K_q]:
            depth_change += DEPTH_CHANGE_RATE * dt
        if keys[pygame.K_e]:
            depth_change -= DEPTH_CHANGE_RATE * dt

        self.velocity += self.acceleration * dt
        self.velocity *= MOMENTUM_DECAY

        if self.velocity.length() > 0:
            self.velocity.scale_to_length(max(self.velocity.length(), MINIMUM_MOVEMENT))

        self.depth += math.copysign(abs(depth_change) ** 1.2, depth_change)
        self.depth = max(MIN_DEPTH, min(MAX_DEPTH, self.depth))
        self.depth = MAX_DEPTH + MIN_DEPTH - self.depth

        return depth_change

def draw_target_box(surface, pos, size, color, thickness=2):
    """
    Draw a retro-style targeting box around a position.

    Args:
        surface (pygame.Surface): The surface to draw on.
        pos (Vector2): The position around which to draw the box.
        size (int): The size of the box.
        color (Tuple[int, int, int]): The color of the box.
        thickness (int, optional): The thickness of the lines.
    """
    half_size = size // 2
    left = pos.x - half_size
    right = pos.x + half_size
    top = pos.y - half_size
    bottom = pos.y + half_size
    corner_length = size // 4

    pygame.draw.line(surface, color, (left, top), (left + corner_length, top), thickness)
    pygame.draw.line(surface, color, (left, top), (left, top + corner_length), thickness)
    pygame.draw.line(surface, color, (right, top), (right - corner_length, top), thickness)
    pygame.draw.line(surface, color, (right, top), (right, top + corner_length), thickness)
    pygame.draw.line(surface, color, (left, bottom), (left + corner_length, bottom), thickness)
    pygame.draw.line(surface, color, (left, bottom), (left, bottom - corner_length), thickness)
    pygame.draw.line(surface, color, (right, bottom), (right - corner_length, bottom), thickness)
    pygame.draw.line(surface, color, (right, bottom), (right, bottom - corner_length), thickness)


def handle_mouse_click(mouse_pos, stars):
    """
    Handle mouse click events to target stars.

    Args:
        mouse_pos (Tuple[int, int]): The position of the mouse click.
        stars (List[Star]): The list of stars.

    Returns:
        Optional[Star]: The targeted star if any.
    """
    mouse_vector = Vector2(mouse_pos)
    clicked_star = None

    for star in reversed(stars):
        size = star.get_draw_size()
        if (star.pos - mouse_vector).length() <= size:
            clicked_star = star
            break

    return clicked_star

class SpatialGrid:
    """Simple grid-based spatial partitioning for efficient culling."""

    def __init__(self, width: int, height: int, grid_size: int):
        """
        Initialize the spatial grid.

        Args:
            width (int): Width of the screen.
            height (int): Height of the screen.
            grid_size (int): Size of each grid cell.
        """
        self.grid_size = grid_size
        self.cols = math.ceil(width / grid_size)
        self.rows = math.ceil(height / grid_size)
        self.grid = defaultdict(list)

    def add_star(self, star: Star) -> None:
        """
        Add a star to the appropriate grid cell.

        Args:
            star (Star): The star to add.
        """
        col = int(star.pos.x // self.grid_size)
        row = int(star.pos.y // self.grid_size)
        self.grid[(col, row)].append(star)

    def get_visible_stars(self, visible_rect: pygame.Rect) -> List[Star]:
        """
        Retrieve stars within the visible rectangle.

        Args:
            visible_rect (pygame.Rect): The visible area.

        Returns:
            List[Star]: List of visible stars.
        """
        visible_stars = []
        start_col = int(max(visible_rect.left // self.grid_size, 0))
        end_col = int(min((visible_rect.right // self.grid_size) + 1, self.cols))
        start_row = int(max(visible_rect.top // self.grid_size, 0))
        end_row = int(min((visible_rect.bottom // self.grid_size) + 1, self.rows))

        for col in range(start_col, end_col):
            for row in range(start_row, end_row):
                visible_stars.extend(self.grid.get((col, row), []))

        return visible_stars

    def clear(self) -> None:
        """
        Clear the spatial grid.
        """
        self.grid.clear()

class Game:
    """The main game class that manages the game loop, rendering, and updates."""

    def __init__(self):
        """
        Initialize the game.
        """
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT)) if not fullscreen else pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN)
        pygame.display.set_caption("Generative Universe v1.2.2")
        self.clock = pygame.time.Clock()
        self.FPS = 60
        self.running = True
        self.player = Player()
        self.simulation_time = 0.0
        self.stars: List[Star] = []
        galaxy_count = 0

        for _ in range(NUM_STARS):
            x = random.uniform(0, WIDTH)
            y = random.uniform(0, HEIGHT)

            if random.random() < 0.1:
                phenomenon = Phenomenon(x=x, y=y)
                if phenomenon.name == "Galaxy" and galaxy_count < MAX_GALAXIES:
                    self.stars.append(phenomenon)
                    galaxy_count += 1
                elif phenomenon.name != "Galaxy":
                    self.stars.append(phenomenon)
            else:
                depth = random.uniform(MIN_DEPTH, MAX_DEPTH)
                star = Star(x=x, y=y, depth=depth)
                self.stars.append(star)

        self.spatial_grid = SpatialGrid(WIDTH, HEIGHT, GRID_SIZE)
        self.font = pygame.freetype.SysFont("Arial", 16)
        self.targeted_star: Optional[Star] = None

    def center_and_zoom_target(self, dt: float) -> None:
        """
        Gradually bring the targeted star to the center and increase its depth.
        """
        if self.targeted_star is None:
            return

        center = Vector2(WIDTH / 2, HEIGHT / 2)
        to_center = center - self.targeted_star.pos
        move_speed = 1.0  # Adjust to control centering speed
        displacement = to_center * move_speed * dt

        # Adjust positions of all stars to center the targeted object
        for star in self.stars:
            star.pos += displacement

        # Skip depth adjustment for phenomena
        if isinstance(self.targeted_star, Phenomenon):
            return

        # Gradually adjust the depth to bring the star closer
        zoom_speed = 1.0  # Adjust to control zoom speed
        depth_delta = (MIN_DEPTH - self.targeted_star.depth) * zoom_speed * dt

        for star in self.stars:
            star.update(Vector2(0, 0), depth_delta, dt)

    def update_and_draw_stars(
        self, sim_time: float, depth_change: float, dt: float
    ) -> None:
        """
        Update and draw stars and their orbital bodies.
        """
        margin = GRID_SIZE * 2
        visible_rect = pygame.Rect(-margin, -margin, WIDTH + margin * 2, HEIGHT + margin * 2)
        self.update_spatial_grid()
        visible_stars = self.spatial_grid.get_visible_stars(visible_rect)
        drawable_objects = []
        for star in visible_stars:
            drawable_objects.append((star.depth, star))
            for planet in star.planets:
                drawable_objects.append((planet.depth, planet))
                for moon in planet.moons:
                    drawable_objects.append((moon.depth, moon))
        drawable_objects.sort(key=lambda obj: obj[0], reverse=True)

        for _, obj in drawable_objects:
            if isinstance(obj, Star):
                obj.update(self.player.velocity, depth_change, dt)
                obj.draw(surface=self.screen, time=sim_time)
            elif isinstance(obj, OrbitalBody):
                obj.update(dt)
                obj.draw(surface=self.screen, time=sim_time)

    def run(self) -> None:
        """
        The main game loop.
        """
        while self.running:
            dt = self.clock.tick(self.FPS) / 1000
            self.simulation_time += dt

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self.targeted_star = handle_mouse_click(pygame.mouse.get_pos(), self.stars)

            depth_change = self.player.handle_input(dt)

            # Update stars
            if self.targeted_star:
                self.center_and_zoom_target(dt)

            self.screen.fill((5, 11, 22))
            self.update_and_draw_stars(self.simulation_time, depth_change, dt)

            if self.targeted_star:
                pos = self.targeted_star.pos
                size = self.targeted_star.get_draw_size()
                box_size = size * 8
                draw_target_box(self.screen, pos, box_size, TARGET_COLOR, thickness=2)

            pygame.display.flip()

        pygame.quit()

    def update_spatial_grid(self) -> None:
        """
        Update the spatial grid with current star positions.
        """
        self.spatial_grid.clear()
        for star in self.stars:
            self.spatial_grid.add_star(star)

    def draw_tooltip(self, star: Star, mouse_pos: Vector2) -> None:
        """
        Draw a tooltip near the mouse cursor with star information.

        Args:
            star (Star): The star being hovered.
            mouse_pos (Vector2): Current mouse position.
        """
        def wrap_text(text: str, max_width: int) -> list:
            wrapper = textwrap.TextWrapper(width=30)
            return wrapper.wrap(text)

        if isinstance(star, Phenomenon):
            title = f"Phenomenon ID: #{star.id}"
            type_info = f"Type: {star.name}"
            effect_info = f"Effect: {star.effect}"
        else:
            title = f"Star ID: #{star.id}"
            type_info = f"Type: {star.name}"
            effect_info = "Effect: None"

        composition_lines = []
        for element, percentage in star.composition.items():
            line = f"{element}: {percentage:.2f}%"
            composition_lines.extend(wrap_text(line, 250))

        tooltip_lines = [
            title,
            type_info,
            effect_info,
            f"Color: {star.color}",
            "Composition:",
        ] + composition_lines

        text_surfaces = []
        for i, line in enumerate(tooltip_lines):
            if i == 0:
                surface, _ = self.font.render(
                    line, fgcolor=(255, 255, 255), style=pygame.freetype.STYLE_STRONG
                )
            else:
                surface, _ = self.font.render(line, fgcolor=(255, 255, 255))
            text_surfaces.append(surface)

        line_height = self.font.get_sized_height() + 4
        padding = 10
        width = max(surface.get_width() for surface in text_surfaces) + padding * 2
        height = len(text_surfaces) * line_height + padding * 2
        tooltip_x = mouse_pos.x + 15
        tooltip_y = mouse_pos.y + 15

        if tooltip_x + width > WIDTH:
            tooltip_x = mouse_pos.x - width - 15
        if tooltip_y + height > HEIGHT:
            tooltip_y = mouse_pos.y - height - 15

        tooltip_surface = pygame.Surface((width, height), pygame.SRCALPHA)
        pygame.draw.rect(
            tooltip_surface, (0, 0, 0, 200), (0, 0, width, height), border_radius=8
        )
        self.screen.blit(tooltip_surface, (tooltip_x, tooltip_y))

        current_y = tooltip_y + padding
        for surface in text_surfaces:
            self.screen.blit(surface, (tooltip_x + padding, current_y))
            current_y += line_height


if __name__ == "__main__":
    game = Game()
    game.run()

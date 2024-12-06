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
from helper_methods import *

@dataclass
class OrbitalBody:
    """Represents an orbital body orbiting a parent object."""

    parent: Union["Star", "Planet"]
    size_scale: float
    semi_major_scale: Tuple[float, float]
    semi_minor_scale: Tuple[float, float]
    orbital_period_range: Tuple[float, float]
    pos: Vector2 = field(init=False)
    eccentricity: float = field(init=False)
    inclination: float = field(init=False)
    orbital_period: float = field(init=False)
    current_anomaly: float = field(init=False)
    orbital_speed: float = field(init=False)
    size: int = field(init=False, default=1)
    color: Tuple[int, int, int] = field(init=False)
    semi_major_factor: float = field(init=False)
    z: float = field(init=False, default=0.0)
    precession_rate: float = field(init=False)
    eccentricity_variation: float = field(init=False)

    def __post_init__(self):
        parent_size = (
            self.parent.get_draw_size()
            if isinstance(self.parent, Star)
            else self.parent.get_scaled_size()
        )
        self.semi_major_factor = random.uniform(*self.semi_major_scale)
        self.eccentricity = random.uniform(0.0, 0.6)
        self.inclination = random.uniform(0, math.pi / 6)
        self.orbital_period = random.uniform(*self.orbital_period_range)
        self.current_anomaly = random.uniform(0, 2 * math.pi)
        self.orbital_speed = (2 * math.pi) / self.orbital_period

        self.color = (
            random.randint(200, 255),
            random.randint(200, 255),
            random.randint(200, 255),
        )

        self.precession_rate = random.uniform(-0.001, 0.001)
        self.eccentricity_variation = random.uniform(-0.0005, 0.0005)

    @property
    def depth(self) -> float:
        """
        Calculate the depth of the orbital body based on the parent's depth and z position.

        Returns:
            float: The depth of the orbital body.
        """
        return self.parent.depth + self.z

    def get_scaled_size(self) -> int:
        """
        Calculate the scaled size based on the parent's size and scaling factors.

        Returns:
            int: The scaled size.
        """
        parent_size = (
            self.parent.get_draw_size()
            if isinstance(self.parent, Star)
            else self.parent.get_scaled_size()
        )
        scaled_size = parent_size * self.size_scale * 1
        return max(1, int(scaled_size))

    def draw(self, surface: pygame.Surface, time: float) -> None:
        """
        Draw the orbital body with opacity based on its z-position.

        Args:
            surface (pygame.Surface): The surface to draw on.
            time (float): Current simulation time.
        """
        pos = self.get_position()
        scaled_size = self.get_scaled_size()

        opacity = 255
        if self.z > 0:
            opacity = max(50, 255 - int((self.z / MAX_DEPTH) * 200))

        color_with_opacity = (
            self.color[0],
            self.color[1],
            self.color[2],
            opacity,
        )

        positions_to_draw = get_wrapped_positions(pos, scaled_size)

        for draw_pos in positions_to_draw:
            body_surface = pygame.Surface((scaled_size * 2, scaled_size * 2), pygame.SRCALPHA)
            pygame.draw.circle(body_surface, color_with_opacity, (scaled_size, scaled_size), scaled_size)
            surface.blit(body_surface, (int(draw_pos.x) - scaled_size, int(draw_pos.y) - scaled_size))

    def get_position(self) -> Vector2:
        """
        Calculate and set the position of the orbital body, including the z-axis for depth.

        Returns:
            Vector2: The position of the orbital body.
        """
        parent_size = (
            self.parent.get_draw_size()
            if isinstance(self.parent, Star)
            else self.parent.get_scaled_size()
        )

        semi_major_axis = self.semi_major_factor * (parent_size / 10)
        semi_minor_axis = semi_major_axis * math.sqrt(1 - self.eccentricity ** 2)

        mean_anomaly = self.current_anomaly
        E = self.solve_keplers_equation(mean_anomaly, self.eccentricity)
        true_anomaly = 2 * math.atan2(
            math.sqrt(1 + self.eccentricity) * math.sin(E / 2),
            math.sqrt(1 - self.eccentricity) * math.cos(E / 2),
        )
        r = semi_major_axis * (1 - self.eccentricity * math.cos(E))
        x = r * math.cos(true_anomaly)
        y = r * math.sin(true_anomaly)

        self.z = r * math.sin(true_anomaly) * math.sin(self.inclination)
        y *= math.cos(self.inclination)

        depth_factor = max(self.parent.depth, MIN_DEPTH)
        x /= math.sqrt(depth_factor)
        y /= math.sqrt(depth_factor)

        self.pos = Vector2(self.parent.pos.x + x, self.parent.pos.y + y)
        return self.pos

    def update(self, dt: float) -> None:
        """
        Update the body's anomaly, position, depth, and orbital variations based on the elapsed time.

        Args:
            dt (float): Delta time in seconds.
        """
        self.current_anomaly = (self.current_anomaly + self.orbital_speed * dt + self.precession_rate) % (2 * math.pi)
        self.eccentricity += self.eccentricity_variation
        self.eccentricity = max(0.0, min(self.eccentricity, 0.6))
        self.get_position()

    @staticmethod
    def solve_keplers_equation(M: float, e: float, tolerance: float = 1e-5) -> float:
        """
        Solve Kepler's Equation M = E - e*sin(E) for E given M and e.

        Args:
            M (float): Mean anomaly.
            e (float): Eccentricity.
            tolerance (float, optional): The tolerance for convergence.

        Returns:
            float: Eccentric anomaly.
        """
        E = M if e < 0.8 else math.pi
        F = E - e * math.sin(E) - M
        iteration = 0
        while abs(F) > tolerance and iteration < 100:
            E -= F / (1 - e * math.cos(E))
            F = E - e * math.sin(E) - M
            iteration += 1
        return E


@dataclass(init=False)
class Planet(OrbitalBody):
    """Represents a planet orbiting a star."""

    moons: List["Moon"] = field(default_factory=list, init=False)

    def __init__(self, parent: "Star"):
        super().__init__(
            parent=parent,
            size_scale=0.02,
            semi_major_scale=(4, 24),
            semi_minor_scale=(0.7, 1.3),
            orbital_period_range=(50, 200),
        )
        num_moons = random.randint(0, 2)
        self.moons = [Moon(self) for _ in range(num_moons)]


@dataclass(init=False)
class Moon(OrbitalBody):
    """Represents a moon orbiting a planet."""

    def __init__(self, parent: Planet):
        super().__init__(
            parent=parent,
            size_scale=0.01,
            semi_major_scale=(2, 12),
            semi_minor_scale=(0.7, 1.3),
            orbital_period_range=(20, 50),
        )


@dataclass
class Star:
    """Represents a star in the universe."""

    x: float
    y: float
    depth: float
    type_info: dict = field(init=False)
    name: str = field(init=False)
    color: Union[Tuple[int, int, int], Callable[[], Tuple[int, int, int]]] = field(init=False)
    base_size: int = field(init=False)
    twinkle_offset: float = field(init=False)
    composition: dict = field(init=False)
    planets: List[Planet] = field(default_factory=list, init=False)
    fixed_depth: bool = field(default=False)
    _id_counter: int = field(default=1, init=False, repr=False)
    pos: Vector2 = field(init=False)

    def __post_init__(self):
        self.id = Star._id_counter
        Star._id_counter += 1

        self.pos = Vector2(self.x, self.y)
        self.type_info = random.choice(STAR_TYPES)
        self.name = self.type_info["name"]
        self.color = self.type_info["color"]
        self.base_size = self.type_info["base_size"]
        self.twinkle_offset = random.uniform(0, 2 * math.pi)
        self.composition = self.generate_composition()
        self.depth = MAX_DEPTH + MIN_DEPTH - self.depth

        if not self.fixed_depth and random.random() < 0.2:
            num_planets = random.randint(1, 3)
            self.planets = [Planet(self) for _ in range(num_planets)]

    def generate_composition(self) -> dict:
        """
        Generates a random composition of elements for the star.

        Returns:
            dict: Composition percentages of elements.
        """
        elements = ["Hydrogen", "Helium", "Carbon", "Oxygen", "Iron", "Neon", "Nitrogen"]
        composition = {}
        remaining_percentage = 100.0

        for element in elements[:-1]:
            percentage = round(random.uniform(0, remaining_percentage), 2)
            composition[element] = percentage
            remaining_percentage -= percentage

        composition[elements[-1]] = round(remaining_percentage, 2)
        return composition

    def update(self, player_velocity: Vector2, depth_change: float, dt: float) -> None:
        """
        Update the star's position and depth with inversion-based wrapping.

        Args:
            player_velocity (Vector2): The player's current velocity.
            depth_change (float): Change in depth.
            dt (float): Delta time in seconds.
        """
        if not self.fixed_depth:
            new_depth = self.depth + math.copysign(abs(depth_change) ** 1.1, depth_change)
            wrapped_depth = False

            if new_depth > MAX_DEPTH:
                self.depth = wrap_depth(new_depth)
                wrapped_depth = True
            elif new_depth < MIN_DEPTH:
                self.depth = wrap_depth(new_depth)
                wrapped_depth = True
            else:
                self.depth = new_depth

            if wrapped_depth:
                self.pos.x = WIDTH - self.pos.x
                self.pos.y = HEIGHT - self.pos.y
                self.pos.x = max(0.1, min(self.pos.x, WIDTH - 0.1))
                self.pos.y = max(0.1, min(self.pos.y, HEIGHT - 0.1))

        depth_factor = max(self.depth, MIN_DEPTH)
        parallax_speed = 1 / depth_factor
        self.pos.x -= (player_velocity.x * parallax_speed) * dt
        self.pos.y -= (player_velocity.y * parallax_speed) * dt

        if self.pos.x < 0:
            self.pos.x += WIDTH
            self.pos.y = HEIGHT - self.pos.y
        elif self.pos.x > WIDTH:
            self.pos.x -= WIDTH
            self.pos.y = HEIGHT - self.pos.y

        if self.pos.y < 0:
            self.pos.y += HEIGHT
            self.pos.x = WIDTH - self.pos.x
        elif self.pos.y > HEIGHT:
            self.pos.y -= HEIGHT
            self.pos.x = WIDTH - self.pos.x

        for planet in self.planets:
            planet.update(dt)

    def get_draw_size(self) -> int:
        """
        Calculate the draw size of the star based on its depth.

        Returns:
            int: Scaled size of the star.
        """
        center = Vector2(WIDTH / 2, HEIGHT / 2)
        return calculate_scale(self.base_size, self.depth, self.pos, center) * 1

    def draw(self, surface: pygame.Surface, time: float) -> None:
        """
        Draw the star on the given surface.

        Args:
            surface (pygame.Surface): The surface to draw on.
            time (float): Current simulation time.
        """
        center = Vector2(WIDTH / 2, HEIGHT / 2)
        size = calculate_scale(self.base_size, self.depth, self.pos, center)
        glow_radius = size * 2 if size > 1 else 1.5
        fog_alpha = calculate_fog_alpha(self.depth)
        dynamic_color = self.color() if callable(self.color) else self.color
        current_color = (
            min(255, int(dynamic_color[0])),
            min(255, int(dynamic_color[1])),
            min(255, int(dynamic_color[2])),
            fog_alpha,
        )
        vibrant_color = (
            min(255, int(current_color[0] * 1.1)),
            min(255, int(current_color[1] * 1.1)),
            min(255, int(current_color[2] * 1.1)),
            current_color[3],
        )
        positions_to_draw = get_wrapped_positions(self.pos, size, glow_size=int(glow_radius))

        for draw_pos in positions_to_draw:
            pygame.draw.circle(surface, vibrant_color, (int(draw_pos.x), int(draw_pos.y)), size)
            if glow_radius > 0:
                glow_surface = pygame.Surface(
                    (int(glow_radius) * 2, int(glow_radius) * 2), pygame.SRCALPHA
                )
                glow_color = (*vibrant_color[:3], EFFECT_CONFIG.get("pulsing", {}).get("glow_strength", 64))
                pygame.draw.circle(
                    glow_surface, glow_color, (int(glow_radius), int(glow_radius)), int(glow_radius)
                )
                surface.blit(
                    glow_surface,
                    (int(draw_pos.x) - int(glow_radius), int(draw_pos.y) - int(glow_radius)),
                )

    def is_hovered(self, mouse_pos: Vector2) -> bool:
        """
        Check if the star is hovered by the mouse.

        Args:
            mouse_pos (Vector2): Current mouse position.

        Returns:
            bool: True if hovered, False otherwise.
        """
        size = self.get_draw_size()
        distance = self.pos.distance_to(mouse_pos)
        return distance <= size


@dataclass(init=False)
class Phenomenon(Star):
    """Represents a celestial phenomenon in the universe."""

    type_info: dict = field(init=False)
    effect: str = field(init=False)
    twinkle_speed: float = field(init=False)
    squash_x: float = field(init=False, default=0.3)
    squash_y: float = field(init=False, default=0.7)
    orientation_angle: float = field(init=False, default=0.314)
    render_strategy: Callable = field(init=False)

    def __init__(self, x: float, y: float, depth: Optional[float] = None):
        super().__init__(x=x, y=y, depth=depth if depth is not None else MIN_PHENOMENA_DEPTH, fixed_depth=True)
        self.type_info = random.choice(PHENOMENA_TYPES)
        self.name = self.type_info["name"]
        self.color = (
            self.type_info["color"]()
            if callable(self.type_info["color"])
            else self.type_info["color"]
        )
        self.base_size = self.type_info["base_size"]
        self.effect = self.type_info["effect"]
        self.twinkle_speed = EFFECT_CONFIG.get(self.effect, {}).get("twinkle_speed", 1)

        if self.name == "Galaxy":
            self.depth = random.uniform(MAX_DEPTH * 0.5, MAX_DEPTH)
            self.squash_x = random.uniform(0.6, 1.4)
            self.squash_y = random.uniform(0.6, 1.4)
            self.orientation_angle = random.uniform(0, 2 * math.pi)
        else:
            self.depth = random.uniform(MIN_PHENOMENA_DEPTH, MAX_DEPTH)

        self.render_strategy = self.get_render_strategy()

    def get_render_strategy(self) -> Callable:
        """
        Determine the rendering strategy based on the phenomenon's effect.

        Returns:
            Callable: The appropriate render method.
        """
        strategies = {
            "pulsing": self.draw_pulsing,
            "radiating": self.draw_radiating,
            "swirling": self.draw_swirling,
            "nebula": self.draw_nebula,
            "galaxy": self.draw_galaxy,
        }
        return strategies.get(self.effect, self.draw_default)

    def update(self, player_velocity: Vector2, depth_change: float, dt: float) -> None:
        """
        Update the phenomenon's position. Depth remains fixed.

        Args:
            player_velocity (Vector2): The player's current velocity.
            depth_change (float): Change in depth.
            dt (float): Delta time in seconds.
        """
        depth_factor = max(self.depth, MIN_DEPTH)
        parallax_speed = 1 / depth_factor
        self.pos.x -= (player_velocity.x * parallax_speed) * dt
        self.pos.y -= (player_velocity.y * parallax_speed) * dt

        if self.pos.x < 0:
            self.pos.x += WIDTH
            self.pos.y = HEIGHT - self.pos.y
        elif self.pos.x > WIDTH:
            self.pos.x -= WIDTH
            self.pos.y = HEIGHT - self.pos.y

        if self.pos.y < 0:
            self.pos.y += HEIGHT
            self.pos.x = WIDTH - self.pos.x
        elif self.pos.y > HEIGHT:
            self.pos.y -= HEIGHT
            self.pos.x = WIDTH - self.pos.x

    def draw(self, surface: pygame.Surface, time: float) -> None:
        """
        Draw the phenomenon on the given surface.

        Args:
            surface (pygame.Surface): The surface to draw on.
            time (float): Current simulation time.
        """
        size = self.get_draw_size()
        twinkle = 0.2 * math.sin(time * self.twinkle_speed + self.twinkle_offset) + 0.8
        brightness = twinkle

        current_color = (
            min(255, int(self.color[0] * brightness)),
            min(255, int(self.color[1] * brightness)),
            min(255, int(self.color[2] * brightness)),
        )

        glow_radius = (
            int(size * 2)
            if self.effect == "galaxy"
            else (int(size * 1.5) if self.effect == "nebula" else 0)
        )

        positions_to_draw = get_wrapped_positions(self.pos, size, glow_size=glow_radius)

        for draw_pos in positions_to_draw:
            self.render_strategy(surface, draw_pos, size, time)

            if glow_radius > 0:
                glow_surface = pygame.Surface(
                    (glow_radius * 2, glow_radius * 2), pygame.SRCALPHA
                )
                glow_color = (*current_color, EFFECT_CONFIG.get(self.effect, {}).get("glow_strength", 8))
                pygame.draw.circle(
                    glow_surface, glow_color, (glow_radius, glow_radius), glow_radius
                )
                surface.blit(
                    glow_surface,
                    (int(draw_pos.x) - glow_radius, int(draw_pos.y) - glow_radius),
                )

    def draw_pulsing(self, surface: pygame.Surface, pos: Vector2, size: int, time: float) -> None:
        """
        Draw a pulsing phenomenon.

        Args:
            surface (pygame.Surface): The surface to draw on.
            pos (Vector2): Position to draw.
            size (int): Base size.
            time (float): Current simulation time.
        """
        pulsate_size = size + int(3 * math.sin(time * 3))
        pygame.draw.circle(
            surface, self.color, (int(pos.x), int(pos.y)), pulsate_size
        )

    def draw_radiating(self, surface: pygame.Surface, pos: Vector2, size: int, time: float = 0) -> None:
        """
        Draw a radiating phenomenon.

        Args:
            surface (pygame.Surface): The surface to draw on.
            pos (Vector2): Position to draw.
            size (int): Base size.
            time (float, optional): Current simulation time.
        """
        for i in range(3):
            radiate_radius = size * (4 - i)
            radiate_surface = pygame.Surface(
                (radiate_radius * 2, radiate_radius * 2), pygame.SRCALPHA
            )
            radiate_color = (*self.color, 50 // (i + 1))
            pygame.draw.circle(
                radiate_surface, radiate_color, (radiate_radius, radiate_radius), radiate_radius
            )
            surface.blit(
                radiate_surface,
                (int(pos.x) - radiate_radius, int(pos.y) - radiate_radius),
            )

    def draw_swirling(self, surface: pygame.Surface, pos: Vector2, size: int, time: float) -> None:
        """
        Draw a swirling phenomenon.

        Args:
            surface (pygame.Surface): The surface to draw on.
            pos (Vector2): Position to draw.
            size (int): Base size.
            time (float): Current simulation time.
        """
        num_rings = 1
        for i in range(num_rings):
            angle = time * 2 + (i * math.pi / num_rings)
            ring_radius = size + (i * 5)
            ring_surface = pygame.Surface(
                (ring_radius * 2, ring_radius * 2), pygame.SRCALPHA
            )
            ring_color = (*self.color, 100 // (i + 1))
            pygame.draw.circle(
                ring_surface, ring_color, (ring_radius, ring_radius), ring_radius, width=3
            )
            surface.blit(
                ring_surface,
                (int(pos.x - ring_radius), int(pos.y - ring_radius)),
            )

    def draw_nebula(self, surface: pygame.Surface, pos: Vector2, size: int, time: float = 0) -> None:
        """
        Draw a nebula phenomenon.

        Args:
            surface (pygame.Surface): The surface to draw on.
            pos (Vector2): Position to draw.
            size (int): Base size.
            time (float, optional): Current simulation time.
        """
        layers = 1
        for i in range(layers):
            layer_size = int(size * (1.2 + i * 0.4))
            glow_surface = pygame.Surface(
                (layer_size * 2, layer_size * 2), pygame.SRCALPHA
            )

            for _ in range(int(layer_size * 2 * layer_size * 0.1)):
                x = random.randint(0, layer_size * 2 - 1)
                y = random.randint(0, layer_size * 2 - 1)
                glow_surface.set_at((x, y), (*self.color, random.randint(10, 40)))

            layer_color = (*self.color, max(25, 30 - i * 5))
            pygame.draw.circle(glow_surface, layer_color, (layer_size, layer_size), layer_size)
            surface.blit(
                glow_surface,
                (int(pos.x) - layer_size, int(pos.y) - layer_size),
            )

    def draw_galaxy(self, surface: pygame.Surface, pos: Vector2, size: int, time: float) -> None:
        """
        Draw a galaxy phenomenon.

        Args:
            surface (pygame.Surface): The surface to draw on.
            pos (Vector2): Position to draw.
            size (int): Base size.
            time (float): Current simulation time.
        """
        arms = 8
        arm_length = 44
        core_size = max(3, int(self.base_size / 10))
        pulse = 0.5 * math.sin(time * EFFECT_CONFIG["galaxy"]["twinkle_speed"]) + 0.95

        for arm in range(arms):
            angle_offset = arm * (2 * math.pi / arms)
            for i in range(arm_length):
                progress = i / arm_length
                radius = progress * (self.get_draw_size() / 10) * 1.5
                angle = angle_offset + progress * 4
                angle += random.uniform(-0.00005, 0.00005)
                radius += random.uniform(-1, 1)
                rotated_angle = angle + self.orientation_angle
                x = pos.x + radius * math.cos(rotated_angle) * self.squash_x
                y = pos.y + radius * math.sin(rotated_angle) * self.squash_y
                star_size = max(
                    1,
                    int(
                        (self.get_draw_size() / 10)
                        / 30
                        * (1 - progress)
                    ),
                )
                arm_color = (
                    min(255, int(self.color[0] * pulse)),
                    min(255, int(self.color[1] * pulse)),
                    min(255, int(self.color[2] * pulse)),
                    max(50, int(255 * (1 - progress))),
                )
                pygame.draw.circle(surface, arm_color, (int(x), int(y)), star_size)

    def draw_default(self, surface: pygame.Surface, pos: Vector2, size: int, time: float) -> None:
        """
        Default draw method for unknown effects.

        Args:
            surface (pygame.Surface): The surface to draw on.
            pos (Vector2): Position to draw.
            size (int): Base size.
            time (float): Current simulation time.
        """
        pygame.draw.circle(surface, self.color, (int(pos.x), int(pos.y)), size)

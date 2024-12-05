# Generative Universe v1.2.2

import pygame
import random
import math
from pygame.math import Vector2
import pygame.freetype  # For better font rendering
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Tuple, Callable, Union, Optional
import textwrap

# Initialize Pygame
pygame.init()

# =========================
# Constants and Configuration
# =========================

# Screen dimensions
WIDTH, HEIGHT = 1920, 1080
fullscreen = True

# Other constants
NUM_STARS = 255
MIN_DEPTH = .00001
MAX_DEPTH = 100.0  # Depth range

# Grid size for spatial partitioning
GRID_SIZE = 100  # Increased grid size for better performance

# Scroll speeds
PLAYER_MOVEMENT_SPEED = 100  # Pixels per second for WASD

# Player movement parameters
MOMENTUM_DECAY = 0.95  # Momentum decay rate per frame
MINIMUM_MOVEMENT = 0.05  # Minimum movement value to prevent complete stop

# Depth change rate
DEPTH_CHANGE_RATE = 1  # Adjust this value to control depth change speed

# Fog configuration
FOG_START = MAX_DEPTH // 4
FOG_END = MAX_DEPTH

# Phenomena configuration
MIN_PHENOMENA_DEPTH = MAX_DEPTH * .8  # Ensures all phenomena are at deep depths
MAX_GALAXIES = 8  # Increased limit for galaxies

# Effect configurations
EFFECT_CONFIG = {
    "pulsing": {
        "twinkle_speed": 5,
        "glow_strength": 8,
    },
    "radiating": {
        "glow_strength": 50,
    },
    "swirling": {
        "glow_strength": 0,
    },
    "nebula": {
        "glow_strength": 30,
    },
    "galaxy": {
        "twinkle_speed": 0.1,
        "glow_strength": 8,
    },
}

# Define star types with names, colors, and base sizes
STAR_TYPES = [
    {"name": "Red Dwarf", "color": (255, 80, 80), "base_size": 2},
    {"name": "Yellow Dwarf", "color": (255, 255, 180), "base_size": 3},
    {"name": "White Dwarf", "color": (255, 255, 255), "base_size": 1},
    {"name": "Blue Giant", "color": (140, 180, 255), "base_size": 5},
    {"name": "Red Giant", "color": (255, 100, 100), "base_size": 5},
    {"name": "Supergiant", "color": (255, 150, 150), "base_size": 7},
    {"name": "Neutron Star", "color": (180, 180, 255), "base_size": 1},
]

# Define phenomena types with names, colors, and sizes
PHENOMENA_TYPES = [
    {"name": "Pulsar", "color": (150, 150, 255), "base_size": 10, "effect": "pulsing"},
    {"name": "Quasar", "color": (255, 150, 255), "base_size": 12, "effect": "radiating"},
    {"name": "Wormhole", "color": (100, 100, 255), "base_size": 16, "effect": "swirling"},
    {"name": "Nebula", "color": (255, 100, 100), "base_size": 50, "effect": "nebula"},
    {
        "name": "Galaxy",
        "color": lambda: (
            random.randint(100, 200),
            random.randint(100, 200),
            random.randint(200, 255),
        ),
        "base_size": 80,
        "effect": "galaxy",
    },
]

# Scaling factors
SCALE_EXPONENT = 0.33
SCALE_FACTOR = 2

# =========================
# Utility Functions
# =========================

def wrap_depth(depth: float) -> float:
    """
    Wrap depth within [MIN_DEPTH, MAX_DEPTH] toroidally.

    Args:
        depth (float): The current depth.

    Returns:
        float: Wrapped depth.
    """
    depth_range = MAX_DEPTH - MIN_DEPTH
    return MIN_DEPTH + (depth - MIN_DEPTH) % depth_range


def calculate_scale(
    base_size: float,
    depth: float,
    position: Optional[Vector2] = None,
    center: Optional[Vector2] = None,
    exponent: float = SCALE_EXPONENT,
    scale_factor: float = SCALE_FACTOR,
) -> int:
    """
    Calculate the scaled size of an object based on its depth and distance from the center.

    Args:
        base_size (float): The base size of the object.
        depth (float): The depth of the object.
        position (Vector2, optional): The position of the object.
        center (Vector2, optional): The center of the screen.
        exponent (float): Controls scaling sensitivity.
        scale_factor (float): Multiplier to adjust the overall size.

    Returns:
        int: The scaled size, ensuring a minimum size of 1.
    """
    perspective = (1 / depth) ** exponent
    scaled_size = base_size * perspective * scale_factor

    if position is not None and center is not None:
        distance_from_center = position.distance_to(center)
        max_distance = math.sqrt((WIDTH / 2) ** 2 + (HEIGHT / 2) ** 2)
        distance_scale = max(0.5, 1.5 - (distance_from_center / max_distance))
    else:
        distance_scale = 1.0

    final_scaled_size = int(scaled_size * distance_scale)
    return max(1, final_scaled_size)


def calculate_fog_alpha(depth: float) -> int:
    """
    Calculate the fog alpha based on the depth.

    Args:
        depth (float): The depth of the object.

    Returns:
        int: Fog alpha value.
    """
    if FOG_END == FOG_START:
        return 255 if depth >= FOG_END else 0
    fog_alpha = 255 - int(255 * (depth - FOG_START) / (FOG_END - FOG_START))
    return max(0, min(255, fog_alpha))  # Ensure alpha stays within bounds


def get_wrapped_positions(
    pos: Vector2, size: int, glow_size: int = 0
) -> List[Vector2]:
    """
    Given a position, size, and optional glow size, return a list of positions
    where the object and its glow should be drawn to account for screen wrapping.

    Args:
        pos (Vector2): The original position of the object.
        size (int): The main size of the object.
        glow_size (int, optional): The maximum radius of the glow effect. Defaults to 0.

    Returns:
        List[Vector2]: Positions where the object should be drawn.
    """
    positions = [pos]
    total_size = size + glow_size

    # Check horizontal wrapping
    if pos.x - total_size < 0:
        positions.append(Vector2(pos.x + WIDTH, pos.y))
    if pos.x + total_size > WIDTH:
        positions.append(Vector2(pos.x - WIDTH, pos.y))

    # Check vertical wrapping
    if pos.y - total_size < 0:
        positions.append(Vector2(pos.x, pos.y + HEIGHT))
    if pos.y + total_size > HEIGHT:
        positions.append(Vector2(pos.x, pos.y - HEIGHT))

    # Check for corner wrapping
    if (pos.x - total_size < 0 and pos.y - total_size < 0):
        positions.append(Vector2(pos.x + WIDTH, pos.y + HEIGHT))
    if (pos.x - total_size < 0 and pos.y + total_size > HEIGHT):
        positions.append(Vector2(pos.x + WIDTH, pos.y - HEIGHT))
    if (pos.x + total_size > WIDTH and pos.y - total_size < 0):
        positions.append(Vector2(pos.x - WIDTH, pos.y + HEIGHT))
    if (pos.x + total_size > WIDTH and pos.y + total_size > HEIGHT):
        positions.append(Vector2(pos.x - WIDTH, pos.y - HEIGHT))

    return positions

# =========================
# Data Classes
# =========================

@dataclass
class OrbitalBody:
    """
    Represents an orbital body, such as a planet or moon, orbiting a parent object.
    """
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
    z: float = field(init=False, default=0.0)  # Initialize `z` to 0.0 by default

    # New attributes for orbital variations
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

        # Initialize orbital variations
        self.precession_rate = random.uniform(-0.001, 0.001)  # Small precession
        self.eccentricity_variation = random.uniform(-0.0005, 0.0005)  # Small eccentricity changes

    @property
    def depth(self) -> float:
        """
        Calculate the depth of the orbital body based on the parent's depth and z position.
        """
        return self.parent.depth + self.z

    def get_scaled_size(self) -> int:
        """
        Calculate the scaled size based on the parent's size and scaling factors.
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
        """
        pos = self.get_position()
        scaled_size = self.get_scaled_size()

        # Calculate opacity based on z-position for transit effect
        opacity = 255
        if self.z > 0:
            opacity = max(50, 255 - int((self.z / MAX_DEPTH) * 200))  # Adjust as needed

        color_with_opacity = (
            self.color[0],
            self.color[1],
            self.color[2],
            opacity
        )

        # Handle screen wrapping
        positions_to_draw = get_wrapped_positions(pos, scaled_size)

        for draw_pos in positions_to_draw:
            # Create a surface with per-pixel alpha for opacity
            body_surface = pygame.Surface((scaled_size * 2, scaled_size * 2), pygame.SRCALPHA)
            pygame.draw.circle(body_surface, color_with_opacity, (scaled_size, scaled_size), scaled_size)
            surface.blit(body_surface, (int(draw_pos.x) - scaled_size, int(draw_pos.y) - scaled_size))

    def get_position(self) -> Vector2:
        """
        Calculate and set the position of the orbital body, including the z-axis for depth.
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

        # Calculate z based on inclination
        self.z = r * math.sin(true_anomaly) * math.sin(self.inclination)  # Calculate `z`

        # Adjust y to simulate inclination
        y *= math.cos(self.inclination)

        # Use parent's depth to adjust position scaling
        depth_factor = max(self.parent.depth, MIN_DEPTH)  # Prevent division by zero
        x /= math.sqrt(depth_factor)
        y /= math.sqrt(depth_factor)

        self.pos = Vector2(self.parent.pos.x + x, self.parent.pos.y + y)
        return self.pos

    def update(self, dt: float) -> None:
        """
        Update the body's anomaly, position, depth, and orbital variations based on the elapsed time.
        """
        # Update orbital anomaly with precession
        self.current_anomaly = (self.current_anomaly + self.orbital_speed * dt + self.precession_rate) % (2 * math.pi)
        
        # Update eccentricity with variation
        self.eccentricity += self.eccentricity_variation
        self.eccentricity = max(0.0, min(self.eccentricity, 0.6))  # Clamp between 0.0 and 0.6

        self.get_position()  # Update position and depth based on the new anomaly

    @staticmethod
    def solve_keplers_equation(M: float, e: float, tolerance: float = 1e-5) -> float:
        """
        Solve Kepler's Equation M = E - e*sin(E) for E given M and e.
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
    """
    Represents a planet orbiting a star.
    """
    moons: List["Moon"] = field(default_factory=list, init=False)

    def __init__(self, parent: "Star"):
        super().__init__(
            parent=parent,
            size_scale=0.02,  # Adjusted based on scaling logic
            semi_major_scale=(4, 24),
            semi_minor_scale=(0.7, 1.3),
            orbital_period_range=(50, 200),
        )
        num_moons = random.randint(0, 2)
        self.moons = [Moon(self) for _ in range(num_moons)]


@dataclass(init=False)
class Moon(OrbitalBody):
    """
    Represents a moon orbiting a planet.
    """

    def __init__(self, parent: Planet):
        super().__init__(
            parent=parent,
            size_scale=0.01,  # Adjusted based on scaling logic
            semi_major_scale=(2, 12),
            semi_minor_scale=(0.7, 1.3),
            orbital_period_range=(20, 50),
        )


@dataclass
class Star:
    """
    Represents a star in the universe.
    """
    # Remove 'id' from the constructor parameters
    id: int = field(init=False)
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

    # New Attribute:
    fixed_depth: bool = field(default=False)  # Determines if depth_change is applied

    # Class variable for unique IDs
    _id_counter: int = field(default=1, init=False, repr=False)

    def __post_init__(self):
        """
        Initialize the star's properties.
        """
        self.id = Star._id_counter
        Star._id_counter += 1

        self.pos = Vector2(self.x, self.y)
        self.type_info = random.choice(STAR_TYPES)
        self.name = self.type_info["name"]
        self.color = self.type_info["color"]
        self.base_size = self.type_info["base_size"]
        self.twinkle_offset = random.uniform(0, 2 * math.pi)
        self.composition = self.generate_composition()

        # Planet Generation Based on `fixed_depth`:
        if not self.fixed_depth and random.random() < 0.3:
            num_planets = random.randint(1, 4)
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
        Update the star's position and depth.

        Args:
            player_velocity (Vector2): The player's current velocity.
            depth_change (float): Change in depth.
            dt (float): Delta time in seconds.
        """
        if not self.fixed_depth:
            # Apply non-linear depth changes using exponential scaling
            self.depth += math.copysign(abs(depth_change) ** 1.2, depth_change)
            self.depth = wrap_depth(self.depth)

        depth_factor = max(self.depth, 1.0)  # Prevent division by zero

        # Update position with parallax effect
        self.pos.x -= (player_velocity.x / depth_factor) * dt * 50
        self.pos.y -= (player_velocity.y / depth_factor) * dt * 50

        # Screen wrapping with toroidal logic
        self.pos.x %= WIDTH
        self.pos.y %= HEIGHT

        # Update planets
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

        # Calculate glow size based on the main size
        glow_radius = int(size * math.pi) if size > 3 else 0

        # Depth fog
        fog_alpha = calculate_fog_alpha(self.depth)

        # Handle dynamic color if color is callable
        dynamic_color = self.color() if callable(self.color) else self.color
        current_color = (
            min(255, int(dynamic_color[0])),
            min(255, int(dynamic_color[1])),
            min(255, int(dynamic_color[2])),
            fog_alpha,  # Apply fog alpha
        )

        # Make the colors more vibrant for a video game style
        vibrant_color = (
            min(255, int(current_color[0] * 1.1)),
            min(255, int(current_color[1] * 1.1)),
            min(255, int(current_color[2] * 1.1)),
            current_color[3],
        )

        # Get all necessary positions for wrapping, including glow size
        positions_to_draw = get_wrapped_positions(self.pos, size, glow_size=glow_radius)

        for draw_pos in positions_to_draw:
            # Draw the star
            pygame.draw.circle(surface, vibrant_color, (int(draw_pos.x), int(draw_pos.y)), size)

            # Glow effect for larger stars
            if glow_radius > 0:
                glow_surface = pygame.Surface(
                    (glow_radius * 2, glow_radius * 2), pygame.SRCALPHA
                )
                glow_color = (vibrant_color[0], vibrant_color[1], vibrant_color[2], EFFECT_CONFIG.get("pulsing", {}).get("glow_strength", 8))
                pygame.draw.circle(
                    glow_surface, glow_color, (glow_radius, glow_radius), glow_radius
                )
                surface.blit(
                    glow_surface,
                    (int(draw_pos.x) - glow_radius, int(draw_pos.y) - glow_radius),
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
    """
    Represents a celestial phenomenon in the universe.
    """
    type_info: dict = field(init=False)
    effect: str = field(init=False)
    twinkle_speed: float = field(init=False)
    squash_x: float = field(init=False, default=0.3)
    squash_y: float = field(init=False, default=0.7)
    orientation_angle: float = field(init=False, default=0.314)

    # New attribute for rendering strategy
    render_strategy: Callable = field(init=False)

    def __init__(self, x: float, y: float, depth: Optional[float] = None):
        """
        Initialize a Phenomenon object.

        Args:
            x (float): X-coordinate.
            y (float): Y-coordinate.
            depth (float, optional): Depth value. Ignored to enforce deep depth.
        """
        # Set `fixed_depth` to True
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

        # Set twinkle speed based on the phenomenon type
        self.twinkle_speed = EFFECT_CONFIG.get(self.effect, {}).get("twinkle_speed", 1)

        # Enforce Deep Depths:
        if self.name == "Galaxy":
            # For galaxies, set depth within a specific deep range
            self.depth = random.uniform(MAX_DEPTH * 0.5, MAX_DEPTH)
            self.squash_x = random.uniform(0.6, 1.4)
            self.squash_y = random.uniform(0.6, 1.4)
            self.orientation_angle = random.uniform(0, 2 * math.pi)
        else:
            # For other phenomena, ensure depth is at least MIN_PHENOMENA_DEPTH
            self.depth = random.uniform(MIN_PHENOMENA_DEPTH, MAX_DEPTH)

        # Assign rendering strategy based on effect
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
            depth_change (float): Change in depth. Ignored.
            dt (float): Delta time in seconds.
        """
        # Ignore `depth_change` by not calling the superclass update method
        depth_factor = max(self.depth, 1.0)
        self.pos.x -= (player_velocity.x / depth_factor) * dt * 50
        self.pos.y -= (player_velocity.y / depth_factor) * dt * 50

        # Screen wrapping
        self.pos.x %= WIDTH
        self.pos.y %= HEIGHT

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

        # Determine glow size based on effect
        glow_radius = (
            int(size * 2)
            if self.effect == "galaxy"
            else (int(size * 1.5) if self.effect == "nebula" else 0)
        )

        # Get all necessary positions for wrapping, including glow size
        positions_to_draw = get_wrapped_positions(self.pos, size, glow_size=glow_radius)

        for draw_pos in positions_to_draw:
            # Use the rendering strategy
            self.render_strategy(surface, draw_pos, size, time)

            # Draw glow effects if applicable
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
            time (float, optional): Current simulation time. Defaults to 0.
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
            time (float, optional): Current simulation time. Defaults to 0.
        """
        layers = 1
        for i in range(layers):
            layer_size = int(size * (1.2 + i * 0.4))
            glow_surface = pygame.Surface(
                (layer_size * 2, layer_size * 2), pygame.SRCALPHA
            )

            # Add grainy noise
            for _ in range(int(layer_size * 2 * layer_size * 0.1)):
                x = random.randint(0, layer_size * 2 - 1)
                y = random.randint(0, layer_size * 2 - 1)
                glow_surface.set_at((x, y), (*self.color, random.randint(10, 40)))

            # More transparent glow
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
        arms = 8  # Number of spiral arms
        arm_length = 44  # Number of stars per arm
        core_size = max(3, int(self.base_size / 10))  # Core size of the galaxy

        # Slow pulsing effect for the galaxy core and arms
        pulse = 0.5 * math.sin(time * EFFECT_CONFIG["galaxy"]["twinkle_speed"]) + 0.95  # Slow pulse modifier

        # Draw spiral arms
        for arm in range(arms):
            angle_offset = arm * (2 * math.pi / arms)  # Fixed offset for each arm
            for i in range(arm_length):
                progress = i / arm_length  # Star's position along the arm
                radius = progress * (self.get_draw_size() / 10) * 1.5  # Distance from core
                angle = angle_offset + progress * 4  # Fixed spiral curve

                # Add slight controlled randomness for natural appearance
                angle += random.uniform(-0.00005, 0.00005)
                radius += random.uniform(-1, 1)

                # Apply squash and rotation
                rotated_angle = angle + self.orientation_angle
                x = pos.x + radius * math.cos(rotated_angle) * self.squash_x
                y = pos.y + radius * math.sin(rotated_angle) * self.squash_y

                # Dynamic star size and fading
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
                    max(50, int(255 * (1 - progress))),  # Fade stars toward the outer edges
                )

                # Draw each star in the arm
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


# =========================
# Player Class
# =========================

@dataclass
class Player:
    """
    Represents the player in the universe.
    """
    depth: float = 1.0  # Starting depth
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

        if keys[pygame.K_w]:  # Move up
            self.acceleration.y = -PLAYER_MOVEMENT_SPEED
        if keys[pygame.K_s]:  # Move down
            self.acceleration.y = PLAYER_MOVEMENT_SPEED
        if keys[pygame.K_a]:  # Move left
            self.acceleration.x = -PLAYER_MOVEMENT_SPEED
        if keys[pygame.K_d]:  # Move right
            self.acceleration.x = PLAYER_MOVEMENT_SPEED
        if keys[pygame.K_q]:  # Move out (increase depth)
            depth_change += DEPTH_CHANGE_RATE * dt
        if keys[pygame.K_e]:  # Move in (decrease depth)
            depth_change -= DEPTH_CHANGE_RATE * dt

        # Apply acceleration to velocity
        self.velocity += self.acceleration * dt

        # Apply momentum decay
        self.velocity *= MOMENTUM_DECAY

        # Apply non-linear scaling to velocity for smoother movement
        if self.velocity.length() > 0:
            self.velocity.scale_to_length(max(self.velocity.length(), MINIMUM_MOVEMENT))

        # Update player depth with clamping
        self.depth += math.copysign(abs(depth_change) ** 1.2, depth_change)  # Non-linear depth change
        self.depth = max(MIN_DEPTH, min(MAX_DEPTH, self.depth))  # Clamp depth

        return depth_change  # Return depth change for updating stars


# =========================
# Spatial Partitioning
# =========================

class SpatialGrid:
    """
    Simple grid-based spatial partitioning for efficient culling.
    """

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


# =========================
# Game Class
# =========================

class Game:
    """
    The main game class that manages the game loop, rendering, and updates.
    """

    def __init__(self):
        """
        Initialize the game.
        """
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT)) if not fullscreen else pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN)
        pygame.display.set_caption("Generative Universe v1.2.2")
        self.clock = pygame.time.Clock()
        self.FPS = 60
        self.running = True

        # Initialize player
        self.player = Player()
        self.simulation_time = 0.0  # Initialize simulation time

        # Initialize stars and phenomena
        self.stars: List[Star] = []
        galaxy_count = 0  # Counter for galaxies

        for _ in range(NUM_STARS):
            x = random.uniform(0, WIDTH)
            y = random.uniform(0, HEIGHT)

            # Decide whether to spawn a phenomenon or a star
            if random.random() < 0.1:  # Increased chance for phenomena
                phenomenon = Phenomenon(x=x, y=y)  # No depth passed
                if phenomenon.name == "Galaxy" and galaxy_count < MAX_GALAXIES:
                    self.stars.append(phenomenon)
                    galaxy_count += 1
                elif phenomenon.name != "Galaxy":
                    self.stars.append(phenomenon)
            else:
                # Regular star with depth in [MIN_DEPTH, MAX_DEPTH]
                depth = random.uniform(MIN_DEPTH, MAX_DEPTH)
                star = Star(x=x, y=y, depth=depth)
                self.stars.append(star)

        # Initialize spatial grid
        self.spatial_grid = SpatialGrid(WIDTH, HEIGHT, GRID_SIZE)

        # Font for tooltips
        self.font = pygame.freetype.SysFont("Arial", 16)

        # =========================
        # Panning Variables
        # =========================
        self.is_panning = False          # Flag to indicate if panning is active
        self.pan_start_pos = Vector2(0, 0)  # Starting mouse position for panning
        self.pan_last_pos = Vector2(0, 0)   # Last mouse position during panning

    def update_spatial_grid(self) -> None:
        """
        Update the spatial grid with current star positions.
        """
        self.spatial_grid.clear()
        for star in self.stars:
            self.spatial_grid.add_star(star)

    def update_and_draw_stars(
        self, sim_time: float, depth_change: float, dt: float
    ) -> None:
        """
        Update and draw all objects sorted by depth, ensuring correct layering.

        Args:
            sim_time (float): Current simulation time.
            depth_change (float): Change in depth.
            dt (float): Delta time in seconds.
        """
        # Define visible area with a margin
        margin = GRID_SIZE * 2
        visible_rect = pygame.Rect(-margin, -margin, WIDTH + margin * 2, HEIGHT + margin * 2)

        # Update spatial grid
        self.update_spatial_grid()

        # Retrieve visible stars using spatial partitioning
        visible_stars = self.spatial_grid.get_visible_stars(visible_rect)

        # Flatten all objects (stars, planets, moons) into a single list with their depth
        drawable_objects = []
        for star in visible_stars:
            drawable_objects.append((star.depth, star))  # Add the star itself
            for planet in star.planets:
                drawable_objects.append((planet.depth, planet))  # Add the planet
                for moon in planet.moons:
                    drawable_objects.append((moon.depth, moon))  # Add the moon

        # Sort all drawable objects by depth in descending order
        drawable_objects.sort(key=lambda obj: obj[0], reverse=True)

        # Draw all objects in sorted order
        for _, obj in drawable_objects:
            if isinstance(obj, Star):
                obj.update(self.player.velocity, depth_change, dt)
                obj.draw(surface=self.screen, time=sim_time)
            elif isinstance(obj, OrbitalBody):
                obj.update(dt)
                obj.draw(surface=self.screen, time=sim_time)  # Pass sim_time if needed

    def run(self) -> None:
        """
        The main game loop.
        """
        while self.running:
            dt = self.clock.tick(self.FPS) / 1000  # Delta time in seconds
            time_elapsed = pygame.time.get_ticks() / 1000.0  # Current time in seconds
            self.simulation_time += dt  # Increment simulation time

            # Event handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

                # Handle Mouse Button Down for panning
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left mouse button
                        self.is_panning = True
                        self.pan_start_pos = Vector2(event.pos)
                        self.pan_last_pos = Vector2(event.pos)

                # Handle Mouse Button Up to stop panning
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:  # Left mouse button
                        self.is_panning = False

                # Handle Mouse Motion for panning
                elif event.type == pygame.MOUSEMOTION:
                    if self.is_panning:
                        current_mouse_pos = Vector2(event.pos)
                        delta = current_mouse_pos - self.pan_last_pos
                        self.pan_last_pos = current_mouse_pos

                        # Update player's velocity based on mouse movement
                        # Invert delta to move the view in the opposite direction of the mouse drag
                        self.player.velocity -= delta * 5  # The multiplier adjusts panning speed

            # Handle player input (keyboard)
            depth_change = self.player.handle_input(dt)

            # Fill the background
            self.screen.fill((5, 11, 22))  # Dark background to represent space

            # Update and draw stars with culling, passing depth_change
            self.update_and_draw_stars(self.simulation_time, depth_change, dt)

            # Check for hovered star
            mouse_pos = Vector2(pygame.mouse.get_pos())
            hovered_star = None
            for star in self.stars:
                if star.is_hovered(mouse_pos):
                    hovered_star = star
                    break

            # Draw tooltip if a star is hovered
            if hovered_star:
                self.draw_tooltip(hovered_star, mouse_pos)

            # Update display
            pygame.display.flip()

        pygame.quit()

    def draw_tooltip(self, star: Star, mouse_pos: Vector2) -> None:
        """
        Draw a tooltip near the mouse cursor with star information.

        Args:
            star (Star): The star being hovered.
            mouse_pos (Vector2): Current mouse position.
        """
        def wrap_text(text: str, max_width: int) -> list:
            """
            Wrap text for longer lines.

            Args:
                text (str): The text to wrap.
                max_width (int): Maximum width of a line.

            Returns:
                list: Wrapped lines.
            """
            wrapper = textwrap.TextWrapper(width=30)  # Adjust width as needed
            return wrapper.wrap(text)

        # Prepare tooltip text
        if isinstance(star, Phenomenon):
            title = f"Phenomenon ID: #{star.id}"
            type_info = f"Type: {star.name}"
            effect_info = f"Effect: {star.effect}"
        else:
            title = f"Star ID: #{star.id}"
            type_info = f"Type: {star.name}"
            effect_info = "Effect: None"

        # Wrapping the composition text to fit in smaller tooltips
        composition_lines = []
        for element, percentage in star.composition.items():
            line = f"{element}: {percentage:.2f}%"
            composition_lines.extend(wrap_text(line, 250))  # Assuming 250 as max width

        tooltip_lines = [
            title,
            type_info,
            effect_info,
            f"Color: {star.color}",
            "Composition:",
        ] + composition_lines

        # Render text surfaces
        text_surfaces = []
        for i, line in enumerate(tooltip_lines):
            if i == 0:  # Make the title bold
                surface, _ = self.font.render(
                    line, fgcolor=(255, 255, 255), style=pygame.freetype.STYLE_STRONG
                )
            else:
                surface, _ = self.font.render(line, fgcolor=(255, 255, 255))
            text_surfaces.append(surface)

        line_height = self.font.get_sized_height() + 4  # More spacing between lines
        padding = 10

        # Determine tooltip size
        width = max(surface.get_width() for surface in text_surfaces) + padding * 2
        height = len(text_surfaces) * line_height + padding * 2

        # Tooltip position
        tooltip_x = mouse_pos.x + 15
        tooltip_y = mouse_pos.y + 15

        # Ensure tooltip doesn't go off-screen
        if tooltip_x + width > WIDTH:
            tooltip_x = mouse_pos.x - width - 15
        if tooltip_y + height > HEIGHT:
            tooltip_y = mouse_pos.y - height - 15

        # Draw semi-transparent background with rounded corners
        tooltip_surface = pygame.Surface((width, height), pygame.SRCALPHA)
        pygame.draw.rect(
            tooltip_surface, (0, 0, 0, 200), (0, 0, width, height), border_radius=8
        )
        self.screen.blit(tooltip_surface, (tooltip_x, tooltip_y))

        # Blit text
        current_y = tooltip_y + padding
        for surface in text_surfaces:
            self.screen.blit(surface, (tooltip_x + padding, current_y))
            current_y += line_height


# =========================
# Main Execution
# =========================

if __name__ == "__main__":
    game = Game()
    game.run()

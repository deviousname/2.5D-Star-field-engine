from constants import *
from celestial_objects import *

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
        position (Optional[Vector2]): The position of the object.
        center (Optional[Vector2]): The center point for distance calculation.
        exponent (float): The exponent for scaling.
        scale_factor (float): The scaling factor.

    Returns:
        int: The final scaled size.
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
        depth (float): The depth value.

    Returns:
        int: The fog alpha value.
    """
    if FOG_END == FOG_START:
        return 255 if depth >= FOG_END else 0
    fog_alpha = 255 - int(255 * (depth - FOG_START) / (FOG_END - FOG_START))
    return max(0, min(255, fog_alpha))


def get_wrapped_positions(
    pos: Vector2, size: int, glow_size: int = 0
) -> List[Vector2]:
    """
    Given a position, size, and optional glow size, return a list of positions
    where the object and its glow should be drawn to account for screen wrapping.

    Args:
        pos (Vector2): The original position.
        size (int): The size of the object.
        glow_size (int, optional): The size of the glow.

    Returns:
        List[Vector2]: List of wrapped positions.
    """
    positions = [pos]
    total_size = size + glow_size

    if pos.x - total_size < 0:
        positions.append(Vector2(pos.x + WIDTH, HEIGHT - pos.y))
    if pos.x + total_size > WIDTH:
        positions.append(Vector2(pos.x - WIDTH, HEIGHT - pos.y))
    if pos.y - total_size < 0:
        positions.append(Vector2(WIDTH - pos.x, pos.y + HEIGHT))
    if pos.y + total_size > HEIGHT:
        positions.append(Vector2(WIDTH - pos.x, pos.y - HEIGHT))

    if (pos.x - total_size < 0 and pos.y - total_size < 0):
        positions.append(Vector2(pos.x + WIDTH, HEIGHT - pos.y + HEIGHT))
    if (pos.x - total_size < 0 and pos.y + total_size > HEIGHT):
        positions.append(Vector2(pos.x + WIDTH, HEIGHT - pos.y - HEIGHT))
    if (pos.x + total_size > WIDTH and pos.y - total_size < 0):
        positions.append(Vector2(pos.x - WIDTH, HEIGHT - pos.y + HEIGHT))
    if (pos.x + total_size > WIDTH and pos.y + total_size > HEIGHT):
        positions.append(Vector2(pos.x - WIDTH, HEIGHT - pos.y - HEIGHT))

    return positions

def wrap_depth(depth: float) -> float:
    """
    Wrap depth within [MIN_DEPTH, MAX_DEPTH] toroidally.

    Args:
        depth (float): The depth value.

    Returns:
        float: Wrapped depth value.
    """
    depth_range = MAX_DEPTH - MIN_DEPTH
    return MIN_DEPTH + (depth - MIN_DEPTH) % depth_range

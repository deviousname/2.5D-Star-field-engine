WIDTH, HEIGHT = 1920, 1080
import random

fullscreen = False
NUM_STARS = 100
GRID_SIZE = 100
PLAYER_MOVEMENT_SPEED = 100
MOMENTUM_DECAY = 0.95
MINIMUM_MOVEMENT = 0.05
DEPTH_CHANGE_RATE = 1
MIN_DEPTH = 0.1
MAX_DEPTH = 100.0
FOG_START = MAX_DEPTH / 4
FOG_END = MAX_DEPTH
MIN_PHENOMENA_DEPTH = MAX_DEPTH * 1
MAX_GALAXIES = 8
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
starscale = 2
STAR_TYPES = [
    {"name": "Red Dwarf", "color": (255, 80, 80), "base_size": 2 * starscale},
    {"name": "Yellow Dwarf", "color": (255, 255, 180), "base_size": 3 * starscale},
    {"name": "White Dwarf", "color": (255, 255, 255), "base_size": 1 * starscale},
    {"name": "Blue Giant", "color": (140, 180, 255), "base_size": 5 * starscale},
    {"name": "Red Giant", "color": (255, 100, 100), "base_size": 5 * starscale},
    {"name": "Supergiant", "color": (255, 150, 150), "base_size": 7 * starscale},
    {"name": "Neutron Star", "color": (180, 180, 255), "base_size": 1 * starscale},
]
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
SCALE_EXPONENT = 0.33
SCALE_FACTOR = 2
TARGET_COLOR = (255, 0, 0)

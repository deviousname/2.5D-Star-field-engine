import pygame

PIXEL_COLORS = {
    1: (220, 220, 220),
    2: (180, 180, 180),
    3: (140, 140, 140),
    4: (100, 100, 100),
    5: (57, 255, 20),
    6: (0, 255, 255),
    7: (255, 20, 147)
}

PIXEL_SIZE = 5

def draw_spaceship(surface, matrix, position):
    """
    Draw the spaceship on the given surface based on the provided matrix and position.

    Args:
        surface (pygame.Surface): The surface to draw the spaceship on.
        matrix (list of list of int): The matrix representing the spaceship shape.
        position (tuple of int): The (x, y) position to draw the spaceship.
    """
    x, y = position
    for row_index, row in enumerate(matrix):
        for col_index, pixel in enumerate(row):
            if pixel in PIXEL_COLORS:
                color = PIXEL_COLORS[pixel]
                pygame.draw.rect(
                    surface,
                    color,
                    (
                        x + col_index * PIXEL_SIZE,
                        y + row_index * PIXEL_SIZE,
                        PIXEL_SIZE,
                        PIXEL_SIZE,
                    ),
                )


RAW_SPACESHIP_SHAPES = {
    "up_outward": """
    0002000
    0022200
    0223220
    2255222
    0022200
    0002000
    """,
    "down_outward": """
    0002000
    0022200
    0223220
    2255222
    0022200
    0002000
    """,
    "left_outward": """
    0000200
    0002220
    0022522
    0223220
    0002220
    0000200
    """,
    "right_outward": """
    0020000
    0222000
    2252200
    0223220
    0222000
    0020000
    """,
    "up-right_outward": """
    000220
    002222
    022522
    225522
    002220
    000220
    """,
    "up-left_outward": """
    022000
    222220
    225520
    225522
    222220
    022000
    """,
    "down-right_outward": """
    000220
    002222
    022522
    225522
    002220
    000220
    """,
    "down-left_outward": """
    220000
    222220
    225520
    225522
    222220
    022000
    """,
    "up_inward": """
    0004000
    0044400
    0445540
    0044400
    0004000
    0000000
    """,
    "down_inward": """
    0000000
    0004000
    0044400
    0445540
    0044400
    0004000
    """,
    "left_inward": """
    0000000
    0000400
    0004554
    0045554
    0004554
    0000400
    """,
    "right_inward": """
    0000000
    0040000
    4555400
    4555540
    4555400
    0040000
    """,
    "up-right_inward": """
    000040
    000440
    004454
    004454
    000440
    000040
    """,
    "up-left_inward": """
    040000
    044400
    044554
    044554
    044400
    040000
    """,
    "down-right_inward": """
    000040
    000440
    004554
    004454
    000440
    000000
    """,
    "down-left_inward": """
    040000
    044400
    455540
    045454
    044400
    000000
    """,
    "up": """
    0002000
    0022200
    0223220
    0225220
    0022200
    0002000
    """,
    "down": """
    0002000
    0022200
    0225220
    0223220
    0022200
    0002000
    """,
    "left": """
    0000200
    0002220
    0022522
    0223220
    0002220
    0000200
    """,
    "right": """
    0020000
    0222000
    2252200
    0223220
    0222000
    0020000
    """,
    "up-right": """
    000220
    002222
    022522
    022522
    002220
    000220
    """,
    "up-left": """
    022000
    222220
    225520
    225520
    222220
    022000
    """,
    "down-right": """
    000220
    002222
    022522
    022522
    002220
    000220
    """,
    "down-left": """
    220000
    222220
    225520
    225520
    222220
    022000
    """,
    "up-right_outward": """
    000330
    003333
    033533
    335533
    003333
    000330
    """,
    "up-left_outward": """
    033000
    333330
    335530
    335533
    333330
    033000
    """,
    "down-right_outward": """
    000330
    003333
    033533
    335533
    003333
    000330
    """,
    "down-left_outward": """
    330000
    333330
    335530
    335533
    333330
    033000
    """,
    "up-right_inward": """
    000440
    004444
    044554
    445554
    004444
    000440
    """,
    "up-left_inward": """
    440000
    444440
    445540
    445554
    444440
    440000
    """,
    "down-right_inward": """
    000440
    004444
    044554
    445554
    004444
    000440
    """,
    "down-left_inward": """
    440000
    444440
    445540
    445554
    444440
    440000
    """
}

SPACESHIP_SHAPES = {
    direction: [
        [int(char) for char in row.strip()]
        for row in shape.strip().split("\n")
    ]
    for direction, shape in RAW_SPACESHIP_SHAPES.items()
}

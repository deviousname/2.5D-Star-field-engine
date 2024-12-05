# Generative Universe v1.2.2

Generative Universe is a procedural simulation of a dynamic and expansive 2D universe with interactive celestial objects. Explore an endless space populated by stars, planets, moons, and mysterious phenomena, all rendered with vibrant visuals and smooth animations.

## Features

- **Procedural Celestial Objects**: 
  - Stars of different types (e.g., Red Dwarfs, Blue Giants) with unique characteristics.
  - Planets and moons orbiting stars dynamically.
  - Phenomena like Pulsars, Quasars, Galaxies, and Nebulae with distinct effects.

- **Interactive Exploration**: 
  - Move through the universe with WASD controls and explore varying depths using Q/E.
  - Panning feature for intuitive drag-based exploration.

- **Dynamic Rendering**: 
  - Depth scaling for a realistic 3D-like parallax effect.
  - Fog effects for objects at extreme depths.
  - Vibrant glow and animation effects for stars and phenomena.

- **Spatial Partitioning**: 
  - Efficient culling using a spatial grid for smooth performance, even with hundreds of objects.

- **Tooltips**: 
  - Hover over objects to see detailed information about their type, composition, and effects.

- **Customizable Gameplay**: 
  - Adjust constants like movement speed, number of stars, and depth ranges to tailor the experience.

---

## Controls

- **W/A/S/D**: Move up, left, down, or right.
- **Q/E**: Change depth (zoom in/out).
- **Mouse Drag**: Pan the universe view.
- **Hover**: View detailed information about stars and phenomena.

---

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/generative-universe.git
   cd generative-universe
   ```

2. **Install Dependencies**:
   Make sure you have Python installed, then install the required libraries:
   ```bash
   pip install pygame
   ```

3. **Run the Simulation**:
   Execute the main Python script:
   ```bash
   python generative_universe.py
   ```

---

## Configuration

Customize the simulation in the `Generative Universe v1.2.2` script by modifying constants:
- **`NUM_STARS`**: Adjust the number of stars in the universe.
- **`MAX_GALAXIES`**: Set the limit for galaxies in the simulation.
- **`FOG_START` and `FOG_END`**: Control the fog effect's start and end depths.
- **`PLAYER_MOVEMENT_SPEED`**: Change the player's navigation speed.

---

## Requirements

- **Python 3.7+**
- **Pygame 2.0+**

---

## Contributing

1. Fork the repository.
2. Create a feature branch:
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add feature"
   ```
4. Push to the branch:
   ```bash
   git push origin feature-name
   ```
5. Open a pull request.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Screenshots

Coming Soon!

---

## Acknowledgments

Special thanks to:
- The **Pygame** community for their excellent game development framework.
- Space and science enthusiasts for inspiring the development of this simulation. 

Enjoy your journey through the stars! 🌌✨

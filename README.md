# Satellite Beam Projection Calculation

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

Accurately calculate the coverage area of satellite beams on the Earth's surface. Supports any sub-satellite point position and beam pointing, outputs latitude/longitude coordinates of beam boundaries and coverage area. Unlike traditional methods that approximate beams as standard circles, this project calculates the **true coverage area** of beams.

## Roadmap

- ✅ Accurate plotting of single beam
- ⏲️ Multiple beam arrangement planning (in progress)
- ......

## ✨ Key Features

- **Accurate coordinate transformation**: Directly constructs orthogonal rotation matrix, ensures axes alignment with geographic north/east directions, solves rotation errors and azimuth offset problems in traditional methods
- **Flexible input**: Supports both offset angles (polar angle + azimuth) and direct input of beam center latitude/longitude
- **Accurate area calculation**: Uses Gauss-Bonnet theorem to calculate spherical polygon area
- **Visualization**: Provides 2D geographic projection
- **General compatibility**: Supports any orbital altitude, not limited to geosynchronous orbit

## 🔧 Implementation

**Coordinate System Design**: Directly constructs orthogonal rotation matrix to ensure:
- Local x-axis → **Geographic North** at the sub-satellite point tangent plane
- Local y-axis → **Geographic East** at the sub-satellite point tangent plane
- Local z-axis → Direction from Earth center to sub-satellite point

**Computation Flow**:
1. Find intersection points between beam cone and Earth sphere in local coordinates
2. Transform to global geocentric coordinate system via rotation
3. Convert to geographic latitude/longitude coordinates
4. Calculate coverage area using Gauss-Bonnet theorem

## 📦 Dependencies

```bash
pip install numpy matplotlib
```

## 🚀 Quick Start

### Interactive Mode

```bash
python beam_projection.py
```

Follow the prompts to input: satellite orbital altitude, sub-satellite point latitude/longitude, beam half-angle, and select beam center input method.

### Module Usage

```python
from beam_projection import calculate_beam_boundary, calculate_beam_area, H_GEO

# Calculate beam boundary (input beam center coordinates)
boundary = calculate_beam_boundary(
    satellite_height=H_GEO,
    subpoint_lat=39.9,
    subpoint_lon=116.4,
    beam_half_angle=2.0,
    beam_center_lat=39.9,
    beam_center_lon=116.4
)

# Calculate coverage area
area = calculate_beam_area(boundary)
print(f"Coverage area: {area:.0f} km²")
```

`boundary` is a numpy array of shape `(n_points, 2)`, each row stores `(lat, lon)` of a boundary point.

## 📊 Examples

| Example | Parameters | Result |
|---------|------------|--------|
| Centered Beam | GEO orbit, subpoint Beijing (39.9°N, 116.4°E), half-angle 2.0° | Coverage: **4,985,051 km²**, range: 28.6°N~51.3°N, 101.6°E~131.2°E |
| North-offset Beam | GEO orbit, subpoint Beijing, beam center 44.9°N, half-angle 1.0° | Coverage: **1,238,444 km²**, center error < 0.1° |

## 🔧 Bug Fixes

Fixed two critical errors in traditional implementations:

1. **Rodrigues rotation formula coefficient error**: Lost sin(θ) coefficient after normalizing rotation axis, causing incorrect rotation matrix
2. **Misaligned axes**: Traditional methods only ensure z-axis points to sub-satellite, leaving x/y axes misaligned with geographic directions, causing azimuth error

**Solution**: Abandon Rodrigues rotation, directly construct orthogonal rotation matrix with columns corresponding to north, east, and sub-satellite directions. Coordinate system is perfectly aligned, calculation results are accurate.

## 📄 License

MIT License

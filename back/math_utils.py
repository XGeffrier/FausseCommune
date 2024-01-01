import numpy as np
from geopy.distance import geodesic

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import shapely.geometry


def coords_dist(coords_a: tuple[float, float], coords_b: tuple[float, float]) -> float:
    """
    Compute geodesic distance between (latitude, longitude) coordinates.
    """
    return geodesic(coords_a, coords_b).km


def generate_grid_coords(mask_shape: "shapely.geometry.shape", size_grid: int = 50) -> list[tuple[float, float]]:
    """
    Return (latitude, longitude) coords of points evenly distributed in France.
    """
    from shapely.geometry import Point

    min_long, min_lat, max_long, max_lat = mask_shape.bounds
    lats = np.linspace(min_lat, max_lat, size_grid)
    longs = np.linspace(min_long, max_long, size_grid)
    lats, longs = np.meshgrid(lats, longs)
    grid_coords = np.array([lats.flatten(), longs.flatten()]).T
    # we reverse coords to test them in France: (lat, long) is the norm, but it is equivalent to (y, x)
    french_coords = [tuple(c) for c in grid_coords if Point((c[1], c[0])).within(mask_shape)]
    return french_coords


if __name__ == '__main__':
    from back.external import load_france_shape

    france_shape = load_france_shape()
    print(generate_grid_coords(france_shape))

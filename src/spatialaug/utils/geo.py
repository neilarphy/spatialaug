"""Geographic helpers shared across the package."""

from __future__ import annotations

import math

KM_PER_DEGREE = 111.0


def km_to_deg(lat: float, km: float) -> tuple[float, float]:
    """Convert a distance in km to (deg_lat, deg_lon) at the given latitude.

    Latitudinal degrees are roughly constant (≈ KM_PER_DEGREE).
    Longitudinal degrees shrink with latitude (multiplier cos(lat)),
    going from KM_PER_DEGREE at the equator to 0 at the poles.
    """
    deg_lat = km / KM_PER_DEGREE
    deg_lon = km / (KM_PER_DEGREE * max(math.cos(math.radians(lat)), 1e-6))
    
    return deg_lat, deg_lon

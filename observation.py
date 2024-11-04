from dataclasses import dataclass
import numpy as np

# a single observation of an iceberg
@dataclass
class Observation:
    lat: float
    lon: float
    time: np.datetime64
    length: float
    height: float
    depth: float
    in_tow: bool
    grounded: bool

    def __init__(self, lat, lon, time, length = 100, grounded = False, in_tow = False):
        self.lat = lat
        self.lon = lon
        self.time = time
        self.length = length
        self.height = length**0.68
        self.depth = 2.5*length**0.68
        self.grounded = grounded
        self.in_tow = in_tow

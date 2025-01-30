
from dataclasses import dataclass
import numpy as np

@dataclass

class Observations:
    id: str
    lat: float
    lon: float
    time: np.datetime64
    length: float
    in_tow: bool
    grounded: bool

    def __init__(self, lat, lon, time, length=100., grounded=False, in_tow=False, id='0000'):
        self.id = id
        self.lat = lat
        self.lon = lon
        self.time = time
        self.length = length
        self.grounded = grounded
        self.in_tow = in_tow


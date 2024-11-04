import numpy as np
from observation import Observation

def forecast(obs: Observation, t1: np.datetime64 ):
    lat0 = obs.lat
    lon0 = obs.lon
    t0 = obs.time
    tint = np.arange(t0,t1+np.timedelta64(3600,'s'),np.timedelta64(3600,'s'))
    latint = np.full((tint.size,),lat0)
    lonint = np.full((tint.size,), lon0)

    return (tint,latint,lonint)


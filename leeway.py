import numpy as np
from observation import Observation
#import pull_data.pull_hycom
import pull_cfsr_v2, pull_ora5
import matplotlib.pyplot as plt

earthRadius = 6381000

def forecast(obs: Observation, t1: np.datetime64 ) -> (np.array, np.array, np.array):
    lat0 = obs.lat
    lon0 = obs.lon
    t0 = obs.time

    tint = np.arange(t0,t1+np.timedelta64(3600,'s'),np.timedelta64(3600,'s'))
    latint = np.full((tint.size,), lat0)
    lonint = np.full((tint.size,), lon0)

    ui = np.full((tint.size),0.0)
    vi = np.full((tint.size),0.0)

    [fuw,fvw] = pull_ora5.get_interpolators(obs.time,t1,obs.depth)
    [fua,fva] = pull_cfsr_v2.get_global_interpolators()

    alpha = 0.014

    # Get interpolators from forecast
    for i,t in enumerate(tint):
        p = ((t.astype(float),latint[i],lonint[i]))
        uw = fuw(p)
        vw = fvw(p)
        ua = fua(p)
        va = fva(p)

        ui[i] = uw + alpha * ua
        vi[i] = vw + alpha * va

        if ui[i]>1.0:
            print("error")

        x = np.trapz(ui[:i+1], dx = 3600)
        y = np.trapz(vi[:i+1], dx = 3600)

        latint[i] = lat0 + y / earthRadius * 180.0 / np.pi
        lonint[i] = lon0 + x / earthRadius * 180.0 / np.pi / np.cos(lat0 * np.pi/180.0)

    return (tint,latint,lonint)


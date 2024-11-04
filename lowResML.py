import numpy as np
from observation import Observation
import pull_ora5
import pull_cfsr_v2
from catboost import Pool, CatBoostRegressor
import matplotlib.pyplot as plt

earthRadius = 6381000
model = CatBoostRegressor()
model.load_model("coarse")

def forecast(obs: Observation, t1: np.datetime64 ) -> (np.array, np.array, np.array):
    lat0 = obs.lat
    lon0 = obs.lon
    t0 = obs.time

    tint = np.arange(t0,t1+np.timedelta64(3600,'s'),np.timedelta64(3600,'s'))
    latint = np.full((tint.size,), lat0)
    lonint = np.full((tint.size,), lon0)


    [fuw,fvw] = pull_ora5.get_interpolators(obs.time,t1,obs.depth)
    [fua,fva] = pull_cfsr_v2.get_global_interpolators()

    l = obs.length

    p = (tint.astype(float), latint, lonint)
    uw = np.nanmean(fuw(p))
    vw = np.nanmean(fvw(p))
    ua = np.nanmean(fua(p))
    va = np.nanmean(fva(p))

    dt = (t1-t0).astype(float)/3600

    temp1 = (dt, l, uw, vw, ua, va)
    x = Pool([temp1])
    y = model.predict(x)[0]

    latint = lat0 + y[1] * 180.0 / np.pi / 6381000.0
    lonint = lon0 + y[0] * 180.0 / np.pi / 6381000.0 / np.cos(lat0 * np.pi / 180.0)

    return (np.array([t0,t1]),np.array([lat0,latint]),np.array([lon0,lonint]))


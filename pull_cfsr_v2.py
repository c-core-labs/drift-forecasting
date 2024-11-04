import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

def get_limits(t, lat, lon):
    min_lat = np.min(lat) - 1.0
    max_lat = np.max(lat) + 1.0
    min_lon = np.min(lon) - 0.5
    max_lon = np.max(lon) + 0.5
    t1 = t[0] - np.timedelta64(3, 'h')
    t2 = t[-1] + np.timedelta64(3, 'h')

    return [t1,t2, min_lat,max_lat,min_lon,max_lon]

def get_global_interpolators():
    url = "../pull_data/CFSRv2/2023.nc"

    ds = xr.open_dataset(url)

    time = ds.variables['MT'][:].to_numpy().astype('datetime64[s]')
    lat = ds.variables['Latitude'][:].to_numpy()
    lon = ds.variables['Longitude'][:].to_numpy()

    u = ds.wndewd.to_numpy()#*0.001953125
    v = ds.wndnwd.to_numpy()#*0.001953125

    FUA = RegularGridInterpolator((time.astype('datetime64[s]').astype(float), lat, lon - 360), u, fill_value=0)
    FVA = RegularGridInterpolator((time.astype('datetime64[s]').astype(float), lat, lon - 360), v, fill_value=0)

    return [FUA, FVA]

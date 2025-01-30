import copernicusmarine
import numpy as np
from scipy.integrate import trapezoid
import xarray as xr
import time as tm
from scipy.interpolate import RegularGridInterpolator
from glob import glob
import earthkit
import pandas as pd
import matplotlib.pyplot as plt
import os

plt.style.use('default')

def pull_data():

  copernicusmarine.subset(
      dataset_id="cmems_mod_glo_phy-cur_anfc_0.083deg_PT6H-i",
      variables=["uo", "vo"],
      minimum_longitude=-58,
      maximum_longitude=-48,
      minimum_latitude=49,
      maximum_latitude=55,
      start_datetime="2024-05-07T00:00:00",
      end_datetime="2024-08-10T00:00:00",
      minimum_depth=0,
      maximum_depth=200,
      output_filename ="2024.nc",
      output_directory = "copernicus-data"
    )


def pull_data_operational(t0):
  copernicusmarine.subset(
    username='ryulmetov',
    password='d8ZNwJRB',
    dataset_id="cmems_mod_glo_phy-cur_anfc_0.083deg_PT6H-i",
    variables=["uo", "vo"],
    minimum_longitude=-64.5,
    maximum_longitude=-46.75,
    minimum_latitude=48,
    maximum_latitude=61.2,
    start_datetime=str(t0-np.timedelta64(6,'h')),
    end_datetime=(t0+np.timedelta64(3,'D')).astype(str),
    minimum_depth=0,
    maximum_depth=200,
    output_filename=t0.astype('datetime64[D]').astype(str)+".nc",
    output_directory="copernicus-data"
  )

def get_interpolators(t0: np.datetime64, t1: np.datetime64, d: float):
    date = str(t0.astype('datetime64[D]'))
    fname = './copernicus-data/' + date + '.nc'

    if not os.path.isfile(fname):
        pull_data_operational(t0)

    # Get dimensions first
    ds = xr.open_dataset(fname, decode_cf=False)

    time_ref = np.datetime64('1950-01-01T00:00:00')
    t = ds.variables['time'].to_numpy().astype('timedelta64[h]')+time_ref
    lat = ds.variables['latitude']
    lon = ds.variables['longitude']

    depth = ds.variables['depth'].to_numpy()
    inddepth = np.squeeze(np.argwhere(depth <= d))

    u0 = ds.variables['uo'].to_numpy()
    v0 = ds.variables['vo'].to_numpy()

    if inddepth.size == 1:
        u = np.squeeze(u0[:, 0, :, :])
    else:
        u_mean = trapezoid(u0[:, inddepth, :, :], x=np.squeeze(depth[inddepth]), axis=1) / np.squeeze(depth[inddepth[-1]])
        u = np.squeeze(u_mean)

    u[np.nonzero(np.abs(u)>2.0)] = np.nan

    if inddepth.size == 1:
        v = np.squeeze(v0[:, 0, :, :])
    else:
        v_mean = trapezoid(v0[:, inddepth, :, :], x=np.squeeze(depth[inddepth]), axis=1) / np.squeeze(depth[inddepth[-1]])
        v = np.squeeze(v_mean)

    v[np.nonzero(np.abs(v) > 2.0)] = np.nan

    fuw = RegularGridInterpolator((t.astype('datetime64[s]').astype(float), lat, lon), u, fill_value = 0)
    fvw = RegularGridInterpolator((t.astype('datetime64[s]').astype(float), lat, lon), v, fill_value = 0)

    return [fuw, fvw]

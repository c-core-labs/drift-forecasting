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

def pull_data_new():
    ds = earthkit.data.from_source("ecmwf-open-data",
                                   param = ["10u", "10v"],
                                   step = [0,3,6,9,12,15,18,21,24,27,30,33,36,39,42,45,48,51,54,57,60,63,66,69,72],
                                   model = "ifs"
                                   )

    t0 = np.datetime64('now').astype('datetime64[D]').astype(str)
    fname = ('./copernicus-data/'+t0+'_ifs.grib')#.replace('-','')
    ds.save(fname)

def get_global_interpolators(t0: np.datetime64):
    date = str(t0.astype('datetime64[D]'))
    fname = './copernicus-data/' + date + '_ifs.grib'
    if not os.path.isfile(fname):
        pull_data_new()

    ds = earthkit.data.from_source("file", fname).to_xarray()

    step = ds.variables['step'][:].to_numpy().astype('timedelta64[ns]')
    lat = ds.variables['latitude'][:].to_numpy()
    lon = ds.variables['longitude'][:].to_numpy()
    u = ds.variables['10u'].to_numpy()#*0.001953125
    v = ds.variables['10u'].to_numpy()#*0.001953125

    refdate = str(ds.attrs['date'])
    reftime = ds.attrs['time']
    t0 = np.datetime64(refdate[0:4]+'-'+refdate[4:6]+'-'+refdate[6:]).astype('datetime64[s]')+np.timedelta64(reftime//100,'h')

    time = (t0 + step.astype('timedelta64[s]'))

    FUA = RegularGridInterpolator((time.astype('datetime64[s]').astype(float), lat, lon), u, fill_value=0)
    FVA = RegularGridInterpolator((time.astype('datetime64[s]').astype(float), lat, lon), v, fill_value=0)

    return [FUA, FVA]

#pull_data_new()
get_global_interpolators(np.datetime64('now'))


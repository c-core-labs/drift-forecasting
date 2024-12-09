
from subprocess import run
import numpy as np
import requests
import os
import warnings
warnings.simplefilter(action='ignore')

# 43.0 GB of daily metocean forecast data if downloaded only once daily.
# 2 hours and 11 minutes to download 84 hours (3.5 days ) of forecast metocean data.

rootpath_to_metdata = './RCM_Iceberg_Metocean_Data/'
wgrib_path = './wgrib/'
forecast_hours = 84

dirname_today = str(np.datetime64('today'))
d_today = dirname_today.replace('-', '')
wind_waves_ocean_hours = np.arange(0, forecast_hours + 1, 1)
airT_sw_rad_hours = np.arange(0, forecast_hours + 1, 3)

if not os.path.isdir(rootpath_to_metdata):
    os.mkdir(rootpath_to_metdata)

if not os.path.isdir(rootpath_to_metdata + 'GDPS_airT_sw_rad_forecast_files/'):
    os.mkdir(rootpath_to_metdata + 'GDPS_airT_sw_rad_forecast_files/')

if not os.path.isdir(rootpath_to_metdata + 'GDWPS_wind_wave_forecast_files/'):
    os.mkdir(rootpath_to_metdata + 'GDWPS_wind_wave_forecast_files/')

if not os.path.isdir(rootpath_to_metdata + 'RIOPS_ocean_forecast_files/'):
    os.mkdir(rootpath_to_metdata + 'RIOPS_ocean_forecast_files/')

if not os.path.isdir(rootpath_to_metdata + 'GDPS_airT_sw_rad_forecast_files/' + dirname_today):
    os.mkdir(rootpath_to_metdata + 'GDPS_airT_sw_rad_forecast_files/' + dirname_today)

if not os.path.isdir(rootpath_to_metdata + 'GDWPS_wind_wave_forecast_files/' + dirname_today):
    os.mkdir(rootpath_to_metdata + 'GDWPS_wind_wave_forecast_files/' + dirname_today)

if not os.path.isdir(rootpath_to_metdata + 'RIOPS_ocean_forecast_files/' + dirname_today):
    os.mkdir(rootpath_to_metdata + 'RIOPS_ocean_forecast_files/' + dirname_today)

url = 'https://dd.meteo.gc.ca/model_gem_global/15km/grib2/lat_lon/12/240/CMC_glb_TMP_TGL_2_latlon.15x.15_' + d_today + '12_P240.grib2'
response = requests.head(url)

if response.status_code == 200:
    hour_utc_str_airT_sw_rad = '12'
else:
    hour_utc_str_airT_sw_rad = '00'

url = 'https://dd.weather.gc.ca/model_riops/netcdf/forecast/polar_stereographic/3d/18/084/' + d_today + 'T18Z_MSC_RIOPS_VOZOCRTX_DBS-all_PS5km_P084.nc'
response = requests.head(url)

if response.status_code == 200:
    hour_utc_str_ocean = '18'
else:
    url = 'https://dd.weather.gc.ca/model_riops/netcdf/forecast/polar_stereographic/3d/12/084/' + d_today + 'T12Z_MSC_RIOPS_VOZOCRTX_DBS-all_PS5km_P084.nc'
    response = requests.head(url)

    if response.status_code == 200:
        hour_utc_str_ocean = '12'
    else:
        url = 'https://dd.weather.gc.ca/model_riops/netcdf/forecast/polar_stereographic/3d/06/084/' + d_today + 'T06Z_MSC_RIOPS_VOZOCRTX_DBS-all_PS5km_P084.nc'
        response = requests.head(url)

        if response.status_code == 200:
            hour_utc_str_ocean = '06'
        else:
            hour_utc_str_ocean = '00'

url = 'https://dd.weather.gc.ca/model_gdwps/25km/12/' + d_today + 'T12Z_MSC_GDWPS_HTSGW_Sfc_LatLon0.25_PT240H.grib2'
response = requests.head(url)

if response.status_code == 200:
    hour_utc_str_wind_waves = '12'
else:
    hour_utc_str_wind_waves = '00'

directory = rootpath_to_metdata + 'GDPS_airT_sw_rad_forecast_files/'

for i in range(len(airT_sw_rad_hours)):
    url = 'https://dd.meteo.gc.ca/model_gem_global/15km/grib2/lat_lon/' + hour_utc_str_airT_sw_rad + '/' + \
        str(airT_sw_rad_hours[i]).zfill(3) + '/CMC_glb_TMP_TGL_2_latlon.15x.15_' + \
        d_today + hour_utc_str_airT_sw_rad + '_P' + str(airT_sw_rad_hours[i]).zfill(3) + '.grib2'
    fname = directory + dirname_today + '/CMC_glb_TMP_TGL_2_latlon.15x.15_' + d_today + hour_utc_str_airT_sw_rad + '_P' + \
            str(airT_sw_rad_hours[i]).zfill(3) + '.grib2'
    flag = True

    while flag:
        try:
            print('Obtaining forecast air temperature file CMC_glb_TMP_TGL_2_latlon.15x.15_' + d_today + hour_utc_str_airT_sw_rad + '_P' + \
                str(airT_sw_rad_hours[i]).zfill(3) + '.grib2')
            r = requests.get(url, allow_redirects=True, timeout=5.0)
            open(fname, 'wb').write(r.content)
            flag = False
        except:
            print('Error: could not download forecast air temperature file CMC_glb_TMP_TGL_2_latlon.15x.15_' + d_today + hour_utc_str_airT_sw_rad + '_P' + \
                str(airT_sw_rad_hours[i]).zfill(3) + '.grib2, retrying...')

    fname = directory + dirname_today + '/CMC_glb_TMP_TGL_2_latlon.15x.15_' + d_today + hour_utc_str_airT_sw_rad + '_P' + \
            str(airT_sw_rad_hours[i]).zfill(3) + '.grib2'
    print('Converting forecast air temperature file CMC_glb_TMP_TGL_2_latlon.15x.15_' + d_today + hour_utc_str_airT_sw_rad + '_P' + \
        str(airT_sw_rad_hours[i]).zfill(3) + '.grib2 to NetCDF')
    run(wgrib_path + 'wgrib2.exe ' + fname + ' -netcdf ' + directory + dirname_today + '/CMC_glb_TMP_TGL_2_latlon.15x.15_' + \
        d_today + hour_utc_str_airT_sw_rad + '_P' + str(airT_sw_rad_hours[i]).zfill(3) + '.nc')
    os.remove(fname)

    url = 'https://dd.meteo.gc.ca/model_gem_global/15km/grib2/lat_lon/' + hour_utc_str_airT_sw_rad + '/' + \
          str(airT_sw_rad_hours[i]).zfill(3) + '/CMC_glb_DSWRF_SFC_0_latlon.15x.15_' + \
          d_today + hour_utc_str_airT_sw_rad + '_P' + str(airT_sw_rad_hours[i]).zfill(3) + '.grib2'
    fname = directory + dirname_today + '/CMC_glb_DSWRF_SFC_0_latlon.15x.15_' + d_today + hour_utc_str_airT_sw_rad + '_P' + \
            str(airT_sw_rad_hours[i]).zfill(3) + '.grib2'
    flag = True

    while flag:
        try:
            print('Obtaining forecast solar radiation file CMC_glb_DSWRF_SFC_0_latlon.15x.15_' + d_today + hour_utc_str_airT_sw_rad + '_P' + \
                str(airT_sw_rad_hours[i]).zfill(3) + '.grib2')
            r = requests.get(url, allow_redirects=True, timeout=5.0)
            open(fname, 'wb').write(r.content)
            flag = False
        except:
            print('Error: could not download forecast solar radiation file CMC_glb_DSWRF_SFC_0_latlon.15x.15_' + d_today + hour_utc_str_airT_sw_rad + '_P' + \
                str(airT_sw_rad_hours[i]).zfill(3) + '.grib2, retrying...')

    fname = directory + dirname_today + '/CMC_glb_DSWRF_SFC_0_latlon.15x.15_' + d_today + hour_utc_str_airT_sw_rad + '_P' + \
            str(airT_sw_rad_hours[i]).zfill(3) + '.grib2'
    print('Converting forecast solar radiation file CMC_glb_DSWRF_SFC_0_latlon.15x.15_' + d_today + hour_utc_str_airT_sw_rad + '_P' + \
        str(airT_sw_rad_hours[i]).zfill(3) + '.grib2 to NetCDF')
    run(wgrib_path + 'wgrib2.exe ' + fname + ' -netcdf ' + directory + dirname_today + '/CMC_glb_DSWRF_SFC_0_latlon.15x.15_' + \
        d_today + hour_utc_str_airT_sw_rad + '_P' + str(airT_sw_rad_hours[i]).zfill(3) + '.nc')
    os.remove(fname)

for i in range(len(wind_waves_ocean_hours)):
    directory = rootpath_to_metdata + 'GDWPS_wind_wave_forecast_files/'

    url = 'https://dd.weather.gc.ca/model_gdwps/25km/' + hour_utc_str_wind_waves + '/' + d_today + 'T' + hour_utc_str_wind_waves + \
        'Z_MSC_GDWPS_UGRD_AGL-10m_LatLon0.25_PT' + str(wind_waves_ocean_hours[i]).zfill(3) + 'H.grib2'
    fname = directory + dirname_today + '/' + d_today + 'T' + hour_utc_str_wind_waves + \
        'Z_MSC_GDWPS_UGRD_AGL-10m_LatLon0.25_PT' + str(wind_waves_ocean_hours[i]).zfill(3) + 'H.grib2'
    flag = True

    while flag:
        try:
            print('Obtaining forecast zonal wind velocity file ' + d_today + 'T' + hour_utc_str_wind_waves + \
                'Z_MSC_GDWPS_UGRD_AGL-10m_LatLon0.25_PT' + str(wind_waves_ocean_hours[i]).zfill(3) + 'H.grib2')
            r = requests.get(url, allow_redirects=True, timeout=5.0)
            open(fname, 'wb').write(r.content)
            flag = False
        except:
            print('Error: could not download forecast zonal wind velocity file ' + d_today + 'T' + hour_utc_str_wind_waves + \
                'Z_MSC_GDWPS_UGRD_AGL-10m_LatLon0.25_PT' + str(wind_waves_ocean_hours[i]).zfill(3) + 'H.grib2, retrying...')

    fname = directory + dirname_today + '/' + d_today + 'T' + hour_utc_str_wind_waves + \
        'Z_MSC_GDWPS_UGRD_AGL-10m_LatLon0.25_PT' + str(wind_waves_ocean_hours[i]).zfill(3) + 'H.grib2'
    print('Converting forecast zonal wind velocity file ' + d_today + 'T' + hour_utc_str_wind_waves + \
          'Z_MSC_GDWPS_UGRD_AGL-10m_LatLon0.25_PT' + str(wind_waves_ocean_hours[i]).zfill(3) + 'H.grib2 to NetCDF')
    run(wgrib_path + 'wgrib2.exe ' + fname + ' -netcdf ' + directory + dirname_today + '/' + d_today + 'T' + hour_utc_str_wind_waves + \
        'Z_MSC_GDWPS_UGRD_AGL-10m_LatLon0.25_PT' + str(wind_waves_ocean_hours[i]).zfill(3) + 'H.nc')
    os.remove(fname)

    url = 'https://dd.weather.gc.ca/model_gdwps/25km/' + hour_utc_str_wind_waves + '/' + d_today + 'T' + hour_utc_str_wind_waves + \
          'Z_MSC_GDWPS_VGRD_AGL-10m_LatLon0.25_PT' + str(wind_waves_ocean_hours[i]).zfill(3) + 'H.grib2'
    fname = directory + dirname_today + '/' + d_today + 'T' + hour_utc_str_wind_waves + \
            'Z_MSC_GDWPS_VGRD_AGL-10m_LatLon0.25_PT' + str(wind_waves_ocean_hours[i]).zfill(3) + 'H.grib2'
    flag = True

    while flag:
        try:
            print('Obtaining forecast meridional wind velocity file ' + d_today + 'T' + hour_utc_str_wind_waves + \
                  'Z_MSC_GDWPS_VGRD_AGL-10m_LatLon0.25_PT' + str(wind_waves_ocean_hours[i]).zfill(3) + 'H.grib2')
            r = requests.get(url, allow_redirects=True, timeout=5.0)
            open(fname, 'wb').write(r.content)
            flag = False
        except:
            print('Error: could not download forecast meridional wind velocity file ' + d_today + 'T' + hour_utc_str_wind_waves + \
                'Z_MSC_GDWPS_VGRD_AGL-10m_LatLon0.25_PT' + str(wind_waves_ocean_hours[i]).zfill(3) + 'H.grib2, retrying...')

    fname = directory + dirname_today + '/' + d_today + 'T' + hour_utc_str_wind_waves + \
            'Z_MSC_GDWPS_VGRD_AGL-10m_LatLon0.25_PT' + str(wind_waves_ocean_hours[i]).zfill(3) + 'H.grib2'
    print('Converting forecast meridional wind velocity file ' + d_today + 'T' + hour_utc_str_wind_waves + \
          'Z_MSC_GDWPS_VGRD_AGL-10m_LatLon0.25_PT' + str(wind_waves_ocean_hours[i]).zfill(3) + 'H.grib2 to NetCDF')
    run(wgrib_path + 'wgrib2.exe ' + fname + ' -netcdf ' + directory + dirname_today + '/' + d_today + 'T' + hour_utc_str_wind_waves + \
        'Z_MSC_GDWPS_VGRD_AGL-10m_LatLon0.25_PT' + str(wind_waves_ocean_hours[i]).zfill(3) + 'H.nc')
    os.remove(fname)

    url = 'https://dd.weather.gc.ca/model_gdwps/25km/' + hour_utc_str_wind_waves + '/' + d_today + 'T' + hour_utc_str_wind_waves + \
          'Z_MSC_GDWPS_HTSGW_Sfc_LatLon0.25_PT' + str(wind_waves_ocean_hours[i]).zfill(3) + 'H.grib2'
    fname = directory + dirname_today + '/' + d_today + 'T' + hour_utc_str_wind_waves + \
            'Z_MSC_GDWPS_HTSGW_Sfc_LatLon0.25_PT' + str(wind_waves_ocean_hours[i]).zfill(3) + 'H.grib2'
    flag = True

    while flag:
        try:
            print('Obtaining forecast significant wave height file ' + d_today + 'T' + hour_utc_str_wind_waves + \
                  'Z_MSC_GDWPS_HTSGW_Sfc_LatLon0.25_PT' + str(wind_waves_ocean_hours[i]).zfill(3) + 'H.grib2')
            r = requests.get(url, allow_redirects=True, timeout=5.0)
            open(fname, 'wb').write(r.content)
            flag = False
        except:
            print('Error: could not download forecast significant wave height file ' + d_today + 'T' + hour_utc_str_wind_waves + \
                'Z_MSC_GDWPS_HTSGW_Sfc_LatLon0.25_PT' + str(wind_waves_ocean_hours[i]).zfill(3) + 'H.grib2, retrying...')

    fname = directory + dirname_today + '/' + d_today + 'T' + hour_utc_str_wind_waves + \
            'Z_MSC_GDWPS_HTSGW_Sfc_LatLon0.25_PT' + str(wind_waves_ocean_hours[i]).zfill(3) + 'H.grib2'
    print('Converting forecast significant wave height file ' + d_today + 'T' + hour_utc_str_wind_waves + \
          'Z_MSC_GDWPS_HTSGW_Sfc_LatLon0.25_PT' + str(wind_waves_ocean_hours[i]).zfill(3) + 'H.grib2 to NetCDF')
    run(wgrib_path + 'wgrib2.exe ' + fname + ' -netcdf ' + directory + dirname_today + '/' + d_today + 'T' + hour_utc_str_wind_waves + \
        'Z_MSC_GDWPS_HTSGW_Sfc_LatLon0.25_PT' + str(wind_waves_ocean_hours[i]).zfill(3) + 'H.nc')
    os.remove(fname)

    url = 'https://dd.weather.gc.ca/model_gdwps/25km/' + hour_utc_str_wind_waves + '/' + d_today + 'T' + hour_utc_str_wind_waves + \
          'Z_MSC_GDWPS_WVDIR_Sfc_LatLon0.25_PT' + str(wind_waves_ocean_hours[i]).zfill(3) + 'H.grib2'
    fname = directory + dirname_today + '/' + d_today + 'T' + hour_utc_str_wind_waves + \
            'Z_MSC_GDWPS_WVDIR_Sfc_LatLon0.25_PT' + str(wind_waves_ocean_hours[i]).zfill(3) + 'H.grib2'
    flag = True

    while flag:
        try:
            print('Obtaining forecast wave direction file ' + d_today + 'T' + hour_utc_str_wind_waves + \
                  'Z_MSC_GDWPS_WVDIR_Sfc_LatLon0.25_PT' + str(wind_waves_ocean_hours[i]).zfill(3) + 'H.grib2')
            r = requests.get(url, allow_redirects=True, timeout=5.0)
            open(fname, 'wb').write(r.content)
            flag = False
        except:
            print('Error: could not download forecast wave direction file ' + d_today + 'T' + hour_utc_str_wind_waves + \
                'Z_MSC_GDWPS_WVDIR_Sfc_LatLon0.25_PT' + str(wind_waves_ocean_hours[i]).zfill(3) + 'H.grib2, retrying...')

    fname = directory + dirname_today + '/' + d_today + 'T' + hour_utc_str_wind_waves + \
            'Z_MSC_GDWPS_WVDIR_Sfc_LatLon0.25_PT' + str(wind_waves_ocean_hours[i]).zfill(3) + 'H.grib2'
    print('Converting forecast wave direction file ' + d_today + 'T' + hour_utc_str_wind_waves + \
          'Z_MSC_GDWPS_WVDIR_Sfc_LatLon0.25_PT' + str(wind_waves_ocean_hours[i]).zfill(3) + 'H.grib2 to NetCDF')
    run(wgrib_path + 'wgrib2.exe ' + fname + ' -netcdf ' + directory + dirname_today + '/' + d_today + 'T' + hour_utc_str_wind_waves + \
        'Z_MSC_GDWPS_WVDIR_Sfc_LatLon0.25_PT' + str(wind_waves_ocean_hours[i]).zfill(3) + 'H.nc')
    os.remove(fname)

    url = 'https://dd.weather.gc.ca/model_gdwps/25km/' + hour_utc_str_wind_waves + '/' + d_today + 'T' + hour_utc_str_wind_waves + \
          'Z_MSC_GDWPS_MZWPER_Sfc_LatLon0.25_PT' + str(wind_waves_ocean_hours[i]).zfill(3) + 'H.grib2'
    fname = directory + dirname_today + '/' + d_today + 'T' + hour_utc_str_wind_waves + \
            'Z_MSC_GDWPS_MZWPER_Sfc_LatLon0.25_PT' + str(wind_waves_ocean_hours[i]).zfill(3) + 'H.grib2'
    flag = True

    while flag:
        try:
            print('Obtaining forecast mean wave period file ' + d_today + 'T' + hour_utc_str_wind_waves + \
                  'Z_MSC_GDWPS_MZWPER_Sfc_LatLon0.25_PT' + str(wind_waves_ocean_hours[i]).zfill(3) + 'H.grib2')
            r = requests.get(url, allow_redirects=True, timeout=5.0)
            open(fname, 'wb').write(r.content)
            flag = False
        except:
            print('Error: could not download forecast mean wave period file ' + d_today + 'T' + hour_utc_str_wind_waves + \
                'Z_MSC_GDWPS_MZWPER_Sfc_LatLon0.25_PT' + str(wind_waves_ocean_hours[i]).zfill(3) + 'H.grib2, retrying...')

    fname = directory + dirname_today + '/' + d_today + 'T' + hour_utc_str_wind_waves + \
        'Z_MSC_GDWPS_MZWPER_Sfc_LatLon0.25_PT' + str(wind_waves_ocean_hours[i]).zfill(3) + 'H.grib2'
    print('Converting forecast mean wave period file ' + d_today + 'T' + hour_utc_str_wind_waves + \
          'Z_MSC_GDWPS_MZWPER_Sfc_LatLon0.25_PT' + str(wind_waves_ocean_hours[i]).zfill(3) + 'H.grib2 to NetCDF')
    run(wgrib_path + 'wgrib2.exe ' + fname + ' -netcdf ' + directory + dirname_today + '/' + d_today + 'T' + hour_utc_str_wind_waves + \
        'Z_MSC_GDWPS_MZWPER_Sfc_LatLon0.25_PT' + str(wind_waves_ocean_hours[i]).zfill(3) + 'H.nc')
    os.remove(fname)

    directory = rootpath_to_metdata + 'RIOPS_ocean_forecast_files/'

    url = 'https://dd.weather.gc.ca/model_riops/netcdf/forecast/polar_stereographic/3d/' + hour_utc_str_ocean + '/' + \
        str(wind_waves_ocean_hours[i]).zfill(3) + '/' + d_today + 'T' + hour_utc_str_ocean + \
        'Z_MSC_RIOPS_VOZOCRTX_DBS-all_PS5km_P' + str(wind_waves_ocean_hours[i]).zfill(3) + '.nc'
    fname = directory + dirname_today + '/' + d_today + 'T' + hour_utc_str_ocean + \
        'Z_MSC_RIOPS_VOZOCRTX_DBS-all_PS5km_P' + str(wind_waves_ocean_hours[i]).zfill(3) + '.nc'
    flag = True

    while flag:
        try:
            print('Obtaining forecast zonal ocean current file ' + d_today + 'T' + hour_utc_str_ocean + \
                  'Z_MSC_RIOPS_VOZOCRTX_DBS-all_PS5km_P' + str(wind_waves_ocean_hours[i]).zfill(3) + '.nc')
            r = requests.get(url, allow_redirects=True, timeout=5.0)
            open(fname, 'wb').write(r.content)
            flag = False
        except:
            print('Error: could not download forecast zonal ocean current file ' + d_today + 'T' + hour_utc_str_ocean + \
                'Z_MSC_RIOPS_VOZOCRTX_DBS-all_PS5km_P' + str(wind_waves_ocean_hours[i]).zfill(3) + '.nc, retrying...')

    url = 'https://dd.weather.gc.ca/model_riops/netcdf/forecast/polar_stereographic/3d/' + hour_utc_str_ocean + '/' + \
          str(wind_waves_ocean_hours[i]).zfill(3) + '/' + d_today + 'T' + hour_utc_str_ocean + \
          'Z_MSC_RIOPS_VOMECRTY_DBS-all_PS5km_P' + str(wind_waves_ocean_hours[i]).zfill(3) + '.nc'
    fname = directory + dirname_today + '/' + d_today + 'T' + hour_utc_str_ocean + \
            'Z_MSC_RIOPS_VOMECRTY_DBS-all_PS5km_P' + str(wind_waves_ocean_hours[i]).zfill(3) + '.nc'
    flag = True

    while flag:
        try:
            print('Obtaining forecast meridional ocean current file ' + d_today + 'T' + hour_utc_str_ocean + \
                  'Z_MSC_RIOPS_VOMECRTY_DBS-all_PS5km_P' + str(wind_waves_ocean_hours[i]).zfill(3) + '.nc')
            r = requests.get(url, allow_redirects=True, timeout=5.0)
            open(fname, 'wb').write(r.content)
            flag = False
        except:
            print('Error: could not download forecast meridional ocean current file ' + d_today + 'T' + hour_utc_str_ocean + \
                  'Z_MSC_RIOPS_VOMECRTY_DBS-all_PS5km_P' + str(wind_waves_ocean_hours[i]).zfill(3) + '.nc, retrying...')

    url = 'https://dd.weather.gc.ca/model_riops/netcdf/forecast/polar_stereographic/3d/' + hour_utc_str_ocean + '/' + \
          str(wind_waves_ocean_hours[i]).zfill(3) + '/' + d_today + 'T' + hour_utc_str_ocean + \
          'Z_MSC_RIOPS_VOSALINE_DBS-all_PS5km_P' + str(wind_waves_ocean_hours[i]).zfill(3) + '.nc'
    fname = directory + dirname_today + '/' + d_today + 'T' + hour_utc_str_ocean + \
            'Z_MSC_RIOPS_VOSALINE_DBS-all_PS5km_P' + str(wind_waves_ocean_hours[i]).zfill(3) + '.nc'
    flag = True

    while flag:
        try:
            print('Obtaining forecast ocean salinity file ' + d_today + 'T' + hour_utc_str_ocean + \
                  'Z_MSC_RIOPS_VOSALINE_DBS-all_PS5km_P' + str(wind_waves_ocean_hours[i]).zfill(3) + '.nc')
            r = requests.get(url, allow_redirects=True, timeout=5.0)
            open(fname, 'wb').write(r.content)
            flag = False
        except:
            print('Error: could not download forecast ocean salinity file ' + d_today + 'T' + hour_utc_str_ocean + \
                  'Z_MSC_RIOPS_VOSALINE_DBS-all_PS5km_P' + str(wind_waves_ocean_hours[i]).zfill(3) + '.nc, retrying...')

    url = 'https://dd.weather.gc.ca/model_riops/netcdf/forecast/polar_stereographic/3d/' + hour_utc_str_ocean + '/' + \
          str(wind_waves_ocean_hours[i]).zfill(3) + '/' + d_today + 'T' + hour_utc_str_ocean + \
          'Z_MSC_RIOPS_VOTEMPER_DBS-all_PS5km_P' + str(wind_waves_ocean_hours[i]).zfill(3) + '.nc'
    fname = directory + dirname_today + '/' + d_today + 'T' + hour_utc_str_ocean + \
            'Z_MSC_RIOPS_VOTEMPER_DBS-all_PS5km_P' + str(wind_waves_ocean_hours[i]).zfill(3) + '.nc'
    flag = True

    while flag:
        try:
            print('Obtaining forecast ocean potential temperature file ' + d_today + 'T' + hour_utc_str_ocean + \
                  'Z_MSC_RIOPS_VOTEMPER_DBS-all_PS5km_P' + str(wind_waves_ocean_hours[i]).zfill(3) + '.nc')
            r = requests.get(url, allow_redirects=True, timeout=5.0)
            open(fname, 'wb').write(r.content)
            flag = False
        except:
            print('Error: could not download forecast ocean potential temperature file ' + d_today + 'T' + hour_utc_str_ocean + \
                  'Z_MSC_RIOPS_VOTEMPER_DBS-all_PS5km_P' + str(wind_waves_ocean_hours[i]).zfill(3) + '.nc, retrying...')

    url = 'https://dd.weather.gc.ca/model_riops/netcdf/forecast/polar_stereographic/2d/' + hour_utc_str_ocean + '/' + \
          str(wind_waves_ocean_hours[i]).zfill(3) + '/' + d_today + 'T' + hour_utc_str_ocean + \
          'Z_MSC_RIOPS_SOSSHEIG_SFC_PS5km_P' + str(wind_waves_ocean_hours[i]).zfill(3) + '.nc'
    fname = directory + dirname_today + '/' + d_today + 'T' + hour_utc_str_ocean + \
            'Z_MSC_RIOPS_SOSSHEIG_SFC_PS5km_P' + str(wind_waves_ocean_hours[i]).zfill(3) + '.nc'
    flag = True

    while flag:
        try:
            print('Obtaining forecast sea surface height file ' + d_today + 'T' + hour_utc_str_ocean + \
                  'Z_MSC_RIOPS_SOSSHEIG_SFC_PS5km_P' + str(wind_waves_ocean_hours[i]).zfill(3) + '.nc')
            r = requests.get(url, allow_redirects=True, timeout=5.0)
            open(fname, 'wb').write(r.content)
            flag = False
        except:
            print('Error: could not download forecast sea surface height file ' + d_today + 'T' + hour_utc_str_ocean + \
                  'Z_MSC_RIOPS_SOSSHEIG_SFC_PS5km_P' + str(wind_waves_ocean_hours[i]).zfill(3) + '.nc, retrying...')



from subprocess import run
import numpy as np
import netCDF4 as nc
import requests
import os
import warnings
warnings.simplefilter(action='ignore')

rootpath_to_data = './RCM_Iceberg_Metocean_Data/'
wgrib_path = './wgrib/'

def dist_bearing(Re, lat1, lat2, lon1, lon2):
    def arccot(x):
        return np.pi / 2 - np.arctan(x)

    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)
    lon1 = np.radians(lon1)
    lon2 = np.radians(lon2)
    L = Re * np.arccos(np.sin(lat1) * np.sin(lat2) + np.cos(lat1) * np.cos(lat2) * np.cos(lon2 - lon1))
    Az = np.degrees(arccot((np.cos(lat1) * np.tan(lat2) - np.sin(lat1) * np.cos(lon2 - lon1)) / np.sin(lon2 - lon1)))

    if Az < 0:
        Az += 360
    elif Az >= 360:
        Az -= 360

    if lon1 == lon2:
        if lat1 > lat2:
            Az = 180.
        elif lat1 < lat2:
            Az = 0.
        elif lat1 == lat2:
            L = 0.
            Az = 0.

    return L, Az

def dist_course(Re, lat1, lon1, dist, course):
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    course = np.radians(course)
    lat2 = np.arcsin(np.sin(lat1) * np.cos(dist / Re) + np.cos(lat1) * np.sin(dist / Re) * np.cos(course))
    lat2 = np.degrees(lat2)
    lat2_rad = np.radians(lat2)
    lon2 = lon1 + np.arctan2(np.sin(course) * np.sin(dist / Re) * np.cos(lat1), np.cos(dist / Re) - np.sin(lat1) * np.sin(lat2_rad))
    lon2 = np.degrees(lon2)
    return lat2, lon2

dirname_today = str(np.datetime64('today'))
d_today = dirname_today.replace('-', '')

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

wind_waves_ocean_hours = np.arange(0, 85, 1)
airT_sw_rad_hours = np.arange(0, 85, 3)
Re = 6371e3

if not os.path.isdir(rootpath_to_data + 'GDPS_airT_sw_rad_forecast_files/'):
    os.mkdir(rootpath_to_data + 'GDPS_airT_sw_rad_forecast_files/')

if not os.path.isdir(rootpath_to_data + 'GDWPS_wind_wave_forecast_files/'):
    os.mkdir(rootpath_to_data + 'GDWPS_wind_wave_forecast_files/')

if not os.path.isdir(rootpath_to_data + 'RIOPS_ocean_forecast_files/'):
    os.mkdir(rootpath_to_data + 'RIOPS_ocean_forecast_files/')

if not os.path.isdir(rootpath_to_data + 'GDPS_airT_sw_rad_forecast_files/' + dirname_today):
    os.mkdir(rootpath_to_data + 'GDPS_airT_sw_rad_forecast_files/' + dirname_today)

if not os.path.isdir(rootpath_to_data + 'GDWPS_wind_wave_forecast_files/' + dirname_today):
    os.mkdir(rootpath_to_data + 'GDWPS_wind_wave_forecast_files/' + dirname_today)

if not os.path.isdir(rootpath_to_data + 'RIOPS_ocean_forecast_files/' + dirname_today):
    os.mkdir(rootpath_to_data + 'RIOPS_ocean_forecast_files/' + dirname_today)

for i in range(len(wind_waves_ocean_hours)):
    directory = rootpath_to_data + 'GDWPS_wind_wave_forecast_files/'

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
    run(wgrib_path + 'wgrib2.exe ' + fname + ' -netcdf ' + directory + dirname_today + '/' + d_today + 'T' + hour_utc_str_wind_waves + \
        'Z_MSC_GDWPS_MZWPER_Sfc_LatLon0.25_PT' + str(wind_waves_ocean_hours[i]).zfill(3) + 'H.nc')
    os.remove(fname)

    directory = rootpath_to_data + 'RIOPS_ocean_forecast_files/'

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

for i in range(len(airT_sw_rad_hours)):
    directory = rootpath_to_data + 'GDPS_airT_sw_rad_forecast_files/'

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
    run(wgrib_path + 'wgrib2.exe ' + fname + ' -netcdf ' + directory + dirname_today + '/CMC_glb_DSWRF_SFC_0_latlon.15x.15_' + \
        d_today + hour_utc_str_airT_sw_rad + '_P' + str(airT_sw_rad_hours[i]).zfill(3) + '.nc')
    os.remove(fname)

for i in range(len(wind_waves_ocean_hours)):
    directory = rootpath_to_data + 'RIOPS_ocean_forecast_files/'
    fname = (directory + dirname_today + '/' + d_today + 'T' + hour_utc_str_ocean +
             'Z_MSC_RIOPS_SOSSHEIG_SFC_PS5km_P' + str(wind_waves_ocean_hours[i]).zfill(3) + '.nc')
    ssh_data = nc.Dataset(fname)
    lat_ocean = ssh_data.variables['latitude'][:]  # lat x lon
    lon_ocean = ssh_data.variables['longitude'][:]  # lat x lon
    ssh = np.squeeze(ssh_data.variables['sossheig'][:])  # lat x lon
    ssh_data.close()

    ssh_grad_x = np.empty((len(lat_ocean[:, 0]), len(lon_ocean[0, :]) - 1))
    ssh_grad_y = np.empty((len(lat_ocean[:, 0]) - 1, len(lon_ocean[0, :])))

    ssh_grad_x_lat = np.empty((len(lat_ocean[:, 0]), len(lon_ocean[0, :]) - 1))
    ssh_grad_y_lat = np.empty((len(lat_ocean[:, 0]) - 1, len(lon_ocean[0, :])))

    ssh_grad_x_lon = np.empty((len(lat_ocean[:, 0]), len(lon_ocean[0, :]) - 1))
    ssh_grad_y_lon = np.empty((len(lat_ocean[:, 0]) - 1, len(lon_ocean[0, :])))

    fname = (directory + dirname_today + '/' + d_today + 'T' + hour_utc_str_ocean +
             'Z_MSC_RIOPS_SOSSHEIG_SFC_GRAD_PS5km_P' + str(wind_waves_ocean_hours[i]).zfill(3) + '.nc')

    print('Computing forecast sea surface height gradients for ' + d_today + 'T' + hour_utc_str_ocean + \
          'Z_MSC_RIOPS_SOSSHEIG_SFC_GRAD_PS5km_P' + str(wind_waves_ocean_hours[i]).zfill(3) + '.nc')

    for k in range(len(lat_ocean[:, 0])):
        for n in range(len(lon_ocean[0, :]) - 1):
            grid_pt_dist, grid_pt_bearing = dist_bearing(Re, lat_ocean[k, n], lat_ocean[k, n + 1], lon_ocean[k, n], lon_ocean[k, n + 1])
            ssh_grad_x_lat[k, n], ssh_grad_x_lon[k, n] = dist_course(Re, lat_ocean[k, n], lon_ocean[k, n], grid_pt_dist / 2., grid_pt_bearing)
            ssh_grad = (ssh[k, n + 1] - ssh[k, n]) / grid_pt_dist
            ssh_grad_x[k, n] = ssh_grad * np.sin(np.deg2rad(grid_pt_bearing))

    for k in range(len(lat_ocean[:, 0]) - 1):
        for n in range(len(lon_ocean[0, :])):
            grid_pt_dist, grid_pt_bearing = dist_bearing(Re, lat_ocean[k, n], lat_ocean[k + 1, n], lon_ocean[k, n], lon_ocean[k + 1, n])
            ssh_grad_y_lat[k, n], ssh_grad_y_lon[k, n] = dist_course(Re, lat_ocean[k, n], lon_ocean[k, n], grid_pt_dist / 2., grid_pt_bearing)
            ssh_grad = (ssh[k + 1, n] - ssh[k, n]) / grid_pt_dist
            ssh_grad_y[k, n] = -ssh_grad * np.cos(np.deg2rad(grid_pt_bearing))

    print('Writing forecast sea surface height gradient file ' + d_today + 'T' + hour_utc_str_ocean + \
          'Z_MSC_RIOPS_SOSSHEIG_SFC_GRAD_PS5km_P' + str(wind_waves_ocean_hours[i]).zfill(3) + '.nc')

    with nc.Dataset(fname, 'w', format='NETCDF4') as ncfile:
        ncfile.createDimension('x_gradient_latitude', len(lat_ocean[:, 0]))
        ncfile.createDimension('x_gradient_longitude', len(lon_ocean[0, :]) - 1)
        ncfile.createDimension('y_gradient_latitude', len(lat_ocean[:, 0]) - 1)
        ncfile.createDimension('y_gradient_longitude', len(lon_ocean[0, :]))

        x_gradient_latitude_var = ncfile.createVariable('ssh_grad_x_lat', 'f8',('x_gradient_latitude', 'x_gradient_longitude',))
        x_gradient_longitude_var = ncfile.createVariable('ssh_grad_x_lon', 'f8',('x_gradient_latitude', 'x_gradient_longitude',))
        y_gradient_latitude_var = ncfile.createVariable('ssh_grad_y_lat', 'f8',('y_gradient_latitude', 'y_gradient_longitude',))
        y_gradient_longitude_var = ncfile.createVariable('ssh_grad_y_lon', 'f8',('y_gradient_latitude', 'y_gradient_longitude',))
        ssh_grad_x_var = ncfile.createVariable('ssh_grad_x', 'f4', ('x_gradient_latitude', 'x_gradient_longitude',))
        ssh_grad_y_var = ncfile.createVariable('ssh_grad_y', 'f4', ('y_gradient_latitude', 'y_gradient_longitude',))

        x_gradient_latitude_var[:] = ssh_grad_x_lat
        x_gradient_longitude_var[:] = ssh_grad_x_lon
        y_gradient_latitude_var[:] = ssh_grad_y_lat
        y_gradient_longitude_var[:] = ssh_grad_y_lon
        ssh_grad_x_var[:] = ssh_grad_x
        ssh_grad_y_var[:] = ssh_grad_y

    fname = (directory + dirname_today + '/' + d_today + 'T' + hour_utc_str_ocean +
             'Z_MSC_RIOPS_SOSSHEIG_SFC_PS5km_P' + str(wind_waves_ocean_hours[i]).zfill(3) + '.nc')
    os.remove(fname)

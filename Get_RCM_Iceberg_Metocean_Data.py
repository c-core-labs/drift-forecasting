
import numpy as np
import requests
import os
import xarray as xr
import netCDF4 as nc
import warnings
warnings.simplefilter(action='ignore')

# Without sea ice data:
# 43.0 GB of daily metocean forecast data if downloaded only once daily.
# 2 hours and 11 minutes to download 84 hours (3.5 days) of forecast metocean data.

# With sea ice data:
# 46.4 GB of daily metocean forecast data if downloaded only once daily.
# 2 hours and 56 minutes to download 84 hours (3.5 days) of forecast metocean data.

def get_rcm_metocean_data(date, forecast_hours, minimum_longitude, maximum_longitude, minimum_latitude, maximum_latitude, maximum_iceberg_length, si_toggle):
    rootpath_to_metdata = './RCM_Iceberg_Metocean_Data/'
    dirname_today = date
    Re = 6371e3

    def dist_bearing(Re, lat1, lat2, lon1, lon2):
        def arccot(x):
            return np.pi / 2. - np.arctan(x)

        lat1 = np.radians(lat1)
        lat2 = np.radians(lat2)
        lon1 = np.radians(lon1)
        lon2 = np.radians(lon2)
        L = Re * np.arccos(np.sin(lat1) * np.sin(lat2) + np.cos(lat1) * np.cos(lat2) * np.cos(lon2 - lon1))
        Az = np.degrees(arccot((np.cos(lat1) * np.tan(lat2) - np.sin(lat1) * np.cos(lon2 - lon1)) / np.sin(lon2 - lon1)))

        if Az < 0:
            Az += 360.
        elif Az >= 360:
            Az -= 360.

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
        ds = xr.open_dataset(fname, engine='cfgrib')
        ds.to_netcdf(directory + dirname_today + '/CMC_glb_TMP_TGL_2_latlon.15x.15_' +
                     d_today + hour_utc_str_airT_sw_rad + '_P' + str(airT_sw_rad_hours[i]).zfill(3) + '.nc')
        ds.close()
        os.remove(directory + dirname_today + '/CMC_glb_TMP_TGL_2_latlon.15x.15_' + d_today + hour_utc_str_airT_sw_rad + '_P' + \
                str(airT_sw_rad_hours[i]).zfill(3) + '.grib2.5b7b6.idx')
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
        ds = xr.open_dataset(fname, engine='cfgrib')
        ds.to_netcdf(directory + dirname_today + '/CMC_glb_DSWRF_SFC_0_latlon.15x.15_' +
                     d_today + hour_utc_str_airT_sw_rad + '_P' + str(airT_sw_rad_hours[i]).zfill(3) + '.nc')
        ds.close()
        os.remove(directory + dirname_today + '/CMC_glb_DSWRF_SFC_0_latlon.15x.15_' + d_today + hour_utc_str_airT_sw_rad + '_P' + \
            str(airT_sw_rad_hours[i]).zfill(3) + '.grib2.5b7b6.idx')
        os.remove(fname)

    directory = rootpath_to_metdata + 'GDWPS_wind_wave_forecast_files/'

    for i in range(len(wind_waves_ocean_hours)):
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
        ds = xr.open_dataset(fname, engine='cfgrib')
        ds.to_netcdf(directory + dirname_today + '/' + d_today + 'T' + hour_utc_str_wind_waves + \
            'Z_MSC_GDWPS_UGRD_AGL-10m_LatLon0.25_PT' + str(wind_waves_ocean_hours[i]).zfill(3) + 'H.nc')
        ds.close()
        os.remove(directory + dirname_today + '/' + d_today + 'T' + hour_utc_str_wind_waves + \
            'Z_MSC_GDWPS_UGRD_AGL-10m_LatLon0.25_PT' + str(wind_waves_ocean_hours[i]).zfill(3) + 'H.grib2.5b7b6.idx')
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
        ds = xr.open_dataset(fname, engine='cfgrib')
        ds.to_netcdf(directory + dirname_today + '/' + d_today + 'T' + hour_utc_str_wind_waves + \
                     'Z_MSC_GDWPS_VGRD_AGL-10m_LatLon0.25_PT' + str(wind_waves_ocean_hours[i]).zfill(3) + 'H.nc')
        ds.close()
        os.remove(directory + dirname_today + '/' + d_today + 'T' + hour_utc_str_wind_waves + \
                  'Z_MSC_GDWPS_VGRD_AGL-10m_LatLon0.25_PT' + str(wind_waves_ocean_hours[i]).zfill(3) + 'H.grib2.5b7b6.idx')
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
        ds = xr.open_dataset(fname, engine='cfgrib')
        ds.to_netcdf(directory + dirname_today + '/' + d_today + 'T' + hour_utc_str_wind_waves + \
                     'Z_MSC_GDWPS_HTSGW_Sfc_LatLon0.25_PT' + str(wind_waves_ocean_hours[i]).zfill(3) + 'H.nc')
        ds.close()
        os.remove(directory + dirname_today + '/' + d_today + 'T' + hour_utc_str_wind_waves + \
                  'Z_MSC_GDWPS_HTSGW_Sfc_LatLon0.25_PT' + str(wind_waves_ocean_hours[i]).zfill(3) + 'H.grib2.5b7b6.idx')
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
        ds = xr.open_dataset(fname, engine='cfgrib')
        ds.to_netcdf(directory + dirname_today + '/' + d_today + 'T' + hour_utc_str_wind_waves + \
                     'Z_MSC_GDWPS_WVDIR_Sfc_LatLon0.25_PT' + str(wind_waves_ocean_hours[i]).zfill(3) + 'H.nc')
        ds.close()
        os.remove(directory + dirname_today + '/' + d_today + 'T' + hour_utc_str_wind_waves + \
                  'Z_MSC_GDWPS_WVDIR_Sfc_LatLon0.25_PT' + str(wind_waves_ocean_hours[i]).zfill(3) + 'H.grib2.5b7b6.idx')
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
        ds = xr.open_dataset(fname, engine='cfgrib')
        ds.to_netcdf(directory + dirname_today + '/' + d_today + 'T' + hour_utc_str_wind_waves + \
                     'Z_MSC_GDWPS_MZWPER_Sfc_LatLon0.25_PT' + str(wind_waves_ocean_hours[i]).zfill(3) + 'H.nc')
        ds.close()
        os.remove(directory + dirname_today + '/' + d_today + 'T' + hour_utc_str_wind_waves + \
                  'Z_MSC_GDWPS_MZWPER_Sfc_LatLon0.25_PT' + str(wind_waves_ocean_hours[i]).zfill(3) + 'H.grib2.5b7b6.idx')
        os.remove(fname)

    directory = rootpath_to_metdata + 'RIOPS_ocean_forecast_files/'

    for i in range(len(wind_waves_ocean_hours)):
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

        if si_toggle:
            url = 'https://dd.weather.gc.ca/model_riops/netcdf/forecast/polar_stereographic/2d/' + hour_utc_str_ocean + '/' + \
                  str(wind_waves_ocean_hours[i]).zfill(3) + '/' + d_today + 'T' + hour_utc_str_ocean + \
                  'Z_MSC_RIOPS_IICECONC_SFC_PS5km_P' + str(wind_waves_ocean_hours[i]).zfill(3) + '.nc'
            fname = directory + dirname_today + '/' + d_today + 'T' + hour_utc_str_ocean + \
                    'Z_MSC_RIOPS_IICECONC_SFC_PS5km_P' + str(wind_waves_ocean_hours[i]).zfill(3) + '.nc'
            flag = True

            while flag:
                try:
                    print('Obtaining forecast sea ice concentration file ' + d_today + 'T' + hour_utc_str_ocean + \
                          'Z_MSC_RIOPS_IICECONC_SFC_PS5km_P' + str(wind_waves_ocean_hours[i]).zfill(3) + '.nc')
                    r = requests.get(url, allow_redirects=True, timeout=5.0)
                    open(fname, 'wb').write(r.content)
                    flag = False
                except:
                    print('Error: could not download forecast sea ice concentration file ' + d_today + 'T' + hour_utc_str_ocean + \
                          'Z_MSC_RIOPS_IICECONC_SFC_PS5km_P' + str(wind_waves_ocean_hours[i]).zfill(3) + '.nc, retrying...')

            url = 'https://dd.weather.gc.ca/model_riops/netcdf/forecast/polar_stereographic/2d/' + hour_utc_str_ocean + '/' + \
                  str(wind_waves_ocean_hours[i]).zfill(3) + '/' + d_today + 'T' + hour_utc_str_ocean + \
                  'Z_MSC_RIOPS_IICEVOL_SFC_PS5km_P' + str(wind_waves_ocean_hours[i]).zfill(3) + '.nc'
            fname = directory + dirname_today + '/' + d_today + 'T' + hour_utc_str_ocean + \
                    'Z_MSC_RIOPS_IICEVOL_SFC_PS5km_P' + str(wind_waves_ocean_hours[i]).zfill(3) + '.nc'
            flag = True

            while flag:
                try:
                    print('Obtaining forecast sea ice thickness file ' + d_today + 'T' + hour_utc_str_ocean + \
                          'Z_MSC_RIOPS_IICEVOL_SFC_PS5km_P' + str(wind_waves_ocean_hours[i]).zfill(3) + '.nc')
                    r = requests.get(url, allow_redirects=True, timeout=5.0)
                    open(fname, 'wb').write(r.content)
                    flag = False
                except:
                    print('Error: could not download forecast sea ice thickness file ' + d_today + 'T' + hour_utc_str_ocean + \
                        'Z_MSC_RIOPS_IICEVOL_SFC_PS5km_P' + str(wind_waves_ocean_hours[i]).zfill(3) + '.nc, retrying...')

            url = 'https://dd.weather.gc.ca/model_riops/netcdf/forecast/polar_stereographic/2d/' + hour_utc_str_ocean + '/' + \
                  str(wind_waves_ocean_hours[i]).zfill(3) + '/' + d_today + 'T' + hour_utc_str_ocean + \
                  'Z_MSC_RIOPS_ITZOCRTX_SFC_PS5km_P' + str(wind_waves_ocean_hours[i]).zfill(3) + '.nc'
            fname = directory + dirname_today + '/' + d_today + 'T' + hour_utc_str_ocean + \
                    'Z_MSC_RIOPS_ITZOCRTX_SFC_PS5km_P' + str(wind_waves_ocean_hours[i]).zfill(3) + '.nc'
            flag = True

            while flag:
                try:
                    print('Obtaining forecast sea ice zonal velocity file ' + d_today + 'T' + hour_utc_str_ocean + \
                          'Z_MSC_RIOPS_ITZOCRTX_SFC_PS5km_P' + str(wind_waves_ocean_hours[i]).zfill(3) + '.nc')
                    r = requests.get(url, allow_redirects=True, timeout=5.0)
                    open(fname, 'wb').write(r.content)
                    flag = False
                except:
                    print('Error: could not download forecast sea ice zonal velocity file ' + d_today + 'T' + hour_utc_str_ocean + \
                        'Z_MSC_RIOPS_ITZOCRTX_SFC_PS5km_P' + str(wind_waves_ocean_hours[i]).zfill(3) + '.nc, retrying...')

            url = 'https://dd.weather.gc.ca/model_riops/netcdf/forecast/polar_stereographic/2d/' + hour_utc_str_ocean + '/' + \
                  str(wind_waves_ocean_hours[i]).zfill(3) + '/' + d_today + 'T' + hour_utc_str_ocean + \
                  'Z_MSC_RIOPS_ITMECRTY_SFC_PS5km_P' + str(wind_waves_ocean_hours[i]).zfill(3) + '.nc'
            fname = directory + dirname_today + '/' + d_today + 'T' + hour_utc_str_ocean + \
                    'Z_MSC_RIOPS_ITMECRTY_SFC_PS5km_P' + str(wind_waves_ocean_hours[i]).zfill(3) + '.nc'
            flag = True

            while flag:
                try:
                    print('Obtaining forecast sea ice meridional velocity file ' + d_today + 'T' + hour_utc_str_ocean + \
                          'Z_MSC_RIOPS_ITMECRTY_SFC_PS5km_P' + str(wind_waves_ocean_hours[i]).zfill(3) + '.nc')
                    r = requests.get(url, allow_redirects=True, timeout=5.0)
                    open(fname, 'wb').write(r.content)
                    flag = False
                except:
                    print('Error: could not download forecast sea ice meridional velocity file ' + d_today + 'T' + hour_utc_str_ocean + \
                        'Z_MSC_RIOPS_ITMECRTY_SFC_PS5km_P' + str(wind_waves_ocean_hours[i]).zfill(3) + '.nc, retrying...')

    try:
        if np.isnan(maximum_iceberg_length):
            maximum_iceberg_length = 100.
    except (TypeError, ValueError):
        maximum_iceberg_length = 100.

    max_ib_draft = 1.78 * (maximum_iceberg_length ** 0.71)
    fname = directory + dirname_today + '/' + d_today + 'T' + hour_utc_str_ocean + \
        'Z_MSC_RIOPS_VOZOCRTX_DBS-all_PS5km_P' + str(wind_waves_ocean_hours[0]).zfill(3) + '.nc'
    ocean_data = nc.Dataset(fname)
    ocean_lat = ocean_data.variables['latitude'][:]
    ocean_lon = ocean_data.variables['longitude'][:]
    ocean_depth = ocean_data.variables['depth'][:]
    ocean_depth = ocean_depth.flatten()
    ocean_data.close()
    lat_indices = np.where((ocean_lat[:, 0] >= minimum_latitude) & (ocean_lat[:, 0] <= maximum_latitude))[0]
    lon_indices = np.where((ocean_lon[0, :] >= minimum_longitude + 360.) & (ocean_lon[0, :] <= maximum_longitude + 360.))[0]
    ocean_lat = ocean_lat[np.min(lat_indices):np.max(lat_indices) + 1, np.min(lon_indices):np.max(lon_indices) + 1]
    ocean_lon = ocean_lon[np.min(lat_indices):np.max(lat_indices) + 1, np.min(lon_indices):np.max(lon_indices) + 1]
    loc_depth = np.argwhere(ocean_depth <= max_ib_draft).flatten()

    if loc_depth[-1] + 1 < len(ocean_depth):
        loc_depth = np.append(loc_depth, loc_depth[-1] + 1)

    ocean_depth = ocean_depth[loc_depth]
    ssh_grad_x_lat = np.empty((len(ocean_lat[:, 0]), len(ocean_lon[0, :]) - 1))
    ssh_grad_y_lat = np.empty((len(ocean_lat[:, 0]) - 1, len(ocean_lon[0, :])))
    ssh_grad_x_lon = np.empty((len(ocean_lat[:, 0]), len(ocean_lon[0, :]) - 1))
    ssh_grad_y_lon = np.empty((len(ocean_lat[:, 0]) - 1, len(ocean_lon[0, :])))

    for k in range(len(ocean_lat[:, 0])):
        for n in range(len(ocean_lon[0, :]) - 1):
            grid_pt_dist, grid_pt_bearing = dist_bearing(Re, ocean_lat[k, n], ocean_lat[k, n + 1], ocean_lon[k, n], ocean_lon[k, n + 1])
            ssh_grad_x_lat[k, n], ssh_grad_x_lon[k, n] = dist_course(Re, ocean_lat[k, n], ocean_lon[k, n], grid_pt_dist / 2., grid_pt_bearing)

    for k in range(len(ocean_lat[:, 0]) - 1):
        for n in range(len(ocean_lon[0, :])):
            grid_pt_dist, grid_pt_bearing = dist_bearing(Re, ocean_lat[k, n], ocean_lat[k + 1, n], ocean_lon[k, n], ocean_lon[k + 1, n])
            ssh_grad_y_lat[k, n], ssh_grad_y_lon[k, n] = dist_course(Re, ocean_lat[k, n], ocean_lon[k, n], grid_pt_dist / 2., grid_pt_bearing)

    for i in range(len(wind_waves_ocean_hours)):
        fname = rootpath_to_metdata + 'RIOPS_ocean_forecast_files/' + dirname_today + '/' + d_today + 'T' + hour_utc_str_ocean + \
                'Z_MSC_RIOPS_VOZOCRTX_DBS-all_PS5km_P' + str(wind_waves_ocean_hours[i]).zfill(3) + '.nc'
        print('Shrinking forecast zonal ocean current file ' + d_today + 'T' + hour_utc_str_ocean + \
              'Z_MSC_RIOPS_VOZOCRTX_DBS-all_PS5km_P' + str(wind_waves_ocean_hours[i]).zfill(3) + '.nc')
        u_curr_data = nc.Dataset(fname)
        u_curr = np.squeeze(u_curr_data.variables['vozocrtx'][:])
        u_curr_data.close()
        u_curr = u_curr[loc_depth, np.min(lat_indices):np.max(lat_indices) + 1, np.min(lon_indices):np.max(lon_indices) + 1]
        fname = (directory + '/' + dirname_today + '/' + d_today + 'T' + hour_utc_str_ocean + 'Z_MSC_RIOPS_VOZOCRTX_DBS-all_PS5km_P' +
                 str(wind_waves_ocean_hours[i]).zfill(3) + '.nc')

        with nc.Dataset(fname, 'w', format='NETCDF4') as ncfile:
            ncfile.createDimension('latitude', len(ocean_lat[:, 0]))
            ncfile.createDimension('longitude', len(ocean_lon[0, :]))
            ncfile.createDimension('depth', len(ocean_depth))

            latitude_var = ncfile.createVariable('latitude', 'f8', ('latitude', 'longitude',))
            longitude_var = ncfile.createVariable('longitude', 'f8', ('latitude', 'longitude',))
            depth_var = ncfile.createVariable('depth', 'f8', ('depth',))
            u_curr_var = ncfile.createVariable('vozocrtx', 'f4', ('depth', 'latitude', 'longitude',))

            latitude_var[:] = ocean_lat
            longitude_var[:] = ocean_lon
            depth_var[:] = ocean_depth
            u_curr_var[:] = u_curr

        fname = rootpath_to_metdata + 'RIOPS_ocean_forecast_files/' + dirname_today + '/' + d_today + 'T' + hour_utc_str_ocean + \
                'Z_MSC_RIOPS_VOMECRTY_DBS-all_PS5km_P' + str(wind_waves_ocean_hours[i]).zfill(3) + '.nc'
        print('Shrinking forecast meridional ocean current file ' + d_today + 'T' + hour_utc_str_ocean + \
              'Z_MSC_RIOPS_VOMECRTY_DBS-all_PS5km_P' + str(wind_waves_ocean_hours[i]).zfill(3) + '.nc')
        v_curr_data = nc.Dataset(fname)
        v_curr = np.squeeze(v_curr_data.variables['vomecrty'][:])
        v_curr_data.close()
        v_curr = v_curr[loc_depth, np.min(lat_indices):np.max(lat_indices) + 1, np.min(lon_indices):np.max(lon_indices) + 1]
        fname = (directory + '/' + dirname_today + '/' + d_today + 'T' + hour_utc_str_ocean + 'Z_MSC_RIOPS_VOMECRTY_DBS-all_PS5km_P' +
                 str(wind_waves_ocean_hours[i]).zfill(3) + '.nc')

        with nc.Dataset(fname, 'w', format='NETCDF4') as ncfile:
            ncfile.createDimension('latitude', len(ocean_lat[:, 0]))
            ncfile.createDimension('longitude', len(ocean_lon[0, :]))
            ncfile.createDimension('depth', len(ocean_depth))

            latitude_var = ncfile.createVariable('latitude', 'f8', ('latitude', 'longitude',))
            longitude_var = ncfile.createVariable('longitude', 'f8', ('latitude', 'longitude',))
            depth_var = ncfile.createVariable('depth', 'f8', ('depth',))
            v_curr_var = ncfile.createVariable('vomecrty', 'f4', ('depth', 'latitude', 'longitude',))

            latitude_var[:] = ocean_lat
            longitude_var[:] = ocean_lon
            depth_var[:] = ocean_depth
            v_curr_var[:] = v_curr

        fname = rootpath_to_metdata + 'RIOPS_ocean_forecast_files/' + dirname_today + '/' + d_today + 'T' + hour_utc_str_ocean + \
                'Z_MSC_RIOPS_VOTEMPER_DBS-all_PS5km_P' + str(wind_waves_ocean_hours[i]).zfill(3) + '.nc'
        print('Shrinking forecast ocean potential temperature file ' + d_today + 'T' + hour_utc_str_ocean + \
              'Z_MSC_RIOPS_VOTEMPER_DBS-all_PS5km_P' + str(wind_waves_ocean_hours[i]).zfill(3) + '.nc')
        water_pot_temp_data = nc.Dataset(fname)
        water_pot_temp = np.squeeze(water_pot_temp_data.variables['votemper'][:]) - 273.15
        water_pot_temp_data.close()
        water_pot_temp = water_pot_temp[loc_depth, np.min(lat_indices):np.max(lat_indices) + 1, np.min(lon_indices):np.max(lon_indices) + 1]
        fname = (directory + '/' + dirname_today + '/' + d_today + 'T' + hour_utc_str_ocean + 'Z_MSC_RIOPS_VOTEMPER_DBS-all_PS5km_P' +
                 str(wind_waves_ocean_hours[i]).zfill(3) + '.nc')

        with nc.Dataset(fname, 'w', format='NETCDF4') as ncfile:
            ncfile.createDimension('latitude', len(ocean_lat[:, 0]))
            ncfile.createDimension('longitude', len(ocean_lon[0, :]))
            ncfile.createDimension('depth', len(ocean_depth))

            latitude_var = ncfile.createVariable('latitude', 'f8', ('latitude', 'longitude',))
            longitude_var = ncfile.createVariable('longitude', 'f8', ('latitude', 'longitude',))
            depth_var = ncfile.createVariable('depth', 'f8', ('depth',))
            water_pot_temp_var = ncfile.createVariable('votemper', 'f4', ('depth', 'latitude', 'longitude',))

            latitude_var[:] = ocean_lat
            longitude_var[:] = ocean_lon
            depth_var[:] = ocean_depth
            water_pot_temp_var[:] = water_pot_temp

        fname = rootpath_to_metdata + 'RIOPS_ocean_forecast_files/' + dirname_today + '/' + d_today + 'T' + hour_utc_str_ocean + \
                'Z_MSC_RIOPS_VOSALINE_DBS-all_PS5km_P' + str(wind_waves_ocean_hours[i]).zfill(3) + '.nc'
        print('Shrinking forecast ocean salinity file ' + d_today + 'T' + hour_utc_str_ocean + \
              'Z_MSC_RIOPS_VOSALINE_DBS-all_PS5km_P' + str(wind_waves_ocean_hours[i]).zfill(3) + '.nc')
        water_sal_data = nc.Dataset(fname)
        water_sal = np.squeeze(water_sal_data.variables['vosaline'][:])
        water_sal_data.close()
        water_sal = water_sal[loc_depth, np.min(lat_indices):np.max(lat_indices) + 1, np.min(lon_indices):np.max(lon_indices) + 1]
        fname = (directory + '/' + dirname_today + '/' + d_today + 'T' + hour_utc_str_ocean + 'Z_MSC_RIOPS_VOSALINE_DBS-all_PS5km_P' +
                 str(wind_waves_ocean_hours[i]).zfill(3) + '.nc')

        with nc.Dataset(fname, 'w', format='NETCDF4') as ncfile:
            ncfile.createDimension('latitude', len(ocean_lat[:, 0]))
            ncfile.createDimension('longitude', len(ocean_lon[0, :]))
            ncfile.createDimension('depth', len(ocean_depth))

            latitude_var = ncfile.createVariable('latitude', 'f8', ('latitude', 'longitude',))
            longitude_var = ncfile.createVariable('longitude', 'f8', ('latitude', 'longitude',))
            depth_var = ncfile.createVariable('depth', 'f8', ('depth',))
            water_sal_var = ncfile.createVariable('vosaline', 'f4', ('depth', 'latitude', 'longitude',))

            latitude_var[:] = ocean_lat
            longitude_var[:] = ocean_lon
            depth_var[:] = ocean_depth
            water_sal_var[:] = water_sal

        fname = rootpath_to_metdata + 'RIOPS_ocean_forecast_files/' + dirname_today + '/' + d_today + 'T' + hour_utc_str_ocean + \
                'Z_MSC_RIOPS_SOSSHEIG_SFC_PS5km_P' + str(wind_waves_ocean_hours[i]).zfill(3) + '.nc'
        print('Computing forecast sea surface height gradients and writing file ' + d_today + 'T' + hour_utc_str_ocean + \
              'Z_MSC_RIOPS_SOSSHEIG_SFC_GRAD_PS5km_P' + str(wind_waves_ocean_hours[i]).zfill(3) + '.nc')
        ssh_data = nc.Dataset(fname)
        ssh = np.squeeze(ssh_data.variables['sossheig'][:])
        ssh_data.close()
        ssh = ssh[np.min(lat_indices):np.max(lat_indices) + 1, np.min(lon_indices):np.max(lon_indices) + 1]
        ssh_grad_x = np.empty((len(ocean_lat[:, 0]), len(ocean_lon[0, :]) - 1))
        ssh_grad_y = np.empty((len(ocean_lat[:, 0]) - 1, len(ocean_lon[0, :])))

        for k in range(len(ocean_lat[:, 0])):
            for n in range(len(ocean_lon[0, :]) - 1):
                grid_pt_dist, grid_pt_bearing = dist_bearing(Re, ocean_lat[k, n], ocean_lat[k, n + 1], ocean_lon[k, n], ocean_lon[k, n + 1])
                ssh_grad = (ssh[k, n + 1] - ssh[k, n]) / grid_pt_dist
                ssh_grad_x[k, n] = ssh_grad * np.sin(np.deg2rad(grid_pt_bearing))

        for k in range(len(ocean_lat[:, 0]) - 1):
            for n in range(len(ocean_lon[0, :])):
                grid_pt_dist, grid_pt_bearing = dist_bearing(Re, ocean_lat[k, n], ocean_lat[k + 1, n], ocean_lon[k, n], ocean_lon[k + 1, n])
                ssh_grad = (ssh[k + 1, n] - ssh[k, n]) / grid_pt_dist
                ssh_grad_y[k, n] = -ssh_grad * np.cos(np.deg2rad(grid_pt_bearing))

        fname = (directory + '/' + dirname_today + '/' + d_today + 'T' + hour_utc_str_ocean + 'Z_MSC_RIOPS_SOSSHEIG_SFC_GRAD_PS5km_P' +
                 str(wind_waves_ocean_hours[i]).zfill(3) + '.nc')

        with nc.Dataset(fname, 'w', format='NETCDF4') as ncfile:
            ncfile.createDimension('x_gradient_latitude', len(ocean_lat[:, 0]))
            ncfile.createDimension('x_gradient_longitude', len(ocean_lon[0, :]) - 1)
            ncfile.createDimension('y_gradient_latitude', len(ocean_lat[:, 0]) - 1)
            ncfile.createDimension('y_gradient_longitude', len(ocean_lon[0, :]))

            x_gradient_latitude_var = ncfile.createVariable('ssh_grad_x_lat', 'f8', ('x_gradient_latitude', 'x_gradient_longitude',))
            x_gradient_longitude_var = ncfile.createVariable('ssh_grad_x_lon', 'f8', ('x_gradient_latitude', 'x_gradient_longitude',))
            y_gradient_latitude_var = ncfile.createVariable('ssh_grad_y_lat', 'f8', ('y_gradient_latitude', 'y_gradient_longitude',))
            y_gradient_longitude_var = ncfile.createVariable('ssh_grad_y_lon', 'f8', ('y_gradient_latitude', 'y_gradient_longitude',))
            ssh_grad_x_var = ncfile.createVariable('ssh_grad_x', 'f4', ('x_gradient_latitude', 'x_gradient_longitude',))
            ssh_grad_y_var = ncfile.createVariable('ssh_grad_y', 'f4', ('y_gradient_latitude', 'y_gradient_longitude',))

            x_gradient_latitude_var[:] = ssh_grad_x_lat
            x_gradient_longitude_var[:] = ssh_grad_x_lon
            y_gradient_latitude_var[:] = ssh_grad_y_lat
            y_gradient_longitude_var[:] = ssh_grad_y_lon
            ssh_grad_x_var[:] = ssh_grad_x
            ssh_grad_y_var[:] = ssh_grad_y

        os.remove(rootpath_to_metdata + 'RIOPS_ocean_forecast_files/' + dirname_today + '/' + d_today + 'T' + hour_utc_str_ocean + \
            'Z_MSC_RIOPS_SOSSHEIG_SFC_PS5km_P' + str(wind_waves_ocean_hours[i]).zfill(3) + '.nc')

        if si_toggle:
            fname = rootpath_to_metdata + 'RIOPS_ocean_forecast_files/' + dirname_today + '/' + d_today + 'T' + hour_utc_str_ocean + \
                    'Z_MSC_RIOPS_IICECONC_SFC_PS5km_P' + str(wind_waves_ocean_hours[i]).zfill(3) + '.nc'
            print('Shrinking forecast sea ice concentration file ' + d_today + 'T' + hour_utc_str_ocean + \
                  'Z_MSC_RIOPS_IICECONC_SFC_PS5km_P' + str(wind_waves_ocean_hours[i]).zfill(3) + '.nc')
            siconc_data = nc.Dataset(fname)
            siconc = np.squeeze(siconc_data.variables['iiceconc'][:])
            siconc_data.close()
            siconc = siconc[np.min(lat_indices):np.max(lat_indices) + 1, np.min(lon_indices):np.max(lon_indices) + 1]
            fname = (directory + '/' + dirname_today + '/' + d_today + 'T' + hour_utc_str_ocean + 'Z_MSC_RIOPS_IICECONC_SFC_PS5km_P' +
                     str(wind_waves_ocean_hours[i]).zfill(3) + '.nc')

            with nc.Dataset(fname, 'w', format='NETCDF4') as ncfile:
                ncfile.createDimension('latitude', len(ocean_lat[:, 0]))
                ncfile.createDimension('longitude', len(ocean_lon[0, :]))

                latitude_var = ncfile.createVariable('latitude', 'f8', ('latitude', 'longitude',))
                longitude_var = ncfile.createVariable('longitude', 'f8', ('latitude', 'longitude',))
                siconc_var = ncfile.createVariable('iiceconc', 'f4', ('latitude', 'longitude',))

                latitude_var[:] = ocean_lat
                longitude_var[:] = ocean_lon
                siconc_var[:] = siconc

            fname = rootpath_to_metdata + 'RIOPS_ocean_forecast_files/' + dirname_today + '/' + d_today + 'T' + hour_utc_str_ocean + \
                    'Z_MSC_RIOPS_IICEVOL_SFC_PS5km_P' + str(wind_waves_ocean_hours[i]).zfill(3) + '.nc'
            print('Shrinking forecast sea ice thickness file ' + d_today + 'T' + hour_utc_str_ocean + \
                  'Z_MSC_RIOPS_IICEVOL_SFC_PS5km_P' + str(wind_waves_ocean_hours[i]).zfill(3) + '.nc')
            sithick_data = nc.Dataset(fname)
            sithick = np.squeeze(sithick_data.variables['iicevol'][:])
            sithick_data.close()
            sithick = sithick[np.min(lat_indices):np.max(lat_indices) + 1, np.min(lon_indices):np.max(lon_indices) + 1]
            fname = (directory + '/' + dirname_today + '/' + d_today + 'T' + hour_utc_str_ocean + 'Z_MSC_RIOPS_IICEVOL_SFC_PS5km_P' +
                     str(wind_waves_ocean_hours[i]).zfill(3) + '.nc')

            with nc.Dataset(fname, 'w', format='NETCDF4') as ncfile:
                ncfile.createDimension('latitude', len(ocean_lat[:, 0]))
                ncfile.createDimension('longitude', len(ocean_lon[0, :]))

                latitude_var = ncfile.createVariable('latitude', 'f8', ('latitude', 'longitude',))
                longitude_var = ncfile.createVariable('longitude', 'f8', ('latitude', 'longitude',))
                sithick_var = ncfile.createVariable('iicevol', 'f4', ('latitude', 'longitude',))

                latitude_var[:] = ocean_lat
                longitude_var[:] = ocean_lon
                sithick_var[:] = sithick

            fname = rootpath_to_metdata + 'RIOPS_ocean_forecast_files/' + dirname_today + '/' + d_today + 'T' + hour_utc_str_ocean + \
                    'Z_MSC_RIOPS_ITZOCRTX_SFC_PS5km_P' + str(wind_waves_ocean_hours[i]).zfill(3) + '.nc'
            print('Shrinking forecast sea ice zonal velocity file ' + d_today + 'T' + hour_utc_str_ocean + \
                  'Z_MSC_RIOPS_ITZOCRTX_SFC_PS5km_P' + str(wind_waves_ocean_hours[i]).zfill(3) + '.nc')
            usi_data = nc.Dataset(fname)
            usi = np.squeeze(usi_data.variables['itzocrtx'][:])
            usi_data.close()
            usi = usi[np.min(lat_indices):np.max(lat_indices) + 1, np.min(lon_indices):np.max(lon_indices) + 1]
            fname = (directory + '/' + dirname_today + '/' + d_today + 'T' + hour_utc_str_ocean + 'Z_MSC_RIOPS_ITZOCRTX_SFC_PS5km_P' +
                     str(wind_waves_ocean_hours[i]).zfill(3) + '.nc')

            with nc.Dataset(fname, 'w', format='NETCDF4') as ncfile:
                ncfile.createDimension('latitude', len(ocean_lat[:, 0]))
                ncfile.createDimension('longitude', len(ocean_lon[0, :]))

                latitude_var = ncfile.createVariable('latitude', 'f8', ('latitude', 'longitude',))
                longitude_var = ncfile.createVariable('longitude', 'f8', ('latitude', 'longitude',))
                usi_var = ncfile.createVariable('itzocrtx', 'f4', ('latitude', 'longitude',))

                latitude_var[:] = ocean_lat
                longitude_var[:] = ocean_lon
                usi_var[:] = usi

            fname = rootpath_to_metdata + 'RIOPS_ocean_forecast_files/' + dirname_today + '/' + d_today + 'T' + hour_utc_str_ocean + \
                    'Z_MSC_RIOPS_ITMECRTY_SFC_PS5km_P' + str(wind_waves_ocean_hours[i]).zfill(3) + '.nc'
            print('Shrinking forecast sea ice meridional velocity file ' + d_today + 'T' + hour_utc_str_ocean + \
                  'Z_MSC_RIOPS_ITMECRTY_SFC_PS5km_P' + str(wind_waves_ocean_hours[i]).zfill(3) + '.nc')
            vsi_data = nc.Dataset(fname)
            vsi = np.squeeze(vsi_data.variables['itmecrty'][:])
            vsi_data.close()
            vsi = vsi[np.min(lat_indices):np.max(lat_indices) + 1, np.min(lon_indices):np.max(lon_indices) + 1]
            fname = (directory + '/' + dirname_today + '/' + d_today + 'T' + hour_utc_str_ocean + 'Z_MSC_RIOPS_ITMECRTY_SFC_PS5km_P' +
                     str(wind_waves_ocean_hours[i]).zfill(3) + '.nc')

            with nc.Dataset(fname, 'w', format='NETCDF4') as ncfile:
                ncfile.createDimension('latitude', len(ocean_lat[:, 0]))
                ncfile.createDimension('longitude', len(ocean_lon[0, :]))

                latitude_var = ncfile.createVariable('latitude', 'f8', ('latitude', 'longitude',))
                longitude_var = ncfile.createVariable('longitude', 'f8', ('latitude', 'longitude',))
                vsi_var = ncfile.createVariable('itmecrty', 'f4', ('latitude', 'longitude',))

                latitude_var[:] = ocean_lat
                longitude_var[:] = ocean_lon
                vsi_var[:] = vsi

date = str(np.datetime64('today'))
forecast_hours = 84
minimum_longitude = -64.5
maximum_longitude = -46.75
minimum_latitude = 48
maximum_latitude = 61.2
maximum_iceberg_length = 100.
si_toggle = True

get_rcm_metocean_data(date, forecast_hours, minimum_longitude, maximum_longitude, minimum_latitude, maximum_latitude, maximum_iceberg_length, si_toggle)


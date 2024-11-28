
from scipy.interpolate import RegularGridInterpolator, griddata, interp1d
from scipy.integrate import solve_ivp
from tempfile import TemporaryDirectory
from subprocess import run
import numpy as np
import netCDF4 as nc
import requests
import shutil
import os

def rcm_iceberg_drift_forecaster(iceberg_lat0, iceberg_lon0, rcm_datetime0, iceberg_length, grounded_status, next_rcm_time):
    use_temporary_directory = True
    wgrib_path = './wgrib/'
    bathy_data_path = './GEBCO_Bathymetric_Data/gebco_2024.nc'
    deg_radius = 10
    g = 9.80665
    rho_water = 1023.6
    rho_air = 1.225
    rho_ice = 910.
    omega = 7.292115e-5
    Cw = 0.7867
    Ca = 1.1857
    C_wave = 0.6
    am = 0.5
    Re = 6371e3

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

    def iceberg_acc(iceberg_lat, iceberg_u, iceberg_v, iceberg_sail, iceberg_draft, iceberg_length, iceberg_mass, dt, am, omega, Cw, Ca, C_wave, g, rho_air, rho_water,
                      u_wind, v_wind, u_curr, v_curr, ssh_grad_x, ssh_grad_y, Hs, wave_dir):
        iceberg_keel = iceberg_length * iceberg_draft
        wind_dir = 90. - np.rad2deg(np.arctan2(v_wind, u_wind))

        if wind_dir < 0:
            wind_dir = wind_dir + 360.

        if np.isnan(wave_dir):
            wave_dir = wind_dir

        Fa_E = 0.5 * rho_air * Ca * iceberg_sail * np.sqrt((u_wind - iceberg_u) ** 2 + (v_wind - iceberg_v) ** 2) * (u_wind - iceberg_u)
        Fw_E = 0.5 * rho_water * Cw * iceberg_keel * np.sqrt((u_curr[0] - iceberg_u) ** 2 + (v_curr[0] - iceberg_v) ** 2) * (u_curr[0] - iceberg_u)
        f = 2. * omega * np.sin(np.deg2rad(iceberg_lat))
        Fc_E = (iceberg_mass + am * iceberg_mass) * f * iceberg_v
        Fp_E = (iceberg_mass + am * iceberg_mass) * ((u_curr[1] - u_curr[0]) / dt + f * v_curr[0])
        Fs_E = -(iceberg_mass + am * iceberg_mass) * g * ssh_grad_x
        Fr_E = 0.25 * C_wave * rho_water * g * ((0.5 * Hs) ** 2) * iceberg_length * np.sin(np.deg2rad(wave_dir))
        Fa_N = 0.5 * rho_air * Ca * iceberg_sail * np.sqrt((u_wind - iceberg_u) ** 2 + (v_wind - iceberg_v) ** 2) * (v_wind - iceberg_v)
        Fw_N = 0.5 * rho_water * Cw * iceberg_keel * np.sqrt((u_curr[0] - iceberg_u) ** 2 + (v_curr[0] - iceberg_v) ** 2) * (v_curr[0] - iceberg_v)
        Fc_N = -(iceberg_mass + am * iceberg_mass) * f * iceberg_u
        Fp_N = (iceberg_mass + am * iceberg_mass) * ((v_curr[1] - v_curr[0]) / dt - f * u_curr[0])
        Fs_N = -(iceberg_mass + am * iceberg_mass) * g * ssh_grad_y
        Fr_N = 0.25 * C_wave * rho_water * g * ((0.5 * Hs) ** 2) * iceberg_length * np.cos(np.deg2rad(wave_dir))
        F_sum_E = Fa_E + Fw_E + Fc_E + Fp_E + Fs_E + Fr_E
        F_sum_N = Fa_N + Fw_N + Fc_N + Fp_N + Fs_N + Fr_N
        ib_acc_E = F_sum_E / (iceberg_mass + am * iceberg_mass)
        ib_acc_N = F_sum_N / (iceberg_mass + am * iceberg_mass)
        return ib_acc_E, ib_acc_N

    iceberg_u0 = 0.
    iceberg_v0 = 0.
    rcm_datetime0 = np.datetime64(rcm_datetime0)

    if not isinstance(iceberg_length, (int, float)) or iceberg_length <= 0:
        iceberg_length = 100.

    iceberg_draft = 1.78 * (iceberg_length ** 0.71) # meters
    # iceberg_mass = 0.43 * (iceberg_length ** 2.9) * 1000. # kg
    iceberg_mass = 0.45 * rho_ice * (iceberg_length ** 3) # kg
    iceberg_sail = 0.077 * (iceberg_length ** 2) # m ** 2
    bathy_data = nc.Dataset(bathy_data_path)
    bathy_lat = bathy_data.variables['lat'][:]
    bathy_lon = bathy_data.variables['lon'][:]
    bathy_depth = -bathy_data.variables['elevation'][:]  # lat x lon
    bathy_data.close()
    bathy_interp = RegularGridInterpolator((bathy_lat, bathy_lon), bathy_depth, method='linear', bounds_error=True, fill_value=np.nan)
    iceberg_bathy_depth0 = bathy_interp([[iceberg_lat0, iceberg_lon0]])[0]

    if grounded_status == 'not grounded' and iceberg_bathy_depth0 <= iceberg_draft:
        iceberg_draft = iceberg_bathy_depth0 - 1.

    next_rcm_time = np.datetime64(next_rcm_time)
    forecast_time = rcm_datetime0
    # dirname_wind_waves = str(np.datetime64('today'))
    dirname_wind_waves = np.datetime_as_string(forecast_time, unit='D')
    d_wind_waves = dirname_wind_waves.replace('-', '')
    # dirname_curr_ssh = str(np.datetime64('today'))
    dirname_curr_ssh = np.datetime_as_string(forecast_time, unit='D')
    d_curr_ssh = dirname_curr_ssh.replace('-', '')

    iceberg_times = [forecast_time]

    # Add hourly intervals until close to the end time
    current_time = forecast_time

    while current_time + np.timedelta64(1, 'h') < next_rcm_time:
        current_time += np.timedelta64(1, 'h')
        iceberg_times.append(current_time)

    # Append the exact end time for the remainder interval
    if iceberg_times[-1] < next_rcm_time:
        iceberg_times.append(next_rcm_time)

    # Convert to list for easier inspection, if desired
    iceberg_times = list(iceberg_times)
    iceberg_times_dt = [float((iceberg_times[i + 1] - iceberg_times[i]) / np.timedelta64(1, 's')) for i in
                        range(len(iceberg_times) - 1)]

    # Convert to list for easier inspection if desired
    iceberg_times_dt = list(iceberg_times_dt)

    iceberg_lats = np.empty((len(iceberg_times),))
    iceberg_lons = np.empty((len(iceberg_times),))

    if grounded_status == 'not grounded':
        # url = 'https://dd.meteo.gc.ca/model_gem_global/15km/grib2/lat_lon/12/240/CMC_glb_UGRD_TGL_10_latlon.15x.15_' + d_wind_waves + '12_P240.grib2'
        # response = requests.head(url)
        #
        # if response.status_code == 200 and forecast_time >= np.datetime64(str(np.datetime64('today')) + 'T12:00:00'):
        #     hour_utc_str_wind = '12'
        # else:
        #     hour_utc_str_wind = '00'

        url = 'https://dd.weather.gc.ca/model_riops/netcdf/forecast/polar_stereographic/3d/18/084/' + d_curr_ssh + 'T18Z_MSC_RIOPS_VOZOCRTX_DBS-all_PS5km_P084.nc'
        response = requests.head(url)

        if response.status_code == 200 and forecast_time >= np.datetime64(dirname_curr_ssh + 'T18:00:00'):
            hour_utc_str_curr_ssh = '18'
        else:
            url = 'https://dd.weather.gc.ca/model_riops/netcdf/forecast/polar_stereographic/3d/12/084/' + d_curr_ssh + 'T12Z_MSC_RIOPS_VOZOCRTX_DBS-all_PS5km_P084.nc'
            response = requests.head(url)

            if response.status_code == 200 and forecast_time >= np.datetime64(dirname_curr_ssh + 'T12:00:00'):
                hour_utc_str_curr_ssh = '12'
            else:
                url = ('https://dd.weather.gc.ca/model_riops/netcdf/forecast/polar_stereographic/3d/06/084/' + d_curr_ssh +
                       'T06Z_MSC_RIOPS_VOZOCRTX_DBS-all_PS5km_P084.nc')
                response = requests.head(url)

                if response.status_code == 200 and forecast_time >= np.datetime64(dirname_curr_ssh + 'T06:00:00'):
                    hour_utc_str_curr_ssh = '06'
                else:
                    hour_utc_str_curr_ssh = '00'

        url = 'https://dd.weather.gc.ca/model_gdwps/25km/12/' + d_wind_waves + 'T12Z_MSC_GDWPS_HTSGW_Sfc_LatLon0.25_PT240H.grib2'
        response = requests.head(url)

        if response.status_code == 200 and forecast_time >= np.datetime64(dirname_wind_waves + 'T12:00:00'):
            hour_utc_str_waves = '12'
            hour_utc_str_wind = '12'
        else:
            hour_utc_str_waves = '00'
            hour_utc_str_wind = '00'

        forecast_times_wind = []
        forecast_times_waves = []
        forecast_times_curr_ssh = []
        forecast_start_time_wind = np.datetime64(dirname_wind_waves + 'T' + hour_utc_str_wind + ':00:00')
        forecast_start_time_waves = np.datetime64(dirname_wind_waves + 'T' + hour_utc_str_waves + ':00:00')
        forecast_start_time_curr_ssh = np.datetime64(dirname_curr_ssh + 'T' + hour_utc_str_curr_ssh + ':00:00')
        time_count_wind = forecast_start_time_wind
        time_count_waves = forecast_start_time_waves
        time_count_curr_ssh = forecast_start_time_curr_ssh

        while time_count_wind <= next_rcm_time + np.timedelta64(1, 'h'):
            forecast_times_wind.append(time_count_wind)
            time_count_wind = time_count_wind + np.timedelta64(1, 'h')

        while time_count_waves <= next_rcm_time + np.timedelta64(1, 'h'):
            forecast_times_waves.append(time_count_waves)
            time_count_waves = time_count_waves + np.timedelta64(1, 'h')

        while time_count_curr_ssh <= next_rcm_time + np.timedelta64(1, 'h'):
            forecast_times_curr_ssh.append(time_count_curr_ssh)
            time_count_curr_ssh = time_count_curr_ssh + np.timedelta64(1, 'h')

        forecast_times_wind = np.array(forecast_times_wind, dtype='datetime64[s]')
        forecast_times_waves = np.array(forecast_times_waves, dtype='datetime64[s]')
        forecast_times_curr_ssh = np.array(forecast_times_curr_ssh, dtype='datetime64[s]')

        forecast_times_wind_hours = (forecast_times_wind.astype('datetime64[h]') - forecast_start_time_wind.astype('datetime64[h]')).astype(int)
        forecast_times_waves_hours = (forecast_times_waves.astype('datetime64[h]') - forecast_start_time_waves.astype('datetime64[h]')).astype(int)
        forecast_times_curr_ssh_hours = (forecast_times_curr_ssh.astype('datetime64[h]') - forecast_start_time_curr_ssh.astype('datetime64[h]')).astype(int)

        with (TemporaryDirectory() as directory):
            if not use_temporary_directory:
                directory = './GDPS_wind_forecast_grib2_files'

                if not os.path.isdir('./GDPS_wind_forecast_grib2_files/'):
                    os.mkdir('./GDPS_wind_forecast_grib2_files/')

                if not os.path.isdir('./GDPS_wind_forecast_netcdf_files/'):
                    os.mkdir('./GDPS_wind_forecast_netcdf_files/')

                if not os.path.isdir('./GDWPS_wave_forecast_grib2_files/'):
                    os.mkdir('./GDWPS_wave_forecast_grib2_files/')

                if not os.path.isdir('./GDWPS_wave_forecast_netcdf_files/'):
                    os.mkdir('./GDWPS_wave_forecast_netcdf_files/')

                if not os.path.isdir('./RIOPS_ocean_forecast_netcdf_files/'):
                    os.mkdir('./RIOPS_ocean_forecast_netcdf_files/')

            if not os.path.isdir(directory + '/' + dirname_wind_waves):
                os.mkdir(directory + '/' + dirname_wind_waves)

            for i in range(len(forecast_times_wind)):
                # url = 'https://dd.meteo.gc.ca/model_gem_global/15km/grib2/lat_lon/' + hour_utc_str_wind + '/' + \
                # str(forecast_times_wind_hours[i]).zfill(3) + '/CMC_glb_UGRD_TGL_10_latlon.15x.15_' + \
                # d_wind_waves + hour_utc_str_wind + '_P' + str(forecast_times_wind_hours[i]).zfill(3) + '.grib2'
                # fname = directory + '/' + dirname_wind_waves + '/CMC_glb_UGRD_TGL_10_latlon.15x.15_' + d_wind_waves + hour_utc_str_wind + '_P' + \
                #         str(forecast_times_wind_hours[i]).zfill(3) + '.grib2'

                url = 'https://dd.weather.gc.ca/model_gdwps/25km/' + hour_utc_str_wind + '/' + d_wind_waves + 'T' + hour_utc_str_wind + \
                      'Z_MSC_GDWPS_UGRD_AGL-10m_LatLon0.25_PT' + str(forecast_times_wind_hours[i]).zfill(3) + 'H.grib2'
                fname = directory + '/' + dirname_wind_waves + '/' + d_wind_waves + 'T' + hour_utc_str_wind + \
                        'Z_MSC_GDWPS_UGRD_AGL-10m_LatLon0.25_PT' + str(forecast_times_wind_hours[i]).zfill(3) + 'H.grib2'
                flag = True

                while flag:
                    try:
                        print('Obtaining forecast zonal wind velocity file ' + fname)
                        r = requests.get(url, allow_redirects=True, timeout=5.0)
                        open(fname, 'wb').write(r.content)
                        flag = False
                    except:
                        print('Error: could not download forecast zonal wind velocity file, retrying...')

                # fname = directory + '/' + dirname_wind_waves + '/CMC_glb_UGRD_TGL_10_latlon.15x.15_' + d_wind_waves + hour_utc_str_wind + '_P' + \
                #         str(forecast_times_wind_hours[i]).zfill(3) + '.grib2'
                # run(wgrib_path + 'wgrib2.exe ' + fname + ' -netcdf ' + directory + '/' + dirname_wind_waves + '/CMC_glb_UGRD_TGL_10_latlon.15x.15_' + \
                #     d_wind_waves + hour_utc_str_wind + '_P' + str(forecast_times_wind_hours[i]).zfill(3) + '.nc')

                fname = directory + '/' + dirname_wind_waves + '/' + d_wind_waves + 'T' + hour_utc_str_wind + \
                        'Z_MSC_GDWPS_UGRD_AGL-10m_LatLon0.25_PT' + str(forecast_times_wind_hours[i]).zfill(3) + 'H.grib2'
                run(wgrib_path + 'wgrib2.exe ' + fname + ' -netcdf ' + directory + '/' + dirname_wind_waves + '/' + d_wind_waves + 'T' + hour_utc_str_wind + \
                    'Z_MSC_GDWPS_UGRD_AGL-10m_LatLon0.25_PT' + str(forecast_times_wind_hours[i]).zfill(3) + 'H.nc')

                if not use_temporary_directory:
                    if not os.path.isdir('./GDPS_wind_forecast_netcdf_files/' + dirname_wind_waves):
                        os.mkdir('./GDPS_wind_forecast_netcdf_files/' + dirname_wind_waves)

                    # shutil.move(fname[:-6] + '.nc', './GDPS_wind_forecast_netcdf_files/' + dirname_wind_waves + '/CMC_glb_UGRD_TGL_10_latlon.15x.15_' +
                    #             d_wind_waves + hour_utc_str_wind + '_P' + str(forecast_times_wind_hours[i]).zfill(3) + '.nc')

                    shutil.move(fname[:-6] + '.nc', './GDPS_wind_forecast_netcdf_files/' + dirname_wind_waves + '/' + d_wind_waves + 'T' + hour_utc_str_wind + \
                                'Z_MSC_GDWPS_UGRD_AGL-10m_LatLon0.25_PT' + str(forecast_times_wind_hours[i]).zfill(3) + 'H.nc')

                # url = 'https://dd.meteo.gc.ca/model_gem_global/15km/grib2/lat_lon/' + hour_utc_str_wind + '/' + \
                #       str(forecast_times_wind_hours[i]).zfill(3) + '/CMC_glb_VGRD_TGL_10_latlon.15x.15_' + \
                #       d_wind_waves + hour_utc_str_wind + '_P' + str(forecast_times_wind_hours[i]).zfill(3) + '.grib2'
                # fname = directory + '/' + dirname_wind_waves + '/CMC_glb_VGRD_TGL_10_latlon.15x.15_' + d_wind_waves + hour_utc_str_wind + '_P' + \
                #         str(forecast_times_wind_hours[i]).zfill(3) + '.grib2'

                url = 'https://dd.weather.gc.ca/model_gdwps/25km/' + hour_utc_str_wind + '/' + d_wind_waves + 'T' + hour_utc_str_wind + \
                      'Z_MSC_GDWPS_VGRD_AGL-10m_LatLon0.25_PT' + str(forecast_times_wind_hours[i]).zfill(3) + 'H.grib2'
                fname = directory + '/' + dirname_wind_waves + '/' + d_wind_waves + 'T' + hour_utc_str_wind + \
                        'Z_MSC_GDWPS_VGRD_AGL-10m_LatLon0.25_PT' + str(forecast_times_wind_hours[i]).zfill(3) + 'H.grib2'
                flag = True

                while flag:
                    try:
                        print('Obtaining forecast meridional wind velocity file ' + fname)
                        r = requests.get(url, allow_redirects=True, timeout=5.0)
                        open(fname, 'wb').write(r.content)
                        flag = False
                    except:
                        print('Error: could not download forecast meridional wind velocity file, retrying...')

                # fname = directory + '/' + dirname_wind_waves + '/CMC_glb_VGRD_TGL_10_latlon.15x.15_' + d_wind_waves + hour_utc_str_wind + '_P' + \
                #         str(forecast_times_wind_hours[i]).zfill(3) + '.grib2'
                # run(wgrib_path + 'wgrib2.exe ' + fname + ' -netcdf ' + directory + '/' + dirname_wind_waves + '/CMC_glb_VGRD_TGL_10_latlon.15x.15_' + \
                #     d_wind_waves + hour_utc_str_wind + '_P' + str(forecast_times_wind_hours[i]).zfill(3) + '.nc')

                fname = directory + '/' + dirname_wind_waves + '/' + d_wind_waves + 'T' + hour_utc_str_wind + \
                        'Z_MSC_GDWPS_VGRD_AGL-10m_LatLon0.25_PT' + str(forecast_times_wind_hours[i]).zfill(3) + 'H.grib2'
                run(wgrib_path + 'wgrib2.exe ' + fname + ' -netcdf ' + directory + '/' + dirname_wind_waves + '/' + d_wind_waves + 'T' + hour_utc_str_wind + \
                    'Z_MSC_GDWPS_VGRD_AGL-10m_LatLon0.25_PT' + str(forecast_times_wind_hours[i]).zfill(3) + 'H.nc')

                if not use_temporary_directory:
                    if not os.path.isdir('./GDPS_wind_forecast_netcdf_files/' + dirname_wind_waves):
                        os.mkdir('./GDPS_wind_forecast_netcdf_files/' + dirname_wind_waves)

                    # shutil.move(fname[:-6] + '.nc', './GDPS_wind_forecast_netcdf_files/' + dirname_wind_waves + '/CMC_glb_VGRD_TGL_10_latlon.15x.15_' +
                    #             d_wind_waves + hour_utc_str_wind + '_P' + str(forecast_times_wind_hours[i]).zfill(3) + '.nc')

                    shutil.move(fname[:-6] + '.nc', './GDPS_wind_forecast_netcdf_files/' + dirname_wind_waves + '/' + d_wind_waves + 'T' + hour_utc_str_wind + \
                                'Z_MSC_GDWPS_VGRD_AGL-10m_LatLon0.25_PT' + str(forecast_times_wind_hours[i]).zfill(3) + 'H.nc')

            if not use_temporary_directory:
                directory = './GDWPS_wave_forecast_grib2_files'

            if not os.path.isdir(directory + '/' + dirname_wind_waves):
                os.mkdir(directory + '/' + dirname_wind_waves)

            for i in range(len(forecast_times_waves)):
                url = 'https://dd.weather.gc.ca/model_gdwps/25km/' + hour_utc_str_waves + '/' + d_wind_waves + 'T' + hour_utc_str_waves + \
                    'Z_MSC_GDWPS_HTSGW_Sfc_LatLon0.25_PT' + str(forecast_times_waves_hours[i]).zfill(3) + 'H.grib2'
                fname = directory + '/' + dirname_wind_waves + '/' + d_wind_waves + 'T' + hour_utc_str_waves + \
                    'Z_MSC_GDWPS_HTSGW_Sfc_LatLon0.25_PT' + str(forecast_times_waves_hours[i]).zfill(3) + 'H.grib2'
                flag = True

                while flag:
                    try:
                        print('Obtaining forecast significant wave height file ' + fname)
                        r = requests.get(url, allow_redirects=True, timeout=5.0)
                        open(fname, 'wb').write(r.content)
                        flag = False
                    except:
                        print('Error: could not download forecast significant wave height file, retrying...')

                fname = directory + '/' + dirname_wind_waves + '/' + d_wind_waves + 'T' + hour_utc_str_waves + \
                    'Z_MSC_GDWPS_HTSGW_Sfc_LatLon0.25_PT' + str(forecast_times_waves_hours[i]).zfill(3) + 'H.grib2'
                run(wgrib_path + 'wgrib2.exe ' + fname + ' -netcdf ' + directory + '/' + dirname_wind_waves + '/' + d_wind_waves + 'T' + hour_utc_str_waves + \
                    'Z_MSC_GDWPS_HTSGW_Sfc_LatLon0.25_PT' + str(forecast_times_waves_hours[i]).zfill(3) + 'H.nc')

                if not use_temporary_directory:
                    if not os.path.isdir('./GDWPS_wave_forecast_netcdf_files/' + dirname_wind_waves):
                        os.mkdir('./GDWPS_wave_forecast_netcdf_files/' + dirname_wind_waves)

                    shutil.move(fname[:-6] + '.nc', './GDWPS_wave_forecast_netcdf_files/' + dirname_wind_waves + '/' + d_wind_waves + 'T' + hour_utc_str_waves + \
                    'Z_MSC_GDWPS_HTSGW_Sfc_LatLon0.25_PT' + str(forecast_times_waves_hours[i]).zfill(3) + 'H.nc')

                url = 'https://dd.weather.gc.ca/model_gdwps/25km/' + hour_utc_str_waves + '/' + d_wind_waves + 'T' + hour_utc_str_waves + \
                      'Z_MSC_GDWPS_WVDIR_Sfc_LatLon0.25_PT' + str(forecast_times_waves_hours[i]).zfill(3) + 'H.grib2'
                fname = directory + '/' + dirname_wind_waves + '/' + d_wind_waves + 'T' + hour_utc_str_waves + \
                        'Z_MSC_GDWPS_WVDIR_Sfc_LatLon0.25_PT' + str(forecast_times_waves_hours[i]).zfill(3) + 'H.grib2'
                flag = True

                while flag:
                    try:
                        print('Obtaining forecast wave direction file ' + fname)
                        r = requests.get(url, allow_redirects=True, timeout=5.0)
                        open(fname, 'wb').write(r.content)
                        flag = False
                    except:
                        print('Error: could not download forecast wave direction file, retrying...')

                fname = directory + '/' + dirname_wind_waves + '/' + d_wind_waves + 'T' + hour_utc_str_waves + \
                        'Z_MSC_GDWPS_WVDIR_Sfc_LatLon0.25_PT' + str(forecast_times_waves_hours[i]).zfill(3) + 'H.grib2'
                run(wgrib_path + 'wgrib2.exe ' + fname + ' -netcdf ' + directory + '/' + dirname_wind_waves + '/' + d_wind_waves + 'T' + hour_utc_str_waves + \
                    'Z_MSC_GDWPS_WVDIR_Sfc_LatLon0.25_PT' + str(forecast_times_waves_hours[i]).zfill(3) + 'H.nc')

                if not use_temporary_directory:
                    if not os.path.isdir('./GDWPS_wave_forecast_netcdf_files/' + dirname_wind_waves):
                        os.mkdir('./GDWPS_wave_forecast_netcdf_files/' + dirname_wind_waves)

                    shutil.move(fname[:-6] + '.nc', './GDWPS_wave_forecast_netcdf_files/' + dirname_wind_waves + '/' + d_wind_waves + 'T' + hour_utc_str_waves + \
                                'Z_MSC_GDWPS_WVDIR_Sfc_LatLon0.25_PT' + str(forecast_times_waves_hours[i]).zfill(3) + 'H.nc')

            if not use_temporary_directory:
                directory = './RIOPS_ocean_forecast_netcdf_files'

            if not os.path.isdir(directory + '/' + dirname_curr_ssh):
                os.mkdir(directory + '/' + dirname_curr_ssh)

            for i in range(len(forecast_times_curr_ssh)):
                url = 'https://dd.weather.gc.ca/model_riops/netcdf/forecast/polar_stereographic/3d/' + hour_utc_str_curr_ssh + '/' + \
                      str(forecast_times_curr_ssh_hours[i]).zfill(3) + '/' + d_curr_ssh + 'T' + hour_utc_str_curr_ssh + \
                      'Z_MSC_RIOPS_VOZOCRTX_DBS-all_PS5km_P' + str(forecast_times_curr_ssh_hours[i]).zfill(3) + '.nc'
                fname = directory + '/' + dirname_curr_ssh + '/' + d_curr_ssh + 'T' + hour_utc_str_curr_ssh + \
                      'Z_MSC_RIOPS_VOZOCRTX_DBS-all_PS5km_P' + str(forecast_times_curr_ssh_hours[i]).zfill(3) + '.nc'
                flag = True

                while flag:
                    try:
                        print('Obtaining forecast zonal ocean current file ' + fname)
                        r = requests.get(url, allow_redirects=True, timeout=5.0)
                        open(fname, 'wb').write(r.content)
                        flag = False
                    except:
                        print('Error: could not download forecast zonal ocean current file, retrying...')

                url = 'https://dd.weather.gc.ca/model_riops/netcdf/forecast/polar_stereographic/3d/' + hour_utc_str_curr_ssh + '/' + \
                      str(forecast_times_curr_ssh_hours[i]).zfill(3) + '/' + d_curr_ssh + 'T' + hour_utc_str_curr_ssh + \
                      'Z_MSC_RIOPS_VOMECRTY_DBS-all_PS5km_P' + str(forecast_times_curr_ssh_hours[i]).zfill(3) + '.nc'
                fname = directory + '/' + dirname_curr_ssh + '/' + d_curr_ssh + 'T' + hour_utc_str_curr_ssh + \
                        'Z_MSC_RIOPS_VOMECRTY_DBS-all_PS5km_P' + str(forecast_times_curr_ssh_hours[i]).zfill(3) + '.nc'
                flag = True

                while flag:
                    try:
                        print('Obtaining forecast meridional ocean current file ' + fname)
                        r = requests.get(url, allow_redirects=True, timeout=5.0)
                        open(fname, 'wb').write(r.content)
                        flag = False
                    except:
                        print('Error: could not download forecast meridional ocean current file, retrying...')

                url = 'https://dd.weather.gc.ca/model_riops/netcdf/forecast/polar_stereographic/2d/' + hour_utc_str_curr_ssh + '/' + \
                      str(forecast_times_curr_ssh_hours[i]).zfill(3) + '/' + d_curr_ssh + 'T' + hour_utc_str_curr_ssh + \
                      'Z_MSC_RIOPS_SOSSHEIG_SFC_PS5km_P' + str(forecast_times_curr_ssh_hours[i]).zfill(3) + '.nc'
                fname = directory + '/' + dirname_curr_ssh + '/' + d_curr_ssh + 'T' + hour_utc_str_curr_ssh + \
                        'Z_MSC_RIOPS_SOSSHEIG_SFC_PS5km_P' + str(forecast_times_curr_ssh_hours[i]).zfill(3) + '.nc'
                flag = True

                while flag:
                    try:
                        print('Obtaining forecast sea surface height file ' + fname)
                        r = requests.get(url, allow_redirects=True, timeout=5.0)
                        open(fname, 'wb').write(r.content)
                        flag = False
                    except:
                        print('Error: could not download forecast sea surface height file, retrying...')

            for i in range(len(forecast_times_curr_ssh)):
                fname = directory + '/' + dirname_curr_ssh + '/' + d_curr_ssh + 'T' + hour_utc_str_curr_ssh + \
                        'Z_MSC_RIOPS_SOSSHEIG_SFC_PS5km_P' + str(forecast_times_curr_ssh_hours[i]).zfill(3) + '.nc'
                ssh_data = nc.Dataset(fname)
                curr_ssh_lat = ssh_data.variables['latitude'][:]  # lat x lon
                curr_ssh_lon = ssh_data.variables['longitude'][:] # lat x lon
                ssh = np.squeeze(ssh_data.variables['sossheig'][:]) # lat x lon
                ssh_data.close()

                fname = directory + '/' + dirname_curr_ssh + '/' + d_curr_ssh + 'T' + hour_utc_str_curr_ssh + \
                        'Z_MSC_RIOPS_VOZOCRTX_DBS-all_PS5km_P' + str(forecast_times_curr_ssh_hours[i]).zfill(3) + '.nc'
                u_curr_data = nc.Dataset(fname)
                depth_curr = u_curr_data.variables['depth'][:]
                u_curr = np.squeeze(u_curr_data.variables['vozocrtx'][:])  # depth x lat x lon
                u_curr_data.close()

                fname = directory + '/' + dirname_curr_ssh + '/' + d_curr_ssh + 'T' + hour_utc_str_curr_ssh + \
                        'Z_MSC_RIOPS_VOMECRTY_DBS-all_PS5km_P' + str(forecast_times_curr_ssh_hours[i]).zfill(3) + '.nc'
                v_curr_data = nc.Dataset(fname)
                v_curr = np.squeeze(v_curr_data.variables['vomecrty'][:])  # depth x lat x lon
                v_curr_data.close()

                distance = np.sqrt((curr_ssh_lat - iceberg_lat0) ** 2 + (curr_ssh_lon - (iceberg_lon0 + 360.)) ** 2)

                # Find the indices of the nearest grid point
                nearest_idx = np.unravel_index(np.argmin(distance, axis=None), distance.shape)
                nearest_lat_idx, nearest_lon_idx = nearest_idx

                # Define the index range to restrict to within 10 indices in all directions
                lat_min = max(nearest_lat_idx - deg_radius, 0)
                lat_max = min(nearest_lat_idx + deg_radius, curr_ssh_lat.shape[0] - 1)
                lon_min = max(nearest_lon_idx - deg_radius, 0)
                lon_max = min(nearest_lon_idx + deg_radius, curr_ssh_lon.shape[1] - 1)

                # Slice the lat, lon, and ssh arrays within the 10x10 index range
                curr_ssh_lat = curr_ssh_lat[lat_min:lat_max + 1, lon_min:lon_max + 1]
                curr_ssh_lon = curr_ssh_lon[lat_min:lat_max + 1, lon_min:lon_max + 1]
                ssh = ssh[lat_min:lat_max + 1, lon_min:lon_max + 1]
                u_curr = u_curr[:, lat_min:lat_max + 1, lon_min:lon_max + 1]
                v_curr = v_curr[:, lat_min:lat_max + 1, lon_min:lon_max + 1]

                ssh_grad_x = np.empty((len(curr_ssh_lat[:, 0]), len(curr_ssh_lon[0, :]) - 1))
                ssh_grad_y = np.empty((len(curr_ssh_lat[:, 0]) - 1, len(curr_ssh_lon[0, :])))

                ssh_grad_x_lat = np.empty((len(curr_ssh_lat[:, 0]), len(curr_ssh_lon[0, :]) - 1))
                ssh_grad_y_lat = np.empty((len(curr_ssh_lat[:, 0]) - 1, len(curr_ssh_lon[0, :])))

                ssh_grad_x_lon = np.empty((len(curr_ssh_lat[:, 0]), len(curr_ssh_lon[0, :]) - 1))
                ssh_grad_y_lon = np.empty((len(curr_ssh_lat[:, 0]) - 1, len(curr_ssh_lon[0, :])))

                for k in range(len(curr_ssh_lat[:, 0])):
                    for n in range(len(curr_ssh_lon[0, :]) - 1):
                        grid_pt_dist, grid_pt_bearing = dist_bearing(Re, curr_ssh_lat[k, n], curr_ssh_lat[k, n + 1], curr_ssh_lon[k, n], curr_ssh_lon[k, n + 1])
                        ssh_grad_x_lat[k, n], ssh_grad_x_lon[k, n] = dist_course(Re, curr_ssh_lat[k, n], curr_ssh_lon[k, n], grid_pt_dist / 2., grid_pt_bearing)
                        ssh_grad = (ssh[k, n + 1] - ssh[k, n]) / grid_pt_dist
                        ssh_grad_x[k, n] = ssh_grad * np.sin(np.deg2rad(grid_pt_bearing))

                for k in range(len(curr_ssh_lat[:, 0]) - 1):
                    for n in range(len(curr_ssh_lon[0, :])):
                        grid_pt_dist, grid_pt_bearing = dist_bearing(Re, curr_ssh_lat[k, n], curr_ssh_lat[k + 1, n], curr_ssh_lon[k, n], curr_ssh_lon[k + 1, n])
                        ssh_grad_y_lat[k, n], ssh_grad_y_lon[k, n] = dist_course(Re, curr_ssh_lat[k, n], curr_ssh_lon[k, n], grid_pt_dist / 2., grid_pt_bearing)
                        ssh_grad = (ssh[k + 1, n] - ssh[k, n]) / grid_pt_dist
                        ssh_grad_y[k, n] = -ssh_grad * np.cos(np.deg2rad(grid_pt_bearing))

                fname = directory + '/' + dirname_curr_ssh + '/' + d_curr_ssh + 'T' + hour_utc_str_curr_ssh + \
                        'Z_MSC_RIOPS_SOSSHEIG_SFC_GRAD_PS5km_P' + str(forecast_times_curr_ssh_hours[i]).zfill(3) + '.nc'

                print('Writing forecast sea surface height gradient file ' + fname)

                with nc.Dataset(fname, 'w', format='NETCDF4') as ncfile:
                    ncfile.createDimension('x_gradient_latitude', len(curr_ssh_lat[:, 0]))
                    ncfile.createDimension('x_gradient_longitude', len(curr_ssh_lon[0, :]) - 1)
                    ncfile.createDimension('y_gradient_latitude', len(curr_ssh_lat[:, 0]) - 1)
                    ncfile.createDimension('y_gradient_longitude', len(curr_ssh_lon[0, :]))

                    x_gradient_latitude_var = ncfile.createVariable('ssh_grad_x_lat', 'f8', ('x_gradient_latitude', 'x_gradient_longitude',))
                    x_gradient_longitude_var = ncfile.createVariable('ssh_grad_x_lon', 'f8',('x_gradient_latitude', 'x_gradient_longitude',))
                    y_gradient_latitude_var = ncfile.createVariable('ssh_grad_y_lat', 'f8', ('y_gradient_latitude', 'y_gradient_longitude',))
                    y_gradient_longitude_var = ncfile.createVariable('ssh_grad_y_lon', 'f8',('y_gradient_latitude', 'y_gradient_longitude',))
                    ssh_grad_x_var = ncfile.createVariable('ssh_grad_x', 'f4', ('x_gradient_latitude', 'x_gradient_longitude',))
                    ssh_grad_y_var = ncfile.createVariable('ssh_grad_y', 'f4', ('y_gradient_latitude', 'y_gradient_longitude',))

                    x_gradient_latitude_var[:] = ssh_grad_x_lat
                    x_gradient_longitude_var[:] = ssh_grad_x_lon
                    y_gradient_latitude_var[:] = ssh_grad_y_lat
                    y_gradient_longitude_var[:] = ssh_grad_y_lon
                    ssh_grad_x_var[:] = ssh_grad_x
                    ssh_grad_y_var[:] = ssh_grad_y

                fname = directory + '/' + dirname_curr_ssh + '/' + d_curr_ssh + 'T' + hour_utc_str_curr_ssh + \
                        'Z_MSC_RIOPS_VOZOCRTX_DBS-all_PS5km_P' + str(forecast_times_curr_ssh_hours[i]).zfill(3) + '.nc'

                print('Shrinking forecast zonal ocean current file ' + fname)

                with nc.Dataset(fname, 'w', format='NETCDF4') as ncfile:
                    ncfile.createDimension('latitude', len(curr_ssh_lat[:, 0]))
                    ncfile.createDimension('longitude', len(curr_ssh_lon[0, :]))
                    ncfile.createDimension('depth', len(depth_curr))

                    latitude_var = ncfile.createVariable('latitude', 'f8', ('latitude', 'longitude',))
                    longitude_var = ncfile.createVariable('longitude', 'f8', ('latitude', 'longitude',))
                    depth_var = ncfile.createVariable('depth', 'f8', ('depth',))
                    u_curr_var = ncfile.createVariable('vozocrtx', 'f4', ('depth', 'latitude', 'longitude',))

                    latitude_var[:] = curr_ssh_lat
                    longitude_var[:] = curr_ssh_lon
                    depth_var[:] = depth_curr
                    u_curr_var[:] = u_curr

                fname = directory + '/' + dirname_curr_ssh + '/' + d_curr_ssh + 'T' + hour_utc_str_curr_ssh + \
                        'Z_MSC_RIOPS_VOMECRTY_DBS-all_PS5km_P' + str(forecast_times_curr_ssh_hours[i]).zfill(3) + '.nc'

                print('Shrinking forecast meridional ocean current file ' + fname)

                with nc.Dataset(fname, 'w', format='NETCDF4') as ncfile:
                    ncfile.createDimension('latitude', len(curr_ssh_lat[:, 0]))
                    ncfile.createDimension('longitude', len(curr_ssh_lon[0, :]))
                    ncfile.createDimension('depth', len(depth_curr))

                    latitude_var = ncfile.createVariable('latitude', 'f8', ('latitude', 'longitude',))
                    longitude_var = ncfile.createVariable('longitude', 'f8', ('latitude', 'longitude',))
                    depth_var = ncfile.createVariable('depth', 'f8', ('depth',))
                    v_curr_var = ncfile.createVariable('vomecrty', 'f4', ('depth', 'latitude', 'longitude',))

                    latitude_var[:] = curr_ssh_lat
                    longitude_var[:] = curr_ssh_lon
                    depth_var[:] = depth_curr
                    v_curr_var[:] = v_curr

            base_time_wind = forecast_times_wind[0]
            time_increments_wind = np.arange(forecast_times_wind_hours[0], forecast_times_wind_hours[-1], 3)
            file_times_wind = base_time_wind + time_increments_wind.astype('timedelta64[h]')
            date_only_wind = str(base_time_wind.astype('datetime64[D]')).replace('-', '')

            base_time_waves = forecast_times_waves[0]
            time_increments_waves = np.arange(forecast_times_waves_hours[0], forecast_times_waves_hours[-1], 3)
            file_times_waves = base_time_waves + time_increments_waves.astype('timedelta64[h]')
            date_only_waves = str(base_time_waves.astype('datetime64[D]')).replace('-', '')

            base_time_curr_ssh = forecast_times_curr_ssh[0]
            time_increments_curr_ssh = np.arange(forecast_times_curr_ssh_hours[0], forecast_times_curr_ssh_hours[-1], 1)
            file_times_curr_ssh = base_time_curr_ssh + time_increments_curr_ssh.astype('timedelta64[h]')
            date_only_curr_ssh = str(base_time_curr_ssh.astype('datetime64[D]')).replace('-', '')

            if not use_temporary_directory:
                directory = './GDPS_wind_forecast_netcdf_files'

            # fname = directory + '/' + dirname_wind_waves + '/CMC_glb_UGRD_TGL_10_latlon.15x.15_' + d_wind_waves + hour_utc_str_wind + '_P' + \
            #         str(forecast_times_wind_hours[0]).zfill(3) + '.nc'
            fname = directory + '/' + dirname_wind_waves + '/' + d_wind_waves + 'T' + hour_utc_str_waves + \
                    'Z_MSC_GDWPS_UGRD_AGL-10m_LatLon0.25_PT' + str(forecast_times_waves_hours[0]).zfill(3) + 'H.nc'
            wind_data = nc.Dataset(fname)
            lat_wind = wind_data.variables['latitude'][:]
            lon_wind = wind_data.variables['longitude'][:]
            wind_data.close()

            if not use_temporary_directory:
                directory = './GDWPS_wave_forecast_netcdf_files'

            fname = directory + '/' + dirname_wind_waves + '/' + d_wind_waves + 'T' + hour_utc_str_waves + \
                    'Z_MSC_GDWPS_HTSGW_Sfc_LatLon0.25_PT' + str(forecast_times_waves_hours[0]).zfill(3) + 'H.nc'
            wave_data = nc.Dataset(fname)
            lat_waves = wave_data.variables['latitude'][:]
            lon_waves = wave_data.variables['longitude'][:]
            wave_data.close()

            if not use_temporary_directory:
                directory = './RIOPS_ocean_forecast_netcdf_files'

            fname = directory + '/' + dirname_curr_ssh + '/' + d_curr_ssh + 'T' + hour_utc_str_curr_ssh + \
                    'Z_MSC_RIOPS_VOZOCRTX_DBS-all_PS5km_P' + str(forecast_times_curr_ssh_hours[0]).zfill(3) + '.nc'
            curr_data = nc.Dataset(fname)
            lat_curr = curr_data.variables['latitude'][:]  # lat x lon
            lon_curr = curr_data.variables['longitude'][:]  # lat x lon
            depth_curr = curr_data.variables['depth'][:]
            curr_data.close()

            points_curr = np.array([lat_curr.ravel(), lon_curr.ravel()]).T  # Shape (n_points, 2)

            loc_depth = np.argwhere(depth_curr <= iceberg_draft)
            loc_depth = np.append(loc_depth, loc_depth[-1] + 1)
            depth_curr_ib = list(depth_curr[loc_depth])
            depth_curr_ib_interp = np.arange(0., iceberg_draft, 0.001)

            fname = directory + '/' + dirname_curr_ssh + '/' + d_curr_ssh + 'T' + hour_utc_str_curr_ssh + \
                    'Z_MSC_RIOPS_SOSSHEIG_SFC_GRAD_PS5km_P' + str(forecast_times_curr_ssh_hours[0]).zfill(3) + '.nc'
            ssh_grad_data = nc.Dataset(fname)
            ssh_grad_x_lat = ssh_grad_data.variables['ssh_grad_x_lat'][:]  # lat x lon
            ssh_grad_x_lon = ssh_grad_data.variables['ssh_grad_x_lon'][:]  # lat x lon
            ssh_grad_y_lat = ssh_grad_data.variables['ssh_grad_y_lat'][:]  # lat x lon
            ssh_grad_y_lon = ssh_grad_data.variables['ssh_grad_y_lon'][:]  # lat x lon
            ssh_grad_data.close()

            points_ssh_grad_x = np.array([ssh_grad_x_lat.ravel(), ssh_grad_x_lon.ravel()]).T  # Shape (n_points, 2)
            points_ssh_grad_y = np.array([ssh_grad_y_lat.ravel(), ssh_grad_y_lon.ravel()]).T  # Shape (n_points, 2)

            iceberg_us = np.empty((len(iceberg_times),))
            iceberg_vs = np.empty((len(iceberg_times),))

            iceberg_lats[0] = iceberg_lat0
            iceberg_lons[0] = iceberg_lon0
            iceberg_us[0] = iceberg_u0
            iceberg_vs[0] = iceberg_v0

            for i in range(len(iceberg_times) - 1):
                iceberg_lat = iceberg_lats[i]
                iceberg_lon = iceberg_lons[i]
                iceberg_u = iceberg_us[i]
                iceberg_v = iceberg_vs[i]

                iceberg_time = iceberg_times[i]
                iceberg_time2 = iceberg_times[i + 1]

                # Find the time just before and just after the forecast_time
                before_idx = np.where(file_times_wind <= iceberg_time)[0][-1]

                try:
                    after_idx = np.where(file_times_wind > iceberg_time)[0][0]
                except:
                    after_idx = -1

                # The corresponding NetCDF files
                # u_wind_file_before = 'CMC_glb_UGRD_TGL_10_latlon.15x.15_' + date_only_wind + hour_utc_str_wind + '_P' + \
                #                      str(time_increments_wind[before_idx]).zfill(3) + '.nc'
                # u_wind_file_after = 'CMC_glb_UGRD_TGL_10_latlon.15x.15_' + date_only_wind + hour_utc_str_wind + '_P' + \
                #                     str(time_increments_wind[after_idx]).zfill(3) + '.nc'
                u_wind_file_before = date_only_wind + 'T' + hour_utc_str_wind + 'Z_MSC_GDWPS_UGRD_AGL-10m_LatLon0.25_PT' + \
                                 str(time_increments_wind[before_idx]).zfill(3) + 'H.nc'
                u_wind_file_after = date_only_wind + 'T' + hour_utc_str_wind + 'Z_MSC_GDWPS_UGRD_AGL-10m_LatLon0.25_PT' + \
                                str(time_increments_wind[after_idx]).zfill(3) + 'H.nc'
                # v_wind_file_before = 'CMC_glb_VGRD_TGL_10_latlon.15x.15_' + date_only_wind + hour_utc_str_wind + '_P' + \
                #                      str(time_increments_wind[before_idx]).zfill(3) + '.nc'
                # v_wind_file_after = 'CMC_glb_VGRD_TGL_10_latlon.15x.15_' + date_only_wind + hour_utc_str_wind + '_P' + \
                #                     str(time_increments_wind[after_idx]).zfill(3) + '.nc'
                v_wind_file_before = date_only_wind + 'T' + hour_utc_str_wind + 'Z_MSC_GDWPS_VGRD_AGL-10m_LatLon0.25_PT' + \
                                     str(time_increments_wind[before_idx]).zfill(3) + 'H.nc'
                v_wind_file_after = date_only_wind + 'T' + hour_utc_str_wind + 'Z_MSC_GDWPS_VGRD_AGL-10m_LatLon0.25_PT' + \
                                    str(time_increments_wind[after_idx]).zfill(3) + 'H.nc'

                forecast_time_wind_before = forecast_times_wind[before_idx]
                forecast_time_wind_after = forecast_times_wind[after_idx]

                # Find the time just before and just after the forecast_time
                before_idx = np.where(file_times_waves <= iceberg_time)[0][-1]

                try:
                    after_idx = np.where(file_times_waves > iceberg_time)[0][0]
                except:
                    after_idx = -1

                # The corresponding NetCDF files
                Hs_file_before = date_only_waves + 'T' + hour_utc_str_waves + 'Z_MSC_GDWPS_HTSGW_Sfc_LatLon0.25_PT' + \
                                 str(time_increments_waves[before_idx]).zfill(3) + 'H.nc'
                Hs_file_after = date_only_waves + 'T' + hour_utc_str_waves + 'Z_MSC_GDWPS_HTSGW_Sfc_LatLon0.25_PT' + \
                                str(time_increments_waves[after_idx]).zfill(3) + 'H.nc'
                wave_dir_file_before = date_only_waves + 'T' + hour_utc_str_waves + 'Z_MSC_GDWPS_WVDIR_Sfc_LatLon0.25_PT' + \
                                       str(time_increments_waves[before_idx]).zfill(3) + 'H.nc'
                wave_dir_file_after = date_only_waves + 'T' + hour_utc_str_waves + 'Z_MSC_GDWPS_WVDIR_Sfc_LatLon0.25_PT' + \
                                      str(time_increments_waves[after_idx]).zfill(3) + 'H.nc'

                forecast_time_waves_before = forecast_times_waves[before_idx]
                forecast_time_waves_after = forecast_times_waves[after_idx]

                # Find the time just before and just after the forecast_time
                before_idx = np.where(file_times_curr_ssh <= iceberg_time)[0][-1]

                try:
                    after_idx = np.where(file_times_curr_ssh > iceberg_time)[0][0]
                except:
                    after_idx = -1

                # The corresponding NetCDF files
                u_curr_file_before = date_only_curr_ssh + 'T' + hour_utc_str_curr_ssh + 'Z_MSC_RIOPS_VOZOCRTX_DBS-all_PS5km_P' + \
                                     str(time_increments_curr_ssh[before_idx]).zfill(3) + '.nc'
                u_curr_file_after = date_only_curr_ssh + 'T' + hour_utc_str_curr_ssh + 'Z_MSC_RIOPS_VOZOCRTX_DBS-all_PS5km_P' + \
                                    str(time_increments_curr_ssh[after_idx]).zfill(3) + '.nc'
                v_curr_file_before = date_only_curr_ssh + 'T' + hour_utc_str_curr_ssh + 'Z_MSC_RIOPS_VOMECRTY_DBS-all_PS5km_P' + \
                                     str(time_increments_curr_ssh[before_idx]).zfill(3) + '.nc'
                v_curr_file_after = date_only_curr_ssh + 'T' + hour_utc_str_curr_ssh + 'Z_MSC_RIOPS_VOMECRTY_DBS-all_PS5km_P' + \
                                    str(time_increments_curr_ssh[after_idx]).zfill(3) + '.nc'
                ssh_grad_file_before = date_only_curr_ssh + 'T' + hour_utc_str_curr_ssh + 'Z_MSC_RIOPS_SOSSHEIG_SFC_GRAD_PS5km_P' + \
                str(time_increments_curr_ssh[before_idx]).zfill(3) + '.nc'
                ssh_grad_file_after = date_only_curr_ssh + 'T' + hour_utc_str_curr_ssh + 'Z_MSC_RIOPS_SOSSHEIG_SFC_GRAD_PS5km_P' + \
                str(time_increments_curr_ssh[after_idx]).zfill(3) + '.nc'

                forecast_time_curr_ssh_before = forecast_times_curr_ssh[before_idx]
                forecast_time_curr_ssh_after = forecast_times_curr_ssh[after_idx]

                before_idx = np.where(file_times_curr_ssh <= iceberg_time2)[0][-1]

                try:
                    after_idx = np.where(file_times_curr_ssh > iceberg_time2)[0][0]
                except:
                    after_idx = -1

                # The corresponding NetCDF files
                u_curr_file_before2 = date_only_curr_ssh + 'T' + hour_utc_str_curr_ssh + 'Z_MSC_RIOPS_VOZOCRTX_DBS-all_PS5km_P' + \
                str(time_increments_curr_ssh[before_idx]).zfill(3) + '.nc'
                u_curr_file_after2 = date_only_curr_ssh + 'T' + hour_utc_str_curr_ssh + 'Z_MSC_RIOPS_VOZOCRTX_DBS-all_PS5km_P' + \
                                     str(time_increments_curr_ssh[after_idx]).zfill(3) + '.nc'
                v_curr_file_before2 = date_only_curr_ssh + 'T' + hour_utc_str_curr_ssh + 'Z_MSC_RIOPS_VOMECRTY_DBS-all_PS5km_P' + \
                                      str(time_increments_curr_ssh[before_idx]).zfill(3) + '.nc'
                v_curr_file_after2 = date_only_curr_ssh + 'T' + hour_utc_str_curr_ssh + 'Z_MSC_RIOPS_VOMECRTY_DBS-all_PS5km_P' + \
                                     str(time_increments_curr_ssh[after_idx]).zfill(3) + '.nc'

                forecast_time_curr_ssh_before2 = forecast_times_curr_ssh[before_idx]
                forecast_time_curr_ssh_after2 = forecast_times_curr_ssh[after_idx]

                if not use_temporary_directory:
                    directory = './GDPS_wind_forecast_netcdf_files'

                fname = directory + '/' + dirname_wind_waves + '/' + u_wind_file_before
                u_wind_data_before = nc.Dataset(fname)
                u_wind_before = np.squeeze(u_wind_data_before.variables['UGRD_10maboveground'][:])  # lat x lon
                u_wind_data_before.close()

                fname = directory + '/' + dirname_wind_waves + '/' + u_wind_file_after
                u_wind_data_after = nc.Dataset(fname)
                u_wind_after = np.squeeze(u_wind_data_after.variables['UGRD_10maboveground'][:])  # lat x lon
                u_wind_data_after.close()

                f_u_wind_before = RegularGridInterpolator((lat_wind, lon_wind), u_wind_before, method='linear', bounds_error=True, fill_value=np.nan)
                f_u_wind_after = RegularGridInterpolator((lat_wind, lon_wind), u_wind_after, method='linear', bounds_error=True, fill_value=np.nan)
                u_wind_before_ib = float(f_u_wind_before([iceberg_lat, iceberg_lon + 360.]))
                u_wind_after_ib = float(f_u_wind_after([iceberg_lat, iceberg_lon + 360.]))

                t1 = (forecast_time_wind_before - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
                t2 = (forecast_time_wind_after - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
                t_new = (iceberg_time - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')

                # Calculate the interpolation weight
                weight = (t_new - t1) / (t2 - t1)

                # Perform the linear interpolation for the scalar value
                u_wind_ib = u_wind_before_ib + weight * (u_wind_after_ib - u_wind_before_ib)

                fname = directory + '/' + dirname_wind_waves + '/' + v_wind_file_before
                v_wind_data_before = nc.Dataset(fname)
                v_wind_before = np.squeeze(v_wind_data_before.variables['VGRD_10maboveground'][:])  # lat x lon
                v_wind_data_before.close()

                fname = directory + '/' + dirname_wind_waves + '/' + v_wind_file_after
                v_wind_data_after = nc.Dataset(fname)
                v_wind_after = np.squeeze(v_wind_data_after.variables['VGRD_10maboveground'][:])  # lat x lon
                v_wind_data_after.close()

                f_v_wind_before = RegularGridInterpolator((lat_wind, lon_wind), v_wind_before, method='linear', bounds_error=True, fill_value=np.nan)
                f_v_wind_after = RegularGridInterpolator((lat_wind, lon_wind), v_wind_after, method='linear', bounds_error=True, fill_value=np.nan)
                v_wind_before_ib = float(f_v_wind_before([iceberg_lat, iceberg_lon + 360.]))
                v_wind_after_ib = float(f_v_wind_after([iceberg_lat, iceberg_lon + 360.]))

                # Perform the linear interpolation for the scalar value
                v_wind_ib = v_wind_before_ib + weight * (v_wind_after_ib - v_wind_before_ib)

                if not use_temporary_directory:
                    directory = './GDWPS_wave_forecast_netcdf_files'

                fname = directory + '/' + dirname_wind_waves + '/' + Hs_file_before
                Hs_data_before = nc.Dataset(fname)
                Hs_before = np.squeeze(Hs_data_before.variables['HTSGW_surface'][:])  # lat x lon
                Hs_data_before.close()

                fname = directory + '/' + dirname_wind_waves + '/' + Hs_file_after
                Hs_data_after = nc.Dataset(fname)
                Hs_after = np.squeeze(Hs_data_after.variables['HTSGW_surface'][:])  # lat x lon
                Hs_data_after.close()

                f_Hs_before = RegularGridInterpolator((lat_waves, lon_waves), Hs_before, method='nearest', bounds_error=True, fill_value=np.nan)
                f_Hs_after = RegularGridInterpolator((lat_waves, lon_waves), Hs_after, method='nearest', bounds_error=True, fill_value=np.nan)
                Hs_before_ib = float(f_Hs_before([iceberg_lat, iceberg_lon + 360.]))
                Hs_after_ib = float(f_Hs_after([iceberg_lat, iceberg_lon + 360.]))

                t1 = (forecast_time_waves_before - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
                t2 = (forecast_time_waves_after - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')

                # Calculate the interpolation weight
                weight = (t_new - t1) / (t2 - t1)

                # Perform the linear interpolation for the scalar value
                Hs_ib = Hs_before_ib + weight * (Hs_after_ib - Hs_before_ib)

                fname = directory + '/' + dirname_wind_waves + '/' + wave_dir_file_before
                wave_dir_data_before = nc.Dataset(fname)
                wave_dir_before = np.squeeze(wave_dir_data_before.variables['WVDIR_surface'][:])  # lat x lon
                wave_dir_data_before.close()

                fname = directory + '/' + dirname_wind_waves + '/' + wave_dir_file_after
                wave_dir_data_after = nc.Dataset(fname)
                wave_dir_after = np.squeeze(wave_dir_data_after.variables['WVDIR_surface'][:])  # lat x lon
                wave_dir_data_after.close()

                wave_E_before = np.sin(np.deg2rad(wave_dir_before))
                wave_E_after = np.sin(np.deg2rad(wave_dir_after))
                wave_N_before = np.cos(np.deg2rad(wave_dir_before))
                wave_N_after = np.cos(np.deg2rad(wave_dir_after))

                f_wave_E_before = RegularGridInterpolator((lat_waves, lon_waves), wave_E_before, method='nearest', bounds_error=True, fill_value=np.nan)
                f_wave_E_after = RegularGridInterpolator((lat_waves, lon_waves), wave_E_after, method='nearest', bounds_error=True, fill_value=np.nan)
                wave_E_before_ib = float(f_wave_E_before([iceberg_lat, iceberg_lon + 360.]))
                wave_E_after_ib = float(f_wave_E_after([iceberg_lat, iceberg_lon + 360.]))

                f_wave_N_before = RegularGridInterpolator((lat_waves, lon_waves), wave_N_before, method='nearest', bounds_error=True, fill_value=np.nan)
                f_wave_N_after = RegularGridInterpolator((lat_waves, lon_waves), wave_N_after, method='nearest', bounds_error=True, fill_value=np.nan)
                wave_N_before_ib = float(f_wave_N_before([iceberg_lat, iceberg_lon + 360.]))
                wave_N_after_ib = float(f_wave_N_after([iceberg_lat, iceberg_lon + 360.]))

                wave_E_ib = wave_E_before_ib + weight * (wave_E_after_ib - wave_E_before_ib)
                wave_N_ib = wave_N_before_ib + weight * (wave_N_after_ib - wave_N_before_ib)
                wave_dir_ib = 90. - np.rad2deg(np.arctan2(wave_N_ib, wave_E_ib))

                if wave_dir_ib < 0:
                    wave_dir_ib = wave_dir_ib + 360.

                if not use_temporary_directory:
                    directory = './RIOPS_ocean_forecast_netcdf_files'

                fname = directory + '/' + dirname_curr_ssh + '/' + u_curr_file_before
                u_curr_data_before = nc.Dataset(fname)
                u_curr_before = np.squeeze(u_curr_data_before.variables['vozocrtx'][:]) # depth x lat x lon
                u_curr_data_before.close()

                fname = directory + '/' + dirname_curr_ssh + '/' + u_curr_file_after
                u_curr_data_after = nc.Dataset(fname)
                u_curr_after = np.squeeze(u_curr_data_after.variables['vozocrtx'][:])  # depth x lat x lon
                u_curr_data_after.close()

                fname = directory + '/' + dirname_curr_ssh + '/' + v_curr_file_before
                v_curr_data_before = nc.Dataset(fname)
                v_curr_before = np.squeeze(v_curr_data_before.variables['vomecrty'][:])  # depth x lat x lon
                v_curr_data_before.close()

                fname = directory + '/' + dirname_curr_ssh + '/' + v_curr_file_after
                v_curr_data_after = nc.Dataset(fname)
                v_curr_after = np.squeeze(v_curr_data_after.variables['vomecrty'][:])  # depth x lat x lon
                v_curr_data_after.close()

                u_curr_before = u_curr_before[loc_depth, :, :]
                u_curr_after = u_curr_after[loc_depth, :, :]
                v_curr_before = v_curr_before[loc_depth, :, :]
                v_curr_after = v_curr_after[loc_depth, :, :]

                u_curr_before_depth_list = []
                u_curr_after_depth_list = []
                v_curr_before_depth_list = []
                v_curr_after_depth_list = []

                for n in range(len(depth_curr_ib)):
                    u_curr_before_select = np.squeeze(u_curr_before[n, :, :])
                    u_curr_after_select = np.squeeze(u_curr_after[n, :, :])
                    u_curr_before_temp = griddata(points_curr, u_curr_before_select.ravel(),(iceberg_lat, iceberg_lon + 360.), method='linear')
                    u_curr_after_temp = griddata(points_curr, u_curr_after_select.ravel(),(iceberg_lat, iceberg_lon + 360.), method='linear')
                    u_curr_before_depth_list.append(u_curr_before_temp)
                    u_curr_after_depth_list.append(u_curr_after_temp)
                    v_curr_before_select = np.squeeze(v_curr_before[n, :, :])
                    v_curr_after_select = np.squeeze(v_curr_after[n, :, :])
                    v_curr_before_temp = griddata(points_curr, v_curr_before_select.ravel(),(iceberg_lat, iceberg_lon + 360.), method='linear')
                    v_curr_after_temp = griddata(points_curr, v_curr_after_select.ravel(),(iceberg_lat, iceberg_lon + 360.), method='linear')
                    v_curr_before_depth_list.append(v_curr_before_temp)
                    v_curr_after_depth_list.append(v_curr_after_temp)

                u_curr_before_depth_list = [float(val) for val in u_curr_before_depth_list]
                u_curr_after_depth_list = [float(val) for val in u_curr_after_depth_list]
                interp_func = interp1d(depth_curr_ib, u_curr_before_depth_list, kind='linear', fill_value='extrapolate')
                u_curr_before_depth_list = interp_func(depth_curr_ib_interp)
                interp_func = interp1d(depth_curr_ib, u_curr_after_depth_list, kind='linear', fill_value='extrapolate')
                u_curr_after_depth_list = interp_func(depth_curr_ib_interp)
                u_curr_before_ib = np.nanmean(u_curr_before_depth_list)
                u_curr_after_ib = np.nanmean(u_curr_after_depth_list)

                v_curr_before_depth_list = [float(val) for val in v_curr_before_depth_list]
                v_curr_after_depth_list = [float(val) for val in v_curr_after_depth_list]
                interp_func = interp1d(depth_curr_ib, v_curr_before_depth_list, kind='linear', fill_value='extrapolate')
                v_curr_before_depth_list = interp_func(depth_curr_ib_interp)
                interp_func = interp1d(depth_curr_ib, v_curr_after_depth_list, kind='linear', fill_value='extrapolate')
                v_curr_after_depth_list = interp_func(depth_curr_ib_interp)
                v_curr_before_ib = np.nanmean(v_curr_before_depth_list)
                v_curr_after_ib = np.nanmean(v_curr_after_depth_list)

                t1 = (forecast_time_curr_ssh_before - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
                t2 = (forecast_time_curr_ssh_after - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')

                # Calculate the interpolation weight
                weight = (t_new - t1) / (t2 - t1)

                # Perform the linear interpolation for the scalar value
                u_curr_ib = u_curr_before_ib + weight * (u_curr_after_ib - u_curr_before_ib)
                v_curr_ib = v_curr_before_ib + weight * (v_curr_after_ib - v_curr_before_ib)

                fname = directory + '/' + dirname_curr_ssh + '/' + u_curr_file_before2
                u_curr_data_before2 = nc.Dataset(fname)
                u_curr_before2 = np.squeeze(u_curr_data_before2.variables['vozocrtx'][:])  # depth x lat x lon
                u_curr_data_before2.close()

                fname = directory + '/' + dirname_curr_ssh + '/' + u_curr_file_after2
                u_curr_data_after2 = nc.Dataset(fname)
                u_curr_after2 = np.squeeze(u_curr_data_after2.variables['vozocrtx'][:])  # depth x lat x lon
                u_curr_data_after2.close()

                fname = directory + '/' + dirname_curr_ssh + '/' + v_curr_file_before2
                v_curr_data_before2 = nc.Dataset(fname)
                v_curr_before2 = np.squeeze(v_curr_data_before2.variables['vomecrty'][:])  # depth x lat x lon
                v_curr_data_before2.close()

                fname = directory + '/' + dirname_curr_ssh + '/' + v_curr_file_after2
                v_curr_data_after2 = nc.Dataset(fname)
                v_curr_after2 = np.squeeze(v_curr_data_after2.variables['vomecrty'][:])  # depth x lat x lon
                v_curr_data_after2.close()

                u_curr_before2 = u_curr_before2[loc_depth, :, :]
                u_curr_after2 = u_curr_after2[loc_depth, :, :]
                v_curr_before2 = v_curr_before2[loc_depth, :, :]
                v_curr_after2 = v_curr_after2[loc_depth, :, :]

                u_curr_before2_depth_list = []
                u_curr_after2_depth_list = []
                v_curr_before2_depth_list = []
                v_curr_after2_depth_list = []

                for n in range(len(depth_curr_ib)):
                    u_curr_before2_select = np.squeeze(u_curr_before2[n, :, :])
                    u_curr_after2_select = np.squeeze(u_curr_after2[n, :, :])
                    u_curr_before2_temp = griddata(points_curr, u_curr_before2_select.ravel(),(iceberg_lat, iceberg_lon + 360.), method='linear')
                    u_curr_after2_temp = griddata(points_curr, u_curr_after2_select.ravel(), (iceberg_lat, iceberg_lon + 360.), method='linear')
                    u_curr_before2_depth_list.append(u_curr_before2_temp)
                    u_curr_after2_depth_list.append(u_curr_after2_temp)
                    v_curr_before2_select = np.squeeze(v_curr_before2[n, :, :])
                    v_curr_after2_select = np.squeeze(v_curr_after2[n, :, :])
                    v_curr_before2_temp = griddata(points_curr, v_curr_before2_select.ravel(),(iceberg_lat, iceberg_lon + 360.), method='linear')
                    v_curr_after2_temp = griddata(points_curr, v_curr_after2_select.ravel(), (iceberg_lat, iceberg_lon + 360.), method='linear')
                    v_curr_before2_depth_list.append(v_curr_before2_temp)
                    v_curr_after2_depth_list.append(v_curr_after2_temp)

                u_curr_before2_depth_list = [float(val) for val in u_curr_before2_depth_list]
                u_curr_after2_depth_list = [float(val) for val in u_curr_after2_depth_list]
                interp_func = interp1d(depth_curr_ib, u_curr_before2_depth_list, kind='linear', fill_value='extrapolate')
                u_curr_before2_depth_list = interp_func(depth_curr_ib_interp)
                interp_func = interp1d(depth_curr_ib, u_curr_after2_depth_list, kind='linear', fill_value='extrapolate')
                u_curr_after2_depth_list = interp_func(depth_curr_ib_interp)
                u_curr_before2_ib = np.nanmean(u_curr_before2_depth_list)
                u_curr_after2_ib = np.nanmean(u_curr_after2_depth_list)

                v_curr_before2_depth_list = [float(val) for val in v_curr_before2_depth_list]
                v_curr_after2_depth_list = [float(val) for val in v_curr_after2_depth_list]
                interp_func = interp1d(depth_curr_ib, v_curr_before2_depth_list, kind='linear', fill_value='extrapolate')
                v_curr_before2_depth_list = interp_func(depth_curr_ib_interp)
                interp_func = interp1d(depth_curr_ib, v_curr_after2_depth_list, kind='linear', fill_value='extrapolate')
                v_curr_after2_depth_list = interp_func(depth_curr_ib_interp)
                v_curr_before2_ib = np.nanmean(v_curr_before2_depth_list)
                v_curr_after2_ib = np.nanmean(v_curr_after2_depth_list)

                t1 = (forecast_time_curr_ssh_before2 - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
                t2 = (forecast_time_curr_ssh_after2 - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')

                # Calculate the interpolation weight
                weight = (t_new - t1) / (t2 - t1)

                # Perform the linear interpolation for the scalar value
                u_curr_ib2 = u_curr_before2_ib + weight * (u_curr_after2_ib - u_curr_before2_ib)
                v_curr_ib2 = v_curr_before2_ib + weight * (v_curr_after2_ib - v_curr_before2_ib)

                fname = directory + '/' + dirname_curr_ssh + '/' + ssh_grad_file_before
                ssh_grad_data_before = nc.Dataset(fname)
                ssh_grad_x_before = np.squeeze(ssh_grad_data_before.variables['ssh_grad_x'][:])  # lat x lon
                ssh_grad_y_before = np.squeeze(ssh_grad_data_before.variables['ssh_grad_y'][:])  # lat x lon
                ssh_grad_data_before.close()

                fname = directory + '/' + dirname_curr_ssh + '/' + ssh_grad_file_after
                ssh_grad_data_after = nc.Dataset(fname)
                ssh_grad_x_after = np.squeeze(ssh_grad_data_after.variables['ssh_grad_x'][:])  # lat x lon
                ssh_grad_y_after = np.squeeze(ssh_grad_data_after.variables['ssh_grad_y'][:])  # lat x lon
                ssh_grad_data_after.close()

                ssh_grad_x_before_ib = griddata(points_ssh_grad_x, ssh_grad_x_before.ravel(), (iceberg_lat, iceberg_lon + 360.), method='linear')
                ssh_grad_y_before_ib = griddata(points_ssh_grad_y, ssh_grad_y_before.ravel(), (iceberg_lat, iceberg_lon + 360.), method='linear')

                ssh_grad_x_after_ib = griddata(points_ssh_grad_x, ssh_grad_x_after.ravel(), (iceberg_lat, iceberg_lon + 360.), method='linear')
                ssh_grad_y_after_ib = griddata(points_ssh_grad_y, ssh_grad_y_after.ravel(), (iceberg_lat, iceberg_lon + 360.), method='linear')

                ssh_grad_x_ib = ssh_grad_x_before_ib + weight * (ssh_grad_x_after_ib - ssh_grad_x_before_ib)
                ssh_grad_y_ib = ssh_grad_y_before_ib + weight * (ssh_grad_y_after_ib - ssh_grad_y_before_ib)

                def duv_dt(t, uv):
                    iceberg_u_init, iceberg_v_init = uv
                    ib_acc_E, ib_acc_N = iceberg_acc(iceberg_lat, iceberg_u_init, iceberg_v_init, iceberg_sail, iceberg_draft, iceberg_length,
                                                     iceberg_mass, iceberg_times_dt[i], am, omega, Cw, Ca, C_wave, g, rho_air, rho_water,
                                                     u_wind_ib, v_wind_ib, [u_curr_ib, u_curr_ib2], [v_curr_ib, v_curr_ib2],
                                                     ssh_grad_x_ib, ssh_grad_y_ib, Hs_ib, wave_dir_ib)
                    return ib_acc_E, ib_acc_N

                solution = solve_ivp(duv_dt, (0., iceberg_times_dt[i]), [iceberg_u, iceberg_v], method='BDF', t_eval=[0., iceberg_times_dt[i]])

                # Results
                iceberg_u_end = solution.y[0][-1]  # Final u-velocity
                iceberg_v_end = solution.y[1][-1]  # Final v-velocity

                final_speed = np.sqrt(iceberg_u_end ** 2 + iceberg_v_end ** 2)

                if final_speed >= 2:
                    iceberg_u_end = iceberg_u
                    iceberg_v_end = iceberg_v

                if grounded_status == 'grounded':
                    iceberg_u_end = 0.
                    iceberg_v_end = 0.

                iceberg_x = np.nanmean([iceberg_u, iceberg_u_end]) * iceberg_times_dt[i]
                iceberg_y = np.nanmean([iceberg_v, iceberg_v_end]) * iceberg_times_dt[i]
                iceberg_dist = np.sqrt(iceberg_x ** 2 + iceberg_y ** 2)
                iceberg_course = 90. - np.rad2deg(np.arctan2(iceberg_y, iceberg_x))

                if iceberg_course < 0:
                    iceberg_course = iceberg_course + 360.

                iceberg_lat2, iceberg_lon2 = dist_course(Re, iceberg_lat, iceberg_lon, iceberg_dist, iceberg_course)
                iceberg_us[i + 1] = iceberg_u_end
                iceberg_vs[i + 1] = iceberg_v_end
                iceberg_lats[i + 1] = iceberg_lat2
                iceberg_lons[i + 1] = iceberg_lon2

                iceberg_bathy_depth = bathy_interp([[iceberg_lat2, iceberg_lon2]])[0]

                if iceberg_bathy_depth <= iceberg_draft:
                    grounded_status = 'grounded'
                    iceberg_us[i + 1] = 0.
                    iceberg_vs[i + 1] = 0.

            iceberg_lat_final = iceberg_lats[-1]
            iceberg_lon_final = iceberg_lons[-1]
            iceberg_total_displacement, iceberg_overall_course = dist_bearing(Re, iceberg_lat0, iceberg_lat_final, iceberg_lon0, iceberg_lon_final)

    elif grounded_status == 'grounded':
        iceberg_lats[:] = iceberg_lat0
        iceberg_lons[:] = iceberg_lon0
        iceberg_total_displacement = 0.
        iceberg_overall_course = 0.

    iceberg_mass = iceberg_mass / 1000. # Convert back to tonnes
    iceberg_times = np.array(iceberg_times)
    iceberg_times = iceberg_times.astype(str).tolist()
    return (iceberg_lat0, iceberg_lon0, iceberg_lats, iceberg_lons, iceberg_times, iceberg_total_displacement, iceberg_overall_course,
            iceberg_length, iceberg_draft, iceberg_mass, rcm_datetime0, next_rcm_time, grounded_status)


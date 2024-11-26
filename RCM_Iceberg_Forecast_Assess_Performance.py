
from scipy.interpolate import RegularGridInterpolator, interp1d
from scipy.integrate import solve_ivp
from scipy.io import loadmat, savemat
import h5py
import numpy as np
import netCDF4 as nc
import geopandas as gpd
import datetime
import glob
import os
import warnings
warnings.simplefilter(action='ignore')

def assess_rcm_iceberg_drift_forecaster(iceberg_lat0, iceberg_lon0, rcm_datetime0, iceberg_length, grounded_status, next_rcm_time,
                                        wind_wave_data_path, ocean_data_path_ssh_grad_time, ocean_data_path_u_curr, ocean_data_path_v_curr):
    bathy_data_path = './GEBCO_Bathymetric_Data/gebco_2024.nc'
    hycom_lat_lon_depth_data_file = 'C:/Users/idturnbull/Documents/MATLAB/ExxonMobil_RCM_Project_Analysis/hycom_lat_lon_depth.mat'
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

    def iceberg_acc(iceberg_lat, iceberg_u, iceberg_v, iceberg_sail, iceberg_draft, iceberg_length, dt, am, omega, Cw, Ca, C_wave, g, rho_air, rho_water,
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

    def datenum_to_datetime64(datenum):
        # MATLAB base date offset to Unix epoch (1970-01-01)
        days_offset = 719529  # Number of days from 0000-01-01 to 1970-01-01
        # Adjust datenum to days relative to Unix epoch, convert to seconds, and return datetime64
        seconds_since_epoch = (datenum - days_offset) * 86400  # full float precision
        return np.datetime64(int(seconds_since_epoch), 's')

    if grounded_status == 'grounded':
        grounded_status = 'grounded'
    else:
        grounded_status = 'not grounded'

    iceberg_u0 = 0.
    iceberg_v0 = 0.

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

    forecast_time = rcm_datetime0
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
        wind_wave_data = loadmat(wind_wave_data_path)
        lat_wind = wind_wave_data['lat_wind']
        lon_wind = wind_wave_data['lon_wind']
        lat_wave = wind_wave_data['lat_wave']
        lon_wave = wind_wave_data['lon_wave']
        time_wind_wave = wind_wave_data['time']
        u_wind = wind_wave_data['u_wind']
        v_wind = wind_wave_data['v_wind']
        Hs = wind_wave_data['Hs']
        wave_dir = wind_wave_data['wave_dir']
        time_wind_wave_temp = []

        for i in range(len(time_wind_wave)):
            time_temp = datenum_to_datetime64(time_wind_wave[i, 0])
            time_wind_wave_temp.append(time_temp)

        time_wind_wave = time_wind_wave_temp
        del(time_wind_wave_temp, time_temp)
        lat_wind = lat_wind[:, 0]
        lon_wind = lon_wind[0, :]
        lat_wave = lat_wave[:, 0]
        lon_wave = lon_wave[0, :]
        lat_wind = lat_wind.flatten()
        lon_wind = lon_wind.flatten()
        lat_wave = lat_wave.flatten()
        lon_wave = lon_wave.flatten()
        wave_dir_E = np.sin(np.deg2rad(wave_dir))
        wave_dir_N = np.cos(np.deg2rad(wave_dir))

        hycom_lat_lon_depth_data = loadmat(hycom_lat_lon_depth_data_file)
        hycom_data_ssh_grad_time = loadmat(ocean_data_path_ssh_grad_time)

        depth_curr = hycom_lat_lon_depth_data['depth_hycom']
        lat_curr = hycom_lat_lon_depth_data['lat_grid_hycom']
        lon_curr = hycom_lat_lon_depth_data['lon_grid_hycom']
        lat_curr = lat_curr[:, 0]
        lon_curr = lon_curr[0, :]
        lat_curr = lat_curr.flatten()
        lon_curr = lon_curr.flatten()

        hycom_data_u_curr = h5py.File(ocean_data_path_u_curr, 'r')
        u_curr = hycom_data_u_curr['U']
        del(hycom_data_u_curr)
        hycom_data_v_curr = h5py.File(ocean_data_path_v_curr, 'r')
        v_curr = hycom_data_v_curr['V']
        del(hycom_data_v_curr)

        u_curr = np.transpose(u_curr, (3, 2, 1, 0))
        v_curr = np.transpose(v_curr, (3, 2, 1, 0))

        ssh_grad_x = hycom_data_ssh_grad_time['SSH_grad_x']
        ssh_grad_y = hycom_data_ssh_grad_time['SSH_grad_y']
        lat_ssh_grad_x = hycom_data_ssh_grad_time['lat_ssh_grad_x']
        lat_ssh_grad_y = hycom_data_ssh_grad_time['lat_ssh_grad_y']
        lon_ssh_grad_x = hycom_data_ssh_grad_time['lon_ssh_grad_x']
        lon_ssh_grad_y = hycom_data_ssh_grad_time['lon_ssh_grad_y']
        time_curr_ssh = hycom_data_ssh_grad_time['time']
        time_curr_ssh_temp = []

        for i in range(len(time_curr_ssh)):
            time_temp = datenum_to_datetime64(time_curr_ssh[i, 0])
            time_curr_ssh_temp.append(time_temp)

        time_curr_ssh = time_curr_ssh_temp
        del (time_curr_ssh_temp, time_temp)
        lat_ssh_grad_x = lat_ssh_grad_x[:, 0]
        lat_ssh_grad_y = lat_ssh_grad_y[:, 0]
        lon_ssh_grad_x = lon_ssh_grad_x[0, :]
        lon_ssh_grad_y = lon_ssh_grad_y[0, :]
        lat_ssh_grad_x = lat_ssh_grad_x.flatten()
        lat_ssh_grad_y = lat_ssh_grad_y.flatten()
        lon_ssh_grad_x = lon_ssh_grad_x.flatten()
        lon_ssh_grad_y = lon_ssh_grad_y.flatten()

        time_wind_wave_ref = time_wind_wave[0]
        time_wind_wave_hours = (time_wind_wave - time_wind_wave_ref) / np.timedelta64(1, 'h')
        iceberg_time_wind_wave_hours = (iceberg_times - time_wind_wave_ref) / np.timedelta64(1, 'h')

        time_curr_ssh_ref = time_curr_ssh[0]
        time_curr_ssh_hours = (time_curr_ssh - time_curr_ssh_ref) / np.timedelta64(1, 'h')
        iceberg_time_curr_ssh_hours = (iceberg_times - time_curr_ssh_ref) / np.timedelta64(1, 'h')

        loc_depth = np.argwhere(depth_curr <= iceberg_draft)
        loc_depth = np.append(loc_depth, loc_depth[-1] + 1)
        depth_curr_ib = depth_curr[loc_depth]
        depth_curr_ib = list(depth_curr_ib[:, 0])
        depth_curr_ib_interp = np.arange(0., iceberg_draft, 0.001)

        u_curr = u_curr[:, :, loc_depth, :]
        v_curr = v_curr[:, :, loc_depth, :]

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

            f_u_wind = RegularGridInterpolator((lat_wind, lon_wind, time_wind_wave_hours), u_wind, method='linear', bounds_error=True, fill_value=np.nan)
            u_wind_ib = float(f_u_wind([iceberg_lat, iceberg_lon, iceberg_time_wind_wave_hours[i]]))
            f_v_wind = RegularGridInterpolator((lat_wind, lon_wind, time_wind_wave_hours), v_wind, method='linear', bounds_error=True, fill_value=np.nan)
            v_wind_ib = float(f_v_wind([iceberg_lat, iceberg_lon, iceberg_time_wind_wave_hours[i]]))
            f_Hs = RegularGridInterpolator((lat_wave, lon_wave, time_wind_wave_hours), Hs, method='linear', bounds_error=True, fill_value=np.nan)
            Hs_ib = float(f_Hs([iceberg_lat, iceberg_lon, iceberg_time_wind_wave_hours[i]]))
            f_wave_dir_E = RegularGridInterpolator((lat_wave, lon_wave, time_wind_wave_hours), wave_dir_E, method='linear', bounds_error=True, fill_value=np.nan)
            wave_dir_E_ib = float(f_wave_dir_E([iceberg_lat, iceberg_lon, iceberg_time_wind_wave_hours[i]]))
            f_wave_dir_N = RegularGridInterpolator((lat_wave, lon_wave, time_wind_wave_hours), wave_dir_N, method='linear', bounds_error=True, fill_value=np.nan)
            wave_dir_N_ib = float(f_wave_dir_N([iceberg_lat, iceberg_lon, iceberg_time_wind_wave_hours[i]]))

            wave_dir_ib = 90. - np.rad2deg(np.arctan2(wave_dir_N_ib, wave_dir_E_ib))

            if wave_dir_ib < 0:
                wave_dir_ib = wave_dir_ib + 360.

            f_ssh_grad_x = RegularGridInterpolator((lat_ssh_grad_x, lon_ssh_grad_x, time_curr_ssh_hours), ssh_grad_x,
                                                   method='linear', bounds_error=True, fill_value=np.nan)
            ssh_grad_x_ib = float(f_ssh_grad_x([iceberg_lat, iceberg_lon, iceberg_time_curr_ssh_hours[i]]))
            f_ssh_grad_y = RegularGridInterpolator((lat_ssh_grad_y, lon_ssh_grad_y, time_curr_ssh_hours), ssh_grad_y,
                                                   method='linear', bounds_error=True, fill_value=np.nan)
            ssh_grad_y_ib = float(f_ssh_grad_y([iceberg_lat, iceberg_lon, iceberg_time_curr_ssh_hours[i]]))

            u_curr_depth_list = []
            u_curr_depth_list2 = []
            v_curr_depth_list = []
            v_curr_depth_list2 = []

            for n in range(len(depth_curr_ib)):
                f_u_curr = RegularGridInterpolator((lat_curr, lon_curr, time_curr_ssh_hours), np.squeeze(u_curr[:, :, n, :]),
                                                          method='linear', bounds_error=True, fill_value=np.nan)
                u_curr_ib_temp = float(f_u_curr([iceberg_lat, iceberg_lon, iceberg_time_curr_ssh_hours[i]]))
                u_curr_ib_temp2 = float(f_u_curr([iceberg_lat, iceberg_lon, iceberg_time_curr_ssh_hours[i + 1]]))
                u_curr_depth_list.append(u_curr_ib_temp)
                u_curr_depth_list2.append(u_curr_ib_temp2)

                f_v_curr = RegularGridInterpolator((lat_curr, lon_curr, time_curr_ssh_hours), np.squeeze(v_curr[:, :, n, :]),
                                                   method='linear', bounds_error=True, fill_value=np.nan)
                v_curr_ib_temp = float(f_v_curr([iceberg_lat, iceberg_lon, iceberg_time_curr_ssh_hours[i]]))
                v_curr_ib_temp2 = float(f_v_curr([iceberg_lat, iceberg_lon, iceberg_time_curr_ssh_hours[i + 1]]))
                v_curr_depth_list.append(v_curr_ib_temp)
                v_curr_depth_list2.append(v_curr_ib_temp2)

            interp_func = interp1d(depth_curr_ib, u_curr_depth_list, kind='linear', fill_value='extrapolate')
            u_curr_depth_list = interp_func(depth_curr_ib_interp)
            interp_func = interp1d(depth_curr_ib, v_curr_depth_list, kind='linear', fill_value='extrapolate')
            v_curr_depth_list = interp_func(depth_curr_ib_interp)
            u_curr_ib = np.nanmean(u_curr_depth_list)
            v_curr_ib = np.nanmean(v_curr_depth_list)
            u_curr_ib2 = np.nanmean(u_curr_depth_list2)
            v_curr_ib2 = np.nanmean(v_curr_depth_list2)

            def duv_dt(t, uv):
                iceberg_u_init, iceberg_v_init = uv
                ib_acc_E, ib_acc_N = iceberg_acc(iceberg_lat, iceberg_u_init, iceberg_v_init, iceberg_sail, iceberg_draft, iceberg_length,
                                                 iceberg_times_dt[i], am, omega, Cw, Ca, C_wave, g, rho_air, rho_water,
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

    elif grounded_status == 'grounded':
        iceberg_lats[:] = iceberg_lat0
        iceberg_lons[:] = iceberg_lon0

    iceberg_mass = iceberg_mass / 1000. # Convert back to tonnes
    iceberg_times = np.array(iceberg_times)
    return (iceberg_lat0, iceberg_lon0, iceberg_lats, iceberg_lons, iceberg_times, iceberg_length,
            iceberg_draft, iceberg_mass, rcm_datetime0, next_rcm_time, grounded_status)

def datetime64_to_datenum(dt64):
    # Convert numpy.datetime64 to days since 1970-01-01
    days_since_epoch = (dt64 - np.datetime64('1970-01-01')) / np.timedelta64(1, 'D')
    # MATLAB datenum offset to convert 1970-01-01 in numpy to 0000-01-01 in MATLAB
    matlab_offset = 719529  # days from 0000-01-01 to 1970-01-01 in MATLAB
    # Add offset to convert to MATLAB datenum
    return days_since_epoch + matlab_offset

rcm_shapefiles_path_2023 = 'C:/Users/idturnbull/Documents/ExxonMobil_RCM_Project/Drift_SamleData_finished/2023_tracks/'
rcm_shapefiles_path_2024 = 'C:/Users/idturnbull/Documents/ExxonMobil_RCM_Project/Drift_SamleData_finished/2024_tracks/'
output_filepath = 'C:/Users/idturnbull/Documents/MATLAB/ExxonMobil_RCM_Project_Analysis/RCM_Iceberg_Forecaster_Results/'
era5_wind_wave_data_2023_file = 'C:/Users/idturnbull/Documents/MATLAB/ExxonMobil_RCM_Project_Analysis/ERA5_Wind_Wave_Data_2023.mat'
era5_wind_wave_data_2024_file = 'C:/Users/idturnbull/Documents/MATLAB/ExxonMobil_RCM_Project_Analysis/ERA5_Wind_Wave_Data_2024.mat'
hycom_data_2023_file_ssh_grad_time = 'C:/Users/idturnbull/Documents/MATLAB/ExxonMobil_RCM_Project_Analysis/HYCOM_Data/hycom_data_processed_2023_ssh_grad_time.mat'
hycom_data_2023_file_u_curr = 'C:/Users/idturnbull/Documents/MATLAB/ExxonMobil_RCM_Project_Analysis/HYCOM_Data/hycom_data_processed_2023_u.mat'
hycom_data_2023_file_v_curr = 'C:/Users/idturnbull/Documents/MATLAB/ExxonMobil_RCM_Project_Analysis/HYCOM_Data/hycom_data_processed_2023_v.mat'
hycom_data_2024_file_ssh_grad_time = 'C:/Users/idturnbull/Documents/MATLAB/ExxonMobil_RCM_Project_Analysis/HYCOM_Data/hycom_data_processed_2024_ssh_grad_time.mat'
hycom_data_2024_file_u_curr = 'C:/Users/idturnbull/Documents/MATLAB/ExxonMobil_RCM_Project_Analysis/HYCOM_Data/hycom_data_processed_2024_u.mat'
hycom_data_2024_file_v_curr = 'C:/Users/idturnbull/Documents/MATLAB/ExxonMobil_RCM_Project_Analysis/HYCOM_Data/hycom_data_processed_2024_v.mat'

rcm_shapefiles_list_2023 = glob.glob(os.path.join(rcm_shapefiles_path_2023, '*.shp'))
rcm_shapefiles_list_2024 = glob.glob(os.path.join(rcm_shapefiles_path_2024, '*.shp'))

rcm_shapefiles_names_2023 = [os.path.basename(file) for file in rcm_shapefiles_list_2023]
rcm_shapefiles_names_2024 = [os.path.basename(file) for file in rcm_shapefiles_list_2024]

for p in range(len(rcm_shapefiles_names_2023)):
    rcm_shapefile = rcm_shapefiles_names_2023[p]
    gdf = gpd.read_file(os.path.join(rcm_shapefiles_path_2023, rcm_shapefile))
    acqdates = []

    for i in range(len(gdf['acqDate'])):
        date = gdf['acqDate'].iloc[i]

        try:
            date = np.datetime64(datetime.datetime.strptime(date, '%Y%m%d_%H%M%S'))
            acqdates.append(date)
        except:
            pass

    acqdates = np.array(acqdates)
    sorted_indices = np.argsort(acqdates)
    acqdates = acqdates[sorted_indices]

    for m in range(len(sorted_indices) - 1):
        try:
            index0 = sorted_indices[m]
            index1 = sorted_indices[m + 1]
            point = gdf.geometry.iloc[index0]
            iceberg_lat0 = point.y
            iceberg_lon0 = point.x
            point = gdf.geometry.iloc[index1]
            iceberg_lat_end = point.y
            iceberg_lon_end = point.x
            iceberg_length = gdf['WtrLin'].iloc[index0]
            grounded_status = gdf['Grounded'].iloc[index0]
            rcm_datetime0 = gdf['acqDate'].iloc[index0]
            next_rcm_time = gdf['acqDate'].iloc[index1]
            rcm_datetime0 = np.datetime64(datetime.datetime.strptime(rcm_datetime0, '%Y%m%d_%H%M%S'))
            next_rcm_time = np.datetime64(datetime.datetime.strptime(next_rcm_time, '%Y%m%d_%H%M%S'))
            (iceberg_lat0, iceberg_lon0, iceberg_lats, iceberg_lons, iceberg_times, iceberg_length,
             iceberg_draft, iceberg_mass, rcm_datetime0, next_rcm_time, grounded_status) = (
                assess_rcm_iceberg_drift_forecaster(iceberg_lat0, iceberg_lon0, rcm_datetime0, iceberg_length, grounded_status, next_rcm_time,
                                                    era5_wind_wave_data_2023_file, hycom_data_2023_file_ssh_grad_time,
                                                    hycom_data_2023_file_u_curr, hycom_data_2023_file_v_curr))
            iceberg_times = datetime64_to_datenum(iceberg_times)
            rcm_datetime0 = datetime64_to_datenum(rcm_datetime0)
            next_rcm_time = datetime64_to_datenum(next_rcm_time)
            savemat(output_filepath + rcm_shapefile[:-4] + '_' + str(m + 1) + '.mat', {'iceberg_times': iceberg_times, 'iceberg_lat0': iceberg_lat0,
                                                                                                 'iceberg_lon0': iceberg_lon0, 'rcm_datetime0': rcm_datetime0,
                                                                                                 'next_rcm_time': next_rcm_time, 'iceberg_lats': iceberg_lats,
                                                                                                 'iceberg_lons': iceberg_lons, 'iceberg_length': iceberg_length,
                                                                                                 'iceberg_draft': iceberg_draft, 'iceberg_mass': iceberg_mass,
                                                                                                 'grounded_status': grounded_status, 'iceberg_lat_end': iceberg_lat_end,
                                                                                                 'iceberg_lon_end': iceberg_lon_end})
        except:
            pass

for p in range(len(rcm_shapefiles_names_2024)):
    rcm_shapefile = rcm_shapefiles_names_2024[p]
    gdf = gpd.read_file(os.path.join(rcm_shapefiles_path_2024, rcm_shapefile))
    acqdates = []

    for i in range(len(gdf['acqDate'])):
        date = gdf['acqDate'].iloc[i]

        try:
            date = np.datetime64(datetime.datetime.strptime(date, '%Y%m%d_%H%M%S'))
            acqdates.append(date)
        except:
            pass

    acqdates = np.array(acqdates)
    sorted_indices = np.argsort(acqdates)
    acqdates = acqdates[sorted_indices]

    for m in range(len(sorted_indices) - 1):
        try:
            index0 = sorted_indices[m]
            index1 = sorted_indices[m + 1]
            point = gdf.geometry.iloc[index0]
            iceberg_lat0 = point.y
            iceberg_lon0 = point.x
            point = gdf.geometry.iloc[index1]
            iceberg_lat_end = point.y
            iceberg_lon_end = point.x
            iceberg_length = gdf['WtrLin'].iloc[index0]
            grounded_status = gdf['Grounded'].iloc[index0]
            rcm_datetime0 = gdf['acqDate'].iloc[index0]
            next_rcm_time = gdf['acqDate'].iloc[index1]
            rcm_datetime0 = np.datetime64(datetime.datetime.strptime(rcm_datetime0, '%Y%m%d_%H%M%S'))
            next_rcm_time = np.datetime64(datetime.datetime.strptime(next_rcm_time, '%Y%m%d_%H%M%S'))
            (iceberg_lat0, iceberg_lon0, iceberg_lats, iceberg_lons, iceberg_times, iceberg_length,
             iceberg_draft, iceberg_mass, rcm_datetime0, next_rcm_time, grounded_status) = (
                assess_rcm_iceberg_drift_forecaster(iceberg_lat0, iceberg_lon0, rcm_datetime0, iceberg_length, grounded_status, next_rcm_time,
                                                    era5_wind_wave_data_2024_file, hycom_data_2024_file_ssh_grad_time,
                                                    hycom_data_2024_file_u_curr, hycom_data_2024_file_v_curr))
            iceberg_times = datetime64_to_datenum(iceberg_times)
            rcm_datetime0 = datetime64_to_datenum(rcm_datetime0)
            next_rcm_time = datetime64_to_datenum(next_rcm_time)
            savemat(output_filepath + rcm_shapefile[:-4] + '_' + str(m + 1) + '.mat', {'iceberg_times': iceberg_times, 'iceberg_lat0': iceberg_lat0,
                                                                                                 'iceberg_lon0': iceberg_lon0, 'rcm_datetime0': rcm_datetime0,
                                                                                                 'next_rcm_time': next_rcm_time, 'iceberg_lats': iceberg_lats,
                                                                                                 'iceberg_lons': iceberg_lons, 'iceberg_length': iceberg_length,
                                                                                                 'iceberg_draft': iceberg_draft, 'iceberg_mass': iceberg_mass,
                                                                                                 'grounded_status': grounded_status, 'iceberg_lat_end': iceberg_lat_end,
                                                                                                 'iceberg_lon_end': iceberg_lon_end})
        except:
            pass


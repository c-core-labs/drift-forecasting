
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

def assess_rcm_iceberg_drift_deterioration_forecaster(iceberg_lat0, iceberg_lon0, rcm_datetime0, iceberg_length, grounded_status, next_rcm_time,
                                        wind_wave_data_path, ocean_data_path_ssh_grad_time, ocean_data_path_u_curr, ocean_data_path_v_curr,
                                                      ocean_data_path_temp, ocean_data_path_sal):
    bathy_data_path = './GEBCO_Bathymetric_Data/gebco_2024.nc'
    hycom_lat_lon_depth_data_file = 'C:/Users/idturnbull/Documents/MATLAB/ExxonMobil_RCM_Project_Analysis/hycom_lat_lon_depth.mat'
    deg_radius = 30
    g = 9.80665
    rho_water = 1023.6
    rho_air = 1.225
    rho_ice = 910.
    ice_albedo = 0.1
    Lf_ice = 3.36e5
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

    def iceberg_acc(iceberg_lat, iceberg_u, iceberg_v, iceberg_sail, iceberg_draft, iceberg_length, iceberg_mass, dt, am, omega, Cw, Ca, C_wave, g, rho_air, rho_water,
                      u_wind, v_wind, u_curr, v_curr, ssh_grad_x, ssh_grad_y, Hs, wave_dir):
        iceberg_keel = iceberg_length * iceberg_draft

        if (np.any(np.isnan(u_wind)) or np.any(np.isnan(v_wind)) or np.any(np.isinf(u_wind)) or np.any(np.isinf(v_wind))
                or np.any(~np.isreal(u_wind)) or np.any(~np.isreal(v_wind))):
            u_wind = 0.
            v_wind = 0.

        wind_dir = 90. - np.rad2deg(np.arctan2(v_wind, u_wind))

        if wind_dir < 0:
            wind_dir = wind_dir + 360.

        if np.any(np.isnan(wave_dir)) or np.any(np.isinf(wave_dir)) or np.any(~np.isreal(wave_dir)):
            wave_dir = wind_dir

        if np.any(np.isnan(Hs)) or np.any(np.isinf(Hs)) or np.any(~np.isreal(Hs)):
            Hs = 0.001

        if (np.any(np.isnan(iceberg_u)) or np.any(np.isnan(iceberg_v)) or np.any(np.isinf(iceberg_u)) or np.any(np.isinf(iceberg_v))
                or np.any(~np.isreal(iceberg_u)) or np.any(~np.isreal(iceberg_v))):
            iceberg_u = 0.
            iceberg_v = 0.

        if (np.any(np.isnan(u_curr[0])) or np.any(np.isnan(v_curr[0])) or np.any(np.isinf(u_curr[0])) or np.any(np.isinf(v_curr[0]))
                or np.any(~np.isreal(u_curr[0])) or np.any(~np.isreal(v_curr[0]))):
            u_curr[0] = 0.
            v_curr[0] = 0.

        if (np.any(np.isnan(u_curr[1])) or np.any(np.isnan(v_curr[1])) or np.any(np.isinf(u_curr[1])) or np.any(np.isinf(v_curr[1]))
                or np.any(~np.isreal(u_curr[1])) or np.any(~np.isreal(v_curr[1]))):
            u_curr[1] = 0.
            v_curr[1] = 0.

        if (np.any(np.isnan(ssh_grad_x)) or np.any(np.isnan(ssh_grad_y)) or np.any(np.isinf(ssh_grad_x)) or np.any(np.isinf(ssh_grad_y)) or
                np.any(~np.isreal(ssh_grad_x)) or np.any(~np.isreal(ssh_grad_y))):
            ssh_grad_x = 0.
            ssh_grad_y = 0.

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

    def iceberg_det(iceberg_length, iceberg_mass, solar_rad, ice_albedo, Lf_ice, rho_ice, water_temps, water_sals, water_depths, air_temp,
                    u_curr, v_curr, u_wind, v_wind, iceberg_u, iceberg_v, Hs, wave_pd, time_dt):
        water_temps = np.array(water_temps)
        water_sals = np.array(water_sals)
        water_depths = np.array(water_depths)
        water_temps[np.isnan(water_temps) | np.isinf(water_temps) | ~np.isreal(water_temps)] = 0.
        water_sals[np.isnan(water_sals) | np.isinf(water_sals) | ~np.isreal(water_sals)] = 33.

        if np.any(np.isnan(Hs)) or np.any(np.isinf(Hs)) or np.any(~np.isreal(Hs)):
            Hs = 0.001

        if (np.any(np.isnan(iceberg_u)) or np.any(np.isnan(iceberg_v)) or np.any(np.isinf(iceberg_u)) or np.any(np.isinf(iceberg_v))
                or np.any(~np.isreal(iceberg_u)) or np.any(~np.isreal(iceberg_v))):
            iceberg_u = 0.
            iceberg_v = 0.

        if (np.any(np.isnan(u_curr)) or np.any(np.isnan(v_curr)) or np.any(np.isinf(u_curr)) or np.any(np.isinf(v_curr))
                or np.any(~np.isreal(u_curr)) or np.any(~np.isreal(v_curr))):
            u_curr = 0.
            v_curr = 0.

        if np.any(np.isnan(wave_pd)) or np.any(np.isinf(wave_pd)) or np.any(~np.isreal(wave_pd)):
            wave_pd = 16.

        if np.any(np.isnan(solar_rad)) or np.any(np.isinf(solar_rad)) or np.any(~np.isreal(solar_rad)):
            solar_rad = 0.

        if (np.any(np.isnan(u_wind)) or np.any(np.isnan(v_wind)) or np.any(np.isinf(u_wind)) or np.any(np.isinf(v_wind))
                or np.any(~np.isreal(u_wind)) or np.any(~np.isreal(v_wind))):
            u_wind = 0.
            v_wind = 0.

        air_therm_diff_array = np.array([[-100, 7.88e-6], [-50, 13.02e-6], [-30, 15.33e-6], [-20, 16.54e-6], [-10, 17.78e-6], [-5, 18.41e-6],
                                [0, 19.06e-6], [5, 19.71e-6], [10, 20.36e-6], [20, 21.7e-6], [25, 22.39e-6], [30, 23.07e-6], [50, 25.91e-6],
                                [75, 29.61e-6], [100, 33.49e-6], [125, 37.53e-6], [150, 41.73e-6], [175, 46.07e-6], [200, 50.56e-6],
                                [250, 59.93e-6], [300, 69.79e-6]])
        air_kin_vis_array = np.array([[-75, 7.4e-6], [-50, 9.22e-6], [-25, 11.18e-6], [-15, 12.01e-6], [-10, 12.43e-6], [-5, 12.85e-6],
                             [0, 13.28e-6], [5, 13.72e-6], [10, 14.16e-6], [15, 14.61e-6], [20, 15.06e-6], [25, 15.52e-6],
                             [30, 15.98e-6], [40, 16.92e-6], [50, 17.88e-6], [60, 18.86e-6], [80, 20.88e-6], [100, 22.97e-6],
                             [125, 25.69e-6], [150, 28.51e-6], [175, 31.44e-6], [200, 34.47e-6], [225, 37.6e-6], [300, 47.54e-6],
                             [412, 63.82e-6], [500, 77.72e-6], [600, 94.62e-6], [700, 112.6e-6], [800, 131.7e-6], [900, 151.7e-6],
                             [1000, 172.7e-6], [1100, 194.6e-6]])
        air_temp_therm_diff = air_therm_diff_array[:, 0]
        air_temp_kin_vis = air_kin_vis_array[:, 0]
        air_thermal_diffusivity = air_therm_diff_array[:, 1]
        air_kinematic_viscosity = air_kin_vis_array[:, 1]
        air_therm_diff = np.interp(air_temp, air_temp_therm_diff, air_thermal_diffusivity)
        air_kin_vis = np.interp(air_temp, air_temp_kin_vis, air_kinematic_viscosity)
        water_sal_therm_diff_array = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120])
        water_temp_therm_diff_array = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120])
        water_sal_kin_vis_array = np.array([0, 10, 20, 30, 35, 40, 50, 60, 70, 80, 90, 100, 110, 120])
        water_temp_kin_vis_array = np.array([0, 10, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120])
        water_therm_diff_array = np.array([[1.36, 1.37, 1.38, 1.38, 1.39, 1.4, 1.41, 1.41, 1.42, 1.42, 1.43, 1.43, 1.43],
                                  [1.4, 1.41, 1.42, 1.43, 1.43, 1.44, 1.45, 1.45, 1.46, 1.46, 1.47, 1.47, 1.47],
                                  [1.44, 1.45, 1.46, 1.46, 1.47, 1.48, 1.48, 1.49, 1.49, 1.5, 1.5, 1.51, 1.51],
                                  [1.48, 1.49, 1.5, 1.5, 1.51, 1.51, 1.52, 1.52, 1.53, 1.54, 1.54, 1.54, 1.55],
                                  [1.52, 1.52, 1.53, 1.54, 1.54, 1.55, 1.55, 1.56, 1.56, 1.57, 1.57, 1.58, 1.58],
                                  [1.55, 1.56, 1.56, 1.57, 1.57, 1.58, 1.59, 1.59, 1.6, 1.6, 1.61, 1.61, 1.62],
                                  [1.58, 1.59, 1.59, 1.6, 1.6, 1.61, 1.62, 1.62, 1.63, 1.63, 1.64, 1.64, 1.65],
                                  [1.61, 1.61, 1.62, 1.63, 1.63, 1.64, 1.64, 1.65, 1.65, 1.66, 1.67, 1.67, 1.68],
                                  [1.63, 1.64, 1.64, 1.65, 1.66, 1.66, 1.67, 1.67, 1.68, 1.69, 1.69, 1.7, 1.7],
                                  [1.65, 1.66, 1.67, 1.67, 1.68, 1.68, 1.69, 1.7, 1.7, 1.71, 1.71, 1.72, 1.73],
                                  [1.67, 1.68, 1.69, 1.69, 1.7, 1.7, 1.71, 1.72, 1.72, 1.73, 1.74, 1.74, 1.75],
                                  [1.69, 1.7, 1.7, 1.71, 1.71, 1.72, 1.73, 1.73, 1.74, 1.75, 1.75, 1.76, 1.76],
                                  [1.7, 1.71, 1.72, 1.72, 1.73, 1.73, 1.74, 1.75, 1.75, 1.76, 1.77, 1.77, 1.78]]) * 1e7
        water_kin_vis_array = np.array([[17.92, 18.06, 18.23, 18.43, 18.54, 18.65, 18.9, 19.16, 19.46, 19.77, 20.11, 20.46, 20.84, 21.24],
                               [13.07, 13.2, 13.35, 13.51, 13.6, 13.69, 13.89, 14.1, 14.33, 14.57, 14.82, 15.09, 15.38, 15.67],
                               [10.04, 10.16, 10.29, 10.43, 10.5, 10.58, 10.75, 10.92, 11.1, 11.3, 11.5, 11.71, 11.93, 12.17],
                               [8.93, 9.04, 9.17, 9.3, 9.37, 9.44, 9.59, 9.75, 9.92, 10.09, 10.28, 10.47, 10.67, 10.87],
                               [8.01, 8.12, 8.23, 8.36, 8.42, 8.49, 8.63, 8.77, 8.93, 9.09, 9.25, 9.43, 9.61, 9.8],
                               [6.58, 6.68, 6.78, 6.89, 6.95, 7., 7.12, 7.25, 7.38, 7.52, 7.66, 7.81, 7.96, 8.11],
                               [5.53, 5.62, 5.71, 5.81, 5.86, 5.91, 6.02, 6.13, 6.24, 6.36, 6.48, 6.61, 6.74, 6.87],
                               [4.74, 4.82, 4.91, 4.99, 5.04, 5.08, 5.18, 5.28, 5.38, 5.48, 5.59, 5.7, 5.81, 5.93],
                               [4.13, 4.2, 4.28, 4.36, 4.4, 4.44, 4.52, 4.61, 4.7, 4.79, 4.89, 4.98, 5.08, 5.19],
                               [3.65, 3.71, 3.78, 3.85, 3.89, 3.93, 4., 4.08, 4.16, 4.24, 4.33, 4.42, 4.51, 4.6],
                               [3.26, 3.32, 3.38, 3.45, 3.48, 3.51, 3.58, 3.65, 3.73, 3.8, 3.88, 3.96, 4.04, 4.12],
                               [2.94, 3., 3.05, 3.11, 3.14, 3.17, 3.24, 3.3, 3.37, 3.44, 3.51, 3.58, 3.65, 3.73],
                               [2.68, 2.73, 2.78, 2.83, 2.86, 2.89, 2.95, 3.01, 3.07, 3.13, 3.2, 3.26, 3.33, 3.4],
                               [2.46, 2.51, 2.55, 2.6, 2.63, 2.65, 2.71, 2.76, 2.82, 2.87, 2.93, 3., 3.06, 3.12]]) * 1e7
        water_therm_diff_interpolator = RegularGridInterpolator((water_temp_therm_diff_array, water_sal_therm_diff_array), water_therm_diff_array)
        water_kin_vis_interpolator = RegularGridInterpolator((water_temp_kin_vis_array, water_sal_kin_vis_array), water_kin_vis_array)
        water_therm_diffs = []
        water_kin_viss = []

        for k in range(len(water_depths)):
            try:
                water_therm_diff = water_therm_diff_interpolator((water_temps[k], water_sals[k]))
                water_kin_vis = water_kin_vis_interpolator((water_temps[k], water_sals[k]))
            except:
                water_therm_diff = water_therm_diff_interpolator((0., 33.))
                water_kin_vis = water_kin_vis_interpolator((0., 33.))

            water_therm_diffs.append(water_therm_diff)
            water_kin_viss.append(water_kin_vis)

        water_temp_mean = np.nanmean(np.array(water_temps))
        water_therm_diff_mean = np.nanmean(np.array(water_therm_diffs))
        water_kin_vis_mean = np.nanmean(np.array(water_kin_viss))
        water_sal_mean = np.nanmean(np.array(water_sals))
        V_solar = time_dt * solar_rad * (1. - ice_albedo) / (Lf_ice * rho_ice)
        Tf = -0.036 - (0.0499 * water_sal_mean) - (0.000112 * (water_sal_mean ** 2))
        Tfp = Tf * np.exp(-0.19 * (water_temp_mean - Tf))
        delta_T_water = water_temp_mean - Tfp
        Vb = (2.78 * delta_T_water + 0.47 * (delta_T_water ** 2)) * time_dt / (365. * 24. * 3600.)
        Vr_water = np.sqrt((u_curr - iceberg_u) ** 2 + (v_curr - iceberg_v) ** 2)
        Vr_air = np.sqrt((u_wind - iceberg_u) ** 2 + (v_wind - iceberg_v) ** 2)
        Pr_water = water_kin_vis_mean / water_therm_diff_mean
        Pr_air = air_kin_vis / air_therm_diff
        Re_water = Vr_water * iceberg_length / water_kin_vis_mean
        Re_air = Vr_air * iceberg_length / air_kin_vis
        Nu_water = 0.058 * (Re_water ** 0.8) * (Pr_water ** 0.4)
        Nu_air = 0.058 * (Re_air ** 0.8) * (Pr_air ** 0.4)
        qf_water = Nu_water * water_therm_diff_mean * delta_T_water / iceberg_length
        qf_air = Nu_air * air_therm_diff * air_temp / iceberg_length
        Vf_water = time_dt * qf_water / (Lf_ice * rho_ice)
        Vf_air = time_dt * qf_air / (Lf_ice * rho_ice)
        Vwe = 0.000146 * ((0.01 / Hs) ** 0.2) * (Hs / wave_pd) * delta_T_water
        h = 0.196 * iceberg_length
        Fl = 0.33 * np.sqrt(37.5 * Hs + h ** 2)
        tc = Fl / Vwe
        Vc = 0.64 * iceberg_length * Fl * h * time_dt / tc

        if V_solar < 0:
            V_solar = 0.

        if Vb < 0:
            Vb = 0.

        if Vf_water < 0:
            Vf_water = 0.

        if Vf_air < 0:
            Vf_air = 0.

        if Vc < 0:
            Vc = 0.

        iceberg_length_loss = V_solar + Vb + Vf_water + Vf_air
        iceberg_mass_loss_non_calving = 0.45 * rho_ice * (iceberg_length_loss ** 3)
        iceberg_mass_loss_calving = Vc * rho_ice
        iceberg_mass_loss = iceberg_mass_loss_non_calving + iceberg_mass_loss_calving
        new_iceberg_mass = iceberg_mass - iceberg_mass_loss

        if new_iceberg_mass < 0:
            new_iceberg_mass = 0.

        new_iceberg_length = np.cbrt(new_iceberg_mass / (0.45 * rho_ice))
        new_iceberg_draft = 1.78 * (new_iceberg_length ** 0.71)  # meters
        new_iceberg_sail = 0.077 * (new_iceberg_length ** 2)  # m ** 2
        return new_iceberg_length, new_iceberg_draft, new_iceberg_sail, new_iceberg_mass

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
    iceberg_times_dt = [float((iceberg_times[i + 1] - iceberg_times[i]) / np.timedelta64(1, 's')) for i in range(len(iceberg_times) - 1)]

    # Convert to list for easier inspection if desired
    iceberg_times_dt = list(iceberg_times_dt)

    iceberg_lats = np.empty((len(iceberg_times),))
    iceberg_lons = np.empty((len(iceberg_times),))
    iceberg_lengths = np.empty((len(iceberg_times),))
    iceberg_drafts = np.empty((len(iceberg_times),))
    iceberg_sails = np.empty((len(iceberg_times),))
    iceberg_masses = np.empty((len(iceberg_times),))

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
    sw_rad = wind_wave_data['sw_rad']
    airT = wind_wave_data['airT']
    wave_pd = wind_wave_data['wave_pd']
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
    depth_curr = depth_curr.flatten()

    distance = np.sqrt((lat_curr - iceberg_lat0) ** 2 + (lon_curr - iceberg_lon0) ** 2)

    # Find the indices of the nearest grid point
    nearest_idx = np.unravel_index(np.argmin(distance, axis=None), distance.shape)
    nearest_lat_idx, nearest_lon_idx = nearest_idx

    # Define the index range to restrict to within 10 indices in all directions
    lat_min = max(nearest_lat_idx - deg_radius, 0)
    lat_max = min(nearest_lat_idx + deg_radius, lat_curr.shape[0] - 1)
    lon_min = max(nearest_lon_idx - deg_radius, 0)
    lon_max = min(nearest_lon_idx + deg_radius, lon_curr.shape[1] - 1)

    lat_curr = lat_curr[lat_min:lat_max + 1, lon_min:lon_max + 1]
    lon_curr = lon_curr[lat_min:lat_max + 1, lon_min:lon_max + 1]

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
    hycom_data_temp = h5py.File(ocean_data_path_temp, 'r')
    water_temps = hycom_data_temp['Temp']
    del(hycom_data_temp)
    hycom_data_sal = h5py.File(ocean_data_path_sal, 'r')
    water_sals = hycom_data_sal['Salinity']
    del(hycom_data_sal)

    u_curr = np.transpose(u_curr, (3, 2, 1, 0))
    v_curr = np.transpose(v_curr, (3, 2, 1, 0))
    water_temps = np.transpose(water_temps, (3, 2, 1, 0))
    water_sals = np.transpose(water_sals, (3, 2, 1, 0))

    u_curr = u_curr[lat_min:lat_max + 1, lon_min:lon_max + 1, :, :]
    v_curr = v_curr[lat_min:lat_max + 1, lon_min:lon_max + 1, :, :]
    water_temps = water_temps[lat_min:lat_max + 1, lon_min:lon_max + 1, :, :]
    water_sals = water_sals[lat_min:lat_max + 1, lon_min:lon_max + 1, :, :]

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

    distance = np.sqrt((lat_ssh_grad_x - iceberg_lat0) ** 2 + (lon_ssh_grad_x - iceberg_lon0) ** 2)

    nearest_idx = np.unravel_index(np.argmin(distance, axis=None), distance.shape)
    nearest_lat_idx, nearest_lon_idx = nearest_idx

    # Define the index range to restrict to within 10 indices in all directions
    lat_min = max(nearest_lat_idx - deg_radius, 0)
    lat_max = min(nearest_lat_idx + deg_radius, lat_ssh_grad_x.shape[0] - 1)
    lon_min = max(nearest_lon_idx - deg_radius, 0)
    lon_max = min(nearest_lon_idx + deg_radius, lon_ssh_grad_x.shape[1] - 1)

    lat_ssh_grad_x = lat_ssh_grad_x[lat_min:lat_max + 1, lon_min:lon_max + 1]
    lon_ssh_grad_x = lon_ssh_grad_x[lat_min:lat_max + 1, lon_min:lon_max + 1]
    ssh_grad_x = ssh_grad_x[lat_min:lat_max + 1, lon_min:lon_max + 1, :]

    distance = np.sqrt((lat_ssh_grad_y - iceberg_lat0) ** 2 + (lon_ssh_grad_y - iceberg_lon0) ** 2)

    nearest_idx = np.unravel_index(np.argmin(distance, axis=None), distance.shape)
    nearest_lat_idx, nearest_lon_idx = nearest_idx

    # Define the index range to restrict to within 10 indices in all directions
    lat_min = max(nearest_lat_idx - deg_radius, 0)
    lat_max = min(nearest_lat_idx + deg_radius, lat_ssh_grad_y.shape[0] - 1)
    lon_min = max(nearest_lon_idx - deg_radius, 0)
    lon_max = min(nearest_lon_idx + deg_radius, lon_ssh_grad_y.shape[1] - 1)

    lat_ssh_grad_y = lat_ssh_grad_y[lat_min:lat_max + 1, lon_min:lon_max + 1]
    lon_ssh_grad_y = lon_ssh_grad_y[lat_min:lat_max + 1, lon_min:lon_max + 1]
    ssh_grad_y = ssh_grad_y[lat_min:lat_max + 1, lon_min:lon_max + 1, :]

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

    iceberg_us = np.empty((len(iceberg_times),))
    iceberg_vs = np.empty((len(iceberg_times),))

    iceberg_lats[0] = iceberg_lat0
    iceberg_lons[0] = iceberg_lon0
    iceberg_us[0] = iceberg_u0
    iceberg_vs[0] = iceberg_v0
    iceberg_lengths[0] = iceberg_length
    iceberg_drafts[0] = iceberg_draft
    iceberg_sails[0] = iceberg_sail
    iceberg_masses[0] = iceberg_mass

    for i in range(len(iceberg_times) - 1):
        iceberg_lat = iceberg_lats[i]
        iceberg_lon = iceberg_lons[i]
        iceberg_u = iceberg_us[i]
        iceberg_v = iceberg_vs[i]
        ib_length = iceberg_lengths[i]
        ib_draft = iceberg_drafts[i]
        ib_sail = iceberg_sails[i]
        ib_mass = iceberg_masses[i]

        f_u_wind = RegularGridInterpolator((lat_wind, lon_wind, time_wind_wave_hours), u_wind, method='linear', bounds_error=True, fill_value=np.nan)
        u_wind_ib = float(f_u_wind([iceberg_lat, iceberg_lon, iceberg_time_wind_wave_hours[i]]))
        f_v_wind = RegularGridInterpolator((lat_wind, lon_wind, time_wind_wave_hours), v_wind, method='linear', bounds_error=True, fill_value=np.nan)
        v_wind_ib = float(f_v_wind([iceberg_lat, iceberg_lon, iceberg_time_wind_wave_hours[i]]))
        f_sw_rad = RegularGridInterpolator((lat_wind, lon_wind, time_wind_wave_hours), sw_rad, method='linear', bounds_error=True, fill_value=np.nan)
        sw_rad_ib = float(f_sw_rad([iceberg_lat, iceberg_lon, iceberg_time_wind_wave_hours[i]]))
        f_airT = RegularGridInterpolator((lat_wind, lon_wind, time_wind_wave_hours), airT, method='linear', bounds_error=True, fill_value=np.nan)
        airT_ib = float(f_airT([iceberg_lat, iceberg_lon, iceberg_time_wind_wave_hours[i]]))
        f_wave_pd = RegularGridInterpolator((lat_wave, lon_wave, time_wind_wave_hours), wave_pd, method='linear', bounds_error=True, fill_value=np.nan)
        wave_pd_ib = float(f_wave_pd([iceberg_lat, iceberg_lon, iceberg_time_wind_wave_hours[i]]))
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

        if ib_draft > 0:
            loc_depth = np.argwhere(depth_curr <= ib_draft).flatten()

            # Append the next index if it exists
            if loc_depth[-1] + 1 < len(depth_curr):
                loc_depth = np.append(loc_depth, loc_depth[-1] + 1)

            # Slice depth_curr using the valid indices
            depth_curr_ib = depth_curr[loc_depth]
            depth_curr = depth_curr[loc_depth]

            # Convert to a flat list
            depth_curr_ib = depth_curr_ib.tolist()
            depth_curr_ib_interp = np.arange(0., ib_draft, 0.001)
            u_curr = u_curr[:, :, loc_depth, :]
            v_curr = v_curr[:, :, loc_depth, :]
            water_temps = water_temps[:, :, loc_depth, :]
            water_sals = water_sals[:, :, loc_depth, :]
        else:
            depth_curr_ib = list(depth_curr)
            depth_curr_ib_interp = np.arange(0., depth_curr[-1], 0.001)

        u_curr_depth_list = []
        u_curr_depth_list2 = []
        v_curr_depth_list = []
        v_curr_depth_list2 = []
        water_temps_depth_list = []
        water_sals_depth_list = []

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

            f_water_temps = RegularGridInterpolator((lat_curr, lon_curr, time_curr_ssh_hours), np.squeeze(water_temps[:, :, n, :]),
                                               method='linear', bounds_error=True, fill_value=np.nan)
            water_temps_ib_temp = float(f_water_temps([iceberg_lat, iceberg_lon, iceberg_time_curr_ssh_hours[i]]))
            water_temps_depth_list.append(water_temps_ib_temp)

            f_water_sals = RegularGridInterpolator((lat_curr, lon_curr, time_curr_ssh_hours), np.squeeze(water_sals[:, :, n, :]),
                                                    method='linear', bounds_error=True, fill_value=np.nan)
            water_sals_ib_temp = float(f_water_sals([iceberg_lat, iceberg_lon, iceberg_time_curr_ssh_hours[i]]))
            water_sals_depth_list.append(water_sals_ib_temp)

        interp_func = interp1d(depth_curr_ib, u_curr_depth_list, kind='linear', fill_value='extrapolate')
        u_curr_depth_list = interp_func(depth_curr_ib_interp)
        interp_func = interp1d(depth_curr_ib, v_curr_depth_list, kind='linear', fill_value='extrapolate')
        v_curr_depth_list = interp_func(depth_curr_ib_interp)
        u_curr_ib = np.nanmean(u_curr_depth_list)
        v_curr_ib = np.nanmean(v_curr_depth_list)
        u_curr_ib2 = np.nanmean(u_curr_depth_list2)
        v_curr_ib2 = np.nanmean(v_curr_depth_list2)

        interp_func = interp1d(depth_curr_ib, water_temps_depth_list, kind='linear', fill_value='extrapolate')
        water_temps_ib_list = interp_func(depth_curr_ib_interp)
        interp_func = interp1d(depth_curr_ib, water_sals_depth_list, kind='linear', fill_value='extrapolate')
        water_sals_ib_list = interp_func(depth_curr_ib_interp)

        def duv_dt(t, uv):
            iceberg_u_init, iceberg_v_init = uv
            ib_acc_E, ib_acc_N = iceberg_acc(iceberg_lat, iceberg_u_init, iceberg_v_init, ib_sail, ib_draft, ib_length,
                                             ib_mass, iceberg_times_dt[i], am, omega, Cw, Ca, C_wave, g, rho_air, rho_water,
                                             u_wind_ib, v_wind_ib, [u_curr_ib, u_curr_ib2], [v_curr_ib, v_curr_ib2],
                                             ssh_grad_x_ib, ssh_grad_y_ib, Hs_ib, wave_dir_ib)
            return ib_acc_E, ib_acc_N

        if ib_mass > 0:
            solution = solve_ivp(duv_dt, (0., iceberg_times_dt[i]), [iceberg_u, iceberg_v], method='BDF', t_eval=[0., iceberg_times_dt[i]])

            # Results
            iceberg_u_end = solution.y[0][-1]  # Final u-velocity
            iceberg_v_end = solution.y[1][-1]  # Final v-velocity
        else:
            iceberg_u_end = 0.
            iceberg_v_end = 0.

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

        if ib_mass > 0:
            new_iceberg_length, new_iceberg_draft, new_iceberg_sail, new_iceberg_mass = iceberg_det(ib_length, ib_mass,
                                                                                                    sw_rad_ib, ice_albedo, Lf_ice,
                                                                                                    rho_ice, water_temps_ib_list,
                                                                                                    water_sals_ib_list, depth_curr_ib_interp,
                                                                                                    airT_ib, u_curr_ib, v_curr_ib, u_wind_ib,
                                                                                                    v_wind_ib, iceberg_u, iceberg_v, Hs_ib, wave_pd_ib,
                                                                                                    iceberg_times_dt[i])
        else:
            new_iceberg_length = 0.
            new_iceberg_draft = 0.
            new_iceberg_sail = 0.
            new_iceberg_mass = 0.

        iceberg_lengths[i + 1] = new_iceberg_length
        iceberg_drafts[i + 1] = new_iceberg_draft
        iceberg_sails[i + 1] = new_iceberg_sail
        iceberg_masses[i + 1] = new_iceberg_mass

    iceberg_mass = iceberg_mass / 1000. # Convert back to tonnes
    iceberg_masses = iceberg_masses / 1000.
    iceberg_times = np.array(iceberg_times)
    return (iceberg_lat0, iceberg_lon0, iceberg_lats, iceberg_lons, iceberg_times, iceberg_length,
            iceberg_draft, iceberg_mass, rcm_datetime0, next_rcm_time, grounded_status, iceberg_lengths, iceberg_masses)

def datetime64_to_datenum(dt64):
    # Convert numpy.datetime64 to days since 1970-01-01
    days_since_epoch = (dt64 - np.datetime64('1970-01-01')) / np.timedelta64(1, 'D')
    # MATLAB datenum offset to convert 1970-01-01 in numpy to 0000-01-01 in MATLAB
    matlab_offset = 719529  # days from 0000-01-01 to 1970-01-01 in MATLAB
    # Add offset to convert to MATLAB datenum
    return days_since_epoch + matlab_offset

rcm_shapefiles_path_2023 = 'C:/Users/idturnbull/Documents/ExxonMobil_RCM_Project/Drift_SamleData_finished/2023_tracks/'
rcm_shapefiles_path_2024 = 'C:/Users/idturnbull/Documents/ExxonMobil_RCM_Project/Drift_SamleData_finished/2024_tracks/'
output_filepath = 'C:/Users/idturnbull/Documents/MATLAB/ExxonMobil_RCM_Project_Analysis/RCM_Iceberg_Forecaster_Results_2/'
era5_wind_wave_data_2023_file = 'C:/Users/idturnbull/Documents/MATLAB/ExxonMobil_RCM_Project_Analysis/ERA5_Wind_Wave_Data_2023.mat'
era5_wind_wave_data_2024_file = 'C:/Users/idturnbull/Documents/MATLAB/ExxonMobil_RCM_Project_Analysis/ERA5_Wind_Wave_Data_2024.mat'
hycom_data_2023_file_ssh_grad_time = 'C:/Users/idturnbull/Documents/MATLAB/ExxonMobil_RCM_Project_Analysis/HYCOM_Data/hycom_data_processed_2023_ssh_grad_time.mat'
hycom_data_2023_file_u_curr = 'C:/Users/idturnbull/Documents/MATLAB/ExxonMobil_RCM_Project_Analysis/HYCOM_Data/hycom_data_processed_2023_u.mat'
hycom_data_2023_file_v_curr = 'C:/Users/idturnbull/Documents/MATLAB/ExxonMobil_RCM_Project_Analysis/HYCOM_Data/hycom_data_processed_2023_v.mat'
hycom_data_2023_file_temp = 'C:/Users/idturnbull/Documents/MATLAB/ExxonMobil_RCM_Project_Analysis/HYCOM_Data/hycom_data_processed_2023_temp.mat'
hycom_data_2023_file_sal = 'C:/Users/idturnbull/Documents/MATLAB/ExxonMobil_RCM_Project_Analysis/HYCOM_Data/hycom_data_processed_2023_salinity.mat'
hycom_data_2024_file_ssh_grad_time = 'C:/Users/idturnbull/Documents/MATLAB/ExxonMobil_RCM_Project_Analysis/HYCOM_Data/hycom_data_processed_2024_ssh_grad_time.mat'
hycom_data_2024_file_u_curr = 'C:/Users/idturnbull/Documents/MATLAB/ExxonMobil_RCM_Project_Analysis/HYCOM_Data/hycom_data_processed_2024_u.mat'
hycom_data_2024_file_v_curr = 'C:/Users/idturnbull/Documents/MATLAB/ExxonMobil_RCM_Project_Analysis/HYCOM_Data/hycom_data_processed_2024_v.mat'
hycom_data_2024_file_temp = 'C:/Users/idturnbull/Documents/MATLAB/ExxonMobil_RCM_Project_Analysis/HYCOM_Data/hycom_data_processed_2024_temp.mat'
hycom_data_2024_file_sal = 'C:/Users/idturnbull/Documents/MATLAB/ExxonMobil_RCM_Project_Analysis/HYCOM_Data/hycom_data_processed_2024_salinity.mat'

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
            iceberg_length_end = gdf['WtrLin'].iloc[index1]
            grounded_status = gdf['Grounded'].iloc[index0]
            rcm_datetime0 = gdf['acqDate'].iloc[index0]
            next_rcm_time = gdf['acqDate'].iloc[index1]
            rcm_datetime0 = np.datetime64(datetime.datetime.strptime(rcm_datetime0, '%Y%m%d_%H%M%S'))
            next_rcm_time = np.datetime64(datetime.datetime.strptime(next_rcm_time, '%Y%m%d_%H%M%S'))
            (iceberg_lat0, iceberg_lon0, iceberg_lats, iceberg_lons, iceberg_times, iceberg_length,
             iceberg_draft, iceberg_mass, rcm_datetime0, next_rcm_time, grounded_status, iceberg_lengths, iceberg_masses) = (
                assess_rcm_iceberg_drift_deterioration_forecaster(iceberg_lat0, iceberg_lon0, rcm_datetime0, iceberg_length, grounded_status, next_rcm_time,
                                                    era5_wind_wave_data_2023_file, hycom_data_2023_file_ssh_grad_time,
                                                    hycom_data_2023_file_u_curr, hycom_data_2023_file_v_curr, hycom_data_2023_file_temp, hycom_data_2023_file_sal))
            iceberg_times = datetime64_to_datenum(iceberg_times)
            rcm_datetime0 = datetime64_to_datenum(rcm_datetime0)
            next_rcm_time = datetime64_to_datenum(next_rcm_time)
            savemat(output_filepath + rcm_shapefile[:-4] + '_' + str(m + 1) + '.mat', {'iceberg_times': iceberg_times, 'iceberg_lat0': iceberg_lat0,
                                                                                                 'iceberg_lon0': iceberg_lon0, 'rcm_datetime0': rcm_datetime0,
                                                                                                 'next_rcm_time': next_rcm_time, 'iceberg_lats': iceberg_lats,
                                                                                                 'iceberg_lons': iceberg_lons, 'iceberg_length': iceberg_length,
                                                                                                 'iceberg_draft': iceberg_draft, 'iceberg_mass': iceberg_mass,
                                                                                                 'grounded_status': grounded_status, 'iceberg_lat_end': iceberg_lat_end,
                                                                                                 'iceberg_lon_end': iceberg_lon_end, 'iceberg_lengths': iceberg_lengths,
                                                                                                 'iceberg_masses': iceberg_masses,
                                                                                                 'iceberg_length_end': iceberg_length_end})
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
            iceberg_length_end = gdf['WtrLin'].iloc[index1]
            grounded_status = gdf['Grounded'].iloc[index0]
            rcm_datetime0 = gdf['acqDate'].iloc[index0]
            next_rcm_time = gdf['acqDate'].iloc[index1]
            rcm_datetime0 = np.datetime64(datetime.datetime.strptime(rcm_datetime0, '%Y%m%d_%H%M%S'))
            next_rcm_time = np.datetime64(datetime.datetime.strptime(next_rcm_time, '%Y%m%d_%H%M%S'))
            (iceberg_lat0, iceberg_lon0, iceberg_lats, iceberg_lons, iceberg_times, iceberg_length,
             iceberg_draft, iceberg_mass, rcm_datetime0, next_rcm_time, grounded_status, iceberg_lengths, iceberg_masses) = (
                assess_rcm_iceberg_drift_deterioration_forecaster(iceberg_lat0, iceberg_lon0, rcm_datetime0, iceberg_length, grounded_status, next_rcm_time,
                                                    era5_wind_wave_data_2024_file, hycom_data_2024_file_ssh_grad_time,
                                                    hycom_data_2024_file_u_curr, hycom_data_2024_file_v_curr, hycom_data_2024_file_temp, hycom_data_2024_file_sal))
            iceberg_times = datetime64_to_datenum(iceberg_times)
            rcm_datetime0 = datetime64_to_datenum(rcm_datetime0)
            next_rcm_time = datetime64_to_datenum(next_rcm_time)
            savemat(output_filepath + rcm_shapefile[:-4] + '_' + str(m + 1) + '.mat', {'iceberg_times': iceberg_times, 'iceberg_lat0': iceberg_lat0,
                                                                                                 'iceberg_lon0': iceberg_lon0, 'rcm_datetime0': rcm_datetime0,
                                                                                                 'next_rcm_time': next_rcm_time, 'iceberg_lats': iceberg_lats,
                                                                                                 'iceberg_lons': iceberg_lons, 'iceberg_length': iceberg_length,
                                                                                                 'iceberg_draft': iceberg_draft, 'iceberg_mass': iceberg_mass,
                                                                                                 'grounded_status': grounded_status, 'iceberg_lat_end': iceberg_lat_end,
                                                                                                 'iceberg_lon_end': iceberg_lon_end, 'iceberg_lengths': iceberg_lengths,
                                                                                                 'iceberg_masses': iceberg_masses,
                                                                                                 'iceberg_length_end': iceberg_length_end})
        except:
            pass


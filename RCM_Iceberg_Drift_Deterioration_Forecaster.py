
from scipy.interpolate import RegularGridInterpolator, griddata, interp1d
from scipy.integrate import solve_ivp
import numpy as np
import netCDF4 as nc
import os
import gsw
from observation import Observation
# from observations import Observations

def rcm_iceberg_drift_deterioration_forecaster(obs: Observation, t1: np.datetime64, si_toggle):
    deg_radius = 5
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
    Csi = 1.
    rho_si = 875.
    am = 0.5
    Re = 6371e3
    bathy_data_path = './GEBCO_Bathymetric_Data/gebco_2024.nc'
    rootpath_to_metdata = './RCM_Iceberg_Metocean_Data/'
    iceberg_lats0 = obs.lat
    iceberg_lons0 = obs.lon
    rcm_datetime0 = obs.time
    iceberg_lengths0 = obs.length
    iceberg_ids = obs.id
    iceberg_grounded_statuses0 = obs.grounded
    next_rcm_time = t1
    iceberg_lats0 = iceberg_lats0 if isinstance(iceberg_lats0, list) else [iceberg_lats0]
    iceberg_lons0 = iceberg_lons0 if isinstance(iceberg_lons0, list) else [iceberg_lons0]
    iceberg_lengths0 = iceberg_lengths0 if isinstance(iceberg_lengths0, list) else [iceberg_lengths0]
    iceberg_ids = iceberg_ids if isinstance(iceberg_ids, list) else [iceberg_ids]
    iceberg_grounded_statuses0 = iceberg_grounded_statuses0 if isinstance(iceberg_grounded_statuses0, list) else [iceberg_grounded_statuses0]

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
                    u_wind, v_wind, u_curr, v_curr, ssh_grad_x, ssh_grad_y, Hs, wave_dir, siconc, sithick, usi, vsi, Csi, rho_si):
        iceberg_keel = iceberg_length * iceberg_draft

        if (np.any(np.isnan(u_wind)) or np.any(np.isnan(v_wind)) or np.any(np.isinf(u_wind)) or np.any(np.isinf(v_wind))
                or np.any(~np.isreal(u_wind)) or np.any(~np.isreal(v_wind))):
            u_wind = 0.
            v_wind = 0.

        if (np.any(np.isnan(usi)) or np.any(np.isnan(vsi)) or np.any(np.isinf(usi)) or np.any(np.isinf(vsi))
                or np.any(~np.isreal(usi)) or np.any(~np.isreal(vsi))):
            usi = 0.
            vsi = 0.

        wind_dir = 90. - np.rad2deg(np.arctan2(v_wind, u_wind))

        if wind_dir < 0:
            wind_dir = wind_dir + 360.

        if np.any(np.isnan(wave_dir)) or np.any(np.isinf(wave_dir)) or np.any(~np.isreal(wave_dir)):
            wave_dir = wind_dir

        if siconc > 1:
            siconc = 1.

        if np.any(np.isnan(siconc)) or np.any(np.isinf(siconc)) or np.any(~np.isreal(siconc)) or siconc < 0:
            siconc = 0.
            sithick = 0.
            usi = 0.
            vsi = 0.

        if np.any(np.isnan(sithick)) or np.any(np.isinf(sithick)) or np.any(~np.isreal(sithick)) or sithick < 0:
            siconc = 0.
            sithick = 0.
            usi = 0.
            vsi = 0.

        if np.any(np.isnan(Hs)) or np.any(np.isinf(Hs)) or np.any(~np.isreal(Hs)) or Hs < 0 or siconc >= 0.9:
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

        h_min = 660.9 / ((20e3) * np.exp(-20 * (1 - siconc)))
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

        if siconc <= 0.15:
            Fsi_E = 0.
            Fsi_N = 0.
        elif (siconc > 0.15 and siconc < 0.9) or (siconc >= 0.9 and sithick < h_min):
            Fsi_E = 0.5 * rho_si * Csi * sithick * iceberg_length * np.sqrt((usi - iceberg_u) ** 2 + (vsi - iceberg_v) ** 2) * (usi - iceberg_u)
            Fsi_N = 0.5 * rho_si * Csi * sithick * iceberg_length * np.sqrt((usi - iceberg_u) ** 2 + (vsi - iceberg_v) ** 2) * (vsi - iceberg_v)
        elif siconc >= 0.9 and sithick >= h_min:
            Fsi_E = -(Fa_E + Fw_E + Fc_E + Fp_E + Fs_E + Fr_E)
            Fsi_N = -(Fa_N + Fw_N + Fc_N + Fp_N + Fs_N + Fr_N)
        else:
            Fsi_E = 0.
            Fsi_N = 0.

        F_sum_E = Fa_E + Fw_E + Fc_E + Fp_E + Fs_E + Fr_E + Fsi_E
        F_sum_N = Fa_N + Fw_N + Fc_N + Fp_N + Fs_N + Fr_N + Fsi_N
        ib_acc_E = F_sum_E / (iceberg_mass + am * iceberg_mass)
        ib_acc_N = F_sum_N / (iceberg_mass + am * iceberg_mass)
        return ib_acc_E, ib_acc_N

    def duv_dt(t, uv):
        iceberg_u_init, iceberg_v_init = uv
        ib_acc_E, ib_acc_N = iceberg_acc(iceberg_lat, iceberg_u_init, iceberg_v_init, new_iceberg_sail, new_iceberg_draft,
                                         new_iceberg_length, new_iceberg_mass, iceberg_times_dt[i], am, omega, Cw, Ca,
                                         C_wave, g, rho_air, rho_water, u_wind_ib, v_wind_ib, [u_curr_ib, u_curr_ib2], [v_curr_ib, v_curr_ib2],
                                         ssh_grad_x_ib, ssh_grad_y_ib, Hs_ib, wave_dir_ib, siconc_ib, sithick_ib, usi_ib, vsi_ib, Csi, rho_si)
        return ib_acc_E, ib_acc_N

    def iceberg_det(iceberg_length, iceberg_mass, iceberg_lat, solar_rad, ice_albedo, Lf_ice, rho_ice, water_pot_temps, water_sals, water_depths, air_temp,
                    u_curr, v_curr, u_wind, v_wind, iceberg_u, iceberg_v, Hs, wave_pd, time_dt, siconc):
        water_pot_temps = np.array(water_pot_temps)
        water_sals = np.array(water_sals)
        water_depths = np.array(water_depths)
        water_pot_temps[np.isnan(water_pot_temps) | np.isinf(water_pot_temps) | ~np.isreal(water_pot_temps)] = 0.
        water_sals[np.isnan(water_sals) | np.isinf(water_sals) | ~np.isreal(water_sals)] = 33.

        if siconc > 1:
            siconc = 1.

        if np.any(np.isnan(siconc)) or np.any(np.isinf(siconc)) or np.any(~np.isreal(siconc)) or siconc < 0:
            siconc = 0.

        if np.any(np.isnan(Hs)) or np.any(np.isinf(Hs)) or np.any(~np.isreal(Hs)) or Hs < 0 or siconc >= 0.9:
            Hs = 0.001

        if (np.any(np.isnan(iceberg_u)) or np.any(np.isnan(iceberg_v)) or np.any(np.isinf(iceberg_u)) or np.any(np.isinf(iceberg_v))
                or np.any(~np.isreal(iceberg_u)) or np.any(~np.isreal(iceberg_v))):
            iceberg_u = 0.
            iceberg_v = 0.

        if (np.any(np.isnan(u_curr)) or np.any(np.isnan(v_curr)) or np.any(np.isinf(u_curr)) or np.any(np.isinf(v_curr))
                or np.any(~np.isreal(u_curr)) or np.any(~np.isreal(v_curr))):
            u_curr = 0.
            v_curr = 0.

        if np.any(np.isnan(wave_pd)) or np.any(np.isinf(wave_pd)) or np.any(~np.isreal(wave_pd)) or wave_pd < 0:
            wave_pd = 16.

        if np.any(np.isnan(solar_rad)) or np.any(np.isinf(solar_rad)) or np.any(~np.isreal(solar_rad)) or solar_rad < 0:
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
        water_temps = []
        water_therm_diffs = []
        water_kin_viss = []

        for k in range(len(water_depths)):
            water_pressure = gsw.p_from_z(-water_depths[k], iceberg_lat, geo_strf_dyn_height=0, sea_surface_geopotential=0)
            CT = gsw.CT_from_pt(water_sals[k], water_pot_temps[k])
            water_temp = gsw.t_from_CT(water_sals[k], CT, water_pressure)

            try:
                water_therm_diff = water_therm_diff_interpolator((water_temp, water_sals[k]))
                water_kin_vis = water_kin_vis_interpolator((water_temp, water_sals[k]))
            except:
                water_therm_diff = water_therm_diff_interpolator((0., 33.))
                water_kin_vis = water_kin_vis_interpolator((0., 33.))

            water_temps.append(water_temp)
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

    rcm_datetime0 = np.datetime64(rcm_datetime0)
    next_rcm_time = np.datetime64(next_rcm_time)
    forecast_time = rcm_datetime0

    dirname_wind_waves = np.datetime_as_string(forecast_time, unit='D')
    d_wind_waves = dirname_wind_waves.replace('-', '')
    dirname_airT_sw_rad = np.datetime_as_string(forecast_time, unit='D')
    d_airT_sw_rad = dirname_airT_sw_rad.replace('-', '')
    dirname_ocean = np.datetime_as_string(forecast_time, unit='D')
    d_ocean = dirname_ocean.replace('-', '')

    bathy_data = nc.Dataset(bathy_data_path)
    bathy_lat = bathy_data.variables['lat'][:]
    bathy_lon = bathy_data.variables['lon'][:]
    bathy_depth = -bathy_data.variables['elevation'][:]
    bathy_data.close()
    bathy_interp = RegularGridInterpolator((bathy_lat, bathy_lon), bathy_depth, method='linear', bounds_error=True, fill_value=np.nan)

    iceberg_lats0 = np.array(iceberg_lats0)
    iceberg_lons0 = np.array(iceberg_lons0)
    iceberg_lengths0 = np.array(iceberg_lengths0)
    iceberg_grounded_statuses0 = [1 if status == True else 0 for status in iceberg_grounded_statuses0]
    iceberg_grounded_statuses0 = np.array(iceberg_grounded_statuses0)

    iceberg_drafts0 = np.empty((len(iceberg_lats0),))
    iceberg_masses0 = np.empty((len(iceberg_lats0),))
    iceberg_sails0 = np.empty((len(iceberg_lats0),))

    for i in range(len(iceberg_lats0)):
        if (not isinstance(iceberg_lengths0[i], (int, float)) or np.any(np.isnan(iceberg_lengths0[i])) or
                np.any(np.isinf(iceberg_lengths0[i])) or np.any(~np.isreal(iceberg_lengths0[i])) or iceberg_lengths0[i] <= 0):
            iceberg_lengths0[i] = 100.

        iceberg_drafts0[i] = 1.78 * (iceberg_lengths0[i] ** 0.71) # meters
        iceberg_masses0[i] = 0.45 * rho_ice * (iceberg_lengths0[i] ** 3) # kg
        iceberg_sails0[i] = 0.077 * (iceberg_lengths0[i] ** 2) # m ** 2
        iceberg_bathy_depth0 = bathy_interp([[iceberg_lats0[i], iceberg_lons0[i]]])[0]

        if iceberg_grounded_statuses0[i] == 0 and iceberg_bathy_depth0 <= iceberg_drafts0[i]:
            iceberg_drafts0[i] = iceberg_bathy_depth0 - 1.
        elif iceberg_grounded_statuses0[i] == 1:
            iceberg_drafts0[i] = iceberg_bathy_depth0

    iceberg_times = [forecast_time]
    current_time = forecast_time

    while current_time + np.timedelta64(1, 'h') < next_rcm_time:
        current_time += np.timedelta64(1, 'h')
        iceberg_times.append(current_time)

    if iceberg_times[-1] < next_rcm_time:
        iceberg_times.append(next_rcm_time)

    iceberg_times = list(iceberg_times)
    iceberg_times_dt = [float((iceberg_times[i + 1] - iceberg_times[i]) / np.timedelta64(1, 's')) for i in range(len(iceberg_times) - 1)]
    iceberg_times_dt = list(iceberg_times_dt)

    iceberg_lats = np.empty((len(iceberg_times), len(iceberg_lats0)))
    iceberg_lons = np.empty((len(iceberg_times), len(iceberg_lats0)))
    iceberg_lengths = np.empty((len(iceberg_times), len(iceberg_lats0)))
    iceberg_drafts = np.empty((len(iceberg_times), len(iceberg_lats0)))
    iceberg_sails = np.empty((len(iceberg_times), len(iceberg_lats0)))
    iceberg_masses = np.empty((len(iceberg_times), len(iceberg_lats0)))
    iceberg_grounded_statuses = np.empty((len(iceberg_times), len(iceberg_lats0)))
    iceberg_us = np.zeros((len(iceberg_times), len(iceberg_lats0)))
    iceberg_vs = np.zeros((len(iceberg_times), len(iceberg_lats0)))

    iceberg_lats[0, :] = iceberg_lats0
    iceberg_lons[0, :] = iceberg_lons0
    iceberg_lengths[0, :] = iceberg_lengths0
    iceberg_drafts[0, :] = iceberg_drafts0
    iceberg_sails[0, :] = iceberg_sails0
    iceberg_masses[0, :] = iceberg_masses0
    iceberg_grounded_statuses[0, :] = iceberg_grounded_statuses0

    forecast_times_wind_waves = []
    forecast_times_ocean = []
    forecast_times_airT_sw_rad = []

    directory = rootpath_to_metdata + 'RIOPS_ocean_forecast_files/' + dirname_ocean + '/'
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    file = files[0]
    hour_utc_str_ocean = file[9:11]

    directory = rootpath_to_metdata + 'GDWPS_wind_wave_forecast_files/' + dirname_wind_waves + '/'
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    file = files[0]
    hour_utc_str_wind_waves = file[9:11]

    directory = rootpath_to_metdata + 'GDPS_airT_sw_rad_forecast_files/' + dirname_airT_sw_rad + '/'
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    file = files[0]
    hour_utc_str_airT_sw_rad = file[-10:-8]

    forecast_start_time_wind_waves = np.datetime64(dirname_wind_waves + 'T' + hour_utc_str_wind_waves + ':00:00')
    forecast_start_time_ocean = np.datetime64(dirname_ocean + 'T' + hour_utc_str_ocean + ':00:00')
    forecast_start_time_airT_sw_rad = np.datetime64(dirname_airT_sw_rad + 'T' + hour_utc_str_airT_sw_rad + ':00:00')
    time_count_wind_waves = forecast_start_time_wind_waves
    time_count_ocean = forecast_start_time_ocean
    time_count_airT_sw_rad = forecast_start_time_airT_sw_rad

    while time_count_wind_waves <= next_rcm_time + np.timedelta64(1, 'h'):
        forecast_times_wind_waves.append(time_count_wind_waves)
        time_count_wind_waves = time_count_wind_waves + np.timedelta64(1, 'h')

    while time_count_ocean <= next_rcm_time + np.timedelta64(1, 'h'):
        forecast_times_ocean.append(time_count_ocean)
        time_count_ocean = time_count_ocean + np.timedelta64(1, 'h')

    while time_count_airT_sw_rad <= next_rcm_time + np.timedelta64(3, 'h'):
        forecast_times_airT_sw_rad.append(time_count_airT_sw_rad)
        time_count_airT_sw_rad = time_count_airT_sw_rad + np.timedelta64(3, 'h')

    forecast_times_wind_waves = np.array(forecast_times_wind_waves, dtype='datetime64[s]')
    forecast_times_ocean = np.array(forecast_times_ocean, dtype='datetime64[s]')
    forecast_times_airT_sw_rad = np.array(forecast_times_airT_sw_rad, dtype='datetime64[s]')

    forecast_times_wind_waves_hours = (forecast_times_wind_waves.astype('datetime64[h]') - forecast_start_time_wind_waves.astype('datetime64[h]')).astype(int)
    forecast_times_ocean_hours = (forecast_times_ocean.astype('datetime64[h]') - forecast_start_time_ocean.astype('datetime64[h]')).astype(int)
    forecast_times_airT_sw_rad_hours = (forecast_times_airT_sw_rad.astype('datetime64[h]') - forecast_start_time_airT_sw_rad.astype('datetime64[h]')).astype(int)

    fname = rootpath_to_metdata + 'RIOPS_ocean_forecast_files/' + dirname_ocean + '/' + d_ocean + 'T' + hour_utc_str_ocean + \
            'Z_MSC_RIOPS_VOZOCRTX_DBS-all_PS5km_P' + str(forecast_times_ocean_hours[0]).zfill(3) + '.nc'
    ocean_data = nc.Dataset(fname)
    ocean_lat = ocean_data.variables['latitude'][:]
    ocean_lon = ocean_data.variables['longitude'][:]
    ocean_depth = ocean_data.variables['depth'][:]
    ocean_depth = ocean_depth.flatten()
    ocean_data.close()

    fname = rootpath_to_metdata + 'RIOPS_ocean_forecast_files/' + dirname_ocean + '/' + d_ocean + 'T' + hour_utc_str_ocean + \
            'Z_MSC_RIOPS_SOSSHEIG_SFC_GRAD_PS5km_P' + str(forecast_times_ocean_hours[0]).zfill(3) + '.nc'
    ssh_grad_data = nc.Dataset(fname)
    ssh_grad_x_lat = np.squeeze(ssh_grad_data.variables['ssh_grad_x_lat'][:])
    ssh_grad_x_lon = np.squeeze(ssh_grad_data.variables['ssh_grad_x_lon'][:])
    ssh_grad_y_lat = np.squeeze(ssh_grad_data.variables['ssh_grad_y_lat'][:])
    ssh_grad_y_lon = np.squeeze(ssh_grad_data.variables['ssh_grad_y_lon'][:])
    ssh_grad_data.close()

    points_ocean = np.array([ocean_lat.ravel(), ocean_lon.ravel()]).T
    points_ssh_grad_x = np.array([ssh_grad_x_lat.ravel(), ssh_grad_x_lon.ravel()]).T
    points_ssh_grad_y = np.array([ssh_grad_y_lat.ravel(), ssh_grad_y_lon.ravel()]).T

    fname = (rootpath_to_metdata + 'GDWPS_wind_wave_forecast_files/' + dirname_wind_waves + '/' + d_wind_waves + 'T' +
             hour_utc_str_wind_waves + 'Z_MSC_GDWPS_UGRD_AGL-10m_LatLon0.25_PT' + str(forecast_times_wind_waves_hours[0]).zfill(3) + 'H.nc')
    wind_data = nc.Dataset(fname)
    lat_wind_waves = wind_data.variables['latitude'][:]
    lon_wind_waves = wind_data.variables['longitude'][:]
    wind_data.close()

    fname = (rootpath_to_metdata + 'GDPS_airT_sw_rad_forecast_files/' + dirname_airT_sw_rad + '/CMC_glb_TMP_TGL_2_latlon.15x.15_' +
             d_airT_sw_rad + hour_utc_str_airT_sw_rad + '_P' + str(forecast_times_airT_sw_rad_hours[0]).zfill(3) + '.nc')
    airT_data = nc.Dataset(fname)
    lat_airT_sw_rad = airT_data.variables['latitude'][:]
    lon_airT_sw_rad = airT_data.variables['longitude'][:]
    airT_data.close()

    base_time_wind_waves = forecast_times_wind_waves[0]
    time_increments_wind_waves = np.arange(forecast_times_wind_waves_hours[0], forecast_times_wind_waves_hours[-1], 1)
    file_times_wind_waves = base_time_wind_waves + time_increments_wind_waves.astype('timedelta64[h]')
    date_only_wind_waves = str(base_time_wind_waves.astype('datetime64[D]')).replace('-', '')

    base_time_ocean = forecast_times_ocean[0]
    time_increments_ocean = np.arange(forecast_times_ocean_hours[0], forecast_times_ocean_hours[-1], 1)
    file_times_ocean = base_time_ocean + time_increments_ocean.astype('timedelta64[h]')
    date_only_ocean = str(base_time_ocean.astype('datetime64[D]')).replace('-', '')

    base_time_airT_sw_rad = forecast_times_airT_sw_rad[0]
    time_increments_airT_sw_rad = np.arange(forecast_times_airT_sw_rad_hours[0], forecast_times_airT_sw_rad_hours[-1], 3)
    file_times_airT_sw_rad = base_time_airT_sw_rad + time_increments_airT_sw_rad.astype('timedelta64[h]')
    date_only_airT_sw_rad = str(base_time_airT_sw_rad.astype('datetime64[D]')).replace('-', '')

    for i in range(len(iceberg_times) - 1):
        print('Forecasting icebergs at ' + str(iceberg_times[i]) + ' UTC...')
        iceberg_time = iceberg_times[i]
        iceberg_time2 = iceberg_times[i + 1]
        before_idx = np.where(file_times_wind_waves <= iceberg_time)[0][-1]

        try:
            after_idx = np.where(file_times_wind_waves > iceberg_time)[0][0]
        except:
            after_idx = -1

        u_wind_file_before = date_only_wind_waves + 'T' + hour_utc_str_wind_waves + 'Z_MSC_GDWPS_UGRD_AGL-10m_LatLon0.25_PT' + \
                             str(time_increments_wind_waves[before_idx]).zfill(3) + 'H.nc'
        u_wind_file_after = date_only_wind_waves + 'T' + hour_utc_str_wind_waves + 'Z_MSC_GDWPS_UGRD_AGL-10m_LatLon0.25_PT' + \
                            str(time_increments_wind_waves[after_idx]).zfill(3) + 'H.nc'
        v_wind_file_before = date_only_wind_waves + 'T' + hour_utc_str_wind_waves + 'Z_MSC_GDWPS_VGRD_AGL-10m_LatLon0.25_PT' + \
                             str(time_increments_wind_waves[before_idx]).zfill(3) + 'H.nc'
        v_wind_file_after = date_only_wind_waves + 'T' + hour_utc_str_wind_waves + 'Z_MSC_GDWPS_VGRD_AGL-10m_LatLon0.25_PT' + \
                            str(time_increments_wind_waves[after_idx]).zfill(3) + 'H.nc'
        Hs_file_before = date_only_wind_waves + 'T' + hour_utc_str_wind_waves + 'Z_MSC_GDWPS_HTSGW_Sfc_LatLon0.25_PT' + \
                         str(time_increments_wind_waves[before_idx]).zfill(3) + 'H.nc'
        Hs_file_after = date_only_wind_waves + 'T' + hour_utc_str_wind_waves + 'Z_MSC_GDWPS_HTSGW_Sfc_LatLon0.25_PT' + \
                        str(time_increments_wind_waves[after_idx]).zfill(3) + 'H.nc'
        wave_dir_file_before = date_only_wind_waves + 'T' + hour_utc_str_wind_waves + 'Z_MSC_GDWPS_WVDIR_Sfc_LatLon0.25_PT' + \
                               str(time_increments_wind_waves[before_idx]).zfill(3) + 'H.nc'
        wave_dir_file_after = date_only_wind_waves + 'T' + hour_utc_str_wind_waves + 'Z_MSC_GDWPS_WVDIR_Sfc_LatLon0.25_PT' + \
                              str(time_increments_wind_waves[after_idx]).zfill(3) + 'H.nc'
        wave_pd_file_before = date_only_wind_waves + 'T' + hour_utc_str_wind_waves + 'Z_MSC_GDWPS_MZWPER_Sfc_LatLon0.25_PT' + \
                              str(time_increments_wind_waves[before_idx]).zfill(3) + 'H.nc'
        wave_pd_file_after = date_only_wind_waves + 'T' + hour_utc_str_wind_waves + 'Z_MSC_GDWPS_MZWPER_Sfc_LatLon0.25_PT' + \
                             str(time_increments_wind_waves[after_idx]).zfill(3) + 'H.nc'

        forecast_time_wind_waves_before = forecast_times_wind_waves[before_idx]
        forecast_time_wind_waves_after = forecast_times_wind_waves[after_idx]

        before_idx = np.where(file_times_airT_sw_rad <= iceberg_time)[0][-1]

        try:
            after_idx = np.where(file_times_airT_sw_rad > iceberg_time)[0][0]
        except:
            after_idx = -1

        airT_file_before = 'CMC_glb_TMP_TGL_2_latlon.15x.15_' + date_only_airT_sw_rad + hour_utc_str_airT_sw_rad + '_P' + \
                           str(time_increments_airT_sw_rad[before_idx]).zfill(3) + '.nc'
        airT_file_after = 'CMC_glb_TMP_TGL_2_latlon.15x.15_' + date_only_airT_sw_rad + hour_utc_str_airT_sw_rad + '_P' + \
                          str(time_increments_airT_sw_rad[after_idx]).zfill(3) + '.nc'
        solar_rad_file_before = 'CMC_glb_DSWRF_SFC_0_latlon.15x.15_' + date_only_airT_sw_rad + hour_utc_str_airT_sw_rad + '_P' + \
                                str(time_increments_airT_sw_rad[before_idx]).zfill(3) + '.nc'
        solar_rad_file_after = 'CMC_glb_DSWRF_SFC_0_latlon.15x.15_' + date_only_airT_sw_rad + hour_utc_str_airT_sw_rad + '_P' + \
                               str(time_increments_airT_sw_rad[after_idx]).zfill(3) + '.nc'

        forecast_time_airT_sw_rad_before = forecast_times_airT_sw_rad[before_idx]
        forecast_time_airT_sw_rad_after = forecast_times_airT_sw_rad[after_idx]

        before_idx = np.where(file_times_ocean <= iceberg_time)[0][-1]

        try:
            after_idx = np.where(file_times_ocean > iceberg_time)[0][0]
        except:
            after_idx = -1

        u_curr_file_before = date_only_ocean + 'T' + hour_utc_str_ocean + 'Z_MSC_RIOPS_VOZOCRTX_DBS-all_PS5km_P' + \
                             str(time_increments_ocean[before_idx]).zfill(3) + '.nc'
        u_curr_file_after = date_only_ocean + 'T' + hour_utc_str_ocean + 'Z_MSC_RIOPS_VOZOCRTX_DBS-all_PS5km_P' + \
                            str(time_increments_ocean[after_idx]).zfill(3) + '.nc'
        v_curr_file_before = date_only_ocean + 'T' + hour_utc_str_ocean + 'Z_MSC_RIOPS_VOMECRTY_DBS-all_PS5km_P' + \
                             str(time_increments_ocean[before_idx]).zfill(3) + '.nc'
        v_curr_file_after = date_only_ocean + 'T' + hour_utc_str_ocean + 'Z_MSC_RIOPS_VOMECRTY_DBS-all_PS5km_P' + \
                            str(time_increments_ocean[after_idx]).zfill(3) + '.nc'
        salinity_file_before = date_only_ocean + 'T' + hour_utc_str_ocean + 'Z_MSC_RIOPS_VOSALINE_DBS-all_PS5km_P' + \
                               str(time_increments_ocean[before_idx]).zfill(3) + '.nc'
        salinity_file_after = date_only_ocean + 'T' + hour_utc_str_ocean + 'Z_MSC_RIOPS_VOSALINE_DBS-all_PS5km_P' + \
                              str(time_increments_ocean[after_idx]).zfill(3) + '.nc'
        pot_temp_file_before = date_only_ocean + 'T' + hour_utc_str_ocean + 'Z_MSC_RIOPS_VOTEMPER_DBS-all_PS5km_P' + \
                               str(time_increments_ocean[before_idx]).zfill(3) + '.nc'
        pot_temp_file_after = date_only_ocean + 'T' + hour_utc_str_ocean + 'Z_MSC_RIOPS_VOTEMPER_DBS-all_PS5km_P' + \
                              str(time_increments_ocean[after_idx]).zfill(3) + '.nc'
        ssh_grad_file_before = date_only_ocean + 'T' + hour_utc_str_ocean + 'Z_MSC_RIOPS_SOSSHEIG_SFC_GRAD_PS5km_P' + \
                               str(time_increments_ocean[before_idx]).zfill(3) + '.nc'
        ssh_grad_file_after = date_only_ocean + 'T' + hour_utc_str_ocean + 'Z_MSC_RIOPS_SOSSHEIG_SFC_GRAD_PS5km_P' + \
                              str(time_increments_ocean[after_idx]).zfill(3) + '.nc'

        if si_toggle:
            siconc_file_before = date_only_ocean + 'T' + hour_utc_str_ocean + 'Z_MSC_RIOPS_IICECONC_SFC_PS5km_P' + \
                                   str(time_increments_ocean[before_idx]).zfill(3) + '.nc'
            siconc_file_after = date_only_ocean + 'T' + hour_utc_str_ocean + 'Z_MSC_RIOPS_IICECONC_SFC_PS5km_P' + \
                                  str(time_increments_ocean[after_idx]).zfill(3) + '.nc'
            sithick_file_before = date_only_ocean + 'T' + hour_utc_str_ocean + 'Z_MSC_RIOPS_IICEVOL_SFC_PS5km_P' + \
                                 str(time_increments_ocean[before_idx]).zfill(3) + '.nc'
            sithick_file_after = date_only_ocean + 'T' + hour_utc_str_ocean + 'Z_MSC_RIOPS_IICEVOL_SFC_PS5km_P' + \
                                str(time_increments_ocean[after_idx]).zfill(3) + '.nc'
            usi_file_before = date_only_ocean + 'T' + hour_utc_str_ocean + 'Z_MSC_RIOPS_ITZOCRTX_SFC_PS5km_P' + \
                                 str(time_increments_ocean[before_idx]).zfill(3) + '.nc'
            usi_file_after = date_only_ocean + 'T' + hour_utc_str_ocean + 'Z_MSC_RIOPS_ITZOCRTX_SFC_PS5km_P' + \
                                str(time_increments_ocean[after_idx]).zfill(3) + '.nc'
            vsi_file_before = date_only_ocean + 'T' + hour_utc_str_ocean + 'Z_MSC_RIOPS_ITMECRTY_SFC_PS5km_P' + \
                              str(time_increments_ocean[before_idx]).zfill(3) + '.nc'
            vsi_file_after = date_only_ocean + 'T' + hour_utc_str_ocean + 'Z_MSC_RIOPS_ITMECRTY_SFC_PS5km_P' + \
                             str(time_increments_ocean[after_idx]).zfill(3) + '.nc'

        forecast_time_ocean_before = forecast_times_ocean[before_idx]
        forecast_time_ocean_after = forecast_times_ocean[after_idx]

        before_idx = np.where(file_times_ocean <= iceberg_time2)[0][-1]

        try:
            after_idx = np.where(file_times_ocean > iceberg_time2)[0][0]
        except:
            after_idx = -1

        u_curr_file_before2 = date_only_ocean + 'T' + hour_utc_str_ocean + 'Z_MSC_RIOPS_VOZOCRTX_DBS-all_PS5km_P' + \
                             str(time_increments_ocean[before_idx]).zfill(3) + '.nc'
        u_curr_file_after2 = date_only_ocean + 'T' + hour_utc_str_ocean + 'Z_MSC_RIOPS_VOZOCRTX_DBS-all_PS5km_P' + \
                            str(time_increments_ocean[after_idx]).zfill(3) + '.nc'
        v_curr_file_before2 = date_only_ocean + 'T' + hour_utc_str_ocean + 'Z_MSC_RIOPS_VOMECRTY_DBS-all_PS5km_P' + \
                             str(time_increments_ocean[before_idx]).zfill(3) + '.nc'
        v_curr_file_after2 = date_only_ocean + 'T' + hour_utc_str_ocean + 'Z_MSC_RIOPS_VOMECRTY_DBS-all_PS5km_P' + \
                            str(time_increments_ocean[after_idx]).zfill(3) + '.nc'

        forecast_time_ocean_before2 = forecast_times_ocean[before_idx]
        forecast_time_ocean_after2 = forecast_times_ocean[after_idx]

        fname = rootpath_to_metdata + 'GDWPS_wind_wave_forecast_files/' + dirname_wind_waves + '/' + u_wind_file_before
        u_wind_data_before = nc.Dataset(fname)
        u_wind_before = np.squeeze(u_wind_data_before.variables['u10'][:])
        u_wind_data_before.close()

        fname = rootpath_to_metdata + 'GDWPS_wind_wave_forecast_files/' + dirname_wind_waves + '/' + u_wind_file_after
        u_wind_data_after = nc.Dataset(fname)
        u_wind_after = np.squeeze(u_wind_data_after.variables['u10'][:])
        u_wind_data_after.close()

        fname = rootpath_to_metdata + 'GDWPS_wind_wave_forecast_files/' + dirname_wind_waves + '/' + v_wind_file_before
        v_wind_data_before = nc.Dataset(fname)
        v_wind_before = np.squeeze(v_wind_data_before.variables['v10'][:])
        v_wind_data_before.close()

        fname = rootpath_to_metdata + 'GDWPS_wind_wave_forecast_files/' + dirname_wind_waves + '/' + v_wind_file_after
        v_wind_data_after = nc.Dataset(fname)
        v_wind_after = np.squeeze(v_wind_data_after.variables['v10'][:])
        v_wind_data_after.close()

        fname = rootpath_to_metdata + 'GDPS_airT_sw_rad_forecast_files/' + dirname_airT_sw_rad + '/' + airT_file_before
        airT_data_before = nc.Dataset(fname)
        airT_before = np.squeeze(airT_data_before.variables['t2m'][:]) - 273.15
        airT_data_before.close()

        fname = rootpath_to_metdata + 'GDPS_airT_sw_rad_forecast_files/' + dirname_airT_sw_rad + '/' + airT_file_after
        airT_data_after = nc.Dataset(fname)
        airT_after = np.squeeze(airT_data_after.variables['t2m'][:]) - 273.15
        airT_data_after.close()

        fname = rootpath_to_metdata + 'GDPS_airT_sw_rad_forecast_files/' + dirname_airT_sw_rad + '/' + solar_rad_file_before
        solar_rad_data_before = nc.Dataset(fname)
        solar_rad_before = np.squeeze(solar_rad_data_before.variables['ssrd'][:])
        solar_rad_data_before.close()

        fname = rootpath_to_metdata + 'GDPS_airT_sw_rad_forecast_files/' + dirname_airT_sw_rad + '/' + solar_rad_file_after
        solar_rad_data_after = nc.Dataset(fname)
        solar_rad_after = np.squeeze(solar_rad_data_after.variables['ssrd'][:])
        solar_rad_data_after.close()

        fname = rootpath_to_metdata + 'GDWPS_wind_wave_forecast_files/' + dirname_wind_waves + '/' + Hs_file_before
        Hs_data_before = nc.Dataset(fname)
        Hs_before = np.squeeze(Hs_data_before.variables['swh'][:])
        Hs_data_before.close()

        fname = rootpath_to_metdata + 'GDWPS_wind_wave_forecast_files/' + dirname_wind_waves + '/' + Hs_file_after
        Hs_data_after = nc.Dataset(fname)
        Hs_after = np.squeeze(Hs_data_after.variables['swh'][:])
        Hs_data_after.close()

        fname = rootpath_to_metdata + 'GDWPS_wind_wave_forecast_files/' + dirname_wind_waves + '/' + wave_dir_file_before
        wave_dir_data_before = nc.Dataset(fname)
        wave_dir_before = np.squeeze(wave_dir_data_before.variables['wvdir'][:])
        wave_dir_data_before.close()

        fname = rootpath_to_metdata + 'GDWPS_wind_wave_forecast_files/' + dirname_wind_waves + '/' + wave_dir_file_after
        wave_dir_data_after = nc.Dataset(fname)
        wave_dir_after = np.squeeze(wave_dir_data_after.variables['wvdir'][:])
        wave_dir_data_after.close()

        wave_E_before = np.sin(np.deg2rad(wave_dir_before))
        wave_E_after = np.sin(np.deg2rad(wave_dir_after))
        wave_N_before = np.cos(np.deg2rad(wave_dir_before))
        wave_N_after = np.cos(np.deg2rad(wave_dir_after))

        fname = rootpath_to_metdata + 'GDWPS_wind_wave_forecast_files/' + dirname_wind_waves + '/' + wave_pd_file_before
        wave_pd_data_before = nc.Dataset(fname)
        wave_pd_before = np.squeeze(wave_pd_data_before.variables['mp2'][:])
        wave_pd_data_before.close()

        fname = rootpath_to_metdata + 'GDWPS_wind_wave_forecast_files/' + dirname_wind_waves + '/' + wave_pd_file_after
        wave_pd_data_after = nc.Dataset(fname)
        wave_pd_after = np.squeeze(wave_pd_data_after.variables['mp2'][:])
        wave_pd_data_after.close()

        fname = rootpath_to_metdata + 'RIOPS_ocean_forecast_files/' + dirname_ocean + '/' + u_curr_file_before
        u_curr_data_before = nc.Dataset(fname)
        u_curr_before = np.squeeze(u_curr_data_before.variables['vozocrtx'][:])
        u_curr_data_before.close()

        fname = rootpath_to_metdata + 'RIOPS_ocean_forecast_files/' + dirname_ocean + '/' + u_curr_file_after
        u_curr_data_after = nc.Dataset(fname)
        u_curr_after = np.squeeze(u_curr_data_after.variables['vozocrtx'][:])
        u_curr_data_after.close()

        fname = rootpath_to_metdata + 'RIOPS_ocean_forecast_files/' + dirname_ocean + '/' + v_curr_file_before
        v_curr_data_before = nc.Dataset(fname)
        v_curr_before = np.squeeze(v_curr_data_before.variables['vomecrty'][:])
        v_curr_data_before.close()

        fname = rootpath_to_metdata + 'RIOPS_ocean_forecast_files/' + dirname_ocean + '/' + v_curr_file_after
        v_curr_data_after = nc.Dataset(fname)
        v_curr_after = np.squeeze(v_curr_data_after.variables['vomecrty'][:])
        v_curr_data_after.close()

        fname = rootpath_to_metdata + 'RIOPS_ocean_forecast_files/' + dirname_ocean + '/' + u_curr_file_before2
        u_curr_data_before2 = nc.Dataset(fname)
        u_curr_before2 = np.squeeze(u_curr_data_before2.variables['vozocrtx'][:])
        u_curr_data_before2.close()

        fname = rootpath_to_metdata + 'RIOPS_ocean_forecast_files/' + dirname_ocean + '/' + u_curr_file_after2
        u_curr_data_after2 = nc.Dataset(fname)
        u_curr_after2 = np.squeeze(u_curr_data_after2.variables['vozocrtx'][:])
        u_curr_data_after2.close()

        fname = rootpath_to_metdata + 'RIOPS_ocean_forecast_files/' + dirname_ocean + '/' + v_curr_file_before2
        v_curr_data_before2 = nc.Dataset(fname)
        v_curr_before2 = np.squeeze(v_curr_data_before2.variables['vomecrty'][:])
        v_curr_data_before2.close()

        fname = rootpath_to_metdata + 'RIOPS_ocean_forecast_files/' + dirname_ocean + '/' + v_curr_file_after2
        v_curr_data_after2 = nc.Dataset(fname)
        v_curr_after2 = np.squeeze(v_curr_data_after2.variables['vomecrty'][:])
        v_curr_data_after2.close()

        fname = rootpath_to_metdata + 'RIOPS_ocean_forecast_files/' + dirname_ocean + '/' + salinity_file_before
        salinity_data_before = nc.Dataset(fname)
        salinity_before = np.squeeze(salinity_data_before.variables['vosaline'][:])
        salinity_data_before.close()

        fname = rootpath_to_metdata + 'RIOPS_ocean_forecast_files/' + dirname_ocean + '/' + salinity_file_after
        salinity_data_after = nc.Dataset(fname)
        salinity_after = np.squeeze(salinity_data_after.variables['vosaline'][:])
        salinity_data_after.close()

        fname = rootpath_to_metdata + 'RIOPS_ocean_forecast_files/' + dirname_ocean + '/' + pot_temp_file_before
        pot_temp_data_before = nc.Dataset(fname)
        pot_temp_before = np.squeeze(pot_temp_data_before.variables['votemper'][:])
        pot_temp_data_before.close()

        fname = rootpath_to_metdata + 'RIOPS_ocean_forecast_files/' + dirname_ocean + '/' + pot_temp_file_after
        pot_temp_data_after = nc.Dataset(fname)
        pot_temp_after = np.squeeze(pot_temp_data_after.variables['votemper'][:])
        pot_temp_data_after.close()

        fname = rootpath_to_metdata + 'RIOPS_ocean_forecast_files/' + dirname_ocean + '/' + ssh_grad_file_before
        ssh_grad_data_before = nc.Dataset(fname)
        ssh_grad_x_before = np.squeeze(ssh_grad_data_before.variables['ssh_grad_x'][:])
        ssh_grad_y_before = np.squeeze(ssh_grad_data_before.variables['ssh_grad_y'][:])
        ssh_grad_data_before.close()

        fname = rootpath_to_metdata + 'RIOPS_ocean_forecast_files/' + dirname_ocean + '/' + ssh_grad_file_after
        ssh_grad_data_after = nc.Dataset(fname)
        ssh_grad_x_after = np.squeeze(ssh_grad_data_after.variables['ssh_grad_x'][:])
        ssh_grad_y_after = np.squeeze(ssh_grad_data_after.variables['ssh_grad_y'][:])
        ssh_grad_data_after.close()

        if si_toggle:
            fname = rootpath_to_metdata + 'RIOPS_ocean_forecast_files/' + dirname_ocean + '/' + siconc_file_before
            siconc_data_before = nc.Dataset(fname)
            siconc_before = np.squeeze(siconc_data_before.variables['iiceconc'][:])
            siconc_data_before.close()

            fname = rootpath_to_metdata + 'RIOPS_ocean_forecast_files/' + dirname_ocean + '/' + siconc_file_after
            siconc_data_after = nc.Dataset(fname)
            siconc_after = np.squeeze(siconc_data_after.variables['iiceconc'][:])
            siconc_data_after.close()

            fname = rootpath_to_metdata + 'RIOPS_ocean_forecast_files/' + dirname_ocean + '/' + sithick_file_before
            sithick_data_before = nc.Dataset(fname)
            sithick_before = np.squeeze(sithick_data_before.variables['iicevol'][:])
            sithick_data_before.close()

            fname = rootpath_to_metdata + 'RIOPS_ocean_forecast_files/' + dirname_ocean + '/' + sithick_file_after
            sithick_data_after = nc.Dataset(fname)
            sithick_after = np.squeeze(sithick_data_after.variables['iicevol'][:])
            sithick_data_after.close()

            fname = rootpath_to_metdata + 'RIOPS_ocean_forecast_files/' + dirname_ocean + '/' + usi_file_before
            usi_data_before = nc.Dataset(fname)
            usi_before = np.squeeze(usi_data_before.variables['itzocrtx'][:])
            usi_data_before.close()

            fname = rootpath_to_metdata + 'RIOPS_ocean_forecast_files/' + dirname_ocean + '/' + usi_file_after
            usi_data_after = nc.Dataset(fname)
            usi_after = np.squeeze(usi_data_after.variables['itzocrtx'][:])
            usi_data_after.close()

            fname = rootpath_to_metdata + 'RIOPS_ocean_forecast_files/' + dirname_ocean + '/' + vsi_file_before
            vsi_data_before = nc.Dataset(fname)
            vsi_before = np.squeeze(vsi_data_before.variables['itmecrty'][:])
            vsi_data_before.close()

            fname = rootpath_to_metdata + 'RIOPS_ocean_forecast_files/' + dirname_ocean + '/' + vsi_file_after
            vsi_data_after = nc.Dataset(fname)
            vsi_after = np.squeeze(vsi_data_after.variables['itmecrty'][:])
            vsi_data_after.close()

        f_u_wind_before = RegularGridInterpolator((lat_wind_waves, lon_wind_waves), u_wind_before, method='linear', bounds_error=True, fill_value=np.nan)
        f_u_wind_after = RegularGridInterpolator((lat_wind_waves, lon_wind_waves), u_wind_after, method='linear', bounds_error=True, fill_value=np.nan)

        f_v_wind_before = RegularGridInterpolator((lat_wind_waves, lon_wind_waves), v_wind_before, method='linear', bounds_error=True, fill_value=np.nan)
        f_v_wind_after = RegularGridInterpolator((lat_wind_waves, lon_wind_waves), v_wind_after, method='linear', bounds_error=True, fill_value=np.nan)

        f_Hs_before = RegularGridInterpolator((lat_wind_waves, lon_wind_waves), Hs_before, method='nearest', bounds_error=True, fill_value=np.nan)
        f_Hs_after = RegularGridInterpolator((lat_wind_waves, lon_wind_waves), Hs_after, method='nearest', bounds_error=True, fill_value=np.nan)

        f_wave_E_before = RegularGridInterpolator((lat_wind_waves, lon_wind_waves), wave_E_before, method='nearest', bounds_error=True, fill_value=np.nan)
        f_wave_E_after = RegularGridInterpolator((lat_wind_waves, lon_wind_waves), wave_E_after, method='nearest', bounds_error=True, fill_value=np.nan)

        f_wave_N_before = RegularGridInterpolator((lat_wind_waves, lon_wind_waves), wave_N_before, method='nearest', bounds_error=True, fill_value=np.nan)
        f_wave_N_after = RegularGridInterpolator((lat_wind_waves, lon_wind_waves), wave_N_after, method='nearest', bounds_error=True, fill_value=np.nan)

        f_wave_pd_before = RegularGridInterpolator((lat_wind_waves, lon_wind_waves), wave_pd_before, method='nearest', bounds_error=True, fill_value=np.nan)
        f_wave_pd_after = RegularGridInterpolator((lat_wind_waves, lon_wind_waves), wave_pd_after, method='nearest', bounds_error=True, fill_value=np.nan)

        f_airT_before = RegularGridInterpolator((lat_airT_sw_rad, lon_airT_sw_rad), airT_before, method='linear', bounds_error=True, fill_value=np.nan)
        f_airT_after = RegularGridInterpolator((lat_airT_sw_rad, lon_airT_sw_rad), airT_after, method='linear', bounds_error=True, fill_value=np.nan)

        f_solar_rad_before = RegularGridInterpolator((lat_airT_sw_rad, lon_airT_sw_rad), solar_rad_before, method='linear', bounds_error=True, fill_value=np.nan)
        f_solar_rad_after = RegularGridInterpolator((lat_airT_sw_rad, lon_airT_sw_rad), solar_rad_after, method='linear', bounds_error=True, fill_value=np.nan)

        solar_rad_hr_before = int(solar_rad_file_before[-6:-3])
        solar_rad_hr_after = int(solar_rad_file_after[-6:-3])

        t_new = (iceberg_time - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')

        t1 = (forecast_time_wind_waves_before - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
        t2 = (forecast_time_wind_waves_after - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
        weight_wind_waves = (t_new - t1) / (t2 - t1)

        t1 = (forecast_time_ocean_before - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
        t2 = (forecast_time_ocean_after - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
        weight_ocean = (t_new - t1) / (t2 - t1)

        t1 = (forecast_time_ocean_before2 - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
        t2 = (forecast_time_ocean_after2 - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
        weight_ocean2 = (t_new - t1) / (t2 - t1)

        t1 = (forecast_time_airT_sw_rad_before - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
        t2 = (forecast_time_airT_sw_rad_after - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
        weight_airT_sw_rad = (t_new - t1) / (t2 - t1)

        for k in range(len(iceberg_lats0)):
            print('Forecasting iceberg ' + str(iceberg_ids[k]) + '...')
            iceberg_lat = iceberg_lats[i, k]
            iceberg_lon = iceberg_lons[i, k]
            iceberg_u = iceberg_us[i, k]
            iceberg_v = iceberg_vs[i, k]
            ib_length = iceberg_lengths[i, k]
            ib_draft = iceberg_drafts[i, k]
            ib_mass = iceberg_masses[i, k]
            iceberg_grounded_status = iceberg_grounded_statuses[i, k]

            u_wind_before_ib = float(f_u_wind_before([iceberg_lat, iceberg_lon + 360.]))
            u_wind_after_ib = float(f_u_wind_after([iceberg_lat, iceberg_lon + 360.]))
            u_wind_ib = u_wind_before_ib + weight_wind_waves * (u_wind_after_ib - u_wind_before_ib)

            v_wind_before_ib = float(f_v_wind_before([iceberg_lat, iceberg_lon + 360.]))
            v_wind_after_ib = float(f_v_wind_after([iceberg_lat, iceberg_lon + 360.]))
            v_wind_ib = v_wind_before_ib + weight_wind_waves * (v_wind_after_ib - v_wind_before_ib)

            Hs_before_ib = float(f_Hs_before([iceberg_lat, iceberg_lon + 360.]))
            Hs_after_ib = float(f_Hs_after([iceberg_lat, iceberg_lon + 360.]))
            Hs_ib = Hs_before_ib + weight_wind_waves * (Hs_after_ib - Hs_before_ib)

            wave_E_before_ib = float(f_wave_E_before([iceberg_lat, iceberg_lon + 360.]))
            wave_E_after_ib = float(f_wave_E_after([iceberg_lat, iceberg_lon + 360.]))
            wave_N_before_ib = float(f_wave_N_before([iceberg_lat, iceberg_lon + 360.]))
            wave_N_after_ib = float(f_wave_N_after([iceberg_lat, iceberg_lon + 360.]))

            wave_E_ib = wave_E_before_ib + weight_wind_waves * (wave_E_after_ib - wave_E_before_ib)
            wave_N_ib = wave_N_before_ib + weight_wind_waves * (wave_N_after_ib - wave_N_before_ib)
            wave_dir_ib = 90. - np.rad2deg(np.arctan2(wave_N_ib, wave_E_ib))

            if wave_dir_ib < 0:
                wave_dir_ib = wave_dir_ib + 360.

            wave_pd_before_ib = float(f_wave_pd_before([iceberg_lat, iceberg_lon + 360.]))
            wave_pd_after_ib = float(f_wave_pd_after([iceberg_lat, iceberg_lon + 360.]))
            wave_pd_ib = wave_pd_before_ib + weight_wind_waves * (wave_pd_after_ib - wave_pd_before_ib)

            airT_before_ib = float(f_airT_before([iceberg_lat, iceberg_lon]))
            airT_after_ib = float(f_airT_after([iceberg_lat, iceberg_lon]))
            airT_ib = airT_before_ib + weight_airT_sw_rad * (airT_after_ib - airT_before_ib)

            if solar_rad_hr_before == 0:
                solar_rad_before_ib = float(f_solar_rad_before([iceberg_lat, iceberg_lon]))
            else:
                solar_rad_before_ib = float(f_solar_rad_before([iceberg_lat, iceberg_lon]) / (solar_rad_hr_before * 3600.))

            if solar_rad_hr_after == 0:
                solar_rad_after_ib = float(f_solar_rad_after([iceberg_lat, iceberg_lon]))
            else:
                solar_rad_after_ib = float(f_solar_rad_after([iceberg_lat, iceberg_lon]) / (solar_rad_hr_after * 3600.))

            if solar_rad_before_ib < 0:
                solar_rad_before_ib = -solar_rad_before_ib

            if solar_rad_after_ib < 0:
                solar_rad_after_ib = -solar_rad_after_ib

            if solar_rad_before_ib < 0 or np.isnan(solar_rad_before_ib):
                solar_rad_before_ib = 0.

            if solar_rad_after_ib < 0 or np.isnan(solar_rad_after_ib):
                solar_rad_after_ib = 0.

            solar_rad_ib = solar_rad_before_ib + weight_airT_sw_rad * (solar_rad_after_ib - solar_rad_before_ib)

            if ib_draft > 0:
                try:
                    loc_depth = np.argwhere(ocean_depth <= ib_draft).flatten()

                    if loc_depth[-1] + 1 < len(ocean_depth):
                        loc_depth = np.append(loc_depth, loc_depth[-1] + 1)

                    depth_curr_ib = ocean_depth[loc_depth]
                    depth_curr_ib = depth_curr_ib.tolist()
                    depth_curr_ib_interp = np.arange(0., ib_draft, 0.001)
                except:
                    depth_curr_ib = list(ocean_depth[:1])
                    depth_curr_ib_interp = np.arange(0., ib_draft, 0.001)
            else:
                depth_curr_ib = list(ocean_depth[:1])
                depth_curr_ib_interp = np.arange(0., ocean_depth[-1], 0.001)

            dist = np.sqrt((ocean_lat - iceberg_lat) ** 2 + (ocean_lon - (iceberg_lon + 360.)) ** 2)
            i_center, j_center = np.unravel_index(np.argmin(dist), ocean_lat.shape)
            i_min = max(i_center - deg_radius, 0)
            i_max = min(i_center + deg_radius + 1, ocean_lat.shape[0])
            j_min = max(j_center - deg_radius, 0)
            j_max = min(j_center + deg_radius + 1, ocean_lat.shape[1])
            ocean_lat_ind = np.arange(i_min, i_max)
            ocean_lon_ind = np.arange(j_min, j_max)
            ocean_lat_subset = ocean_lat[ocean_lat_ind, ocean_lon_ind]
            ocean_lon_subset = ocean_lon[ocean_lat_ind, ocean_lon_ind]
            points_ocean = np.array([ocean_lat_subset.ravel(), ocean_lon_subset.ravel()]).T

            u_curr_before_depth_list = []
            u_curr_after_depth_list = []
            v_curr_before_depth_list = []
            v_curr_after_depth_list = []
            u_curr_before2_depth_list = []
            u_curr_after2_depth_list = []
            v_curr_before2_depth_list = []
            v_curr_after2_depth_list = []
            salinity_before_depth_list = []
            salinity_after_depth_list = []
            pot_temp_before_depth_list = []
            pot_temp_after_depth_list = []

            for n in range(len(depth_curr_ib)):
                u_curr_before_select = np.squeeze(u_curr_before[n, ocean_lat_ind, ocean_lon_ind])
                u_curr_after_select = np.squeeze(u_curr_after[n, ocean_lat_ind, ocean_lon_ind])
                u_curr_before_temp = griddata(points_ocean, u_curr_before_select.ravel(),(iceberg_lat, iceberg_lon + 360.), method='linear')
                u_curr_after_temp = griddata(points_ocean, u_curr_after_select.ravel(),(iceberg_lat, iceberg_lon + 360.), method='linear')
                u_curr_before_depth_list.append(u_curr_before_temp)
                u_curr_after_depth_list.append(u_curr_after_temp)
                v_curr_before_select = np.squeeze(v_curr_before[n, ocean_lat_ind, ocean_lon_ind])
                v_curr_after_select = np.squeeze(v_curr_after[n, ocean_lat_ind, ocean_lon_ind])
                v_curr_before_temp = griddata(points_ocean, v_curr_before_select.ravel(),(iceberg_lat, iceberg_lon + 360.), method='linear')
                v_curr_after_temp = griddata(points_ocean, v_curr_after_select.ravel(),(iceberg_lat, iceberg_lon + 360.), method='linear')
                v_curr_before_depth_list.append(v_curr_before_temp)
                v_curr_after_depth_list.append(v_curr_after_temp)
                u_curr_before2_select = np.squeeze(u_curr_before2[n, ocean_lat_ind, ocean_lon_ind])
                u_curr_after2_select = np.squeeze(u_curr_after2[n, ocean_lat_ind, ocean_lon_ind])
                u_curr_before2_temp = griddata(points_ocean, u_curr_before2_select.ravel(),(iceberg_lat, iceberg_lon + 360.), method='linear')
                u_curr_after2_temp = griddata(points_ocean, u_curr_after2_select.ravel(),(iceberg_lat, iceberg_lon + 360.), method='linear')
                u_curr_before2_depth_list.append(u_curr_before2_temp)
                u_curr_after2_depth_list.append(u_curr_after2_temp)
                v_curr_before2_select = np.squeeze(v_curr_before2[n, ocean_lat_ind, ocean_lon_ind])
                v_curr_after2_select = np.squeeze(v_curr_after2[n, ocean_lat_ind, ocean_lon_ind])
                v_curr_before2_temp = griddata(points_ocean, v_curr_before2_select.ravel(),(iceberg_lat, iceberg_lon + 360.), method='linear')
                v_curr_after2_temp = griddata(points_ocean, v_curr_after2_select.ravel(),(iceberg_lat, iceberg_lon + 360.), method='linear')
                v_curr_before2_depth_list.append(v_curr_before2_temp)
                v_curr_after2_depth_list.append(v_curr_after2_temp)
                salinity_before_select = np.squeeze(salinity_before[n, ocean_lat_ind, ocean_lon_ind])
                salinity_after_select = np.squeeze(salinity_after[n, ocean_lat_ind, ocean_lon_ind])
                salinity_before_temp = griddata(points_ocean, salinity_before_select.ravel(),(iceberg_lat, iceberg_lon + 360.), method='linear')
                salinity_after_temp = griddata(points_ocean, salinity_after_select.ravel(),(iceberg_lat, iceberg_lon + 360.), method='linear')
                salinity_before_depth_list.append(salinity_before_temp)
                salinity_after_depth_list.append(salinity_after_temp)
                pot_temp_before_select = np.squeeze(pot_temp_before[n, ocean_lat_ind, ocean_lon_ind])
                pot_temp_after_select = np.squeeze(pot_temp_after[n, ocean_lat_ind, ocean_lon_ind])
                pot_temp_before_temp = griddata(points_ocean, pot_temp_before_select.ravel(),(iceberg_lat, iceberg_lon + 360.), method='linear')
                pot_temp_after_temp = griddata(points_ocean, pot_temp_after_select.ravel(),(iceberg_lat, iceberg_lon + 360.), method='linear')
                pot_temp_before_depth_list.append(pot_temp_before_temp)
                pot_temp_after_depth_list.append(pot_temp_after_temp)

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

            salinity_before_depth_list = [float(val) for val in salinity_before_depth_list]
            salinity_after_depth_list = [float(val) for val in salinity_after_depth_list]
            interp_func = interp1d(depth_curr_ib, salinity_before_depth_list, kind='linear', fill_value='extrapolate')
            salinity_before_depth_list = interp_func(depth_curr_ib_interp)
            interp_func = interp1d(depth_curr_ib, salinity_after_depth_list, kind='linear', fill_value='extrapolate')
            salinity_after_depth_list = interp_func(depth_curr_ib_interp)

            pot_temp_before_depth_list = [float(val) for val in pot_temp_before_depth_list]
            pot_temp_after_depth_list = [float(val) for val in pot_temp_after_depth_list]
            interp_func = interp1d(depth_curr_ib, pot_temp_before_depth_list, kind='linear', fill_value='extrapolate')
            pot_temp_before_depth_list = interp_func(depth_curr_ib_interp)
            interp_func = interp1d(depth_curr_ib, pot_temp_after_depth_list, kind='linear', fill_value='extrapolate')
            pot_temp_after_depth_list = interp_func(depth_curr_ib_interp)

            dist = np.sqrt((ssh_grad_x_lat - iceberg_lat) ** 2 + (ssh_grad_x_lon - (iceberg_lon + 360.)) ** 2)
            i_center, j_center = np.unravel_index(np.argmin(dist), ssh_grad_x_lat.shape)
            i_min = max(i_center - deg_radius, 0)
            i_max = min(i_center + deg_radius + 1, ssh_grad_x_lat.shape[0])
            j_min = max(j_center - deg_radius, 0)
            j_max = min(j_center + deg_radius + 1, ssh_grad_x_lat.shape[1])
            ssh_grad_x_lat_ind = np.arange(i_min, i_max)
            ssh_grad_x_lon_ind = np.arange(j_min, j_max)
            ssh_grad_x_lat_subset = ssh_grad_x_lat[ssh_grad_x_lat_ind, ssh_grad_x_lon_ind]
            ssh_grad_x_lon_subset = ssh_grad_x_lon[ssh_grad_x_lat_ind, ssh_grad_x_lon_ind]
            points_ssh_grad_x = np.array([ssh_grad_x_lat_subset.ravel(), ssh_grad_x_lon_subset.ravel()]).T
            ssh_grad_x_before_subset = ssh_grad_x_before[ssh_grad_x_lat_ind, ssh_grad_x_lon_ind]
            ssh_grad_x_after_subset = ssh_grad_x_after[ssh_grad_x_lat_ind, ssh_grad_x_lon_ind]

            dist = np.sqrt((ssh_grad_y_lat - iceberg_lat) ** 2 + (ssh_grad_y_lon - (iceberg_lon + 360.)) ** 2)
            i_center, j_center = np.unravel_index(np.argmin(dist), ssh_grad_y_lat.shape)
            i_min = max(i_center - deg_radius, 0)
            i_max = min(i_center + deg_radius + 1, ssh_grad_y_lat.shape[0])
            j_min = max(j_center - deg_radius, 0)
            j_max = min(j_center + deg_radius + 1, ssh_grad_y_lat.shape[1])
            ssh_grad_y_lat_ind = np.arange(i_min, i_max)
            ssh_grad_y_lon_ind = np.arange(j_min, j_max)
            ssh_grad_y_lat_subset = ssh_grad_y_lat[ssh_grad_y_lat_ind, ssh_grad_y_lon_ind]
            ssh_grad_y_lon_subset = ssh_grad_y_lon[ssh_grad_y_lat_ind, ssh_grad_y_lon_ind]
            points_ssh_grad_y = np.array([ssh_grad_y_lat_subset.ravel(), ssh_grad_y_lon_subset.ravel()]).T
            ssh_grad_y_before_subset = ssh_grad_y_before[ssh_grad_y_lat_ind, ssh_grad_y_lon_ind]
            ssh_grad_y_after_subset = ssh_grad_y_after[ssh_grad_y_lat_ind, ssh_grad_y_lon_ind]

            ssh_grad_x_before_ib = griddata(points_ssh_grad_x, ssh_grad_x_before_subset.ravel(),(iceberg_lat, iceberg_lon + 360.), method='linear')
            ssh_grad_y_before_ib = griddata(points_ssh_grad_y, ssh_grad_y_before_subset.ravel(),(iceberg_lat, iceberg_lon + 360.), method='linear')

            ssh_grad_x_after_ib = griddata(points_ssh_grad_x, ssh_grad_x_after_subset.ravel(),(iceberg_lat, iceberg_lon + 360.), method='linear')
            ssh_grad_y_after_ib = griddata(points_ssh_grad_y, ssh_grad_y_after_subset.ravel(),(iceberg_lat, iceberg_lon + 360.), method='linear')

            if si_toggle:
                siconc_before_ib = griddata(points_ocean, siconc_before.ravel(),(iceberg_lat, iceberg_lon + 360.), method='linear')
                siconc_after_ib = griddata(points_ocean, siconc_after.ravel(),(iceberg_lat, iceberg_lon + 360.), method='linear')
                sithick_before_ib = griddata(points_ocean, sithick_before.ravel(), (iceberg_lat, iceberg_lon + 360.), method='linear')
                sithick_after_ib = griddata(points_ocean, sithick_after.ravel(), (iceberg_lat, iceberg_lon + 360.), method='linear')
                usi_before_ib = griddata(points_ocean, usi_before.ravel(), (iceberg_lat, iceberg_lon + 360.), method='linear')
                usi_after_ib = griddata(points_ocean, usi_after.ravel(), (iceberg_lat, iceberg_lon + 360.), method='linear')
                vsi_before_ib = griddata(points_ocean, vsi_before.ravel(), (iceberg_lat, iceberg_lon + 360.), method='linear')
                vsi_after_ib = griddata(points_ocean, vsi_after.ravel(), (iceberg_lat, iceberg_lon + 360.), method='linear')
                siconc_ib = siconc_before_ib + weight_ocean * (siconc_after_ib - siconc_before_ib)
                sithick_ib = sithick_before_ib + weight_ocean * (sithick_after_ib - sithick_before_ib)
                usi_ib = usi_before_ib + weight_ocean * (usi_after_ib - usi_before_ib)
                vsi_ib = vsi_before_ib + weight_ocean * (vsi_after_ib - vsi_before_ib)
            else:
                siconc_ib = 0.
                sithick_ib = 0.
                usi_ib = 0.
                vsi_ib = 0.

            u_curr_ib = u_curr_before_ib + weight_ocean * (u_curr_after_ib - u_curr_before_ib)
            v_curr_ib = v_curr_before_ib + weight_ocean * (v_curr_after_ib - v_curr_before_ib)

            u_curr_ib2 = u_curr_before2_ib + weight_ocean2 * (u_curr_after2_ib - u_curr_before2_ib)
            v_curr_ib2 = v_curr_before2_ib + weight_ocean2 * (v_curr_after2_ib - v_curr_before2_ib)

            ssh_grad_x_ib = ssh_grad_x_before_ib + weight_ocean * (ssh_grad_x_after_ib - ssh_grad_x_before_ib)
            ssh_grad_y_ib = ssh_grad_y_before_ib + weight_ocean * (ssh_grad_y_after_ib - ssh_grad_y_before_ib)

            salinity_ib_list = []
            pot_temp_ib_list = []

            for n in range(len(depth_curr_ib_interp)):
                salinity_ib = salinity_before_depth_list[n] + weight_ocean * (salinity_after_depth_list[n] - salinity_before_depth_list[n])
                pot_temp_ib = pot_temp_before_depth_list[n] + weight_ocean * (pot_temp_after_depth_list[n] - pot_temp_before_depth_list[n])
                salinity_ib_list.append(salinity_ib)
                pot_temp_ib_list.append(pot_temp_ib)

            if ib_mass > 0:
                new_iceberg_length, new_iceberg_draft, new_iceberg_sail, new_iceberg_mass = iceberg_det(ib_length, ib_mass, iceberg_lat, solar_rad_ib,
                                                                                                        ice_albedo, Lf_ice, rho_ice, pot_temp_ib_list,
                                                                                                        salinity_ib_list, depth_curr_ib_interp, airT_ib,
                                                                                                        u_curr_ib, v_curr_ib, u_wind_ib, v_wind_ib,
                                                                                                        iceberg_u, iceberg_v, Hs_ib, wave_pd_ib,
                                                                                                        iceberg_times_dt[i], siconc_ib)
                iceberg_bathy_depth = bathy_interp([[iceberg_lat, iceberg_lon]])[0]

                if iceberg_bathy_depth <= new_iceberg_draft:
                    iceberg_grounded_statuses[i, k] = 1
                    iceberg_grounded_status = 1
                    iceberg_us[i, k] = 0.
                    iceberg_vs[i, k] = 0.
                else:
                    iceberg_grounded_statuses[i, k] = 0
                    iceberg_grounded_status = 0
            else:
                new_iceberg_length = 0.
                new_iceberg_draft = 0.
                new_iceberg_sail = 0.
                new_iceberg_mass = 0.

            if new_iceberg_length < 40:
                print('Warning: Iceberg ' + str(iceberg_ids[k]) + ' predicted to deteriorate to ' + str(new_iceberg_length) +
                      ' meters waterline length at ' + str(iceberg_times[i]) + ' UTC at ' + str(iceberg_lat) +
                      u'\N{DEGREE SIGN}' + 'N/' + str(-iceberg_lon) + u'\N{DEGREE SIGN}' + 'W.')

            iceberg_lengths[i + 1, k] = new_iceberg_length
            iceberg_drafts[i + 1, k] = new_iceberg_draft
            iceberg_sails[i + 1, k] = new_iceberg_sail
            iceberg_masses[i + 1, k] = new_iceberg_mass

            if new_iceberg_mass > 0:
                solution = solve_ivp(duv_dt, (0., iceberg_times_dt[i]), [iceberg_u, iceberg_v], method='BDF', t_eval=[0., iceberg_times_dt[i]])
                iceberg_u_end = solution.y[0][-1]
                iceberg_v_end = solution.y[1][-1]
                h_min = 660.9 / ((20e3) * np.exp(-20 * (1 - siconc_ib)))

                if siconc_ib >= 0.9 and sithick_ib >= h_min:
                    iceberg_u_end = usi_ib
                    iceberg_v_end = vsi_ib
            else:
                iceberg_u_end = 0.
                iceberg_v_end = 0.

            final_speed = np.sqrt(iceberg_u_end ** 2 + iceberg_v_end ** 2)

            if final_speed >= 2:
                iceberg_u_end = iceberg_u
                iceberg_v_end = iceberg_v

            if iceberg_grounded_status == 1:
                iceberg_u_end = 0.
                iceberg_v_end = 0.

            iceberg_x = np.nanmean([iceberg_u, iceberg_u_end]) * iceberg_times_dt[i]
            iceberg_y = np.nanmean([iceberg_v, iceberg_v_end]) * iceberg_times_dt[i]
            iceberg_dist = np.sqrt(iceberg_x ** 2 + iceberg_y ** 2)
            iceberg_course = 90. - np.rad2deg(np.arctan2(iceberg_y, iceberg_x))

            if iceberg_course < 0:
                iceberg_course = iceberg_course + 360.

            iceberg_lat2, iceberg_lon2 = dist_course(Re, iceberg_lat, iceberg_lon, iceberg_dist, iceberg_course)
            iceberg_us[i + 1, k] = iceberg_u_end
            iceberg_vs[i + 1, k] = iceberg_v_end
            iceberg_lats[i + 1, k] = iceberg_lat2
            iceberg_lons[i + 1, k] = iceberg_lon2
            iceberg_bathy_depth = bathy_interp([[iceberg_lat2, iceberg_lon2]])[0]

            if iceberg_bathy_depth <= new_iceberg_draft:
                iceberg_grounded_statuses[i + 1, k] = 1
                iceberg_us[i + 1, k] = 0.
                iceberg_vs[i + 1, k] = 0.
            else:
                iceberg_grounded_statuses[i + 1, k] = 0

    iceberg_times = np.array(iceberg_times)
    iceberg_times = iceberg_times.astype(str).tolist()
    iceberg_lats = np.squeeze(iceberg_lats)
    iceberg_lons = np.squeeze(iceberg_lons)
    iceberg_lengths = np.squeeze(iceberg_lengths)
    iceberg_grounded_statuses = np.squeeze(iceberg_grounded_statuses)
    return iceberg_times, iceberg_lats, iceberg_lons, iceberg_lengths, iceberg_grounded_statuses


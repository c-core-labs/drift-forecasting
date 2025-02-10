
from scipy.interpolate import RegularGridInterpolator, griddata, interp1d
from scipy.stats import truncnorm, genlogistic
from scipy.spatial import ConvexHull
import numpy as np
import netCDF4 as nc
import gsw
import os
from observations import Observations
# from observation import Observation

def rcm_bergy_bit_growler_forecaster(obs: Observations, t1: np.datetime64, si_toggle):
    deg_radius = 5
    ens_num = 10
    rho_ice = 910.
    ice_albedo = 0.1
    Lf_ice = 3.36e5
    Re = 6371e3
    alpha_cor = 20.
    mean_wind_speed = 1.66 # in m/s
    std_dev_wind_speed = 2.99 # in m/s
    left_trunc_wind_dir = -180.
    right_trunc_wind_dir = 180.
    mean_wind_dir = 0. # in degrees
    std_dev_wind_dir = 45. / 1.96 # in degrees
    k_curr_speed = -0.24577 # Shape parameter
    scale_curr_speed = 0.27055 # Scale parameter (similar to σ)
    loc_curr_speed = 0.2358 # Location parameter (similar to μ)
    mean_curr_speed = 0.17 # Approximate mean of the distribution
    std_dev_curr_speed = 0.58 # Approximate standard deviation of the distribution
    right_trunc_curr_speed = 1.
    mean_curr_dir = -8. # in degrees
    left_trunc_curr_dir = -180.
    right_trunc_curr_dir = 180.
    mean_Hs = -0.21 # m
    std_dev_Hs = 0.52 # m
    min_length_bb = 0
    min_length_growler = 0
    bathy_data_path = './GEBCO_Bathymetric_Data/gebco_2024.nc'
    rootpath_to_metdata = './RCM_Iceberg_Metocean_Data/'
    iceberg_lats0 = obs.lat
    iceberg_lons0 = obs.lon
    rcm_datetime0 = obs.time
    iceberg_ids = obs.id
    forecast_end_time = t1
    iceberg_lats0 = iceberg_lats0 if isinstance(iceberg_lats0, list) else [iceberg_lats0]
    iceberg_lons0 = iceberg_lons0 if isinstance(iceberg_lons0, list) else [iceberg_lons0]
    iceberg_ids = iceberg_ids if isinstance(iceberg_ids, list) else [iceberg_ids]

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

    def iceberg_vel(u_wind, v_wind, u_curr, v_curr, alpha_coriolis):
        if (np.any(np.isnan(u_wind)) or np.any(np.isnan(v_wind)) or np.any(np.isinf(u_wind)) or np.any(np.isinf(v_wind))
                or np.any(~np.isreal(u_wind)) or np.any(~np.isreal(v_wind))):
            u_wind = 0.
            v_wind = 0.

        if (np.any(np.isnan(u_curr)) or np.any(np.isnan(v_curr)) or np.any(np.isinf(u_curr)) or np.any(np.isinf(v_curr))
                or np.any(~np.isreal(u_curr)) or np.any(~np.isreal(v_curr))):
            u_curr = 0.
            v_curr = 0.

        alpha_coriolis = np.radians(alpha_coriolis)
        cos_alpha = np.cos(alpha_coriolis)
        sin_alpha = np.sin(alpha_coriolis)
        u_wind_rot = cos_alpha * u_wind - sin_alpha * v_wind
        v_wind_rot = sin_alpha * u_wind + cos_alpha * v_wind
        ib_u = 0.018 * u_wind_rot + u_curr
        ib_v = 0.018 * v_wind_rot + v_curr
        return ib_u, ib_v

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
        new_iceberg_draft = 1.78 * (new_iceberg_length ** 0.71) # meters
        new_iceberg_sail = 0.077 * (new_iceberg_length ** 2) # m ** 2
        return new_iceberg_length, new_iceberg_draft, new_iceberg_sail, new_iceberg_mass

    def truncated_normal(mean, std_dev, left_trunc, right_trunc):
        """
        Samples a random number from a truncated normal distribution.

        Args:
        - mean (float): Mean of the normal distribution.
        - std_dev (float): Standard deviation of the normal distribution.
        - left_trunc (float): Left truncation limit.
        - right_trunc (float): Right truncation limit.

        Returns:
        - float: A randomly sampled number within the specified range.
        """
        std_dev = max(std_dev, 1e-6) # Prevent zero or negative std_dev
        a, b = (left_trunc - mean) / std_dev, (right_trunc - mean) / std_dev
        return truncnorm.rvs(a, b, loc=mean, scale=std_dev)

    def sample_scaled_error(mean, std_dev, k, scale, loc, right_trunc):
        """
        Samples a random number from a truncated generalized logistic distribution.

        Args:
        - mean (float): Mean of the distribution.
        - std_dev (float): Standard deviation of the distribution.
        - k (float): Shape parameter of the generalized logistic distribution.
        - scale (float): Scale parameter of the distribution.
        - loc (float): Base location parameter of the distribution.
        - right_trunc (float): Upper truncation limit.

        Returns:
        - float: A random sample within the defined range.
        """
        std_dev = max(std_dev, 1e-6)
        lower_trunc = mean - 5 * std_dev - 10 # Define a lower truncation based on std_dev
        x = np.linspace(lower_trunc, right_trunc, 1000)
        pdf = genlogistic.pdf(x, c=-k, loc=loc, scale=scale)
        truncated_cdf = pdf / np.sum(pdf)
        sample = np.random.choice(x, p=truncated_cdf)
        return sample

    def sample_hs_error(mean, std_dev, left_trunc):
        """
        Samples a random number from a Normal distribution with left truncation and renormalization.

        Args:
        - mean (float): Mean of the Normal distribution.
        - std_dev (float): Standard deviation of the Normal distribution.
        - left_trunc (float): Left truncation point (-Hs).

        Returns:
        - float: A random sample.
        """
       # Calculate the normalized lower bound for truncation
        std_dev = max(std_dev, 1e-6) # Prevent zero or negative std_dev
        a = (left_trunc - mean) / std_dev # Truncation point normalized by the distribution
        trunc_norm_dist = truncnorm(a=a, b=np.inf, loc=mean, scale=std_dev)
        sample = trunc_norm_dist.rvs(size=1)[0]
        return sample

    def calculate_outer_boundaries(lengths, lats, lons, min_length, length_ranges):
        """
        Calculate outer boundaries for iceberg ensemble drift tracks at multiple length ranges.

        Args:
        - lengths (np.ndarray): (i x k x m) Array of iceberg lengths along their ensemble tracks.
        - lats (np.ndarray): (i x k x m) Array of latitudes.
        - lons (np.ndarray): (i x k x m) Array of longitudes.
        - min_length (float): Minimum iceberg length threshold to define the general boundary.
        - length_ranges (list of tuples): List of (lower, upper) length ranges for computing separate boundaries.

        Returns:
        - dict: Dictionary containing:
          - 'min_length_boundary': Outer boundary points for `min_length`.
          - 'length_range_boundaries': Dict mapping each length range (tuple) to its boundary.
        """
        num_icebergs = lengths.shape[1]
        boundaries = {"min_length_boundary": {}, "length_range_boundaries": {rng: {} for rng in length_ranges}}

        for k in range(num_icebergs):
            # === Compute boundary for global min_length ===
            valid_mask_min = lengths[:, k, :] >= min_length
            valid_lats_min = lats[:, k, :][valid_mask_min]
            valid_lons_min = lons[:, k, :][valid_mask_min]

            if valid_lats_min.size > 2:
                points_min = np.column_stack((valid_lons_min, valid_lats_min))
                if not np.all(points_min == points_min[0]):  # Ensure at least 3 unique points
                    hull_min = ConvexHull(points_min)
                    boundaries["min_length_boundary"][k] = points_min[hull_min.vertices]
                else:
                    boundaries["min_length_boundary"][k] = None
            else:
                boundaries["min_length_boundary"][k] = None

            # === Compute boundaries for each length range ===
            for lower, upper in length_ranges:
                valid_mask_range = (lengths[:, k, :] >= lower) & (lengths[:, k, :] < upper)
                valid_lats_range = lats[:, k, :][valid_mask_range]
                valid_lons_range = lons[:, k, :][valid_mask_range]

                if valid_lats_range.size > 2:
                    points_range = np.column_stack((valid_lons_range, valid_lats_range))
                    if not np.all(points_range == points_range[0]):  # Ensure at least 3 unique points
                        hull_range = ConvexHull(points_range)
                        boundaries["length_range_boundaries"][(lower, upper)][k] = points_range[hull_range.vertices]
                    else:
                        boundaries["length_range_boundaries"][(lower, upper)][k] = None
                else:
                    boundaries["length_range_boundaries"][(lower, upper)][k] = None

        return boundaries

    def calculate_overall_bergy_bit_growler_boundary(bergy_bit_bounds, growler_bounds):
        """
        Compute a single overall outer boundary that includes both bergy bits and growlers.

        Args:
        - bergy_bit_bounds (dict): Dictionary containing bergy bit boundaries.
        - growler_bounds (dict): Dictionary containing growler boundaries.

        Returns:
        - np.ndarray or None: Overall outer boundary points (N x 2) if valid points exist, otherwise None.
        """
        all_points = []

        # Collect all valid boundary points from bergy bits
        if "min_length_boundary" in bergy_bit_bounds:
            for k, boundary in bergy_bit_bounds["min_length_boundary"].items():
                if boundary is not None and boundary.size > 2:
                    all_points.append(boundary)

        if "length_range_boundaries" in bergy_bit_bounds:
            for boundary_dict in bergy_bit_bounds["length_range_boundaries"].values():
                for k, boundary in boundary_dict.items():
                    if boundary is not None and boundary.size > 2:
                        all_points.append(boundary)

        # Collect all valid boundary points from growlers
        if "min_length_boundary" in growler_bounds:
            for k, boundary in growler_bounds["min_length_boundary"].items():
                if boundary is not None and boundary.size > 2:
                    all_points.append(boundary)

        if "length_range_boundaries" in growler_bounds:
            for boundary_dict in growler_bounds["length_range_boundaries"].values():
                for k, boundary in boundary_dict.items():
                    if boundary is not None and boundary.size > 2:
                        all_points.append(boundary)

        # If no valid points were collected, return None
        if not all_points:
            return None  # No valid points, return None

        # Convert to a single array
        all_points = np.vstack(all_points)

        # Ensure at least 3 **unique** points exist
        unique_points = np.unique(all_points, axis=0)
        if unique_points.shape[0] < 3:
            return None  # Not enough unique points for a valid boundary

        # Compute the convex hull
        hull = ConvexHull(unique_points)
        return unique_points[hull.vertices]

    def last_valid_length_stats(lengths, min_length, timeseries):
        length_stats = {}

        for iceberg_index in range(lengths.shape[1]): # loop over icebergs (k)
            valid_lengths = []
            latest_times = []

            for ens_member in range(lengths.shape[2]): # loop over ensemble members (m)
               # Find the last valid length and corresponding time
                for time_step in range(lengths.shape[0] - 1, -1, -1): # loop over time steps (i) in reverse
                    if lengths[time_step, iceberg_index, ens_member] > min_length:
                        valid_lengths.append(lengths[time_step, iceberg_index, ens_member])
                        latest_times.append(timeseries[time_step])
                        break # Exit once the last valid length is found for this ensemble member

            if valid_lengths:
               # Save stats and the latest time for this iceberg
                length_stats[iceberg_index] = {'min': min(valid_lengths), 'max': max(valid_lengths), 'mean': np.mean(valid_lengths), 'latest_time': max(latest_times)}
            else:
               # No valid lengths found
                length_stats[iceberg_index] = {'min': None, 'max': None, 'mean': None, 'latest_time': None}

        return length_stats

    def overall_last_valid_length_stats(lengths, min_length, timeseries):
        """
        Calculate overall statistics for the last valid iceberg lengths across all bergy bits and growlers.

        Args:
        - lengths (np.ndarray): (i x k x m) Array of iceberg lengths along their ensemble tracks.
        - min_length (float): Minimum iceberg length threshold to be considered valid.
        - timeseries (np.ndarray): (i,) Array of time steps.

        Returns:
        - dict: Dictionary containing:
            - 'min': Minimum last valid length across all icebergs.
            - 'max': Maximum last valid length across all icebergs.
            - 'mean': Mean of last valid lengths across all icebergs.
            - 'latest_time': Latest time corresponding to the last valid length.
        """
        valid_lengths = []
        latest_times = []

        # Loop over icebergs (k)
        for iceberg_index in range(lengths.shape[1]):
            for ens_member in range(lengths.shape[2]):  # Loop over ensemble members (m)
                # Find the last valid length and corresponding time
                for time_step in range(lengths.shape[0] - 1, -1, -1):  # Reverse loop over time (i)
                    if lengths[time_step, iceberg_index, ens_member] > min_length:
                        valid_lengths.append(lengths[time_step, iceberg_index, ens_member])
                        latest_times.append(timeseries[time_step])
                        break  # Stop once the last valid length is found for this member

        if valid_lengths:
            return {'min': min(valid_lengths), 'max': max(valid_lengths), 'mean': np.mean(valid_lengths), 'latest_time': max(latest_times)}
        else:
            return {'min': None, 'max': None, 'mean': None, 'latest_time': None}

    rcm_datetime0 = np.datetime64(rcm_datetime0)
    forecast_end_time = np.datetime64(forecast_end_time)
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

    bergy_bit_drafts0 = np.empty((len(iceberg_lats0),))
    bergy_bit_masses0 = np.empty((len(iceberg_lats0),))
    bergy_bit_sails0 = np.empty((len(iceberg_lats0),))
    bergy_bit_grounded_statuses0 = np.zeros((len(iceberg_lats0),))

    growler_drafts0 = np.empty((len(iceberg_lats0),))
    growler_masses0 = np.empty((len(iceberg_lats0),))
    growler_sails0 = np.empty((len(iceberg_lats0),))
    growler_grounded_statuses0 = np.zeros((len(iceberg_lats0),))

    for i in range(len(iceberg_lats0)):
        bergy_bit_drafts0[i] = 1.78 * (15. ** 0.71) # meters
        bergy_bit_masses0[i] = 0.45 * rho_ice * (15. ** 3) # kg
        bergy_bit_sails0[i] = 0.077 * (15. ** 2) # m ** 2
        bergy_bit_bathy_depth0 = bathy_interp([[iceberg_lats0[i], iceberg_lons0[i]]])[0]

        if bergy_bit_bathy_depth0 <= bergy_bit_drafts0[i]:
            bergy_bit_drafts0[i] = bergy_bit_bathy_depth0 - 1.

        growler_drafts0[i] = 1.78 * (5. ** 0.71) # meters
        growler_masses0[i] = 0.45 * rho_ice * (5. ** 3) # kg
        growler_sails0[i] = 0.077 * (5. ** 2) # m ** 2
        growler_bathy_depth0 = bathy_interp([[iceberg_lats0[i], iceberg_lons0[i]]])[0]

        if growler_bathy_depth0 <= growler_drafts0[i]:
            growler_drafts0[i] = growler_bathy_depth0 - 1.

    bergy_bit_growler_times = [forecast_time]
    current_time = forecast_time

    while current_time + np.timedelta64(1, 'h') < forecast_end_time:
        current_time += np.timedelta64(1, 'h')
        bergy_bit_growler_times.append(current_time)

    if bergy_bit_growler_times[-1] < forecast_end_time:
        bergy_bit_growler_times.append(forecast_end_time)

    bergy_bit_growler_times = list(bergy_bit_growler_times)
    bergy_bit_growler_times_dt = [float((bergy_bit_growler_times[i + 1] - bergy_bit_growler_times[i]) / np.timedelta64(1, 's'))
                                  for i in range(len(bergy_bit_growler_times) - 1)]
    bergy_bit_growler_times_dt = list(bergy_bit_growler_times_dt)

    bergy_bit_lats = np.empty((len(bergy_bit_growler_times), len(iceberg_lats0), ens_num))
    bergy_bit_lons = np.empty((len(bergy_bit_growler_times), len(iceberg_lats0), ens_num))
    bergy_bit_lengths = np.empty((len(bergy_bit_growler_times), len(iceberg_lats0), ens_num))
    bergy_bit_drafts = np.empty((len(bergy_bit_growler_times), len(iceberg_lats0), ens_num))
    bergy_bit_sails = np.empty((len(bergy_bit_growler_times), len(iceberg_lats0), ens_num))
    bergy_bit_masses = np.empty((len(bergy_bit_growler_times), len(iceberg_lats0), ens_num))
    bergy_bit_grounded_statuses = np.empty((len(bergy_bit_growler_times), len(iceberg_lats0), ens_num))
    bergy_bit_us = np.zeros((len(bergy_bit_growler_times), len(iceberg_lats0), ens_num))
    bergy_bit_vs = np.zeros((len(bergy_bit_growler_times), len(iceberg_lats0), ens_num))

    growler_lats = np.empty((len(bergy_bit_growler_times), len(iceberg_lats0), ens_num))
    growler_lons = np.empty((len(bergy_bit_growler_times), len(iceberg_lats0), ens_num))
    growler_lengths = np.empty((len(bergy_bit_growler_times), len(iceberg_lats0), ens_num))
    growler_drafts = np.empty((len(bergy_bit_growler_times), len(iceberg_lats0), ens_num))
    growler_sails = np.empty((len(bergy_bit_growler_times), len(iceberg_lats0), ens_num))
    growler_masses = np.empty((len(bergy_bit_growler_times), len(iceberg_lats0), ens_num))
    growler_grounded_statuses = np.empty((len(bergy_bit_growler_times), len(iceberg_lats0), ens_num))
    growler_us = np.zeros((len(bergy_bit_growler_times), len(iceberg_lats0), ens_num))
    growler_vs = np.zeros((len(bergy_bit_growler_times), len(iceberg_lats0), ens_num))

    bergy_bit_lats[0, :, :] = np.broadcast_to(np.array(iceberg_lats0)[:, np.newaxis], (len(iceberg_lats0), ens_num))
    bergy_bit_lons[0, :, :] = np.broadcast_to(np.array(iceberg_lons0)[:, np.newaxis], (len(iceberg_lats0), ens_num))
    bergy_bit_lengths[0, :, :] = np.broadcast_to(15., (len(iceberg_lats0), ens_num))
    bergy_bit_drafts[0, :, :] = np.broadcast_to(np.array(bergy_bit_drafts0)[:, np.newaxis], (len(iceberg_lats0), ens_num))
    bergy_bit_sails[0, :, :] = np.broadcast_to(np.array(bergy_bit_sails0)[:, np.newaxis], (len(iceberg_lats0), ens_num))
    bergy_bit_masses[0, :, :] = np.broadcast_to(np.array(bergy_bit_masses0)[:, np.newaxis], (len(iceberg_lats0), ens_num))
    bergy_bit_grounded_statuses[0, :, :] = np.broadcast_to(np.array(bergy_bit_grounded_statuses0)[:, np.newaxis], (len(iceberg_lats0), ens_num))

    growler_lats[0, :, :] = np.broadcast_to(np.array(iceberg_lats0)[:, np.newaxis], (len(iceberg_lats0), ens_num))
    growler_lons[0, :, :] = np.broadcast_to(np.array(iceberg_lons0)[:, np.newaxis], (len(iceberg_lats0), ens_num))
    growler_lengths[0, :, :] = np.broadcast_to(5., (len(iceberg_lats0), ens_num))
    growler_drafts[0, :, :] = np.broadcast_to(np.array(growler_drafts0)[:, np.newaxis], (len(iceberg_lats0), ens_num))
    growler_sails[0, :, :] = np.broadcast_to(np.array(growler_sails0)[:, np.newaxis], (len(iceberg_lats0), ens_num))
    growler_masses[0, :, :] = np.broadcast_to(np.array(growler_masses0)[:, np.newaxis], (len(iceberg_lats0), ens_num))
    growler_grounded_statuses[0, :, :] = np.broadcast_to(np.array(growler_grounded_statuses0)[:, np.newaxis], (len(iceberg_lats0), ens_num))

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

    while time_count_wind_waves <= forecast_end_time + np.timedelta64(1, 'h'):
        forecast_times_wind_waves.append(time_count_wind_waves)
        time_count_wind_waves = time_count_wind_waves + np.timedelta64(1, 'h')

    while time_count_ocean <= forecast_end_time + np.timedelta64(1, 'h'):
        forecast_times_ocean.append(time_count_ocean)
        time_count_ocean = time_count_ocean + np.timedelta64(1, 'h')

    while time_count_airT_sw_rad <= forecast_end_time + np.timedelta64(3, 'h'):
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
    points_ocean = np.array([ocean_lat.ravel(), ocean_lon.ravel()]).T

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

    wind_dir_errors = []
    curr_speed_errors = []

    for i in range(ens_num):
        wind_dir_error = truncated_normal(mean_wind_dir, std_dev_wind_dir, left_trunc_wind_dir, right_trunc_wind_dir)
        curr_speed_error = sample_scaled_error(mean_curr_speed, std_dev_curr_speed, k_curr_speed, scale_curr_speed, loc_curr_speed, right_trunc_curr_speed)
        wind_dir_errors.append(wind_dir_error)
        curr_speed_errors.append(curr_speed_error)

    for i in range(len(bergy_bit_growler_times) - 1):
        print('Forecasting bergy bits and growlers at ' + str(bergy_bit_growler_times[i]) + ' UTC...')
        bergy_bit_growler_time = bergy_bit_growler_times[i]
        before_idx = np.where(file_times_wind_waves <= bergy_bit_growler_time)[0][-1]

        try:
            after_idx = np.where(file_times_wind_waves > bergy_bit_growler_time)[0][0]
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
        wave_pd_file_before = date_only_wind_waves + 'T' + hour_utc_str_wind_waves + 'Z_MSC_GDWPS_MZWPER_Sfc_LatLon0.25_PT' + \
                              str(time_increments_wind_waves[before_idx]).zfill(3) + 'H.nc'
        wave_pd_file_after = date_only_wind_waves + 'T' + hour_utc_str_wind_waves + 'Z_MSC_GDWPS_MZWPER_Sfc_LatLon0.25_PT' + \
                             str(time_increments_wind_waves[after_idx]).zfill(3) + 'H.nc'

        forecast_time_wind_waves_before = forecast_times_wind_waves[before_idx]
        forecast_time_wind_waves_after = forecast_times_wind_waves[after_idx]

        before_idx = np.where(file_times_airT_sw_rad <= bergy_bit_growler_time)[0][-1]

        try:
            after_idx = np.where(file_times_airT_sw_rad > bergy_bit_growler_time)[0][0]
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

        before_idx = np.where(file_times_ocean <= bergy_bit_growler_time)[0][-1]

        try:
            after_idx = np.where(file_times_ocean > bergy_bit_growler_time)[0][0]
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

        if si_toggle:
            siconc_file_before = date_only_ocean + 'T' + hour_utc_str_ocean + 'Z_MSC_RIOPS_IICECONC_SFC_PS5km_P' + \
                                   str(time_increments_ocean[before_idx]).zfill(3) + '.nc'
            siconc_file_after = date_only_ocean + 'T' + hour_utc_str_ocean + 'Z_MSC_RIOPS_IICECONC_SFC_PS5km_P' + \
                                  str(time_increments_ocean[after_idx]).zfill(3) + '.nc'

        forecast_time_ocean_before = forecast_times_ocean[before_idx]
        forecast_time_ocean_after = forecast_times_ocean[after_idx]

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

        if si_toggle:
            fname = rootpath_to_metdata + 'RIOPS_ocean_forecast_files/' + dirname_ocean + '/' + siconc_file_before
            siconc_data_before = nc.Dataset(fname)
            siconc_before = np.squeeze(siconc_data_before.variables['iiceconc'][:])
            siconc_data_before.close()

            fname = rootpath_to_metdata + 'RIOPS_ocean_forecast_files/' + dirname_ocean + '/' + siconc_file_after
            siconc_data_after = nc.Dataset(fname)
            siconc_after = np.squeeze(siconc_data_after.variables['iiceconc'][:])
            siconc_data_after.close()

        f_u_wind_before = RegularGridInterpolator((lat_wind_waves, lon_wind_waves), u_wind_before, method='linear', bounds_error=True, fill_value=np.nan)
        f_u_wind_after = RegularGridInterpolator((lat_wind_waves, lon_wind_waves), u_wind_after, method='linear', bounds_error=True, fill_value=np.nan)

        f_v_wind_before = RegularGridInterpolator((lat_wind_waves, lon_wind_waves), v_wind_before, method='linear', bounds_error=True, fill_value=np.nan)
        f_v_wind_after = RegularGridInterpolator((lat_wind_waves, lon_wind_waves), v_wind_after, method='linear', bounds_error=True, fill_value=np.nan)

        f_Hs_before = RegularGridInterpolator((lat_wind_waves, lon_wind_waves), Hs_before, method='nearest', bounds_error=True, fill_value=np.nan)
        f_Hs_after = RegularGridInterpolator((lat_wind_waves, lon_wind_waves), Hs_after, method='nearest', bounds_error=True, fill_value=np.nan)

        f_wave_pd_before = RegularGridInterpolator((lat_wind_waves, lon_wind_waves), wave_pd_before, method='nearest', bounds_error=True, fill_value=np.nan)
        f_wave_pd_after = RegularGridInterpolator((lat_wind_waves, lon_wind_waves), wave_pd_after, method='nearest', bounds_error=True, fill_value=np.nan)

        f_airT_before = RegularGridInterpolator((lat_airT_sw_rad, lon_airT_sw_rad), airT_before, method='linear', bounds_error=True, fill_value=np.nan)
        f_airT_after = RegularGridInterpolator((lat_airT_sw_rad, lon_airT_sw_rad), airT_after, method='linear', bounds_error=True, fill_value=np.nan)

        f_solar_rad_before = RegularGridInterpolator((lat_airT_sw_rad, lon_airT_sw_rad), solar_rad_before, method='linear', bounds_error=True, fill_value=np.nan)
        f_solar_rad_after = RegularGridInterpolator((lat_airT_sw_rad, lon_airT_sw_rad), solar_rad_after, method='linear', bounds_error=True, fill_value=np.nan)

        solar_rad_hr_before = int(solar_rad_file_before[-6:-3])
        solar_rad_hr_after = int(solar_rad_file_after[-6:-3])

        t_new = (bergy_bit_growler_time - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')

        t1 = (forecast_time_wind_waves_before - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
        t2 = (forecast_time_wind_waves_after - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
        weight_wind_waves = (t_new - t1) / (t2 - t1)

        t1 = (forecast_time_ocean_before - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
        t2 = (forecast_time_ocean_after - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
        weight_ocean = (t_new - t1) / (t2 - t1)

        t1 = (forecast_time_airT_sw_rad_before - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
        t2 = (forecast_time_airT_sw_rad_after - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
        weight_airT_sw_rad = (t_new - t1) / (t2 - t1)

        for k in range(len(iceberg_lats0)):
            print('Forecasting bergy bits and growlers for iceberg ' + str(iceberg_ids[k]) + '...')

            for m in range(ens_num):
                print('Forecasting ensemble track number ' + str(m + 1) + '...')
                bergy_bit_lat = bergy_bit_lats[i, k, m]
                bergy_bit_lon = bergy_bit_lons[i, k, m]
                bergy_bit_u = bergy_bit_us[i, k, m]
                bergy_bit_v = bergy_bit_vs[i, k, m]
                bergy_bit_length = bergy_bit_lengths[i, k, m]
                bergy_bit_draft = bergy_bit_drafts[i, k, m]
                bergy_bit_mass = bergy_bit_masses[i, k, m]
                bergy_bit_grounded_status = bergy_bit_grounded_statuses[i, k, m]

                growler_lat = growler_lats[i, k, m]
                growler_lon = growler_lons[i, k, m]
                growler_u = growler_us[i, k, m]
                growler_v = growler_vs[i, k, m]
                growler_length = growler_lengths[i, k, m]
                growler_draft = growler_drafts[i, k, m]
                growler_mass = growler_masses[i, k, m]
                growler_grounded_status = growler_grounded_statuses[i, k, m]

                u_wind_before_bb = float(f_u_wind_before([bergy_bit_lat, bergy_bit_lon + 360.]))
                u_wind_after_bb = float(f_u_wind_after([bergy_bit_lat, bergy_bit_lon + 360.]))
                u_wind_bb = u_wind_before_bb + weight_wind_waves * (u_wind_after_bb - u_wind_before_bb)

                v_wind_before_bb = float(f_v_wind_before([bergy_bit_lat, bergy_bit_lon + 360.]))
                v_wind_after_bb = float(f_v_wind_after([bergy_bit_lat, bergy_bit_lon + 360.]))
                v_wind_bb = v_wind_before_bb + weight_wind_waves * (v_wind_after_bb - v_wind_before_bb)

                Hs_before_bb = float(f_Hs_before([bergy_bit_lat, bergy_bit_lon + 360.]))
                Hs_after_bb = float(f_Hs_after([bergy_bit_lat, bergy_bit_lon + 360.]))
                Hs_bb = Hs_before_bb + weight_wind_waves * (Hs_after_bb - Hs_before_bb)

                u_wind_before_growler = float(f_u_wind_before([growler_lat, growler_lon + 360.]))
                u_wind_after_growler = float(f_u_wind_after([growler_lat, growler_lon + 360.]))
                u_wind_growler = u_wind_before_growler + weight_wind_waves * (u_wind_after_growler - u_wind_before_growler)

                v_wind_before_growler = float(f_v_wind_before([growler_lat, growler_lon + 360.]))
                v_wind_after_growler = float(f_v_wind_after([growler_lat, growler_lon + 360.]))
                v_wind_growler = v_wind_before_growler + weight_wind_waves * (v_wind_after_growler - v_wind_before_growler)

                Hs_before_growler = float(f_Hs_before([growler_lat, growler_lon + 360.]))
                Hs_after_growler = float(f_Hs_after([growler_lat, growler_lon + 360.]))
                Hs_growler = Hs_before_growler + weight_wind_waves * (Hs_after_growler - Hs_before_growler)

                wave_pd_before_bb = float(f_wave_pd_before([bergy_bit_lat, bergy_bit_lon + 360.]))
                wave_pd_after_bb = float(f_wave_pd_after([bergy_bit_lat, bergy_bit_lon + 360.]))
                wave_pd_bb = wave_pd_before_bb + weight_wind_waves * (wave_pd_after_bb - wave_pd_before_bb)

                wave_pd_before_growler = float(f_wave_pd_before([growler_lat, growler_lon + 360.]))
                wave_pd_after_growler = float(f_wave_pd_after([growler_lat, growler_lon + 360.]))
                wave_pd_growler = wave_pd_before_growler + weight_wind_waves * (wave_pd_after_growler - wave_pd_before_growler)

                airT_before_bb = float(f_airT_before([bergy_bit_lat, bergy_bit_lon]))
                airT_after_bb = float(f_airT_after([bergy_bit_lat, bergy_bit_lon]))
                airT_bb = airT_before_bb + weight_airT_sw_rad * (airT_after_bb - airT_before_bb)

                airT_before_growler = float(f_airT_before([growler_lat, growler_lon]))
                airT_after_growler = float(f_airT_after([growler_lat, growler_lon]))
                airT_growler = airT_before_growler + weight_airT_sw_rad * (airT_after_growler - airT_before_growler)

                if solar_rad_hr_before == 0:
                    solar_rad_before_bb = float(f_solar_rad_before([bergy_bit_lat, bergy_bit_lon]))
                    solar_rad_before_growler = float(f_solar_rad_before([growler_lat, growler_lon]))
                else:
                    solar_rad_before_bb = float(f_solar_rad_before([bergy_bit_lat, bergy_bit_lon]) / (solar_rad_hr_before * 3600.))
                    solar_rad_before_growler = float(f_solar_rad_before([growler_lat, growler_lon]) / (solar_rad_hr_before * 3600.))

                if solar_rad_hr_after == 0:
                    solar_rad_after_bb = float(f_solar_rad_after([bergy_bit_lat, bergy_bit_lon]))
                    solar_rad_after_growler = float(f_solar_rad_after([growler_lat, growler_lon]))
                else:
                    solar_rad_after_bb = float(f_solar_rad_after([bergy_bit_lat, bergy_bit_lon]) / (solar_rad_hr_after * 3600.))
                    solar_rad_after_growler = float(f_solar_rad_after([growler_lat, growler_lon]) / (solar_rad_hr_after * 3600.))

                if solar_rad_before_bb < 0:
                    solar_rad_before_bb = -solar_rad_before_bb

                if solar_rad_before_growler < 0:
                    solar_rad_before_growler = -solar_rad_before_growler

                if solar_rad_after_bb < 0:
                    solar_rad_after_bb = -solar_rad_after_bb

                if solar_rad_after_growler < 0:
                    solar_rad_after_growler = -solar_rad_after_growler

                if solar_rad_before_bb < 0 or np.isnan(solar_rad_before_bb):
                    solar_rad_before_bb = 0.

                if solar_rad_before_growler < 0 or np.isnan(solar_rad_before_growler):
                    solar_rad_before_growler = 0.

                if solar_rad_after_bb < 0 or np.isnan(solar_rad_after_bb):
                    solar_rad_after_bb = 0.

                if solar_rad_after_growler < 0 or np.isnan(solar_rad_after_growler):
                    solar_rad_after_growler = 0.

                solar_rad_bb = solar_rad_before_bb + weight_airT_sw_rad * (solar_rad_after_bb - solar_rad_before_bb)
                solar_rad_growler = solar_rad_before_growler + weight_airT_sw_rad * (solar_rad_after_growler - solar_rad_before_growler)

                if bergy_bit_draft > 0:
                    try:
                        loc_depth = np.argwhere(ocean_depth <= bergy_bit_draft).flatten()

                        if loc_depth[-1] + 1 < len(ocean_depth):
                            loc_depth = np.append(loc_depth, loc_depth[-1] + 1)

                        depth_curr_bb = ocean_depth[loc_depth]
                        depth_curr_bb = depth_curr_bb.tolist()
                        depth_curr_bb_interp = np.arange(0., bergy_bit_draft, 0.001)
                    except:
                        depth_curr_bb = list(ocean_depth[:1])
                        depth_curr_bb_interp = np.arange(0., bergy_bit_draft, 0.001)
                else:
                    depth_curr_bb = list(ocean_depth[:1])
                    depth_curr_bb_interp = np.arange(0., ocean_depth[-1], 0.001)

                if growler_draft > 0:
                    try:
                        loc_depth = np.argwhere(ocean_depth <= growler_draft).flatten()

                        if loc_depth[-1] + 1 < len(ocean_depth):
                            loc_depth = np.append(loc_depth, loc_depth[-1] + 1)

                        depth_curr_growler = ocean_depth[loc_depth]
                        depth_curr_growler = depth_curr_growler.tolist()
                        depth_curr_growler_interp = np.arange(0., growler_draft, 0.001)
                    except:
                        depth_curr_growler = list(ocean_depth[:1])
                        depth_curr_growler_interp = np.arange(0., growler_draft, 0.001)
                else:
                    depth_curr_growler = list(ocean_depth[:1])
                    depth_curr_growler_interp = np.arange(0., ocean_depth[-1], 0.001)

                dist = np.sqrt((ocean_lat - bergy_bit_lat) ** 2 + (ocean_lon - (bergy_bit_lon + 360.)) ** 2)
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

                u_curr_before_bb_depth_list = []
                u_curr_after_bb_depth_list = []
                v_curr_before_bb_depth_list = []
                v_curr_after_bb_depth_list = []
                salinity_before_bb_depth_list = []
                salinity_after_bb_depth_list = []
                pot_temp_before_bb_depth_list = []
                pot_temp_after_bb_depth_list = []

                u_curr_before_growler_depth_list = []
                u_curr_after_growler_depth_list = []
                v_curr_before_growler_depth_list = []
                v_curr_after_growler_depth_list = []
                salinity_before_growler_depth_list = []
                salinity_after_growler_depth_list = []
                pot_temp_before_growler_depth_list = []
                pot_temp_after_growler_depth_list = []

                for n in range(len(depth_curr_bb)):
                    u_curr_before_select = np.squeeze(u_curr_before[n, ocean_lat_ind, ocean_lon_ind])
                    u_curr_after_select = np.squeeze(u_curr_after[n, ocean_lat_ind, ocean_lon_ind])
                    u_curr_before_temp = griddata(points_ocean, u_curr_before_select.ravel(),(bergy_bit_lat, bergy_bit_lon + 360.), method='linear')
                    u_curr_after_temp = griddata(points_ocean, u_curr_after_select.ravel(),(bergy_bit_lat, bergy_bit_lon + 360.), method='linear')
                    u_curr_before_bb_depth_list.append(u_curr_before_temp)
                    u_curr_after_bb_depth_list.append(u_curr_after_temp)
                    v_curr_before_select = np.squeeze(v_curr_before[n, ocean_lat_ind, ocean_lon_ind])
                    v_curr_after_select = np.squeeze(v_curr_after[n, ocean_lat_ind, ocean_lon_ind])
                    v_curr_before_temp = griddata(points_ocean, v_curr_before_select.ravel(),(bergy_bit_lat, bergy_bit_lon + 360.), method='linear')
                    v_curr_after_temp = griddata(points_ocean, v_curr_after_select.ravel(),(bergy_bit_lat, bergy_bit_lon + 360.), method='linear')
                    v_curr_before_bb_depth_list.append(v_curr_before_temp)
                    v_curr_after_bb_depth_list.append(v_curr_after_temp)
                    salinity_before_select = np.squeeze(salinity_before[n, ocean_lat_ind, ocean_lon_ind])
                    salinity_after_select = np.squeeze(salinity_after[n, ocean_lat_ind, ocean_lon_ind])
                    salinity_before_temp = griddata(points_ocean, salinity_before_select.ravel(),(bergy_bit_lat, bergy_bit_lon + 360.), method='linear')
                    salinity_after_temp = griddata(points_ocean, salinity_after_select.ravel(),(bergy_bit_lat, bergy_bit_lon + 360.), method='linear')
                    salinity_before_bb_depth_list.append(salinity_before_temp)
                    salinity_after_bb_depth_list.append(salinity_after_temp)
                    pot_temp_before_select = np.squeeze(pot_temp_before[n, ocean_lat_ind, ocean_lon_ind])
                    pot_temp_after_select = np.squeeze(pot_temp_after[n, ocean_lat_ind, ocean_lon_ind])
                    pot_temp_before_temp = griddata(points_ocean, pot_temp_before_select.ravel(),(bergy_bit_lat, bergy_bit_lon + 360.), method='linear')
                    pot_temp_after_temp = griddata(points_ocean, pot_temp_after_select.ravel(),(bergy_bit_lat, bergy_bit_lon + 360.), method='linear')
                    pot_temp_before_bb_depth_list.append(pot_temp_before_temp)
                    pot_temp_after_bb_depth_list.append(pot_temp_after_temp)

                dist = np.sqrt((ocean_lat - growler_lat) ** 2 + (ocean_lon - (growler_lon + 360.)) ** 2)
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

                for n in range(len(depth_curr_growler)):
                    u_curr_before_select = np.squeeze(u_curr_before[n, ocean_lat_ind, ocean_lon_ind])
                    u_curr_after_select = np.squeeze(u_curr_after[n, ocean_lat_ind, ocean_lon_ind])
                    u_curr_before_temp = griddata(points_ocean, u_curr_before_select.ravel(),(growler_lat, growler_lon + 360.), method='linear')
                    u_curr_after_temp = griddata(points_ocean, u_curr_after_select.ravel(),(growler_lat, growler_lon + 360.), method='linear')
                    u_curr_before_growler_depth_list.append(u_curr_before_temp)
                    u_curr_after_growler_depth_list.append(u_curr_after_temp)
                    v_curr_before_select = np.squeeze(v_curr_before[n, ocean_lat_ind, ocean_lon_ind])
                    v_curr_after_select = np.squeeze(v_curr_after[n, ocean_lat_ind, ocean_lon_ind])
                    v_curr_before_temp = griddata(points_ocean, v_curr_before_select.ravel(),(growler_lat, growler_lon + 360.), method='linear')
                    v_curr_after_temp = griddata(points_ocean, v_curr_after_select.ravel(),(growler_lat, growler_lon + 360.), method='linear')
                    v_curr_before_growler_depth_list.append(v_curr_before_temp)
                    v_curr_after_growler_depth_list.append(v_curr_after_temp)
                    salinity_before_select = np.squeeze(salinity_before[n, ocean_lat_ind, ocean_lon_ind])
                    salinity_after_select = np.squeeze(salinity_after[n, ocean_lat_ind, ocean_lon_ind])
                    salinity_before_temp = griddata(points_ocean, salinity_before_select.ravel(),(growler_lat, growler_lon + 360.), method='linear')
                    salinity_after_temp = griddata(points_ocean, salinity_after_select.ravel(),(growler_lat, growler_lon + 360.), method='linear')
                    salinity_before_growler_depth_list.append(salinity_before_temp)
                    salinity_after_growler_depth_list.append(salinity_after_temp)
                    pot_temp_before_select = np.squeeze(pot_temp_before[n, ocean_lat_ind, ocean_lon_ind])
                    pot_temp_after_select = np.squeeze(pot_temp_after[n, ocean_lat_ind, ocean_lon_ind])
                    pot_temp_before_temp = griddata(points_ocean, pot_temp_before_select.ravel(),(growler_lat, growler_lon + 360.), method='linear')
                    pot_temp_after_temp = griddata(points_ocean, pot_temp_after_select.ravel(),(growler_lat, growler_lon + 360.), method='linear')
                    pot_temp_before_growler_depth_list.append(pot_temp_before_temp)
                    pot_temp_after_growler_depth_list.append(pot_temp_after_temp)

                u_curr_before_bb_depth_list = [float(val) for val in u_curr_before_bb_depth_list]
                u_curr_after_bb_depth_list = [float(val) for val in u_curr_after_bb_depth_list]
                interp_func = interp1d(depth_curr_bb, u_curr_before_bb_depth_list, kind='linear', fill_value='extrapolate')
                u_curr_before_bb_depth_list = interp_func(depth_curr_bb_interp)
                interp_func = interp1d(depth_curr_bb, u_curr_after_bb_depth_list, kind='linear', fill_value='extrapolate')
                u_curr_after_bb_depth_list = interp_func(depth_curr_bb_interp)
                u_curr_before_bb = np.nanmean(u_curr_before_bb_depth_list)
                u_curr_after_bb = np.nanmean(u_curr_after_bb_depth_list)

                v_curr_before_bb_depth_list = [float(val) for val in v_curr_before_bb_depth_list]
                v_curr_after_bb_depth_list = [float(val) for val in v_curr_after_bb_depth_list]
                interp_func = interp1d(depth_curr_bb, v_curr_before_bb_depth_list, kind='linear', fill_value='extrapolate')
                v_curr_before_bb_depth_list = interp_func(depth_curr_bb_interp)
                interp_func = interp1d(depth_curr_bb, v_curr_after_bb_depth_list, kind='linear', fill_value='extrapolate')
                v_curr_after_bb_depth_list = interp_func(depth_curr_bb_interp)
                v_curr_before_bb = np.nanmean(v_curr_before_bb_depth_list)
                v_curr_after_bb = np.nanmean(v_curr_after_bb_depth_list)

                u_curr_before_growler_depth_list = [float(val) for val in u_curr_before_growler_depth_list]
                u_curr_after_growler_depth_list = [float(val) for val in u_curr_after_growler_depth_list]
                interp_func = interp1d(depth_curr_growler, u_curr_before_growler_depth_list, kind='linear', fill_value='extrapolate')
                u_curr_before_growler_depth_list = interp_func(depth_curr_growler_interp)
                interp_func = interp1d(depth_curr_growler, u_curr_after_growler_depth_list, kind='linear', fill_value='extrapolate')
                u_curr_after_growler_depth_list = interp_func(depth_curr_growler_interp)
                u_curr_before_growler = np.nanmean(u_curr_before_growler_depth_list)
                u_curr_after_growler = np.nanmean(u_curr_after_growler_depth_list)

                v_curr_before_growler_depth_list = [float(val) for val in v_curr_before_growler_depth_list]
                v_curr_after_growler_depth_list = [float(val) for val in v_curr_after_growler_depth_list]
                interp_func = interp1d(depth_curr_growler, v_curr_before_growler_depth_list, kind='linear', fill_value='extrapolate')
                v_curr_before_growler_depth_list = interp_func(depth_curr_growler_interp)
                interp_func = interp1d(depth_curr_growler, v_curr_after_growler_depth_list, kind='linear', fill_value='extrapolate')
                v_curr_after_growler_depth_list = interp_func(depth_curr_growler_interp)
                v_curr_before_growler = np.nanmean(v_curr_before_growler_depth_list)
                v_curr_after_growler = np.nanmean(v_curr_after_growler_depth_list)

                salinity_before_bb_depth_list = [float(val) for val in salinity_before_bb_depth_list]
                salinity_after_bb_depth_list = [float(val) for val in salinity_after_bb_depth_list]
                interp_func = interp1d(depth_curr_bb, salinity_before_bb_depth_list, kind='linear', fill_value='extrapolate')
                salinity_before_bb_depth_list = interp_func(depth_curr_bb_interp)
                interp_func = interp1d(depth_curr_bb, salinity_after_bb_depth_list, kind='linear', fill_value='extrapolate')
                salinity_after_bb_depth_list = interp_func(depth_curr_bb_interp)

                pot_temp_before_bb_depth_list = [float(val) for val in pot_temp_before_bb_depth_list]
                pot_temp_after_bb_depth_list = [float(val) for val in pot_temp_after_bb_depth_list]
                interp_func = interp1d(depth_curr_bb, pot_temp_before_bb_depth_list, kind='linear', fill_value='extrapolate')
                pot_temp_before_bb_depth_list = interp_func(depth_curr_bb_interp)
                interp_func = interp1d(depth_curr_bb, pot_temp_after_bb_depth_list, kind='linear', fill_value='extrapolate')
                pot_temp_after_bb_depth_list = interp_func(depth_curr_bb_interp)

                salinity_before_growler_depth_list = [float(val) for val in salinity_before_growler_depth_list]
                salinity_after_growler_depth_list = [float(val) for val in salinity_after_growler_depth_list]
                interp_func = interp1d(depth_curr_growler, salinity_before_growler_depth_list, kind='linear', fill_value='extrapolate')
                salinity_before_growler_depth_list = interp_func(depth_curr_growler_interp)
                interp_func = interp1d(depth_curr_growler, salinity_after_growler_depth_list, kind='linear', fill_value='extrapolate')
                salinity_after_growler_depth_list = interp_func(depth_curr_growler_interp)

                pot_temp_before_growler_depth_list = [float(val) for val in pot_temp_before_growler_depth_list]
                pot_temp_after_growler_depth_list = [float(val) for val in pot_temp_after_growler_depth_list]
                interp_func = interp1d(depth_curr_growler, pot_temp_before_growler_depth_list, kind='linear', fill_value='extrapolate')
                pot_temp_before_growler_depth_list = interp_func(depth_curr_growler_interp)
                interp_func = interp1d(depth_curr_growler, pot_temp_after_growler_depth_list, kind='linear', fill_value='extrapolate')
                pot_temp_after_growler_depth_list = interp_func(depth_curr_growler_interp)

                if si_toggle:
                    siconc_before_bb = griddata(points_ocean, siconc_before.ravel(),(bergy_bit_lat, bergy_bit_lon + 360.), method='linear')
                    siconc_after_bb = griddata(points_ocean, siconc_after.ravel(),(bergy_bit_lat, bergy_bit_lon + 360.), method='linear')
                    siconc_bb = siconc_before_bb + weight_ocean * (siconc_after_bb - siconc_before_bb)
                    siconc_before_growler = griddata(points_ocean, siconc_before.ravel(),(growler_lat, growler_lon + 360.), method='linear')
                    siconc_after_growler = griddata(points_ocean, siconc_after.ravel(),(growler_lat, growler_lon + 360.), method='linear')
                    siconc_growler = siconc_before_growler + weight_ocean * (siconc_after_growler - siconc_before_growler)
                else:
                    siconc_bb = 0.
                    siconc_growler = 0.

                u_curr_bb = u_curr_before_bb + weight_ocean * (u_curr_after_bb - u_curr_before_bb)
                v_curr_bb = v_curr_before_bb + weight_ocean * (v_curr_after_bb - v_curr_before_bb)

                u_curr_growler = u_curr_before_growler + weight_ocean * (u_curr_after_growler - u_curr_before_growler)
                v_curr_growler = v_curr_before_growler + weight_ocean * (v_curr_after_growler - v_curr_before_growler)

                salinity_bb_list = []
                pot_temp_bb_list = []

                salinity_growler_list = []
                pot_temp_growler_list = []

                for n in range(len(depth_curr_bb_interp)):
                    salinity_bb = salinity_before_bb_depth_list[n] + weight_ocean * (salinity_after_bb_depth_list[n] - salinity_before_bb_depth_list[n])
                    pot_temp_bb = pot_temp_before_bb_depth_list[n] + weight_ocean * (pot_temp_after_bb_depth_list[n] - pot_temp_before_bb_depth_list[n])
                    salinity_bb_list.append(salinity_bb)
                    pot_temp_bb_list.append(pot_temp_bb)

                for n in range(len(depth_curr_growler_interp)):
                    salinity_growler = salinity_before_growler_depth_list[n] + weight_ocean * (salinity_after_growler_depth_list[n] -
                                                                                               salinity_before_growler_depth_list[n])
                    pot_temp_growler = pot_temp_before_growler_depth_list[n] + weight_ocean * (pot_temp_after_growler_depth_list[n] -
                                                                                               pot_temp_before_growler_depth_list[n])
                    salinity_growler_list.append(salinity_growler)
                    pot_temp_growler_list.append(pot_temp_growler)

                wind_speed_bb = np.sqrt(u_wind_bb ** 2 + v_wind_bb ** 2)
                wind_speed_growler = np.sqrt(u_wind_growler ** 2 + v_wind_growler ** 2)
                wind_dir_bb = 90. - np.rad2deg(np.arctan2(v_wind_bb, u_wind_bb))
                wind_dir_growler = 90. - np.rad2deg(np.arctan2(v_wind_growler, u_wind_growler))
                curr_speed_bb = np.sqrt(u_curr_bb ** 2 + v_curr_bb ** 2)
                curr_speed_growler = np.sqrt(u_curr_growler ** 2 + v_curr_growler ** 2)
                curr_dir_bb = 90. - np.rad2deg(np.arctan2(v_curr_bb, u_curr_bb))
                curr_dir_growler = 90. - np.rad2deg(np.arctan2(v_curr_growler, u_curr_growler))

                if wind_dir_bb < 0:
                    wind_dir_bb += 360.

                if wind_dir_growler < 0:
                    wind_dir_growler += 360.

                if curr_dir_bb < 0:
                    curr_dir_bb += 360.

                if curr_dir_growler < 0:
                    curr_dir_growler += 360.

                left_trunc_Hs_bb = -Hs_bb
                Hs_bb_error = sample_hs_error(mean_Hs, std_dev_Hs, left_trunc_Hs_bb)
                left_trunc_Hs_growler = -Hs_growler
                Hs_growler_error = sample_hs_error(mean_Hs, std_dev_Hs, left_trunc_Hs_growler)
                Hs_bb += Hs_bb_error
                Hs_growler += Hs_growler_error

                if Hs_bb < 0:
                    Hs_bb = 0.

                if Hs_growler < 0:
                    Hs_growler = 0.

                left_trunc_wind_speed_bb = -wind_speed_bb
                right_trunc_wind_speed_bb = wind_speed_bb
                wind_speed_error_bb = truncated_normal(mean_wind_speed, std_dev_wind_speed, left_trunc_wind_speed_bb, right_trunc_wind_speed_bb)
                left_trunc_wind_speed_growler = -wind_speed_growler
                right_trunc_wind_speed_growler = wind_speed_growler
                wind_speed_error_growler = truncated_normal(mean_wind_speed, std_dev_wind_speed, left_trunc_wind_speed_growler, right_trunc_wind_speed_growler)
                wind_dir_error = wind_dir_errors[m]
                wind_speed_bb += wind_speed_error_bb
                wind_speed_growler += wind_speed_error_growler
                wind_dir_bb += wind_dir_error
                wind_dir_growler += wind_dir_error

                if wind_speed_bb < 0:
                    wind_speed_bb = 0.

                if wind_speed_growler < 0:
                    wind_speed_growler = 0.

                if wind_dir_bb < 0:
                    wind_dir_bb += 360.

                if wind_dir_growler < 0:
                    wind_dir_growler += 360.

                u_wind_bb = wind_speed_bb * np.sin(np.deg2rad(wind_dir_bb))
                v_wind_bb = wind_speed_bb * np.cos(np.deg2rad(wind_dir_bb))
                u_wind_growler = wind_speed_growler * np.sin(np.deg2rad(wind_dir_growler))
                v_wind_growler = wind_speed_growler * np.cos(np.deg2rad(wind_dir_growler))

                curr_speed_error = curr_speed_errors[m]
                std_dev_curr_dir_bb = 5. / (curr_speed_bb ** 2 + 0.05)

                try:
                    curr_dir_error_bb = truncated_normal(mean_curr_dir, std_dev_curr_dir_bb, left_trunc_curr_dir, right_trunc_curr_dir)
                except:
                    curr_dir_error_bb = truncated_normal(mean_curr_dir, 100., left_trunc_curr_dir, right_trunc_curr_dir)

                u_curr_bb_depth_list = []
                v_curr_bb_depth_list = []

                for n in range(len(depth_curr_bb_interp)):
                    u_curr_bb_temp = u_curr_before_bb_depth_list[n] + weight_ocean * (u_curr_after_bb_depth_list[n] - u_curr_before_bb_depth_list[n])
                    v_curr_bb_temp = v_curr_before_bb_depth_list[n] + weight_ocean * (v_curr_after_bb_depth_list[n] - v_curr_before_bb_depth_list[n])
                    curr_speed_bb_temp = np.sqrt(u_curr_bb_temp ** 2 + v_curr_bb_temp ** 2)
                    curr_speed_bb_temp += curr_speed_error

                    if curr_speed_bb_temp < 0:
                        curr_speed_bb_temp = 0.

                    curr_dir_bb_temp = 90. - np.rad2deg(np.arctan2(v_curr_bb_temp, u_curr_bb_temp))

                    if curr_dir_bb_temp < 0:
                        curr_dir_bb_temp += 360.

                    curr_dir_bb_temp += curr_dir_error_bb

                    if curr_dir_bb_temp < 0:
                        curr_dir_bb_temp += 360.

                    u_curr_bb_temp = curr_speed_bb_temp * np.sin(np.deg2rad(curr_dir_bb_temp))
                    v_curr_bb_temp = curr_speed_bb_temp * np.cos(np.deg2rad(curr_dir_bb_temp))
                    u_curr_bb_depth_list.append(u_curr_bb_temp)
                    v_curr_bb_depth_list.append(v_curr_bb_temp)

                u_curr_bb = np.nanmean(u_curr_bb_depth_list)
                v_curr_bb = np.nanmean(v_curr_bb_depth_list)

                std_dev_curr_dir_growler = 5. / (curr_speed_growler ** 2 + 0.05)

                try:
                    curr_dir_error_growler = truncated_normal(mean_curr_dir, std_dev_curr_dir_growler, left_trunc_curr_dir, right_trunc_curr_dir)
                except:
                    curr_dir_error_growler = truncated_normal(mean_curr_dir, 100., left_trunc_curr_dir, right_trunc_curr_dir)

                u_curr_growler_depth_list = []
                v_curr_growler_depth_list = []

                for n in range(len(depth_curr_growler_interp)):
                    u_curr_growler_temp = u_curr_before_growler_depth_list[n] + weight_ocean * \
                        (u_curr_after_growler_depth_list[n] - u_curr_before_growler_depth_list[n])
                    v_curr_growler_temp = v_curr_before_growler_depth_list[n] + weight_ocean * \
                        (v_curr_after_growler_depth_list[n] - v_curr_before_growler_depth_list[n])
                    curr_speed_growler_temp = np.sqrt(u_curr_growler_temp ** 2 + v_curr_growler_temp ** 2)
                    curr_speed_growler_temp += curr_speed_error

                    if curr_speed_growler_temp < 0:
                        curr_speed_growler_temp = 0.

                    curr_dir_growler_temp = 90. - np.rad2deg(np.arctan2(v_curr_growler_temp, u_curr_growler_temp))

                    if curr_dir_growler_temp < 0:
                        curr_dir_growler_temp += 360.

                    curr_dir_growler_temp += curr_dir_error_growler

                    if curr_dir_growler_temp < 0:
                        curr_dir_growler_temp += 360.

                    u_curr_growler_temp = curr_speed_growler_temp * np.sin(np.deg2rad(curr_dir_growler_temp))
                    v_curr_growler_temp = curr_speed_growler_temp * np.cos(np.deg2rad(curr_dir_growler_temp))
                    u_curr_growler_depth_list.append(u_curr_growler_temp)
                    v_curr_growler_depth_list.append(v_curr_growler_temp)

                u_curr_growler = np.nanmean(u_curr_growler_depth_list)
                v_curr_growler = np.nanmean(v_curr_growler_depth_list)

                if bergy_bit_mass > 0:
                    new_bb_length, new_bb_draft, new_bb_sail, new_bb_mass = iceberg_det(bergy_bit_length, bergy_bit_mass, bergy_bit_lat, solar_rad_bb,
                                                                                                            ice_albedo, Lf_ice, rho_ice, pot_temp_bb_list,
                                                                                                            salinity_bb_list, depth_curr_bb_interp, airT_bb,
                                                                                                            u_curr_bb, v_curr_bb, u_wind_bb, v_wind_bb,
                                                                                                            bergy_bit_u, bergy_bit_v, Hs_bb, wave_pd_bb,
                                                                                                            bergy_bit_growler_times_dt[i], siconc_bb)
                    bergy_bit_bathy_depth = bathy_interp([[bergy_bit_lat, bergy_bit_lon]])[0]

                    if bergy_bit_bathy_depth <= new_bb_draft:
                        bergy_bit_grounded_statuses[i, k, m] = 1
                        bergy_bit_grounded_status = 1
                        bergy_bit_us[i, k, m] = 0.
                        bergy_bit_vs[i, k, m] = 0.
                    else:
                        bergy_bit_grounded_statuses[i, k, m] = 0
                        bergy_bit_grounded_status = 0
                else:
                    new_bb_length = 0.
                    new_bb_draft = 0.
                    new_bb_sail = 0.
                    new_bb_mass = 0.

                bergy_bit_lengths[i + 1, k, m] = new_bb_length
                bergy_bit_drafts[i + 1, k, m] = new_bb_draft
                bergy_bit_sails[i + 1, k, m] = new_bb_sail
                bergy_bit_masses[i + 1, k, m] = new_bb_mass

                if new_bb_mass > 0:
                    bergy_bit_u_end, bergy_bit_v_end = iceberg_vel(u_wind_bb, v_wind_bb, u_curr_bb, v_curr_bb, alpha_cor)
                else:
                    bergy_bit_u_end = 0.
                    bergy_bit_v_end = 0.

                final_bb_speed = np.sqrt(bergy_bit_u_end ** 2 + bergy_bit_v_end ** 2)

                if final_bb_speed >= 2:
                    bergy_bit_u_end = bergy_bit_u
                    bergy_bit_v_end = bergy_bit_v

                if bergy_bit_grounded_status == 1:
                    bergy_bit_u_end = 0.
                    bergy_bit_v_end = 0.

                bergy_bit_x = np.nanmean([bergy_bit_u, bergy_bit_u_end]) * bergy_bit_growler_times_dt[i]
                bergy_bit_y = np.nanmean([bergy_bit_v, bergy_bit_v_end]) * bergy_bit_growler_times_dt[i]
                bergy_bit_dist = np.sqrt(bergy_bit_x ** 2 + bergy_bit_y ** 2)
                bergy_bit_course = 90. - np.rad2deg(np.arctan2(bergy_bit_y, bergy_bit_x))

                if bergy_bit_course < 0:
                    bergy_bit_course += 360.

                bergy_bit_lat2, bergy_bit_lon2 = dist_course(Re, bergy_bit_lat, bergy_bit_lon, bergy_bit_dist, bergy_bit_course)
                bergy_bit_us[i + 1, k, m] = bergy_bit_u_end
                bergy_bit_vs[i + 1, k, m] = bergy_bit_v_end
                bergy_bit_lats[i + 1, k, m] = bergy_bit_lat2
                bergy_bit_lons[i + 1, k, m] = bergy_bit_lon2
                bergy_bit_bathy_depth = bathy_interp([[bergy_bit_lat2, bergy_bit_lon2]])[0]

                if bergy_bit_bathy_depth <= new_bb_draft:
                    bergy_bit_grounded_statuses[i + 1, k, m] = 1
                    bergy_bit_us[i + 1, k, m] = 0.
                    bergy_bit_vs[i + 1, k, m] = 0.
                else:
                    bergy_bit_grounded_statuses[i + 1, k, m] = 0

                if growler_mass > 0:
                    new_growler_length, new_growler_draft, new_growler_sail, new_growler_mass = iceberg_det(growler_length, growler_mass, growler_lat,
                                                                                                            solar_rad_growler, ice_albedo, Lf_ice,
                                                                                                            rho_ice, pot_temp_growler_list,
                                                                                                            salinity_growler_list,
                                                                                                            depth_curr_growler_interp, airT_growler,
                                                                                                            u_curr_growler, v_curr_growler,
                                                                                                            u_wind_growler, v_wind_growler,
                                                                                                            growler_u, growler_v, Hs_growler, wave_pd_growler,
                                                                                                            bergy_bit_growler_times_dt[i], siconc_growler)
                    growler_bathy_depth = bathy_interp([[growler_lat, growler_lon]])[0]

                    if growler_bathy_depth <= new_growler_draft:
                        growler_grounded_statuses[i, k, m] = 1
                        growler_grounded_status = 1
                        growler_us[i, k, m] = 0.
                        growler_vs[i, k, m] = 0.
                    else:
                        growler_grounded_statuses[i, k, m] = 0
                        growler_grounded_status = 0
                else:
                    new_growler_length = 0.
                    new_growler_draft = 0.
                    new_growler_sail = 0.
                    new_growler_mass = 0.

                growler_lengths[i + 1, k, m] = new_growler_length
                growler_drafts[i + 1, k, m] = new_growler_draft
                growler_sails[i + 1, k, m] = new_growler_sail
                growler_masses[i + 1, k, m] = new_growler_mass

                if new_growler_mass > 0:
                    growler_u_end, growler_v_end = iceberg_vel(u_wind_growler, v_wind_growler, u_curr_growler, v_curr_growler, alpha_cor)
                else:
                    growler_u_end = 0.
                    growler_v_end = 0.

                final_growler_speed = np.sqrt(growler_u_end ** 2 + growler_v_end ** 2)

                if final_growler_speed >= 2:
                    growler_u_end = growler_u
                    growler_v_end = growler_v

                if growler_grounded_status == 1:
                    growler_u_end = 0.
                    growler_v_end = 0.

                growler_x = np.nanmean([growler_u, growler_u_end]) * bergy_bit_growler_times_dt[i]
                growler_y = np.nanmean([growler_v, growler_v_end]) * bergy_bit_growler_times_dt[i]
                growler_dist = np.sqrt(growler_x ** 2 + growler_y ** 2)
                growler_course = 90. - np.rad2deg(np.arctan2(growler_y, growler_x))

                if growler_course < 0:
                    growler_course += 360.

                growler_lat2, growler_lon2 = dist_course(Re, growler_lat, growler_lon, growler_dist, growler_course)
                growler_us[i + 1, k, m] = growler_u_end
                growler_vs[i + 1, k, m] = growler_v_end
                growler_lats[i + 1, k, m] = growler_lat2
                growler_lons[i + 1, k, m] = growler_lon2
                growler_bathy_depth = bathy_interp([[growler_lat2, growler_lon2]])[0]

                if growler_bathy_depth <= new_growler_draft:
                    growler_grounded_statuses[i + 1, k, m] = 1
                    growler_us[i + 1, k, m] = 0.
                    growler_vs[i + 1, k, m] = 0.
                else:
                    growler_grounded_statuses[i + 1, k, m] = 0

    bergy_bit_length_ranges = [(0, 5), (5, 10), (10, 15)]
    growler_length_ranges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
    bergy_bit_bounds = calculate_outer_boundaries(bergy_bit_lengths, bergy_bit_lats, bergy_bit_lons, min_length_bb, bergy_bit_length_ranges)
    growler_bounds = calculate_outer_boundaries(growler_lengths, growler_lats, growler_lons, min_length_growler, growler_length_ranges)
    bergy_bit_bounds_dict = {}
    growler_bounds_dict = {}

    for k in range(len(bergy_bit_bounds["min_length_boundary"])):
        if bergy_bit_bounds["min_length_boundary"][k] is not None:
            bergy_bit_bounds_dict[k] = np.array(bergy_bit_bounds["min_length_boundary"][k])
        else:
            bergy_bit_bounds_dict[k] = np.empty((0, 2))

    for k in range(len(growler_bounds["min_length_boundary"])):
        if growler_bounds["min_length_boundary"][k] is not None:
            growler_bounds_dict[k] = np.array(growler_bounds["min_length_boundary"][k])
        else:
            growler_bounds_dict[k] = np.empty((0, 2))

    overall_bergy_bit_growler_boundary = calculate_overall_bergy_bit_growler_boundary(bergy_bit_bounds, growler_bounds)
    bergy_bit_length_final_stats = last_valid_length_stats(bergy_bit_lengths, min_length_bb, bergy_bit_growler_times)
    growler_length_final_stats = last_valid_length_stats(growler_lengths, min_length_growler, bergy_bit_growler_times)
    bergy_bit_length_overall_stats = overall_last_valid_length_stats(bergy_bit_lengths, min_length_bb, bergy_bit_growler_times)
    growler_length_overall_stats = overall_last_valid_length_stats(growler_lengths, min_length_growler, bergy_bit_growler_times)
    bergy_bit_growler_times = np.array(bergy_bit_growler_times)
    bergy_bit_growler_times = bergy_bit_growler_times.astype(str).tolist()
    return(bergy_bit_growler_times, bergy_bit_bounds_dict, bergy_bit_bounds, bergy_bit_length_final_stats, growler_bounds_dict,
           growler_bounds, growler_length_final_stats, overall_bergy_bit_growler_boundary, bergy_bit_length_overall_stats, growler_length_overall_stats)



from scipy.interpolate import RegularGridInterpolator, griddata, interp1d
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import numpy as np
import netCDF4 as nc
import os

def rcm_iceberg_drift_optimizer(iceberg_lats0, iceberg_lons0, iceberg_lengths0, iceberg_lats_end, iceberg_lons_end, iceberg_lengths_end,
                                iceberg_ids, rcm_datetime0, next_rcm_time, si_toggle):
    deg_radius = 5
    g = 9.80665
    rho_water = 1023.6
    rho_air = 1.225
    rho_ice = 910.
    omega = 7.292115e-5
    Cw = 0.7867
    Ca = 1.1857
    C_wave = 0.6
    Csi = 1.
    rho_si = 875.
    am = 0.5
    Re = 6371e3
    # u_curr_anc = 0.
    # v_curr_anc = 0.
    bathy_data_path = './GEBCO_Bathymetric_Data/gebco_2024.nc'
    rootpath_to_metdata = './RCM_Iceberg_Metocean_Data/'
    # drag_coeffs_anc_curr_bounds = ((0.01, 2.5), (0.01, 3.5), (-1., 1.), (-1., 1.))
    # drag_coeffs_anc_curr_initial_guess = np.array([Ca, Cw, u_curr_anc, v_curr_anc])
    drag_coeffs_anc_curr_bounds = ((0.01, 2.5), (0.01, 3.5))
    drag_coeffs_anc_curr_initial_guess = np.array([Ca, Cw])
    iceberg_lats0 = iceberg_lats0 if isinstance(iceberg_lats0, list) else [iceberg_lats0]
    iceberg_lons0 = iceberg_lons0 if isinstance(iceberg_lons0, list) else [iceberg_lons0]
    iceberg_lengths0 = iceberg_lengths0 if isinstance(iceberg_lengths0, list) else [iceberg_lengths0]
    iceberg_lats_end = iceberg_lats_end if isinstance(iceberg_lats_end, list) else [iceberg_lats_end]
    iceberg_lons_end = iceberg_lons_end if isinstance(iceberg_lons_end, list) else [iceberg_lons_end]
    iceberg_lengths_end = iceberg_lengths_end if isinstance(iceberg_lengths_end, list) else [iceberg_lengths_end]
    iceberg_ids = iceberg_ids if isinstance(iceberg_ids, list) else [iceberg_ids]

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

    def iceberg_pos_error(drag_coeffs_anc_curr, iceberg_id, Re, iceberg_lat_obs, iceberg_lon_obs, iceberg_times, iceberg_u0, iceberg_v0,
                    date_only_wind_waves, hour_utc_str_wind_waves, time_increments_wind_waves, forecast_times_wind_waves,
                    date_only_ocean, hour_utc_str_ocean, time_increments_ocean, forecast_times_ocean, rootpath_to_metdata,
                    dirname_wind_waves, dirname_ocean, lat_wind_waves, lon_wind_waves, file_times_wind_waves,
                    file_times_ocean, ocean_depth, iceberg_drafts, iceberg_times_dt, iceberg_lengths, iceberg_sails, iceberg_masses, am, omega, C_wave, g,
                    rho_air, rho_water, bathy_interp, Csi, rho_si, si_toggle, deg_radius):

        def duv_dt(t, uv):
            iceberg_u_init, iceberg_v_init = uv
            ib_acc_E, ib_acc_N = iceberg_acc(iceberg_lat, iceberg_u_init, iceberg_v_init, ib_sail, ib_draft,
                                             ib_length, ib_mass, iceberg_times_dt[i], am, omega, Cw, Ca,
                                             C_wave, g, rho_air, rho_water, u_wind_ib, v_wind_ib,
                                             [u_curr_ib, u_curr_ib2], [v_curr_ib, v_curr_ib2],
                                             ssh_grad_x_ib, ssh_grad_y_ib, Hs_ib, wave_dir_ib, siconc_ib, sithick_ib,
                                             usi_ib, vsi_ib, Csi, rho_si)
            return ib_acc_E, ib_acc_N

        Ca = drag_coeffs_anc_curr[0]
        Cw = drag_coeffs_anc_curr[1]
        # u_curr_anc = drag_coeffs_anc_curr[2]
        # v_curr_anc = drag_coeffs_anc_curr[3]

        iceberg_lats = np.empty((len(iceberg_times),))
        iceberg_lons = np.empty((len(iceberg_times),))

        iceberg_us = np.empty((len(iceberg_times),))
        iceberg_vs = np.empty((len(iceberg_times),))

        iceberg_grounded_statuses = np.empty((len(iceberg_times),))

        iceberg_lats[0] = iceberg_lat_obs[0]
        iceberg_lons[0] = iceberg_lon_obs[0]
        iceberg_us[0] = iceberg_u0
        iceberg_vs[0] = iceberg_v0
        iceberg_grounded_statuses[0] = 0

        for i in range(len(iceberg_times) - 1):
            print('Hindcasting iceberg ' + iceberg_id + ' at ' + str(iceberg_times[i]) + ' UTC...')
            iceberg_time = iceberg_times[i]
            iceberg_time2 = iceberg_times[i + 1]
            iceberg_lat = iceberg_lats[i]
            iceberg_lon = iceberg_lons[i]
            iceberg_u = iceberg_us[i]
            iceberg_v = iceberg_vs[i]
            ib_length = iceberg_lengths[i]
            ib_draft = iceberg_drafts[i]
            ib_sail = iceberg_sails[i]
            ib_mass = iceberg_masses[i]
            iceberg_grounded_status = iceberg_grounded_statuses[i]
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

            forecast_time_wind_waves_before = forecast_times_wind_waves[before_idx]
            forecast_time_wind_waves_after = forecast_times_wind_waves[after_idx]

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
                try:
                    siconc_before = siconc_before[ocean_lat_ind, ocean_lon_ind]
                    siconc_after = siconc_after[ocean_lat_ind, ocean_lon_ind]
                    sithick_before = sithick_before[ocean_lat_ind, ocean_lon_ind]
                    sithick_after = sithick_after[ocean_lat_ind, ocean_lon_ind]
                    usi_before = usi_before[ocean_lat_ind, ocean_lon_ind]
                    usi_after = usi_after[ocean_lat_ind, ocean_lon_ind]
                    vsi_before = vsi_before[ocean_lat_ind, ocean_lon_ind]
                    vsi_after = vsi_after[ocean_lat_ind, ocean_lon_ind]
                    siconc_before_ib = griddata(points_ocean, siconc_before.ravel(), (iceberg_lat, iceberg_lon + 360.), method='linear')
                    siconc_after_ib = griddata(points_ocean, siconc_after.ravel(), (iceberg_lat, iceberg_lon + 360.), method='linear')
                    sithick_before_ib = griddata(points_ocean, sithick_before.ravel(),(iceberg_lat, iceberg_lon + 360.), method='linear')
                    sithick_after_ib = griddata(points_ocean, sithick_after.ravel(), (iceberg_lat, iceberg_lon + 360.), method='linear')
                    usi_before_ib = griddata(points_ocean, usi_before.ravel(), (iceberg_lat, iceberg_lon + 360.), method='linear')
                    usi_after_ib = griddata(points_ocean, usi_after.ravel(), (iceberg_lat, iceberg_lon + 360.), method='linear')
                    vsi_before_ib = griddata(points_ocean, vsi_before.ravel(), (iceberg_lat, iceberg_lon + 360.), method='linear')
                    vsi_after_ib = griddata(points_ocean, vsi_after.ravel(), (iceberg_lat, iceberg_lon + 360.), method='linear')
                    siconc_ib = siconc_before_ib + weight_ocean * (siconc_after_ib - siconc_before_ib)
                    sithick_ib = sithick_before_ib + weight_ocean * (sithick_after_ib - sithick_before_ib)
                    usi_ib = usi_before_ib + weight_ocean * (usi_after_ib - usi_before_ib)
                    vsi_ib = vsi_before_ib + weight_ocean * (vsi_after_ib - vsi_before_ib)
                except:
                    siconc_ib = 0.
                    sithick_ib = 0.
                    usi_ib = 0.
                    vsi_ib = 0.
            else:
                siconc_ib = 0.
                sithick_ib = 0.
                usi_ib = 0.
                vsi_ib = 0.

            u_curr_ib = u_curr_before_ib + weight_ocean * (u_curr_after_ib - u_curr_before_ib) # + u_curr_anc
            v_curr_ib = v_curr_before_ib + weight_ocean * (v_curr_after_ib - v_curr_before_ib) # + v_curr_anc

            u_curr_ib2 = u_curr_before2_ib + weight_ocean2 * (u_curr_after2_ib - u_curr_before2_ib) # + u_curr_anc
            v_curr_ib2 = v_curr_before2_ib + weight_ocean2 * (v_curr_after2_ib - v_curr_before2_ib) # + v_curr_anc

            ssh_grad_x_ib = ssh_grad_x_before_ib + weight_ocean * (ssh_grad_x_after_ib - ssh_grad_x_before_ib)
            ssh_grad_y_ib = ssh_grad_y_before_ib + weight_ocean * (ssh_grad_y_after_ib - ssh_grad_y_before_ib)

            solution = solve_ivp(duv_dt, (0., iceberg_times_dt[i]), [iceberg_u, iceberg_v], method='BDF', t_eval=[0., iceberg_times_dt[i]])

            iceberg_u_end = solution.y[0][-1]
            iceberg_v_end = solution.y[1][-1]
            h_min = 660.9 / ((20e3) * np.exp(-20 * (1 - siconc_ib)))

            if siconc_ib >= 0.9 and sithick_ib >= h_min:
                iceberg_u_end = usi_ib
                iceberg_v_end = vsi_ib

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
            iceberg_us[i + 1] = iceberg_u_end
            iceberg_vs[i + 1] = iceberg_v_end
            iceberg_lats[i + 1] = iceberg_lat2
            iceberg_lons[i + 1] = iceberg_lon2
            iceberg_bathy_depth = bathy_interp([[iceberg_lat2, iceberg_lon2]])[0]

            if iceberg_bathy_depth <= ib_draft:
                iceberg_grounded_statuses[i + 1] = 1
                iceberg_us[i + 1] = 0.
                iceberg_vs[i + 1] = 0.
            else:
                iceberg_grounded_statuses[i + 1] = 0

        iceberg_pos_error, Az = dist_bearing(Re, iceberg_lat_obs[-1], iceberg_lats[-1], iceberg_lon_obs[-1], iceberg_lons[-1])
        return iceberg_pos_error

    rcm_datetime0 = np.datetime64(rcm_datetime0)
    next_rcm_time = np.datetime64(next_rcm_time)
    forecast_time = rcm_datetime0

    dirname_wind_waves = np.datetime_as_string(forecast_time, unit='D')
    d_wind_waves = dirname_wind_waves.replace('-', '')
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

        if iceberg_bathy_depth0 <= iceberg_drafts0[i]:
            iceberg_drafts0[i] = iceberg_bathy_depth0 - 1.

    iceberg_lats_end = np.array(iceberg_lats_end)
    iceberg_lons_end = np.array(iceberg_lons_end)
    iceberg_lengths_end = np.array(iceberg_lengths_end)

    iceberg_drafts_end = np.empty((len(iceberg_lats_end),))
    iceberg_masses_end = np.empty((len(iceberg_lats_end),))
    iceberg_sails_end = np.empty((len(iceberg_lats_end),))

    for i in range(len(iceberg_lats_end)):
        if (not isinstance(iceberg_lengths_end[i], (int, float)) or np.any(np.isnan(iceberg_lengths_end[i])) or
                np.any(np.isinf(iceberg_lengths_end[i])) or np.any(~np.isreal(iceberg_lengths_end[i])) or iceberg_lengths_end[i] <= 0):
            iceberg_lengths_end[i] = 100.

        iceberg_drafts_end[i] = 1.78 * (iceberg_lengths_end[i] ** 0.71) # meters
        iceberg_masses_end[i] = 0.45 * rho_ice * (iceberg_lengths_end[i] ** 3) # kg
        iceberg_sails_end[i] = 0.077 * (iceberg_lengths_end[i] ** 2) # m ** 2
        iceberg_bathy_depth_end = bathy_interp([[iceberg_lats_end[i], iceberg_lons_end[i]]])[0]

        if iceberg_bathy_depth_end <= iceberg_drafts_end[i]:
            iceberg_drafts_end[i] = iceberg_bathy_depth_end - 1.

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

    forecast_times_wind_waves = []
    forecast_times_ocean = []

    directory = rootpath_to_metdata + 'RIOPS_ocean_forecast_files/' + dirname_ocean + '/'
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    file = files[0]
    hour_utc_str_ocean = file[9:11]

    directory = rootpath_to_metdata + 'GDWPS_wind_wave_forecast_files/' + dirname_wind_waves + '/'
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    file = files[0]
    hour_utc_str_wind_waves = file[9:11]

    forecast_start_time_wind_waves = np.datetime64(dirname_wind_waves + 'T' + hour_utc_str_wind_waves + ':00:00')
    forecast_start_time_ocean = np.datetime64(dirname_ocean + 'T' + hour_utc_str_ocean + ':00:00')
    time_count_wind_waves = forecast_start_time_wind_waves
    time_count_ocean = forecast_start_time_ocean

    while time_count_wind_waves <= next_rcm_time + np.timedelta64(1, 'h'):
        forecast_times_wind_waves.append(time_count_wind_waves)
        time_count_wind_waves = time_count_wind_waves + np.timedelta64(1, 'h')

    while time_count_ocean <= next_rcm_time + np.timedelta64(1, 'h'):
        forecast_times_ocean.append(time_count_ocean)
        time_count_ocean = time_count_ocean + np.timedelta64(1, 'h')

    forecast_times_wind_waves = np.array(forecast_times_wind_waves, dtype='datetime64[s]')
    forecast_times_ocean = np.array(forecast_times_ocean, dtype='datetime64[s]')

    forecast_times_wind_waves_hours = (forecast_times_wind_waves.astype('datetime64[h]') - forecast_start_time_wind_waves.astype('datetime64[h]')).astype(int)
    forecast_times_ocean_hours = (forecast_times_ocean.astype('datetime64[h]') - forecast_start_time_ocean.astype('datetime64[h]')).astype(int)

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

    fname = (rootpath_to_metdata + 'GDWPS_wind_wave_forecast_files/' + dirname_wind_waves + '/' + d_wind_waves + 'T' +
             hour_utc_str_wind_waves + 'Z_MSC_GDWPS_UGRD_AGL-10m_LatLon0.25_PT' + str(forecast_times_wind_waves_hours[0]).zfill(3) + 'H.nc')
    wind_data = nc.Dataset(fname)
    lat_wind_waves = wind_data.variables['latitude'][:]
    lon_wind_waves = wind_data.variables['longitude'][:]
    wind_data.close()

    base_time_wind_waves = forecast_times_wind_waves[0]
    time_increments_wind_waves = np.arange(forecast_times_wind_waves_hours[0], forecast_times_wind_waves_hours[-1], 1)
    file_times_wind_waves = base_time_wind_waves + time_increments_wind_waves.astype('timedelta64[h]')
    date_only_wind_waves = str(base_time_wind_waves.astype('datetime64[D]')).replace('-', '')

    base_time_ocean = forecast_times_ocean[0]
    time_increments_ocean = np.arange(forecast_times_ocean_hours[0], forecast_times_ocean_hours[-1], 1)
    file_times_ocean = base_time_ocean + time_increments_ocean.astype('timedelta64[h]')
    date_only_ocean = str(base_time_ocean.astype('datetime64[D]')).replace('-', '')

    iceberg_times_interp = np.array(iceberg_times, dtype='datetime64[s]')
    time_numeric = iceberg_times_interp.astype('int')
    time_start = time_numeric[0]
    time_end = time_numeric[-1]

    iceberg_pos_error_list = []
    Ca_list = []
    Cw_list = []
    # u_curr_anc_list = []
    # v_curr_anc_list = []

    for k in range(len(iceberg_lats0)):
        print('Optimizing drag coefficients and currents for iceberg ' + str(iceberg_ids[k]) + '...')
        iceberg_id = str(iceberg_ids[k])
        iceberg_lengths = np.interp(time_numeric, [time_start, time_end], [iceberg_lengths0[k], iceberg_lengths_end[k]])
        iceberg_lat_obs = np.interp(time_numeric, [time_start, time_end], [iceberg_lats0[k], iceberg_lats_end[k]])
        iceberg_lon_obs = np.interp(time_numeric, [time_start, time_end], [iceberg_lons0[k], iceberg_lons_end[k]])
        iceberg_lengths = iceberg_lengths.tolist()
        iceberg_lat_obs = iceberg_lat_obs.tolist()
        iceberg_lon_obs = iceberg_lon_obs.tolist()
        L, Az = dist_bearing(Re, iceberg_lat_obs[0], iceberg_lat_obs[-1], iceberg_lon_obs[0], iceberg_lon_obs[-1])
        delta_time = (iceberg_times[-1] - iceberg_times[0]) / np.timedelta64(1, 's')
        iceberg_u0 = L * np.sin(np.deg2rad(Az)) / delta_time
        iceberg_v0 = L * np.cos(np.deg2rad(Az)) / delta_time
        iceberg_drafts = []
        iceberg_sails = []
        iceberg_masses = []

        for n in range(len(iceberg_lengths)):
            iceberg_drafts.append(1.78 * (iceberg_lengths[n] ** 0.71)) # meters
            iceberg_masses.append(0.45 * rho_ice * (iceberg_lengths[n] ** 3)) # kg
            iceberg_sails.append(0.077 * (iceberg_lengths[n] ** 2)) # m ** 2

        result = minimize(fun=lambda drag_coeffs_anc_curr: iceberg_pos_error(drag_coeffs_anc_curr, iceberg_id, Re, iceberg_lat_obs, iceberg_lon_obs,
                                                                             iceberg_times, iceberg_u0, iceberg_v0, date_only_wind_waves,
                                                                             hour_utc_str_wind_waves, time_increments_wind_waves,
                                                                             forecast_times_wind_waves, date_only_ocean, hour_utc_str_ocean,
                                                                             time_increments_ocean, forecast_times_ocean, rootpath_to_metdata,
                                                                             dirname_wind_waves, dirname_ocean, lat_wind_waves, lon_wind_waves,
                                                                             file_times_wind_waves, file_times_ocean, ocean_depth, iceberg_drafts,
                                                                             iceberg_times_dt, iceberg_lengths, iceberg_sails, iceberg_masses,
                                                                             am, omega, C_wave, g, rho_air, rho_water, bathy_interp, Csi,
                                                                             rho_si, si_toggle, deg_radius),
                          x0=drag_coeffs_anc_curr_initial_guess, method='L-BFGS-B', tol=1e-3, bounds=drag_coeffs_anc_curr_bounds, options={'eps': 0.01})
        Ca = result.x[0]
        Cw = result.x[1]
        # u_curr_anc = result.x[2]
        # v_curr_anc = result.x[3]
        # drag_coeffs_anc_curr_final_array = np.array([Ca, Cw, u_curr_anc, v_curr_anc])
        drag_coeffs_anc_curr_final_array = np.array([Ca, Cw])
        iceberg_pos_error_final = iceberg_pos_error(drag_coeffs_anc_curr_final_array, iceberg_id, Re, iceberg_lat_obs, iceberg_lon_obs,
                                                                             iceberg_times, iceberg_u0, iceberg_v0, date_only_wind_waves,
                                                                             hour_utc_str_wind_waves, time_increments_wind_waves,
                                                                             forecast_times_wind_waves, date_only_ocean, hour_utc_str_ocean,
                                                                             time_increments_ocean, forecast_times_ocean, rootpath_to_metdata,
                                                                             dirname_wind_waves, dirname_ocean, lat_wind_waves, lon_wind_waves,
                                                                             file_times_wind_waves, file_times_ocean, ocean_depth, iceberg_drafts,
                                                                             iceberg_times_dt, iceberg_lengths, iceberg_sails, iceberg_masses,
                                                                             am, omega, C_wave, g, rho_air, rho_water, bathy_interp, Csi,
                                                                             rho_si, si_toggle, deg_radius)
        Ca_list.append(Ca)
        Cw_list.append(Cw)
        # u_curr_anc_list.append(u_curr_anc)
        # v_curr_anc_list.append(v_curr_anc)
        iceberg_pos_error_list.append(iceberg_pos_error_final / 1000.)

    return iceberg_pos_error_list, Ca_list, Cw_list # , u_curr_anc_list, v_curr_anc_list


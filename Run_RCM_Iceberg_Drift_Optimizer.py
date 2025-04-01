
from RCM_Iceberg_Drift_Optimizer import *
import time
import warnings
warnings.simplefilter(action='ignore')

start_time = time.time()

# iceberg_lats0 = [54.56685635, 54.55213921]
# iceberg_lons0 = [-55.49470901, -55.4882172]
# iceberg_lats_end = [54.6, 54.6]
# iceberg_lons_end = [-55.625, -55.55]
# iceberg_lengths0 = [50.15690101, 67.16309007]
# iceberg_lengths_end = [50., 67.]
# iceberg_ids = ['0000', '0001']

iceberg_lats0 = 54.56685635
iceberg_lons0 = -55.49470901
iceberg_lats_end = 54.6
iceberg_lons_end = -55.625
iceberg_lengths0 = 67.16309007
iceberg_lengths_end = 67.
iceberg_ids = '0000'

rcm_datetime0 = np.datetime64('2025-02-06T16:00:00')
next_rcm_time = rcm_datetime0 + np.timedelta64(24, 'h')
si_toggle = True

# iceberg_pos_error_list, Ca_list, Cw_list, iceberg_u0_list, iceberg_v0_list, u_curr_anc_list, v_curr_anc_list = rcm_iceberg_drift_optimizer(iceberg_lats0,
#                                                                                                          iceberg_lons0, iceberg_lengths0,
#                                                                                                          iceberg_lats_end, iceberg_lons_end,
#                                                                                                          iceberg_lengths_end, iceberg_ids, rcm_datetime0,
#                                                                                                          next_rcm_time, si_toggle)
iceberg_pos_error_list, Ca_list, Cw_list, iceberg_u0_list, iceberg_v0_list = rcm_iceberg_drift_optimizer(iceberg_lats0, iceberg_lons0, iceberg_lengths0,
                                                                       iceberg_lats_end, iceberg_lons_end, iceberg_lengths_end, iceberg_ids, rcm_datetime0,
                                                                       next_rcm_time, si_toggle)
iceberg_ids = iceberg_ids if isinstance(iceberg_ids, list) else [iceberg_ids]

for i in range(len(iceberg_ids)):
    print(f"Iceberg {iceberg_ids[i]}:")
    print(f"Optimized hindcast position error: {iceberg_pos_error_list[i]:.2f} km")
    print(f"Optimized air drag coefficient: {Ca_list[i]:.2f}")
    print(f"Optimized water drag coefficient: {Cw_list[i]:.2f}")
    print(f"Iceberg zonal drift velocity: {iceberg_u0_list[i]:.2f} m/s")
    print(f"Iceberg meridional drift velocity: {iceberg_v0_list[i]:.2f} m/s")
    # print(f"Optimized zonal ancillary ocean current velocity: {u_curr_anc_list[i]:.2f} m/s")
    # print(f"Optimized meridional ancillary ocean current velocity: {v_curr_anc_list[i]:.2f} m/s")
    print()

end_time = time.time()

# Calculate the elapsed time
elapsed_time = (end_time - start_time) / 60.
print(f"Script runtime: {elapsed_time:.2f} minutes.")


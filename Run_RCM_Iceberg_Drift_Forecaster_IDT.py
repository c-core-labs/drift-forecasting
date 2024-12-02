
from RCM_Iceberg_Drift_Forecaster_IDT import *
from RCM_Iceberg_Drift_Deterioration_Forecaster_IDT import *
import time
import datetime
import warnings
warnings.simplefilter(action='ignore')

start_time = time.time()

iceberg_lat0 = 47.5
iceberg_lon0 = -46.5
rcm_datetime0 = np.datetime64(datetime.datetime.now(datetime.timezone.utc))  # - np.timedelta64(12, 'h')
next_rcm_time = rcm_datetime0 + np.timedelta64(6, 'h') + np.timedelta64(24, 'm')
iceberg_length = 50.
grounded_status = 'not grounded'
version = 'dynamic - with deterioration' # 'dynamic - no deterioration' or 'dynamic - with deterioration'

if version == 'dynamic - no deterioration':
    (iceberg_lat0, iceberg_lon0, iceberg_lats, iceberg_lons, iceberg_times, iceberg_total_displacement, iceberg_overall_course,
     iceberg_length, iceberg_draft, iceberg_mass, rcm_datetime0, next_rcm_time, grounded_status) = (
        rcm_iceberg_drift_forecaster(iceberg_lat0, iceberg_lon0, rcm_datetime0, iceberg_length, grounded_status, next_rcm_time))

    print(iceberg_lat0)
    print(iceberg_lon0)
    print(iceberg_lats)
    print(iceberg_lons)
    print(iceberg_times)
    print(iceberg_total_displacement / 1000.)
    print(iceberg_overall_course)
    print(grounded_status)

elif version == 'dynamic - with deterioration':
    (iceberg_lat0, iceberg_lon0, iceberg_lats, iceberg_lons, iceberg_times, iceberg_total_displacement, iceberg_overall_course,
            iceberg_length, iceberg_draft, iceberg_mass, rcm_datetime0, next_rcm_time, grounded_status, iceberg_lengths, iceberg_masses,
            iceberg_total_length_loss, iceberg_total_mass_loss) = (
        rcm_iceberg_drift_deterioration_forecaster(iceberg_lat0, iceberg_lon0, rcm_datetime0, iceberg_length, grounded_status, next_rcm_time))

    print(iceberg_lat0)
    print(iceberg_lon0)
    print(iceberg_lats)
    print(iceberg_lons)
    print(iceberg_times)
    print(iceberg_lengths)
    print(iceberg_masses)
    print(iceberg_total_displacement / 1000.)
    print(iceberg_overall_course)
    print(iceberg_total_length_loss)
    print(iceberg_total_mass_loss)
    print(grounded_status)

end_time = time.time()

# Calculate the elapsed time
elapsed_time = (end_time - start_time) / 60.
print(f"Script runtime: {elapsed_time:.2f} minutes.")
# print(f"Script runtime: {elapsed_time * 60.:.2f} seconds.")


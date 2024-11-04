
from RCM_Iceberg_Drift_Forecaster_IDT import *
import time
import datetime
import warnings
warnings.simplefilter(action='ignore')

start_time = time.time()

iceberg_lat0 = 48.6427
iceberg_lon0 = -53.
rcm_datetime0 = np.datetime64(datetime.datetime.now(datetime.timezone.utc))  # - np.timedelta64(12, 'h')
next_rcm_time = rcm_datetime0 + np.timedelta64(16, 'h') # + np.timedelta64(32, 'm')
iceberg_length = 67.
grounded_status = 'not grounded'
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

end_time = time.time()

# Calculate the elapsed time
elapsed_time = (end_time - start_time) / 60.
print(f"Script runtime: {elapsed_time:.2f} minutes.")
# print(f"Script runtime: {elapsed_time * 60.:.2f} seconds.")


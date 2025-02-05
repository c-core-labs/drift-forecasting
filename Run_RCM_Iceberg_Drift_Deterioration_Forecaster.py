
from RCM_Iceberg_Drift_Deterioration_Forecaster import *
import time
# import datetime
import warnings
warnings.simplefilter(action='ignore')

# Without sea ice forcing:
# Script runtime: 13.13 minutes for 2 icebergs forecast to 24 hours.

# With sea ice forcing:
# Script runtime: 13.25 minutes for 2 icebergs forecast to 24 hours.

start_time = time.time()

# iceberg_lats0 = [47.5, 47.5]
# iceberg_lons0 = [-46.9, -46.5]
iceberg_lats0 = 47.5
iceberg_lons0 = -46.5
# rcm_datetime0 = np.datetime64(datetime.datetime.now(datetime.timezone.utc)) # - np.timedelta64(12, 'h')
rcm_datetime0 = np.datetime64('2025-02-05T16:36:48')
next_rcm_time = rcm_datetime0 + np.timedelta64(24, 'h') # + np.timedelta64(23, 'm')
# iceberg_lengths0 = [40., 67.]
iceberg_lengths0 = 67.
# iceberg_ids = ['0000', '0001']
iceberg_ids = '0000'
# iceberg_grounded_statuses0 = [False, False]
iceberg_grounded_statuses0 = False
si_toggle = False

# obs = Observations(iceberg_lats0, iceberg_lons0, rcm_datetime0, iceberg_lengths0, iceberg_grounded_statuses0, [False, False], iceberg_ids)
obs = Observation(iceberg_lats0, iceberg_lons0, rcm_datetime0, iceberg_lengths0, iceberg_grounded_statuses0, False, iceberg_ids)
iceberg_times, iceberg_lats, iceberg_lons, iceberg_lengths, iceberg_grounded_statuses = rcm_iceberg_drift_deterioration_forecaster(obs, next_rcm_time, si_toggle)

print(iceberg_lats)
print(iceberg_lons)
print(iceberg_times)
print(iceberg_lengths)
print(iceberg_grounded_statuses)

end_time = time.time()

# Calculate the elapsed time
elapsed_time = (end_time - start_time) / 60.
print(f"Script runtime: {elapsed_time:.2f} minutes.")
# print(f"Script runtime: {elapsed_time * 60.:.2f} seconds.")


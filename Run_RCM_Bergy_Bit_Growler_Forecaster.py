
from RCM_Bergy_Bit_Growler_Forecaster import *
import matplotlib.pyplot as plt
import time
# import datetime
import warnings
warnings.simplefilter(action='ignore')

# With sea ice forcing:
# Script runtime: 2638.5 minutes (43.975 hours) for 2 icebergs forecast to 84 hours with 250 ensemble members.

start_time = time.time()

iceberg_lats0 = [47.5, 47.5]
iceberg_lons0 = [-46.9, -46.5]
# iceberg_lats0 = [47.5]
# iceberg_lons0 = [-46.5]
# rcm_datetime0 = np.datetime64(datetime.datetime.now(datetime.timezone.utc))  # - np.timedelta64(12, 'h')
# rcm_datetime0 = np.datetime64('2024-12-20T16:46:47.683976')
rcm_datetime0 = np.datetime64('2024-12-20T12:00:00')
# rcm_datetime0 = np.datetime64('2024-12-09T18:01:00')
# forecast_end_time = rcm_datetime0 + np.timedelta64(24, 'h') # + np.timedelta64(23, 'm')
forecast_end_time = np.datetime64('2024-12-20T00:00:00') + np.timedelta64(84, 'h')
iceberg_ids = [0, 1]
# iceberg_ids = [0]
bathy_data_path = './GEBCO_Bathymetric_Data/gebco_2024.nc'
rootpath_to_metdata = './RCM_Iceberg_Metocean_Data/'
hour_utc_str_airT_sw_rad = '12'
hour_utc_str_wind_waves = '00'
hour_utc_str_ocean = '06'
si_toggle = True

bergy_bit_bounds_dict, bergy_bit_length_final_stats, growler_bounds_dict, growler_length_final_stats = rcm_bergy_bit_growler_forecaster(bathy_data_path,
                                                                              rootpath_to_metdata, iceberg_lats0, iceberg_lons0, iceberg_ids, rcm_datetime0,
                                                                              forecast_end_time, hour_utc_str_airT_sw_rad, hour_utc_str_wind_waves,
                                                                              hour_utc_str_ocean, si_toggle)

for iceberg_index, stats in bergy_bit_length_final_stats.items():
    print(f"Iceberg {iceberg_index}:")
    print(f" Min Final Bergy Bit Length: {stats['min']}")
    print(f" Max Final Bergy Bit Length: {stats['max']}")
    print(f" Mean Final Bergy Bit Length: {stats['mean']}")
    print()

for iceberg_index, stats in growler_length_final_stats.items():
    print(f"Iceberg {iceberg_index}:")
    print(f" Min Final Growler Length: {stats['min']}")
    print(f" Max Final Growler Length: {stats['max']}")
    print(f" Mean Final Growler Length: {stats['mean']}")
    print()

end_time = time.time()

# Calculate the elapsed time
elapsed_time = (end_time - start_time) / 60.
print(f"Script runtime: {elapsed_time:.2f} minutes.")
# print(f"Script runtime: {elapsed_time * 60.:.2f} seconds.")

plt.figure(figsize=(10, 8))
colors = ["red", "green", "blue", "orange", "purple"]

# Plot bergy bit boundaries
for k, boundary in bergy_bit_bounds_dict.items():
    if boundary.size > 0: # Check if the boundary is not empty
        # Close the boundary by adding the first point to the end
        closed_boundary = np.vstack([boundary, boundary[0]])
        plt.plot(closed_boundary[:, 0], # Longitude
                 closed_boundary[:, 1], # Latitude
                 label=f"Bergy Bits for Iceberg {k} Boundary",
                 linewidth=2,
                 c=colors[k % len(colors)])

colors = ["blue", "orange", "purple", "red", "green"]

# Plot growler boundaries
for k, boundary in growler_bounds_dict.items():
    if boundary.size > 0: # Check if the boundary is not empty
        # Close the boundary by adding the first point to the end
        closed_boundary = np.vstack([boundary, boundary[0]])
        plt.plot(closed_boundary[:, 0], # Longitude
                 closed_boundary[:, 1], # Latitude
                 label=f"Growlers for Iceberg {k} Boundary",
                 linewidth=2,
                 c=colors[k % len(colors)])

# Correctly label the axes
plt.xlabel("Longitude (°E)")
plt.ylabel("Latitude (°N)")
plt.title("Outer Boundaries of Bergy Bits and Growlers", fontweight='bold', fontsize=10)
plt.legend()
plt.grid(True)
plt.show()


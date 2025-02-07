
from RCM_Bergy_Bit_Growler_Forecaster import *
import matplotlib.pyplot as plt
import time
# import datetime
import warnings
warnings.simplefilter(action='ignore')

start_time = time.time()

iceberg_lats0 = [54.6, 54.6]
iceberg_lons0 = [-55.625, -55.55]
# iceberg_lats0 = 54.6
# iceberg_lons0 = -55.625
# rcm_datetime0 = np.datetime64(datetime.datetime.now(datetime.timezone.utc)) # - np.timedelta64(12, 'h')
rcm_datetime0 = np.datetime64('2025-02-06T16:36:48')
forecast_end_time = rcm_datetime0 + np.timedelta64(24, 'h') # + np.timedelta64(23, 'm')
iceberg_lengths0 = [40., 67.]
# iceberg_lengths0 = 67.
iceberg_ids = ['0000', '0001']
# iceberg_ids = '0000'
iceberg_grounded_statuses0 = [False, False]
# iceberg_grounded_statuses0 = False
si_toggle = False

obs = Observations(iceberg_lats0, iceberg_lons0, rcm_datetime0, iceberg_lengths0, iceberg_grounded_statuses0, [False, False], iceberg_ids)
# obs = Observation(iceberg_lats0, iceberg_lons0, rcm_datetime0, iceberg_lengths0, iceberg_grounded_statuses0, False, iceberg_ids)
(bergy_bit_growler_times, bergy_bit_bounds_dict, bergy_bit_bounds, bergy_bit_length_final_stats, growler_bounds_dict, growler_bounds, growler_length_final_stats,
 overall_bergy_bit_growler_boundary, bergy_bit_length_overall_stats, growler_length_overall_stats) = rcm_bergy_bit_growler_forecaster(obs, forecast_end_time, si_toggle)
bergy_bit_growler_times = np.array(bergy_bit_growler_times, dtype="datetime64[s]")
time_1 = np.min(bergy_bit_growler_times)
time_2 = np.max(bergy_bit_growler_times)

for iceberg_index, stats in bergy_bit_length_final_stats.items():
    print(f"Iceberg {iceberg_index}:")
    print(f" Min Final Bergy Bit Length: {stats['min']}")
    print(f" Max Final Bergy Bit Length: {stats['max']}")
    print(f" Mean Final Bergy Bit Length: {stats['mean']}")
    print(f" Final Bergy Bit Time: {stats['latest_time']}")
    print()

for iceberg_index, stats in growler_length_final_stats.items():
    print(f"Iceberg {iceberg_index}:")
    print(f" Min Final Growler Length: {stats['min']}")
    print(f" Max Final Growler Length: {stats['max']}")
    print(f" Mean Final Growler Length: {stats['mean']}")
    print(f" Final Growler Time: {stats['latest_time']}")
    print()

print("Overall Last Valid Bergy Bit Length Stats:")
print(f"Min Length: {bergy_bit_length_overall_stats['min']} m")
print(f"Max Length: {bergy_bit_length_overall_stats['max']} m")
print(f"Mean Length: {bergy_bit_length_overall_stats['mean']} m")
print(f"Latest Recorded Time: {bergy_bit_length_overall_stats['latest_time']}")
print()

print("Overall Last Valid Growler Length Stats:")
print(f"Min Length: {growler_length_overall_stats['min']} m")
print(f"Max Length: {growler_length_overall_stats['max']} m")
print(f"Mean Length: {growler_length_overall_stats['mean']} m")
print(f"Latest Recorded Time: {growler_length_overall_stats['latest_time']}")
print()

end_time = time.time()

# Calculate the elapsed time
elapsed_time = (end_time - start_time) / 60.
print(f"Script runtime: {elapsed_time:.2f} minutes.")
# print(f"Script runtime: {elapsed_time * 60.:.2f} seconds.")

plt.figure(figsize=(10, 8))

# Define colors for each range
range_colors = ["gold", "cyan", "magenta", "lime", "brown", "navy"]
colors = ["red", "green", "blue", "orange", "purple"]

# Function to get a consistent color for each range
def get_range_color(i):
    return range_colors[i % len(range_colors)]

# --- Plot Bergy Bit Boundaries for min_length ---
for k, boundary in bergy_bit_bounds_dict.items():
    if boundary is not None and boundary.size > 0:
        closed_boundary = np.vstack([boundary, boundary[0]]) # Close the boundary loop
        plt.plot(closed_boundary[:, 0], # Longitude
                 closed_boundary[:, 1], # Latitude
                 label=f"Bergy Bits for Iceberg {k} Boundary", linewidth=2, c=colors[k % len(colors)])

# --- Plot Bergy Bit Ranges ---
if "length_range_boundaries" in bergy_bit_bounds:
    sorted_ranges = sorted(bergy_bit_bounds["length_range_boundaries"].keys()) # Sort (low, high) tuples
    for i, (low, high) in enumerate(sorted_ranges):
        for k, boundary in bergy_bit_bounds["length_range_boundaries"][(low, high)].items():
            if boundary is not None and boundary.size > 0:
                closed_boundary = np.vstack([boundary, boundary[0]]) # Close the loop
                plt.fill(closed_boundary[:, 0], closed_boundary[:, 1], color=get_range_color(i), alpha=0.25, label=f"Bergy Bits {low}m - {high}m", zorder=-i)

colors = ["blue", "orange", "purple", "red", "green"]

# --- Plot Growler Boundaries for min_length ---
for k, boundary in growler_bounds_dict.items():
    if boundary is not None and boundary.size > 0:
        closed_boundary = np.vstack([boundary, boundary[0]]) # Close the boundary loop
        plt.plot(closed_boundary[:, 0], # Longitude
                 closed_boundary[:, 1], # Latitude
                 label=f"Growlers for Iceberg {k} Boundary", linewidth=2, c=colors[k % len(colors)])

# --- Plot Growler Ranges ---
if "length_range_boundaries" in growler_bounds:
    sorted_ranges = sorted(growler_bounds["length_range_boundaries"].keys()) # Sort (low, high) tuples
    for i, (low, high) in enumerate(sorted_ranges):
        for k, boundary in growler_bounds["length_range_boundaries"][(low, high)].items():
            if boundary is not None and boundary.size > 0:
                closed_boundary = np.vstack([boundary, boundary[0]]) # Close the loop
                plt.fill(closed_boundary[:, 0], closed_boundary[:, 1], color=get_range_color(i), alpha=0.25, label=f"Growlers {low}m - {high}m", zorder=-i)

# --- Plot the Overall Bergy Bit + Growler Boundary ---
if overall_bergy_bit_growler_boundary is not None and overall_bergy_bit_growler_boundary.size > 0:
    closed_boundary = np.vstack([overall_bergy_bit_growler_boundary, overall_bergy_bit_growler_boundary[0]]) # Close loop
    plt.plot(closed_boundary[:, 0], closed_boundary[:, 1], label="Overall Bergy Bit + Growler Boundary", linewidth=3, linestyle="dashed", color="black", zorder=5)

# --- Plot Iceberg Initial Positions ---
plt.scatter(iceberg_lons0, iceberg_lats0, c="red", s=100, label="Iceberg Initial Positions", edgecolor="black", zorder=10)
plt.xlabel("Longitude (°E)")
plt.ylabel("Latitude (°N)")
plt.title(f"Bergy Bits and Growlers Drift/Deterioration {time_1} - {time_2}", fontsize=12, fontweight='bold')
plt.legend()
plt.grid(True)
plt.savefig("bergy_bit_growler_outer_boundaries_ranges.png", dpi=300, bbox_inches="tight")
# plt.show()


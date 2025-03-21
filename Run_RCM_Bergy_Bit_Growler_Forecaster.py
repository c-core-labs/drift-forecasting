
from RCM_Bergy_Bit_Growler_Forecaster import *
import matplotlib.pyplot as plt
import time
# import datetime
import warnings
warnings.simplefilter(action='ignore')

start_time = time.time()

iceberg_lats_obs = [54.6, 54.6]
iceberg_lons_obs = [-55.625, -55.55]
iceberg_lats0 = [54.56685635, 54.55213921]
iceberg_lons0 = [-55.49470901, -55.4882172]
# iceberg_lats0 = 54.6
# iceberg_lons0 = -55.625
# rcm_datetime0 = np.datetime64(datetime.datetime.now(datetime.timezone.utc)) # - np.timedelta64(12, 'h')
rcm_datetime0 = np.datetime64('2025-02-06T16:00:00')
forecast_end_time = rcm_datetime0 + np.timedelta64(24, 'h') # + np.timedelta64(23, 'm')
# iceberg_lengths0 = [50., 67.]
iceberg_lengths0 = [50.15690101, 67.16309007]
# iceberg_lengths0 = 67.
iceberg_ids = ['0000', '0001']
# iceberg_ids = '0000'
iceberg_grounded_statuses0 = [False, False]
# iceberg_grounded_statuses0 = False
si_toggle = True

obs = Observations(iceberg_lats0, iceberg_lons0, rcm_datetime0, iceberg_lengths0, iceberg_grounded_statuses0, [False, False], iceberg_ids)
# obs = Observation(iceberg_lats0, iceberg_lons0, rcm_datetime0, iceberg_lengths0, iceberg_grounded_statuses0, False, iceberg_ids)
(bergy_bit_growler_times, bergy_bit_bounds_dict, bergy_bit_bounds, bergy_bit_length_final_stats, growler_bounds_dict, growler_bounds, growler_length_final_stats,
 overall_bergy_bit_growler_boundary, overall_boundary, bergy_bit_length_overall_stats, growler_length_overall_stats) = (
    rcm_bergy_bit_growler_forecaster(obs, forecast_end_time, si_toggle))
bergy_bit_growler_times = np.array(bergy_bit_growler_times, dtype="datetime64[s]")
time_1 = np.min(bergy_bit_growler_times)
time_2 = np.max(bergy_bit_growler_times)

# Ensure observations are NumPy arrays
iceberg_obs_points = np.column_stack((np.array(iceberg_lons_obs).ravel(), np.array(iceberg_lats_obs).ravel()))

# Stack observed iceberg positions with the existing boundary
if overall_boundary is not None and overall_boundary.size > 0:
    all_points = np.vstack([overall_boundary, iceberg_obs_points])
else:
    all_points = iceberg_obs_points # In case the original boundary is empty

# Ensure at least 3 unique points exist
unique_points = np.unique(all_points, axis=0)
if unique_points.shape[0] >= 3:
    hull = ConvexHull(unique_points)
    new_overall_boundary = unique_points[hull.vertices]
else:
    new_overall_boundary = None # Not enough points for a valid boundary

for iceberg_index, stats in bergy_bit_length_final_stats.items():
    print(f"Iceberg {iceberg_index}:")
    print(f" Min Final Bergy Bit Length: {stats['min']:.2f}")
    print(f" Max Final Bergy Bit Length: {stats['max']:.2f}")
    print(f" Mean Final Bergy Bit Length: {stats['mean']:.2f}")
    print(f" Final Bergy Bit Time: {stats['latest_time']}")
    print()

for iceberg_index, stats in growler_length_final_stats.items():
    print(f"Iceberg {iceberg_index}:")
    print(f" Min Final Growler Length: {stats['min']:.2f}")
    print(f" Max Final Growler Length: {stats['max']:.2f}")
    print(f" Mean Final Growler Length: {stats['mean']:.2f}")
    print(f" Final Growler Time: {stats['latest_time']}")
    print()

print("Overall Last Valid Bergy Bit Length Stats:")
print(f"Min Length: {bergy_bit_length_overall_stats['min']:.2f} m")
print(f"Max Length: {bergy_bit_length_overall_stats['max']:.2f} m")
print(f"Mean Length: {bergy_bit_length_overall_stats['mean']:.2f} m")
print(f"Latest Recorded Time: {bergy_bit_length_overall_stats['latest_time']}")
print()

print("Overall Last Valid Growler Length Stats:")
print(f"Min Length: {growler_length_overall_stats['min']:.2f} m")
print(f"Max Length: {growler_length_overall_stats['max']:.2f} m")
print(f"Mean Length: {growler_length_overall_stats['mean']:.2f} m")
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
        iceberg_num = iceberg_ids[k]
        plt.plot(closed_boundary[:, 0], # Longitude
                 closed_boundary[:, 1], # Latitude
                 label=f"Bergy Bits for Iceberg {iceberg_num} Boundary", linewidth=2, c=colors[k % len(colors)])

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
        iceberg_num = iceberg_ids[k]
        plt.plot(closed_boundary[:, 0], # Longitude
                 closed_boundary[:, 1], # Latitude
                 label=f"Growlers for Iceberg {iceberg_num} Boundary", linewidth=2, c=colors[k % len(colors)])

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

# # --- Plot the Overall Bergy Bit + Growler + Iceberg Boundary ---
# if overall_boundary is not None and overall_boundary.size > 0:
#     closed_boundary = np.vstack([overall_boundary, overall_boundary[0]]) # Close loop
#     plt.plot(closed_boundary[:, 0], closed_boundary[:, 1], label="Overall Bergy Bit + Growler + Iceberg Boundary",
#              linewidth=3, linestyle="dashed", color="red", zorder=5)

# --- Plot the Overall Bergy Bit + Growler + Iceberg Boundary ---
if new_overall_boundary is not None and new_overall_boundary.size > 0:
    closed_boundary = np.vstack([new_overall_boundary, new_overall_boundary[0]]) # Close loop
    plt.plot(closed_boundary[:, 0], closed_boundary[:, 1], label="Overall Bergy Water Boundary",
             linewidth=3, linestyle="dashed", color="red", zorder=5)

# --- Plot Iceberg Initial Positions ---
plt.scatter(iceberg_lons_obs, iceberg_lats_obs, c="blue", s=100, label="Iceberg RCM Observed Positions", edgecolor="black", zorder=10)
plt.scatter(iceberg_lons0, iceberg_lats0, c="red", s=100, label="Iceberg Hindcast Positions", edgecolor="black", zorder=10)
plt.xlabel("Longitude (°E)")
plt.ylabel("Latitude (°N)")
plt.title(f"Bergy Bits and Growlers Drift/Deterioration {time_1} - {time_2}", fontsize=12, fontweight='bold')
plt.legend()
plt.grid(True)
# plt.savefig("bergy_water_outer_boundaries_ranges.png", dpi=300, bbox_inches="tight")
plt.show()


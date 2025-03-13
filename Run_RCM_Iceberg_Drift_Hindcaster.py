
from RCM_Iceberg_Drift_Hindcaster import *
import time
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
# import datetime
import warnings
warnings.simplefilter(action='ignore')

def plot_iceberg_tracks(iceberg_lons, iceberg_lats, iceberg_lengths, iceberg_times):
    """
    Plots iceberg tracks, color-coding segments by 5m ranges of iceberg length.

    Parameters:
    - iceberg_lons: 2D (time-steps x iceberg number) or 1D (time-steps,) array of iceberg longitudes
    - iceberg_lats: 2D (time-steps x iceberg number) or 1D (time-steps,) array of iceberg latitudes
    - iceberg_lengths: 2D (time-steps x iceberg number) or 1D (time-steps,) array of iceberg lengths
    - iceberg_times: List of strings representing timestamps in ISO format (e.g., '2025-02-06T16:36:48')
    """
    # Convert times to numpy datetime64
    iceberg_times = np.array(iceberg_times, dtype="datetime64[s]")

    # Ensure 2D arrays for consistency
    if iceberg_lons.ndim == 1:
        iceberg_lons = iceberg_lons[:, np.newaxis]
        iceberg_lats = iceberg_lats[:, np.newaxis]
        iceberg_lengths = iceberg_lengths[:, np.newaxis]

    num_icebergs = iceberg_lons.shape[1]  # Number of icebergs
    cmap = plt.get_cmap("viridis")  # Color map for length ranges
    min_length = np.nanmin(iceberg_lengths)
    max_length = np.nanmax(iceberg_lengths)
    norm = mcolors.BoundaryNorm(boundaries=np.arange(min_length, max_length + 5, 5), ncolors=256)

    plt.figure(figsize=(10, 8))

    all_segments = []
    all_colors = []
    all_lons = []
    all_lats = []

    for k in range(num_icebergs):
        segments = []
        colors = []

        for i in range(iceberg_lons.shape[0] - 1):  # Loop through time steps
            if np.isnan(iceberg_lengths[i, k]) or np.isnan(iceberg_lengths[i + 1, k]):
                continue  # Skip invalid data

            # Define the line segment
            segment = [(iceberg_lons[i, k], iceberg_lats[i, k]), (iceberg_lons[i + 1, k], iceberg_lats[i + 1, k])]
            segments.append(segment)
            length_range = iceberg_lengths[i, k]
            colors.append(cmap(norm(length_range)))

            # Collect all longitude/latitude points for axis limits
            all_lons.extend([iceberg_lons[i, k], iceberg_lons[i + 1, k]])
            all_lats.extend([iceberg_lats[i, k], iceberg_lats[i + 1, k]])

        all_segments.extend(segments)
        all_colors.extend(colors)

        # Plot initial positions as red dots
        plt.scatter(iceberg_lons[0, k], iceberg_lats[0, k], color="red", edgecolor="black", s=100, zorder=10)

    # Create a LineCollection with colors mapped to iceberg lengths
    lc = LineCollection(all_segments, colors=all_colors, linewidth=2, cmap=cmap, norm=norm)
    plt.gca().add_collection(lc)
    plt.xlim(min(all_lons) - 0.05, max(all_lons) + 0.05)
    plt.ylim(min(all_lats) - 0.05, max(all_lats) + 0.05)
    plt.xlabel("Longitude (°E)")
    plt.ylabel("Latitude (°N)")
    time_1 = np.max(iceberg_times)
    time_2 = np.min(iceberg_times)
    plt.title(f"Hindcast Iceberg Tracks {time_1} - {time_2}", fontsize=12, fontweight='bold')
    plt.colorbar(lc, label="Iceberg Waterline Length (m)")
    plt.grid(True)
    # plt.savefig("hindcast_iceberg_tracks.png", dpi=300, bbox_inches="tight")
    plt.show()

start_time = time.time()

iceberg_lats0 = [54.6, 54.6]
iceberg_lons0 = [-55.625, -55.55]
# iceberg_lats0 = 54.6
# iceberg_lons0 = -55.625
# rcm_datetime0 = np.datetime64(datetime.datetime.now(datetime.timezone.utc)) # - np.timedelta64(12, 'h')
rcm_datetime0 = np.datetime64('2025-02-07T16:00:00')
previous_time = rcm_datetime0 - np.timedelta64(24, 'h') # + np.timedelta64(23, 'm')
metocean_folder_date = '2025-02-06'
iceberg_lengths0 = [50., 67.]
# iceberg_lengths0 = 67.
iceberg_ids = ['0000', '0001']
# iceberg_ids = '0000'
iceberg_grounded_statuses0 = [False, False]
# iceberg_grounded_statuses0 = False
si_toggle = True

obs = Observations(iceberg_lats0, iceberg_lons0, rcm_datetime0, iceberg_lengths0, iceberg_grounded_statuses0, [False, False], iceberg_ids)
# obs = Observation(iceberg_lats0, iceberg_lons0, rcm_datetime0, iceberg_lengths0, iceberg_grounded_statuses0, False, iceberg_ids)
iceberg_times, iceberg_lats, iceberg_lons, iceberg_lengths, iceberg_grounded_statuses = rcm_iceberg_drift_hindcaster(obs, previous_time, si_toggle, metocean_folder_date)

print(iceberg_times)
print(iceberg_lats)
print(iceberg_lons)
print(iceberg_lengths)
print(iceberg_grounded_statuses)

end_time = time.time()

# Calculate the elapsed time
elapsed_time = (end_time - start_time) / 60.
print(f"Script runtime: {elapsed_time:.2f} minutes.")
# print(f"Script runtime: {elapsed_time * 60.:.2f} seconds.")

plot_iceberg_tracks(iceberg_lons, iceberg_lats, iceberg_lengths, iceberg_times)


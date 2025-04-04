
from RCM_Iceberg_Drift_Hindcaster import *
import shapefile
import pandas as pd
import os
import time
import warnings
warnings.simplefilter(action='ignore')

start_time = time.time()

# Function to read an existing shapefile into a Pandas DataFrame
def read_shapefile_to_dataframe(shapefile_path):
    sf = shapefile.Reader(shapefile_path)
    fields = [field[0] for field in sf.fields[1:]] # Skip DeletionFlag
    records = [dict(zip(fields, record)) for record in sf.records()]
    shapes = [s.points[0] for s in sf.shapes()] # Extract first point per shape
    for i, (lon, lat) in enumerate(shapes):
        records[i]["longitude"] = lon
        records[i]["latitude"] = lat
    return pd.DataFrame(records)

# Function to write a DataFrame to a shapefile using PyShp
def write_dataframe_to_shapefile(df, shapefile_path):
    with shapefile.Writer(shapefile_path) as shp:
        shp.field("iceberg_id", "N")
        shp.field("time", "C", 25)
        shp.field("length", "F", 10, 5)
        shp.field("grounded", "N")

        for _, row in df.iterrows():
            shp.point(row["longitude"], row["latitude"])
            shp.record(row["iceberg_id"], row["time"], row["length"], row["grounded"])

    # Save projection file (.prj) for EPSG:4326
    with open(shapefile_path.replace(".shp", ".prj"), "w") as prj:
        prj.write('GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],''PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]')

# Path to your shapefile (will create or append to this)
shapefile_path = "C:/Users/idturnbull/Documents/MATLAB/ExxonMobil_RCM_Project_Analysis/20250403_132200_tracks/iceberg_hindcast_output.shp"

# Check if the file exists
shapefile_exists = os.path.isfile(shapefile_path)

rcm_datetime0_list = ['2025-04-03T13:22:00', '2025-04-03T13:31:00', '2025-04-03T13:10:00', '2025-04-03T13:09:00', '2025-04-03T13:09:00',
                      '2025-04-03T13:14:00', '2025-04-03T13:14:00', '2025-04-03T13:31:00', '2025-04-03T13:31:00', '2025-04-03T13:25:00',
                      '2025-04-03T13:31:00', '2025-04-03T13:31:00', '2025-04-03T13:23:00', '2025-04-03T13:25:00', '2025-04-03T13:31:00',
                      '2025-04-03T13:31:00']
next_rcm_time_list = ['2025-04-04T13:22:00', '2025-04-04T13:31:00', '2025-04-04T13:10:00', '2025-04-04T13:09:00', '2025-04-04T13:09:00',
                      '2025-04-04T13:14:00', '2025-04-04T13:14:00', '2025-04-04T13:31:00', '2025-04-04T13:31:00', '2025-04-04T13:25:00',
                      '2025-04-04T13:31:00', '2025-04-04T13:31:00', '2025-04-04T13:23:00', '2025-04-04T13:25:00', '2025-04-04T13:31:00',
                      '2025-04-04T13:31:00']
iceberg_lats0_list = [49.7616, 49.7393, 50.3075, 50.3921, 50.1107, 50.1695, 50.3290, 50.2020, 50.3203, 50.5116, 50.0858, 50.0230, 50.5984, 50.4984, 50.0158, 50.3192]
iceberg_lons0_list = [-48.8603, -49.8658, -48.5382, -48.5881, -48.0356, -48.9316, -49.3904, -49.9956, -50.1222, -50.2040, -50.1807, -50.1269, -50.2389, -50.2806,
                      -50.4641, -50.3450]
iceberg_ids_list = ['0000', '0001', '0002', '0003', '0004', '0005', '0006', '0007', '0008', '0009', '0010', '0011', '0012', '0013', '0014', '0015']
metocean_folder_date = '2025-04-03'

for m in range(len(rcm_datetime0_list)):
    iceberg_lats0 = iceberg_lats0_list[m]
    iceberg_lons0 = iceberg_lons0_list[m]
    previous_time = np.datetime64(rcm_datetime0_list[m])
    rcm_datetime0 = np.datetime64(next_rcm_time_list[m])
    iceberg_lengths0 = 100.
    iceberg_ids = iceberg_ids_list[m]
    iceberg_grounded_statuses0 = False
    si_toggle = False
    obs = Observation(iceberg_lats0, iceberg_lons0, rcm_datetime0, iceberg_lengths0, iceberg_grounded_statuses0, False, iceberg_ids)
    iceberg_times, iceberg_lats, iceberg_lons, iceberg_lengths, iceberg_grounded_statuses = rcm_iceberg_drift_hindcaster(obs, previous_time, si_toggle, metocean_folder_date)

    # Create a Pandas DataFrame
    new_df = pd.DataFrame({"iceberg_id": iceberg_ids, "time": iceberg_times, "latitude": iceberg_lats, "longitude": iceberg_lons, "length": iceberg_lengths,
                           "grounded": iceberg_grounded_statuses})

    # Read existing shapefile if it exists
    if shapefile_exists:
        existing_df = read_shapefile_to_dataframe(shapefile_path)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined_df = new_df
        shapefile_exists = True # Ensure we don't overwrite in future loops

    # Write the updated data to a shapefile
    write_dataframe_to_shapefile(combined_df, shapefile_path)

end_time = time.time()

# Calculate the elapsed time
elapsed_time = (end_time - start_time) / 60.
print(f"Script runtime: {elapsed_time:.2f} minutes.")


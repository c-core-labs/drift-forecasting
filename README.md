Title: Documentation for RCM_Iceberg_Drift_Forecaster_IDT.py
Author: Ian D. Turnbull

Summary:

The RCM_Iceberg_Drift_Forecaster_IDT.py Python script file contains a function called rcm_iceberg_drift_forecaster that is callable in Coresight and runs on a physics-based iceberg drift model. The function takes the following inputs taken from a RADARSAT Constellation Mission (RCM) satellite image: an iceberg latitude/longitude position (iceberg_lat0, iceberg_lon0), the date/time of the RCM image acquisition (rcm_datetime0), the iceberg length (iceberg_length), its status as grounded or not (grounded_status), and the anticipated time of the next RCM image acquisition (next_rcm_time). The function forecasts the iceberg’s new position at some user-specified time in the future assumed to align with the date/time of the next RCM image acquisition. The function will output hourly forecast iceberg latitude/longitude positions up to the next RCM image acquisition date/time. The final time step will be the remainder between the last hourly time step and the next RCM image acquisition date/time if it is less than one hour. This document describes the Python libraries needed to run the function, how the function works, and its key operating assumptions.

Depending on the iceberg drift forecast period, assumed to be on the order of 12-24 hours between RCM image acquisitions, the function can take on the order of 20-40 minutes to complete an iceberg drift forecast. This is mainly because of the time it takes to download and process the necessary ocean forecast files (more details on this are provided below under “How the Function Works”).

Python Libraries Used in Current Version of the Function:

scipy v1.13.1
tempfile
subprocess
numpy v1.26.4
netCDF4 v1.6.2
requests v2.32.3
shutil
os

How the Function Works:

The rcm_iceberg_drift_forecaster function requires the following inputs:

•	An iceberg latitude/longitude position (iceberg_lat0, iceberg_lon0 in degrees North and East, respectively),
•	The date/time of the RCM image acquisition (rcm_datetime0),
•	The iceberg waterline length (iceberg_length),
•	Its status as grounded or not (grounded_status: this should be entered as “grounded” or “not grounded”), and
•	The anticipated time of the next RCM image acquisition (next_rcm_time).

The path to the directory containing the wgrib2.exe tool and associated files for converting grib2 files to netCDF format (wgrib_path) and the path to the bathymetric data file in netCDF format (bathy_data_path) are currently hard coded into the function in the top few lines. These can be changed as needed. The iceberg waterline length is assumed to be in meters. The date/time of the RCM image acquisition (rcm_datetime0) and the anticipated date/time of the next RCM image acquisition (next_rcm_time) should be input as strings in numpy datetime64 format and Universal Time Coordinated (UTC), e.g., for example: ‘2024-11-01T09:42:53.33’.

1.	The very first line of the function, use_temporary_directory = True, sets up the function to store all metocean forecast files in a temporary directory and then delete them once the iceberg drift forecast has been completed. This is a convenient option for running the function on Coresight in the cloud, where dedicated local storage for these files may unavailable. If this is set to False, the function will create local directories for storing the metocean forecast files. Next, the paths to the wgrib2.exe tool and the bathymetric dataset are hard coded.

2.	The next lines of the function define some constants, including densities for air and seawater, the earth gravitational acceleration and angular velocity, drag coefficients for air, water, and wave drift forcing on an iceberg, added mass fraction of an iceberg, and radius of the Earth. The deg_radius constant defines how much of the grid for the ocean current and sea surface tilt forecast files will be saved. It is currently set to deg_radius = 10, meaning that these grids will be saved for points within 10 latitude/longitude points of the original iceberg position. The original grids cover the whole North Atlantic region and have 5 km spatial resolution, so this constant ensures the grids will only be saved within a few tens of km of the original iceberg position at the beginning of the drift forecast, more than enough to cover the region in which it could drift in the next 12-24 hours. This saves memory and processing time for the function.

3.	Three nested functions are defined within the rcm_iceberg_drift_forecaster function:

•	The first function calculates the distance and azimuthal bearing between two latitude/longitude coordinates,
•	The second function calculates the latitude/longitude coordinate at a given distance and course from an initial latitude/longitude coordinate, and
•	The third function calculates the easterly and northerly iceberg acceleration vector components from the sum of the forcings acting on the iceberg.

The forces considered in the function are the (e.g., see Kubat et al., 2005; and Eik, 2009):

•	Air (wind) and water (ocean current) drags,
•	Coriolis,
•	Water pressure gradient,
•	Sea surface tilt (horizontal gravitational), and
•	Wave radiation.

4.	The iceberg drift velocity components are initialized at zero and the RCM image acquisition time is converted to numpy datetime64 format. If no iceberg waterline length is given, the function will assume it is 100 m.

Since the iceberg’s waterline length is the only physical dimension extracted from an RCM image, the iceberg draft, mass, and sail cross-sectional area are calculated from this variable alone. The iceberg draft is calculated next as a function of the iceberg waterline length using the formulation obtained from analysis of C-CORE iceberg profile data. The iceberg mass and cross-sectional sail area are calculated as a function of waterline length from formulations given in Barker et al. (2004).

5.	The function reads the bathymetric data file and interpolates the water depth at the iceberg location. If the iceberg’s status has been entered into the function as “not grounded” but the calculated draft is equal to or greater than the water depth, the iceberg draft is reset to be one meter less than the water depth. If the iceberg’s status is grounded, the function stops here and returns the forecast iceberg latitude/longitude coordinate as its original location, e.g., the iceberg is assumed to remain stationary for the forecast period. The function also returns zero net displacement of the iceberg over the forecast period.

6.	The iceberg drift forecast times are determined. The drift forecast is initialized at the RCM image acquisition date/time, and then hourly time-steps are used until the next RCM image acquisition date/time is reached. The final time-step between the last hourly time-step and the next RCM image acquisition date/time may be less than one hour.

7.	Arrays are initialized for the hourly forecast iceberg latitude/longitude positions.

8.	If the iceberg is not grounded, the next piece of code checks for the latest available wind, wave, ocean current, and sea surface height forecasts that start at or earlier than the iceberg drift forecast start time. The wind velocity, significant wave height, and wave direction forecasts are obtained from the Environment and Climate Change Canada (ECCC) Global Deterministic Wave Prediction System (GDWPS) model. This model releases daily forecasts at 00:00 and 12:00 UTC out to 240 hours (10 days) at hourly resolution. Finally, the Fisheries and Oceans Canada (DFO) Regional Ice Ocean Prediction System (RIOPS) model is used to obtain ocean current velocity at multiple depths down to the seabed and sea surface height forecasts. This model covers the North Atlantic region and has a five-km and hourly spatial and temporal resolution, respectively. The model issues forecasts four times daily at 00:00, 06:00, 12:00, 18:00, and 21:00 UTC out to 84 hours or 3.5 days. This model limits all iceberg drift forecasts to 3.5 days or less.

9.	The forecast times for wind, wave, ocean current, and sea surface height data that fully overlap with the iceberg drift forecast period are determined.

10.	The forecast files for the wind, waves, ocean currents, and sea surface heights are obtained and placed in a temporary directory. Each file contains the full global (wind and waves) or regional (ocean currents and sea surface height) grid and represents one point in time. The wind and wave files are converted from grib2 to netCDF file format using the wgrib tool. The RIOPS ocean current and sea surface height files are already in netCDF format. Note that downloading the RIOPS model files could take some time (on the order of 20-25 minutes depending on the iceberg drift forecast period) and because each hourly file is on the order of 165 MB. The files must first be fully downloaded in order to be read because the RIOPS server does not support OPeNDAP; e.g., the data cannot be read directly from the files on the server.

11.	The ocean current files are stripped down to only retain data within 10 grid points of the original iceberg position. At five-km resolution, this represents around 50 km, which should be more than enough to cover iceberg drift over the next 12-24 hours. This parameter is the deg_radius set at the beginning of (but inside) the function and can be increased if needed. This step is performed to save significant processing time. Using the whole RIOPS grid for the ocean current and sea surface height files was found to greatly increase the function’s runtime to over an hour in many cases due to the more intensive interpolation processes on a larger grid, plus the extra time needed to compute the larger sea surface height gradient grids. The stripped-down forecast sea surface height gradients are also computed in this step and saved into their own netCDF files.

12.	The latitude-longitude grids are loaded for the winds, waves, ocean currents, and sea surface height gradients. The maximum depth of the ocean currents is determined based on the calculated iceberg draft.

13.	Arrays are initialized for the hourly forecast iceberg zonal/meridional drift velocities.

14.	The iceberg drift forecast loop begins. The iceberg position and drift velocity components are set in the first time-step of their respective forecast arrays with the initial iceberg position and drift velocity at the RCM image acquisition date/time.

15.	At each time-step, the forecast wind, wave, ocean current, and sea surface height gradient files are found that align with times just before and after the current iceberg forecast time, or that align exactly with the current iceberg forecast time if they exist.

16.	The forecast wind velocity components, wave parameters, ocean current velocity components, and sea surface height gradient components from each of the before and after files are spatially interpolated to the iceberg forecast position at the present time-step. Ocean current velocities are only processed to a depth equal to or just greater than the calculated iceberg draft. The ocean current velocities are then averaged over the draft depth of the iceberg. The linearly interpolated value of each metocean variable in time is then calculated for the present time-step. Wave variables are interpolated using nearest neighbor interpolation.

17.	The iceberg zonal and meridional acceleration components are computed at the present time-step, and an implicit, unconditionally stable numerical integration algorithm is used to determine the iceberg drift velocity at the next time-step. The iceberg position at the next time-step is then calculated using the Euler forward integration algorithm, integrated using drift velocities averaged between the present and next time-steps.

18.	At each time-step, the water depth is checked at the forecast iceberg location. If the water depth is equal to or less than the iceberg draft, the iceberg drift is halted at that point the iceberg’s status is changed to “grounded.”

19.	The function will finally return the following variables:

•	The original (at the RCM image acquisition date/time) iceberg latitude/longitude coordinates (in degrees North and East, respectively),
•	The hourly forecast iceberg latitude/longitude (in degrees North and East, respectively) positions up to the next RCM image acquisition date/time,
•	The times at which the iceberg positions are forecast (in numpy datetime64 format),
•	The forecast net iceberg displacement (meters) and drift direction (azimuthal degrees),
•	The iceberg waterline length (meters), calculated draft (meters), and calculated mass (tonnes),
•	The forecast beginning and end times (UTC) corresponding to the present RCM image acquisition date/time and the next anticipated image acquisition date/time, respectively, and
•	The final iceberg grounded status.

Key Operating Assumptions:

1.	The bathymetric data file is assumed to be in netCDF format and contain three variables labelled “lat”, “lon”, and “elevation”. The elevation variable is assumed to be negative for water depth and have the dimensions latitude by longitude.

2.	The RCM image acquisition date/time must be within the last day as it is assumed the iceberg forecast start time aligns with presently available metocean forecast data.

3.	The next RCM image acquisition time input to the function must be within the next 3.5 days to remain within the forecast period for the ocean currents and sea surface heights.

4.	The iceberg drift velocity is initialized in the forecast at zero.

5.	The icebergs are in open water and not in sea ice; the current version of the model does not include sea ice forcing on iceberg drift.

6.	Icebergs are entered into the function one at a time.

7.	Iceberg keel cross-sectional areas are assumed to be rectangular.

References:

Barker, A., Sayed, M., and Carrieres, T. (2004). “Determination of Iceberg Draft, Mass and Cross-Sectional Areas,” In Proceedings of the 14th International Offshore and Polar Engineering Conference (ISOPE), Toulon, France, May 23-28.

Eik, K. (2009). “Iceberg drift modelling and validation of applied metocean hindcast data,” Cold Regions Science and Technology, vol. 57, pp. 67-90.

GDPS. (2024). “Global Deterministic Prediction System (GDPS) data in GRIB2 format,” Accessed on the World Wide Web at: https://eccc-msc.github.io/open-data/msc-data/nwp_gdps/readme_gdps-datamart_en/, November 1, 2024.

GDWPS. (2024). “Global Deterministic Wave Prediction System (GDWPS) data in GRIB2 format,” Accessed on the World Wide Web at: https://eccc-msc.github.io/open-data/msc-data/nwp_gdwps/readme_gdwps-datamart_en/, November 1, 2024.

Kubat, I., Sayed, M., Savage, S.B., and Carrieres, T. (2005). “An Operational Model of Iceberg Drift,” International Journal of Offshore and Polar Engineering (IJOPE), vol. 15, no. 2, pp. 125-131.

RIOPS. (2024). “Regional Ice Ocean Prediction System (RIOPS) Data in NetCDF Format,” Accessed on the World Wide Web at: https://eccc-msc.github.io/open-data/msc-data/nwp_riops/readme_riops-datamart_en/, November 1, 2024.

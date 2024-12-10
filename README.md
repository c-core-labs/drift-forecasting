Title: Documentation for RCM_Iceberg_Drift_Deterioration_Forecaster.py
Author: Ian D. Turnbull

Summary:

The RCM_Iceberg_Drift_Deterioration_Forecaster.py Python script file contains a function called rcm_iceberg_drift_deterioration_forecaster and runs on a physics-based iceberg drift and deterioration model. The function takes the following inputs taken from a RADARSAT Constellation Mission (RCM) satellite image: a series of iceberg latitude/longitude positions (iceberg_lats0, iceberg_lons0), the date/time of the RCM image acquisition (rcm_datetime0), the iceberg lengths (iceberg_lengths0), its status as grounded or not (iceberg_grounded_statuses0), the iceberg identification numbers (iceberg_ids), the anticipated time of the next RCM image acquisition (next_rcm_time), and hours at which the metocean forecast data were issued, in string format and Universal Time Coordinated (UTC), for the air temperature and solar radiation, the winds and waves, and the ocean variables (hour_utc_str_airT_sw_rad, hour_utc_str_wind_waves, hour_utc_str_ocean, respectively). The iceberg initial positions, waterline lengths, and grounded statuses should be passed to the function as either lists or numpy arrays. The function forecasts the icebergs’ new positions and waterline lengths at some user-specified time in the future assumed to align with the date/time of the next RCM image acquisition. The function will output hourly forecast iceberg latitude/longitude position and waterline lengths up to the next RCM image acquisition date/time. The final time step will be the remainder between the last hourly time step and the next RCM image acquisition date/time if it is less than one hour. This document lists the Python libraries needed to run the function, describes how the function works, lists its key operating assumptions, and provides an overview of the model’s performance on icebergs tracked in RCM imagery during 2023-2024.

Depending on the iceberg drift forecast period and the number of icebergs to be forecast, assumed to be on the order of 12-24 hours between RCM image acquisitions, the function can take on the order of 20 minutes to complete drift and deterioration forecasts for a set of icebergs.

Python Libraries Used in Current Version of the Function:

scipy v1.13.1
tempfile
numpy v1.26.4
netCDF4 v1.6.2
gsw v3.6.19

How the Function Works:

The rcm_iceberg_drift_deterioration_forecaster function requires the following inputs:

•	The path to the bathymetric data file, assumed to be in NetCDF format (called gebco_2024.nc),
•	The root path to the metocean forecast data files,
•	A set of iceberg latitude/longitude positions in array-like format (iceberg_lats0, iceberg_lons0 in degrees North and East, respectively),
•	The date/time of the RCM image acquisition (rcm_datetime0),
•	The initial iceberg waterline lengths (iceberg_lengths0) in array-like format,
•	The icebergs’ initial statuses as grounded or not (iceberg_grounded_statuses0: these should be entered as a list or array of “grounded” or “not grounded”), and
•	The anticipated time of the next RCM image acquisition (next_rcm_time).

The iceberg waterline lengths are assumed to be in meters. The date/time of the RCM image acquisition (rcm_datetime0) and the anticipated date/time of the next RCM image acquisition (next_rcm_time) should be input as strings in numpy datetime64 format and UTC, e.g., for example: ‘2024-11-01T09:42:53.33’.

1.	The first lines of the function define some constants, including densities for air, seawater, and ice, ice albedo, latent heat of fusion for ice, the earth gravitational acceleration and angular velocity, drag coefficients for air, water, and wave drift forcing on an iceberg, added mass fraction of an iceberg, and radius of the Earth. The deg_radius constant defines how much of the grid for the ocean current, sea surface tilt, and ocean potential temperature and salinity forecast files will be saved. It is currently set to deg_radius = 30, meaning that these grids will be saved for points within 30 latitude/longitude points of the southernmost, northernmost, easternmost, and westernmost initial positions for a set of icebergs. The original grids cover the whole North Atlantic region and have 5 km spatial resolution, so this constant ensures the grids will only be saved within about 150 km of the southernmost, northernmost, easternmost, and westernmost initial iceberg positions at the beginning of the drift forecast, more than enough to cover the region in which they could drift in the next 12-24 hours. This saves memory and processing time for the function.

2.	Five nested functions are defined within the rcm_iceberg_drift_deterioration_forecaster function:

•	The first function calculates the distance and azimuthal bearing between two latitude/longitude coordinates,
•	The second function calculates the latitude/longitude coordinate at a given distance and course from an initial latitude/longitude coordinate,
•	The third function calculates the easterly and northerly iceberg acceleration vector components from the sum of the forcings acting on the iceberg,
•	The fourth function is for numerically solving the iceberg acceleration integration to drift velocity, and
•	The fifth function calculates the new iceberg waterline length, draft, sail cross-sectional area, and total mass after deterioration.

The forces considered in the function for iceberg drift are the (e.g., see Kubat et al., 2005; and Eik, 2009):

•	Air (wind) and water (ocean current) drags,
•	Coriolis,
•	Water pressure gradient,
•	Sea surface tilt (horizontal gravitational), and
•	Wave radiation.

The forcings considered in the function for iceberg deterioration are the (e.g., see Kubat et al., 2007):

•	Surface (freeboard) melting due to solar radiation,
•	Melting of the keel due to buoyant vertical convection,
•	Melting of the keel and freeboard due to forced convection of water and air, respectively, and
•	Calving due to wave erosion.

3.	The rcm_datetime0 (the RCM image acquisition time) and next_rcm_time are ensured to be in numpy datetime64 format and the forecast start time for a set of icebergs passed to the function is set to the rcm_datetime0.

4.	The function reads the bathymetric data file and interpolates the water depth at the iceberg location. If the iceberg’s status has been entered into the function as “not grounded” but the calculated draft is equal to or greater than the water depth, the iceberg draft is reset to be one meter less than the water depth.

5.	If no iceberg waterline length is given for an iceberg, the function will assume it is 100 m. Since the iceberg’s waterline length is the only physical dimension extracted from an RCM image, the iceberg draft, mass, and sail cross-sectional area are calculated from this variable alone. The iceberg draft is calculated as a function of the iceberg waterline length using the formulation obtained from analysis of C-CORE iceberg profile data. The iceberg mass and cross-sectional sail area are calculated as a function of waterline length from formulations given in Barker et al. (2004).

6.	The iceberg drift forecast times are determined. The drift forecast is initialized at the RCM image acquisition date/time, and then hourly time-steps are used until the next RCM image acquisition date/time is reached. The final time-step between the last hourly time-step and the next RCM image acquisition date/time may be less than one hour.

7.	Arrays are initialized for the hourly forecast iceberg latitude/longitude positions, drift velocities, waterline lengths, drafts, and masses.

8.	The forecast times for wind, wave, ocean current, sea surface height, ocean potential temperature, ocean salinity, air temperature, and surface solar radiation data that fully overlap with the iceberg drift forecast period are determined.

9.	The ocean current, potential temperature, and salinity files are stripped down to only retain data within 30 grid points of the original iceberg position. At 5 km resolution, this represents around 150 km, which should be more than enough to cover iceberg drift over the next 12-24 hours. This parameter is the deg_radius set at the beginning of (but inside) the function and can be increased if needed. This step is performed to save significant processing time. Using the whole RIOPS grid for the ocean current and sea surface height files was found to greatly increase the function’s runtime to over an hour in many cases due to the more intensive interpolation processes on a larger grid, plus the extra time needed to compute the larger sea surface height gradient grids. The stripped-down forecast sea surface height gradients are also computed in this step and saved into their own netCDF files. These stripped-down files are placed in a temporary directory.

10.	The iceberg drift and deterioration forecast loop begins. At each time-step, the forecast wind, wave, ocean current, potential temperature, salinity, sea surface height gradient, air temperature, and surface solar radiation files are found that align with times just before and after the current iceberg forecast time, or that align exactly with the current iceberg forecast time if they exist.

11.	The forecast wind velocity components, wave parameters, ocean current velocity components, potential temperatures, salinities, sea surface height gradient components, air temperatures, and surface solar radiation from each of the before and after files are spatially interpolated to the iceberg forecast position at the present time-step. Ocean current velocities, potential temperatures, and salinities are only processed to a depth equal to or just greater than the calculated iceberg draft. The ocean current velocities are then averaged over the draft depth of the iceberg. The linearly interpolated value of each metocean variable in time is then calculated for the present time-step. Wave variables are interpolated using nearest neighbor interpolation.

12.	The iceberg zonal and meridional acceleration components are computed at the present time-step, and an implicit, unconditionally stable numerical integration (backward differentiation) algorithm is used to determine the iceberg drift velocity at the next time-step. The iceberg displacement at the next time-step is then calculated using the Euler forward integration algorithm, integrated using drift velocities averaged between the present and next time-steps (e.g., trapezoidal integration).

13.	At each time-step, the water depth is checked at the forecast iceberg location. If the water depth is equal to or less than the iceberg draft, the iceberg drift is halted at that point and the iceberg’s status is changed to grounded.

14.	The iceberg deterioration between the present and next time-step is calculated. If an iceberg deteriorates completely, the iceberg mass, draft, sail area, and waterline length will be set to zero, and the iceberg drift will stop. A warning will print out if an iceberg is predicted to deteriorate to less than 40 m waterline length, as this iceberg may fall below the minimum spatial resolution of RCM imagery.

15.	The function will finally return the following variables:

•	The hourly forecast iceberg latitude/longitude (in degrees North and East, respectively) positions up to the next RCM image acquisition date/time,
•	The hourly forecast iceberg waterline lengths (in meters) up to the next RCM image acquisition date/time,
•	The hourly forecast iceberg grounded statuses (as one for grounded or zero for not grounded), and
•	The times at which the iceberg positions are forecast (in numpy datetime64 format).

Key Operating Assumptions:

1.	The bathymetric data file is assumed to be in netCDF format and contain three variables labelled “lat”, “lon”, and “elevation”. The elevation variable is assumed to be negative for water depth and have the dimensions latitude by longitude.

2.	The next RCM image acquisition time input to the function must be within the next 3.5 days to remain within the forecast period for the ocean currents and sea surface heights.

3.	The iceberg drift velocities are initialized in the forecast at zero.

4.	The icebergs are in open water and not in sea ice; the current version of the model does not include sea ice forcing on iceberg drift.

5.	Iceberg keel cross-sectional areas are assumed to be rectangular.

Model Performance:

The model performance was assessed on RCM-tracked icebergs over the 2023-2024 seasons. A total of 340 individual iceberg displacements between RCM images were used. Historical winds and waves were obtained from the ERA5 reanalysis (Hersbach et al., 2023) and historical ocean currents, temperatures, salinities, and sea surface heights were obtained from the Hybrid Coordinate Ocean Model (HYCOM, 2024) reanalysis. Hourly observed iceberg positions were linearly interpolated between RCM observations. The average position error at 12 hours is 8.9 km and the average position error at 24 hours is 18.1 km. There were only 10 instances in the 2023-2024 data in which iceberg waterline length decreased between observations. The deterioration model tends to underestimate the iceberg deterioration. This is likely at least partly due to the resolution of the metocean forcing data being too coarse to capture the wave-induced calving events.

References:

Barker, A., Sayed, M., and Carrieres, T. (2004). “Determination of Iceberg Draft, Mass and Cross-Sectional Areas,” In Proceedings of the 14th International Offshore and Polar Engineering Conference (ISOPE), Toulon, France, May 23-28.

Eik, K. (2009). “Iceberg drift modelling and validation of applied metocean hindcast data,” Cold Regions Science and Technology, vol. 57, pp. 67-90.

GDPS. (2024). “Global Deterministic Prediction System (GDPS) data in GRIB2 format,” Accessed on the World Wide Web at: https://eccc-msc.github.io/open-data/msc-data/nwp_gdps/readme_gdps-datamart_en/, November 1, 2024.

GDWPS. (2024). “Global Deterministic Wave Prediction System (GDWPS) data in GRIB2 format,” Accessed on the World Wide Web at: https://eccc-msc.github.io/open-data/msc-data/nwp_gdwps/readme_gdwps-datamart_en/, November 1, 2024.

Hersbach, H., Bell, B., Berrisford, P., Biavati, G., Horányi, A., Muñoz Sabater, J., Nicolas, J., Peubey, C., Radu, R., Rozum, I., Schepers, D., Simmons, A., Soci, C., Dee, D., and Thépaut, J-N. (2023). “ERA5 hourly data on single levels from 1940 to present,” Copernicus Climate Change Service (C3S) Climate Data Store (CDS), DOI: 10.24381/cds.adbb2d47 (Accessed on November 18, 2024).

HYCOM. (2024). “Hybrid Coordinate Ocean Model Global Ocean Forecast System (GOFS) 3.1,” Accessed on the World Wide Web at: https://www.hycom.org/dataserver/gofs-3pt1/analysis, November 18, 2024.

Kubat, I., Sayed, M., Savage, S.B., and Carrieres, T. (2005). “An Operational Model of Iceberg Drift,” International Journal of Offshore and Polar Engineering (IJOPE), vol. 15, no. 2, pp. 125-131.

Kubat, I., Sayed, M., Savage, S., Carrieres, T., and Crocker, G. (2007). “An Operational Iceberg Deterioration Model,” Proceedings of the 16th International Offshore and Polar Engineering Conference (ISOPE), Lisbon, Portugal, July 1-6.

RIOPS. (2024). “Regional Ice Ocean Prediction System (RIOPS) Data in NetCDF Format,” Accessed on the World Wide Web at: https://eccc-msc.github.io/open-data/msc-data/nwp_riops/readme_riops-datamart_en/, November 1, 2024.

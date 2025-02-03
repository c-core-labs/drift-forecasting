Title: Documentation for Get_RCM_Iceberg_Metocean_Data.py
Author: Ian D. Turnbull

Summary:

The Get_RCM_Iceberg_Metocean_Data.py Python script downloads the latest available metocean forecast data needed to run the rcm_iceberg_drift_deterioration_forecaster function contained in the RCM_Iceberg_Drift_Deterioration_Forecaster.py script and the rcm_bergy_bit_growler_forecaster function contained in the RCM_Bergy_Bit_Growler_Forecaster.py script. If the maximum number of forecast hours are entered, the script will take approximately two hours to download the daily forecast files for air temperature, solar radiation, wind velocity, significant wave height, wave direction, mean wave period, ocean potential temperature, salinity, current velocity, and sea surface height. This will require about 43 GB of storage memory. If forecast sea ice data are also requested due to the presence of sea ice, it will take nearly three hours to download 84 hours of data and will require about 46.5 GB of storage memory.

Python Libraries Used in Current Version of the Script:

numpy v1.26.4
requests v2.32.3
xarray v2023.6.0
cfgrib v0.9.15.0
os

How the Script Works:

The script requires only the following four inputs:

•	The root path to be created to store the metocean data (rootpath_to_metdata),
•	The number of forecast hours to be obtained (forecast_hours):
o	This must be at least 3 and not more than 84 since the air temperature and solar radiation data are provided at 3-hour intervals and the ocean data are provided out to 84 hours (3.5 days), and
•	Whether forecast sea ice data should be downloaded (si_toggle=True or False).

1.	The script creates the root path to store the metocean data and then creates three subdirectories for the air temperature and solar radiation data, the wind and wave data, and the ocean data. A directory for today in the format yyyy-mm-dd is created in each subdirectory.

2.	The script checks for the latest available metocean forecast data from each of the three models. Air temperature and solar radiation data are obtained from the Environment and Climate Change Canada (ECCC) Global Deterministic Prediction System (GDPS) which as a 15-km and three-hourly spatial and temporal resolution, respectively. Forecasts are issued twice daily at 00:00 and 12:00 Universal Time Coordinated (UTC). Data for significant wave height, wave direction, mean wave period, and wind velocity are obtained from the ECCC Global Deterministic Wave Prediction System (GDWPS) which as a 25-km and hourly spatial and temporal resolution, respectively. Forecasts are issued twice daily at 00:00 and 12:00 UTC. Data for ocean current velocity, potential temperature, salinity, sea surface height, and sea ice concentration, thickness, and zonal and meridional velocity are obtained from the Fisheries and Oceans Canada (DFO) Regional Ice Ocean Prediction System (RIOPS) which has a 5-km and hourly spatial and temporal resolution, respectively. Forecasts are issued four times daily at 00:00, 06:00, 12:00, and 18:00 UTC.

3.	The grib2 files for the atmospheric, wave, and wind data are converted to NetCDF format. The RIOPS ocean data are already in NetCDF format.

References:

GDPS. (2024). “Global Deterministic Prediction System (GDPS) data in GRIB2 format,” Accessed on the World Wide Web at: https://eccc-msc.github.io/open-data/msc-data/nwp_gdps/readme_gdps-datamart_en/, November 1, 2024.

GDWPS. (2024). “Global Deterministic Wave Prediction System (GDWPS) data in GRIB2 format,” Accessed on the World Wide Web at: https://eccc-msc.github.io/open-data/msc-data/nwp_gdwps/readme_gdwps-datamart_en/, November 1, 2024.

RIOPS. (2024). “Regional Ice Ocean Prediction System (RIOPS) Data in NetCDF Format,” Accessed on the World Wide Web at: https://eccc-msc.github.io/open-data/msc-data/nwp_riops/readme_riops-datamart_en/, November 1, 2024.

Title: Documentation for RCM_Iceberg_Drift_Deterioration_Forecaster.py
Author: Ian D. Turnbull

Summary:

The RCM_Iceberg_Drift_Deterioration_Forecaster.py Python script file contains a function called rcm_iceberg_drift_deterioration_forecaster and runs on a physics-based iceberg drift and deterioration model. The function takes the following inputs taken from a RADARSAT Constellation Mission (RCM) satellite image: a series of iceberg latitude/longitude positions, the date/time of the RCM image acquisition, the iceberg lengths, the iceberg statuses as grounded or not, the iceberg identification numbers, the anticipated time of the next RCM image acquisition, and whether sea ice is present (si_toggle). The iceberg initial positions, time, waterline lengths, id numbers, and grounded statuses should be passed to the function from an instance of the Observations class or an instance of the Observation class if for a single iceberg only. The function forecasts the icebergs’ new positions and waterline lengths at some user-specified time in the future assumed to align with the date/time of the next RCM image acquisition. The function will output hourly forecast iceberg latitude/longitude position and waterline lengths up to the next RCM image acquisition date/time. The final time step will be the remainder between the last hourly time step and the next RCM image acquisition date/time if it is less than one hour. This document lists the Python libraries needed to run the function, describes how the function works, lists its key operating assumptions, and provides an overview of the model’s performance on icebergs tracked in RCM imagery during 2023-2024.

Depending on the iceberg drift forecast period and the number of icebergs to be forecast, assumed to be on the order of 12-24 hours between RCM image acquisitions, the function can take on the order of 20 minutes to complete drift and deterioration forecasts for a set of icebergs.

Python Libraries Used in Current Version of the Function:

scipy v1.13.1
tempfile
numpy v1.26.4
netCDF4 v1.6.2
gsw v3.6.19
os

How the Function Works:

The rcm_iceberg_drift_deterioration_forecaster function requires the following inputs:

•	A set of iceberg latitude/longitude positions in array-like format (in degrees North and East, respectively) passed from an Observations class instance or a single iceberg position passed from an Observation class instance,
•	The date/time of the RCM image acquisition (passed from Observation(s) class instance),
•	The initial iceberg waterline lengths (passed from Observation(s) class instance),
•	The icebergs’ initial statuses as grounded or not (passed from Observation(s) class instance as True/False),
•	The iceberg’s id numbers (passed from Observation(s) class instance) for tracking purposes,
•	The anticipated time of the next RCM image acquisition (t1), and
•	Whether sea is present (si_toggle=True or False).

The iceberg waterline lengths are assumed to be in meters. The date/time of the RCM image acquisition (rcm_datetime0) and the anticipated date/time of the next RCM image acquisition (next_rcm_time) should be strings in numpy datetime64 format and Universal Time Coordinated (UTC), e.g., for example: ‘2024-11-01T09:42:53.33’.

1.	The first lines of the function define some constants, including densities for air, seawater, and ice, ice albedo, latent heat of fusion for ice, the earth gravitational acceleration and angular velocity, drag coefficients for air, water, and wave drift forcing on an iceberg, added mass fraction of an iceberg, and radius of the Earth. The deg_radius constant defines how much of the grid for the ocean current, sea surface tilt, sea ice, and ocean potential temperature and salinity forecast files will be saved. It is currently set to deg_radius = 30, meaning that these grids will be saved for points within 30 latitude/longitude points of the southernmost, northernmost, easternmost, and westernmost initial positions for a set of icebergs. The original grids cover the whole North Atlantic region and have 5 km spatial resolution, so this constant ensures the grids will only be saved within about 150 km of the southernmost, northernmost, easternmost, and westernmost initial iceberg positions at the beginning of the drift forecast, more than enough to cover the region in which they could drift in the next 12-24 hours. This saves memory and processing time for the function. The paths to the bathymetric data (bathy_data_path) and the metocean forecast data files (rootpath_to_metdata) are defined. The iceberg initial latitudes/longitudes, time, waterline lengths, id numbers, grounded statuses, and forecast time are pulled from the Observation(s) class instance (obs) and put into list format if they are not already.

2.	Five nested functions are defined within the rcm_iceberg_drift_deterioration_forecaster function:

•	The first function calculates the distance and azimuthal bearing between two latitude/longitude coordinates,
•	The second function calculates the latitude/longitude coordinate at a given distance and course from an initial latitude/longitude coordinate,
•	The third function calculates the easterly and northerly iceberg acceleration vector components from the sum of the forcings acting on the iceberg,
•	The fourth function is for numerically solving the iceberg acceleration integration to drift velocity, and
•	The fifth function calculates the new iceberg waterline length, draft, sail cross-sectional area, and total mass after deterioration.

The forces considered in the function for iceberg drift are the (e.g., see Lichey and Hellmer, 2001; Kubat et al., 2005; and Eik, 2009):

•	Air (wind) and water (ocean current) drags,
•	Coriolis,
•	Water pressure gradient,
•	Sea surface tilt (horizontal gravitational),
•	Wave radiation, and
•	Sea ice.

The forcings considered in the function for iceberg deterioration are the (e.g., see Kubat et al., 2007):

•	Surface (freeboard) melting due to solar radiation,
•	Melting of the keel due to buoyant vertical convection,
•	Melting of the keel and freeboard due to forced convection of water and air, respectively, and
•	Calving due to wave erosion.

3.	The rcm_datetime0 (the RCM image acquisition time) and next_rcm_time are ensured to be in numpy datetime64 format and the forecast start time for a set of icebergs passed to the function is set to the rcm_datetime0.

4.	The function reads the bathymetric data file and interpolates the water depth at the iceberg location. If the iceberg’s status has been entered into the function as “not grounded” but the calculated draft is equal to or greater than the water depth, the iceberg draft is reset to be one meter less than the water depth. If the iceberg’s initial status is given as “grounded” then the iceberg’s draft will be reset to the water depth at its initial location.

5.	If no iceberg waterline length is given for an iceberg, the function will assume it is 100 m. Since the iceberg’s waterline length is the only physical dimension extracted from an RCM image, the iceberg draft, mass, and sail cross-sectional area are calculated from this variable alone. The iceberg draft is calculated as a function of the iceberg waterline length using the formulation obtained from analysis of C-CORE iceberg profile data. The iceberg mass and cross-sectional sail area are calculated as a function of waterline length from formulations given in Barker et al. (2004).

6.	The iceberg drift forecast times are determined. The drift forecast is initialized at the RCM image acquisition date/time, and then hourly time-steps are used until the next RCM image acquisition date/time is reached. The final time-step between the last hourly time-step and the next RCM image acquisition date/time may be less than one hour.

7.	Arrays are initialized for the hourly forecast iceberg latitude/longitude positions, drift velocities, waterline lengths, drafts, and masses.

8.	The UTC hour strings at which the metocean forecast data start are determined by checking the files in the rootpath_to_metdata.

9.	The forecast times for wind, wave, ocean current, sea surface height, sea ice concentration, thickness, and velocity, ocean potential temperature, ocean salinity, air temperature, and surface solar radiation data that fully overlap with the iceberg drift forecast period are determined.

10.	The ocean current, potential temperature, sea ice, and salinity files are stripped down to only retain data within 30 grid points of the original iceberg position. At 5 km resolution, this represents around 150 km, which should be more than enough to cover iceberg drift over the next 12-24 hours. This parameter is the deg_radius set at the beginning of (but inside) the function and can be increased if needed. This step is performed to save significant processing time. Using the whole RIOPS grid for the ocean files was found to greatly increase the function’s runtime to over an hour in many cases due to the more intensive interpolation processes on a larger grid, plus the extra time needed to compute the larger sea surface height gradient grids. The stripped-down forecast sea surface height gradients are also computed in this step and saved into their own netCDF files. These stripped-down files are placed in a temporary directory.

11.	The iceberg drift and deterioration forecast loop begins. At each time-step, the forecast wind, wave, ocean current, potential temperature, salinity, sea surface height gradient, sea ice concentration, thickness, and zonal and meridional velocity, air temperature, and surface solar radiation files are found that align with times just before and after the current iceberg forecast time, or that align exactly with the current iceberg forecast time if they exist.

12.	The forecast wind velocity components, wave parameters, ocean current velocity components, potential temperatures, salinities, sea surface height gradient components, sea ice concentrations, thicknesses, and zonal and meridional velocities, air temperatures, and surface solar radiation from each of the before and after files are spatially interpolated to the iceberg forecast position at the present time-step. Ocean current velocities, potential temperatures, and salinities are only processed to a depth equal to or just greater than the calculated iceberg draft. The ocean current velocities are then averaged over the draft depth of the iceberg. The linearly interpolated value of each metocean variable in time is then calculated for the present time-step. Wave variables are interpolated using nearest neighbor interpolation.

13.	The iceberg deterioration between the present and next time-step is calculated. If an iceberg deteriorates completely, the iceberg mass, draft, sail area, and waterline length will be set to zero, and the iceberg drift will stop. A warning will print out if an iceberg is predicted to deteriorate to less than 40 m waterline length, as this iceberg may fall below the minimum spatial resolution of RCM imagery.

14.	The iceberg zonal and meridional acceleration components are computed at the present time-step, and an implicit, unconditionally stable numerical integration (backward differentiation) algorithm is used to determine the iceberg drift velocity at the next time-step. The iceberg displacement at the next time-step is then calculated using the Euler forward integration algorithm, integrated using drift velocities averaged between the present and next time-steps (e.g., trapezoidal integration).

15.	At each time-step, the water depth is checked at the forecast iceberg location. If the water depth is equal to or less than the iceberg draft, the iceberg drift is halted at that point and the iceberg’s status is changed to grounded.

16.	The function will finally return the following variables:

•	The hourly forecast iceberg latitude/longitude (in degrees North and East, respectively) positions up to the next RCM image acquisition date/time,
•	The hourly forecast iceberg waterline lengths (in meters) up to the next RCM image acquisition date/time,
•	The hourly forecast iceberg grounded statuses (as one for grounded or zero for not grounded), and
•	The times at which the iceberg positions are forecast (in numpy datetime64 format).

Key Operating Assumptions:

1.	The bathymetric data file is assumed to be in netCDF format and contain three variables labelled “lat”, “lon”, and “elevation”. The elevation variable is assumed to be negative for water depth and have the dimensions latitude by longitude.

2.	The next RCM image acquisition time input to the function must be within the next 3.5 days (84 hours) from the earliest start time of the metocean forecast data to remain within the forecast period for the ocean variables.

3.	The iceberg drift velocities are initialized in the forecast at zero.

4.	Iceberg keel cross-sectional areas are assumed to be rectangular.

References:

Barker, A., Sayed, M., and Carrieres, T. (2004). “Determination of Iceberg Draft, Mass and Cross-Sectional Areas,” In Proceedings of the 14th International Offshore and Polar Engineering Conference (ISOPE), Toulon, France, May 23-28.
Eik, K. (2009). “Iceberg drift modelling and validation of applied metocean hindcast data,” Cold Regions Science and Technology, vol. 57, pp. 67-90.

GDPS. (2024). “Global Deterministic Prediction System (GDPS) data in GRIB2 format,” Accessed on the World Wide Web at: https://eccc-msc.github.io/open-data/msc-data/nwp_gdps/readme_gdps-datamart_en/, November 1, 2024.

GDWPS. (2024). “Global Deterministic Wave Prediction System (GDWPS) data in GRIB2 format,” Accessed on the World Wide Web at: https://eccc-msc.github.io/open-data/msc-data/nwp_gdwps/readme_gdwps-datamart_en/, November 1, 2024.

Hersbach, H., Bell, B., Berrisford, P., Biavati, G., Horányi, A., Muñoz Sabater, J., Nicolas, J., Peubey, C., Radu, R., Rozum, I., Schepers, D., Simmons, A., Soci, C., Dee, D., and Thépaut, J-N. (2023). “ERA5 hourly data on single levels from 1940 to present,” Copernicus Climate Change Service (C3S) Climate Data Store (CDS), DOI: 10.24381/cds.adbb2d47 (Accessed on November 18, 2024).

HYCOM. (2024). “Hybrid Coordinate Ocean Model Global Ocean Forecast System (GOFS) 3.1,” Accessed on the World Wide Web at: https://www.hycom.org/dataserver/gofs-3pt1/analysis, November 18, 2024.

Kubat, I., Sayed, M., Savage, S.B., and Carrieres, T. (2005). “An Operational Model of Iceberg Drift,” International Journal of Offshore and Polar Engineering (IJOPE), vol. 15, no. 2, pp. 125-131.

Kubat, I., Sayed, M., Savage, S., Carrieres, T., and Crocker, G. (2007). “An Operational Iceberg Deterioration Model,” Proceedings of the 16th International Offshore and Polar Engineering Conference (ISOPE), Lisbon, Portugal, July 1-6.

Lichey, C., and Hellmer, H.H. (2001). “Modeling giant iceberg drift under the influence of sea ice in the Weddell Sea, Antarctica,” Journal of Glaciology, vol. 47, no. 158, pp. 452-460.

RIOPS. (2024). “Regional Ice Ocean Prediction System (RIOPS) Data in NetCDF Format,” Accessed on the World Wide Web at: https://eccc-msc.github.io/open-data/msc-data/nwp_riops/readme_riops-datamart_en/, November 1, 2024.

Title: Documentation for RCM_Bergy_Bit_Growler_Forecaster.py
Author: Ian D. Turnbull

Summary:

The RCM_Bergy_Bit_Growler_Forecaster.py Python script file contains a function called rcm_bergy_bit_growler_forecaster and runs on a physics-based ensemble-probabilistic iceberg drift and deterioration model. The function takes the following inputs taken from a RADARSAT Constellation Mission (RCM) satellite image: a series of iceberg latitude/longitude positions, the date/time of the RCM image acquisition, the iceberg identification numbers, the forecast end time (forecast_end_time), and whether sea ice is present (si_toggle). The iceberg initial positions, time, and id numbers should be passed to the function from an instance of the Observations class or an instance of the Observation class if for a single iceberg only. The function assumes each iceberg will calve a bergy bit and a growler and forecasts a cone of uncertainty of possible maximum extents a bergy bit or growler could potentially drift from a given iceberg. The bergy bit is assumed to be 15 m waterline length and the growler 5 m waterline length, representing the maximum sizes of each. The function will output latitude-longitude polygons covering the maximum drift extents of the bergy bit and growler for each iceberg, a latitude-longitude polygon covering the maximum drift extents of the all the bergy bits and growlers  for all icebergs, and their final deteriorated waterline length statistics obtained from the ensemble of tracks (minimum, maximum, and mean) of the hourly forecast iceberg latitude/longitude position and waterline lengths up to the next RCM image acquisition date/time. The final time step will be the remainder between the last hourly time step and the forecast end time if it is less than one hour. This document lists the Python libraries needed to run the function, describes how the function works, and lists its key operating assumptions.

Python Libraries Used in Current Version of the Function:

scipy v1.13.1
tempfile
numpy v1.26.4
netCDF4 v1.6.2
gsw v3.6.19
os

How the Function Works:

The rcm_bergy_bit_growler_forecaster function requires the following inputs:

•	A set of iceberg latitude/longitude positions in array-like format (in degrees North and East, respectively) passed from an Observations class instance or a single iceberg position passed from an Observation class instance,
•	The date/time of the RCM image acquisition (passed from Observation(s) class instance),
•	The iceberg’s id numbers (passed from Observation(s) class instance) for tracking purposes,
•	The anticipated time of the next RCM image acquisition (t1), and
•	Whether sea is present (si_toggle=True or False).

The date/time of the RCM image acquisition (rcm_datetime0) and the forecast end time (forecast_end_time) should be input as strings in numpy datetime64 format and Universal Time Coordinated (UTC), e.g., for example: ‘2024-11-01T09:42:53.33’.

1.	The first lines of the function define some constants, including the density of ice, ice albedo, latent heat of fusion for ice, radius of the Earth, the Coriolis deflection angle, the number of ensemble tracks to produce for each bergy bit and each growler, the minimum bergy bit/growler lengths to be considered as “fully deteriorated,” and parameters that define the probability density functions (pdfs) from which randomized errors in forecast wind and ocean current velocity and significant wave height will be drawn for the ensemble model. These parameters are defined according to Allison et al. (2014). The deg_radius constant defines how much of the grid for the ocean current, sea ice, and ocean potential temperature and salinity forecast files will be saved. It is currently set to deg_radius = 30, meaning that these grids will be saved for points within 30 latitude/longitude points of the southernmost, northernmost, easternmost, and westernmost initial positions for a set of icebergs. The original grids cover the whole North Atlantic region and have 5 km spatial resolution, so this constant ensures the grids will only be saved within about 150 km of the southernmost, northernmost, easternmost, and westernmost initial iceberg positions at the beginning of the drift forecast, more than enough to cover the region in which they could drift in the next 12-24 hours. This saves memory and processing time for the function. The paths to the bathymetric data (bathy_data_path) and the metocean forecast data files (rootpath_to_metdata) are defined. The iceberg initial latitudes/longitudes, time, id numbers, and forecast time are pulled from the Observation(s) class instance (obs) and put into list format if they are not already.

2.	Nine nested functions are defined within the rcm_bergy_bit_growler_forecaster function:

•	The first function calculates the latitude/longitude coordinate at a given distance and course from an initial latitude/longitude coordinate,
•	The second function calculates the easterly and northerly iceberg drift velocity vector components,
•	The third function calculates the new bergy bit/growler waterline length, draft, sail cross-sectional area, and total mass after deterioration,
•	The fourth, fifth, and sixth functions calculate randomized errors for the forecast wind and ocean current speeds and directions and the significant wave heights,
•	The seventh function computes the outer boundaries of the bergy bit/growler ensemble tracks,
•	The eighth and ninth functions compute the final statistics of the bergy bit/growler ensemble modelled deteriorated lengths.

The function for bergy bit/growler drift velocity is a simple linear summation of 2% of the wind velocity vector turned 20 to the right to account for the Coriolis deflection, and the ocean current velocity averaged over the calculated draft of the bergy bit or growler (e.g., see Wagner et al., 2017).

The forcings considered in the function for bergy bit/growler deterioration are the (e.g., see Kubat et al., 2007):

•	Surface (freeboard) melting due to solar radiation,
•	Melting of the keel due to buoyant vertical convection,
•	Melting of the keel and freeboard due to forced convection of water and air, respectively, and
•	Calving due to wave erosion.

3.	The rcm_datetime0 (the RCM image acquisition time) and forecast_end_time are ensured to be in numpy datetime64 format and the forecast start time for a set of icebergs passed to the function is set to the rcm_datetime0.

4.	The function reads the bathymetric data file and interpolates the water depth at the iceberg location. If the calculated bergy bit/growler draft is equal to or greater than the water depth, the bergy bit/growler draft is reset to be one meter less than the water depth.

5.	The bergy bit/growler draft is calculated as a function of the bergy bit/growler waterline length (15 m and 5 m, respectively) using the formulation obtained from analysis of C-CORE iceberg profile data. The bergy bit/growler mass and cross-sectional sail area are calculated as a function of waterline length from formulations given in Barker et al. (2004).

6.	The bergy bit/growler drift forecast times are determined. The drift forecast is initialized at the RCM image acquisition date/time, and then hourly time-steps are used until the forecast end time is reached. The final time-step between the last hourly time-step and the forecast end time may be less than one hour.

7.	Arrays are initialized for the hourly forecast bergy bit/growler latitude/longitude positions, drift velocities, waterline lengths, drafts, and masses.

8.	The UTC hour strings at which the metocean forecast data start are determined by checking the files in the rootpath_to_metdata.

9.	The forecast times for wind, wave, ocean current, sea ice concentration, ocean potential temperature, ocean salinity, air temperature, and surface solar radiation data that fully overlap with the bergy bit/growler drift forecast period are determined.

10.	The ocean current, potential temperature, sea ice, and salinity files are stripped down to only retain data within 30 grid points of the original iceberg position. At 5 km resolution, this represents around 150 km, which should be more than enough to cover bergy bit/growler drift over the next 84 hours. This parameter is the deg_radius set at the beginning of (but inside) the function and can be increased if needed. This step is performed to save significant processing time. Using the whole RIOPS grid for the ocean files was found to greatly increase the function’s runtime to over an hour in many cases due to the more intensive interpolation processes on a larger grid. These stripped-down files are placed in a temporary directory.

11.	The bergy bit/growler drift and deterioration forecast loop begins. At each time-step, the forecast wind, wave, ocean current, potential temperature, salinity, sea ice concentration, air temperature, and surface solar radiation files are found that align with times just before and after the current bergy bit/growler forecast time, or that align exactly with the current bergy bit/growler forecast time if they exist.

12.	The forecast wind velocity components, wave parameters, ocean current velocity components, potential temperatures, salinities, sea ice concentrations, air temperatures, and surface solar radiation from each of the before and after files are spatially interpolated to the bergy bit/growler forecast position at the present time-step. Ocean current velocities, potential temperatures, and salinities are only processed to a depth equal to or just greater than the calculated bergy bit/growler draft. The ocean current velocities are then averaged over the draft depth of the bergy bit/growler. The linearly interpolated value of each metocean variable in time is then calculated for the present time-step. Wave variables are interpolated using nearest neighbor interpolation.

13.	The ensemble forecast loop is initiated and the randomized errors in the forecast wind and ocean current speeds and directions and significant wave heights are computed and added to their respective variables.

14.	The bergy bit/growler deterioration between the present and next time-step is calculated. If a bergy bit/growler deteriorates completely, the bergy bit/growler mass, draft, sail area, and waterline length will be set to zero, and the bergy bit/growler drift will stop.

15.	The bergy bit/growler zonal and meridional drift velocity components are computed at the present time-step. The bergy bit/growler displacement at the next time-step is then calculated using the Euler forward integration algorithm, integrated using drift velocities averaged between the present and next time-steps (e.g., trapezoidal integration).

16.	At each time-step, the water depth is checked at the forecast bergy bit/growler location. If the water depth is equal to or less than the bergy bit/growler draft, the bergy bit/growler drift is halted at that point and the bergy bit’s/growler’s status is changed to grounded.

17.	The outer boundaries of the bergy bit/growler ensemble drift tracks are determined along with their final deteriorated waterline length statistics.

18.	The function will finally return the following variables:

•	Dictionaries containing the bergy bit/growler boundaries, and
•	Dictionaries containing the ensemble statistics of the bergy bit/growler final deteriorated waterline lengths and latest times at which they were reached in the ensemble output.

Key Operating Assumptions:

1.	The bathymetric data file is assumed to be in netCDF format and contain three variables labelled “lat”, “lon”, and “elevation”. The elevation variable is assumed to be negative for water depth and have the dimensions latitude by longitude.

2.	The forecast end time input to the function must be within the next 3.5 days (84 hours) from the earliest start time of the metocean forecast data to remain within the forecast period for the ocean variables.

3.	The bergy bit/growler drift velocities are initialized in the forecast at zero.

4.	Bergy bit/growler keel cross-sectional areas are assumed to be rectangular.

References:

Allison, K., Crocker, G., Tran, H., and Carrieres, T. (2014). “An ensemble forecast model of iceberg drift,” Cold Regions Science and Technology, vol. 108, pp. 1-9.

Barker, A., Sayed, M., and Carrieres, T. (2004). “Determination of Iceberg Draft, Mass and Cross-Sectional Areas,” In Proceedings of the 14th International Offshore and Polar Engineering Conference (ISOPE), Toulon, France, May 23-28.

Eik, K. (2009). “Iceberg drift modelling and validation of applied metocean hindcast data,” Cold Regions Science and Technology, vol. 57, pp. 67-90.

GDPS. (2024). “Global Deterministic Prediction System (GDPS) data in GRIB2 format,” Accessed on the World Wide Web at: https://eccc-msc.github.io/open-data/msc-data/nwp_gdps/readme_gdps-datamart_en/, November 1, 2024.

GDWPS. (2024). “Global Deterministic Wave Prediction System (GDWPS) data in GRIB2 format,” Accessed on the World Wide Web at: https://eccc-msc.github.io/open-data/msc-data/nwp_gdwps/readme_gdwps-datamart_en/, November 1, 2024.

Hersbach, H., Bell, B., Berrisford, P., Biavati, G., Horányi, A., Muñoz Sabater, J., Nicolas, J., Peubey, C., Radu, R., Rozum, I., Schepers, D., Simmons, A., Soci, C., Dee, D., and Thépaut, J-N. (2023). “ERA5 hourly data on single levels from 1940 to present,” Copernicus Climate Change Service (C3S) Climate Data Store (CDS), DOI: 10.24381/cds.adbb2d47 (Accessed on November 18, 2024).

HYCOM. (2024). “Hybrid Coordinate Ocean Model Global Ocean Forecast System (GOFS) 3.1,” Accessed on the World Wide Web at: https://www.hycom.org/dataserver/gofs-3pt1/analysis, November 18, 2024.

Kubat, I., Sayed, M., Savage, S.B., and Carrieres, T. (2005). “An Operational Model of Iceberg Drift,” International Journal of Offshore and Polar Engineering (IJOPE), vol. 15, no. 2, pp. 125-131.

Kubat, I., Sayed, M., Savage, S., Carrieres, T., and Crocker, G. (2007). “An Operational Iceberg Deterioration Model,” Proceedings of the 16th International Offshore and Polar Engineering Conference (ISOPE), Lisbon, Portugal, July 1-6.

Lichey, C., and Hellmer, H.H. (2001). “Modeling giant iceberg drift under the influence of sea ice in the Weddell Sea, Antarctica,” Journal of Glaciology, vol. 47, no. 158, pp. 452-460.

RIOPS. (2024). “Regional Ice Ocean Prediction System (RIOPS) Data in NetCDF Format,” Accessed on the World Wide Web at: https://eccc-msc.github.io/open-data/msc-data/nwp_riops/readme_riops-datamart_en/, November 1, 2024.

Wagner, T.J.W., Dell, R.W., and Eisenman, I. (2017). “An Analytical Model of Iceberg Drift,” Journal of Physical Oceanography, vol. 47, pp. 1,605-1,616.

import pydrift
import matplotlib.pyplot as plt
import utils
from glob import glob
import numpy as np

plt.rcParams['text.usetex'] = True

# TODO: use any model
model = pydrift.HighResML()
context = pydrift.Context(model)

predictions = []

errors = []
errors2 = []
time_deltas = []

tmin = np.datetime64('2025-01-01T00:00:00')
tmax = np.datetime64('2020-12-31T23:59:59')
lonmin = 0
lonmax = -80
latmin = 90
latmax = 0

path_to_shapefiles = './input/2023/*.shp'
for fname in np.sort(glob(path_to_shapefiles)[0:]):

    observations = utils.read_shapefile(fname)
    pairs = utils.process_track(observations)

    print(fname + " " + str(np.size(observations)) + ' targets loaded')

    # plt.ion()

    for p in pairs:
        o = p[0]
        o_next = p[1]

        if (o.time < tmin):
            tmin=o.time
        if (o_next.time > tmax):
            tmax=o_next.time
        if (o.lon < lonmin):
            lonmin=o.lon
        if (o_next.lon > lonmax):
            lonmax=o_next.lon
        if (o.lat < latmin):
            latmin=o.lat
        if (o_next.lat > latmax):
            latmax=o_next.lat

        R = 6371000
        dlat = o.lat - o_next.lat
        dlon = o.lon - o_next.lon

        dx = dlon * np.pi / 180 * R * np.cos(o.lat * np.pi / 180)
        dy = dlat * np.pi / 180 * R

        print("Length: " + str(o.length) + " m")
        print("Image 1: " + str(o.time) + " Image 2: " + str(o_next.time))
        print("Difference: " + str((o_next.time - o.time).astype('timedelta64[h]').astype(int)) + " hours")
        if (o_next.time - o.time).astype('timedelta64[h]').astype(int) > 48:
            print("Average speed: " + str(np.sqrt(dx**2+dy**2)/(o_next.time - o.time).astype(float)) + " m/s\n")

        try:
            t, lat, lon = context.forecast(o, o_next.time)
            predictions.append((lat, lon))

            errors = np.append(errors, utils.get_error(o_next, t, lat, lon))
            errors2 = np.append(errors2,utils.get_dimensionless_error(o,o_next,t,lat,lon))
            time_deltas = np.append(time_deltas, (o_next.time - o.time).astype(float) / 3600)

            if (errors[-1]/time_deltas[-1] > 30000/24):
                print(fname)
                print(o.time)
                print(o.length)

            plt.plot(lon, lat, '-k.', markersize=2)
            plt.plot([o.lon, o_next.lon], [o.lat, o_next.lat], '-ro', markersize=2, linewidth=1)
            plt.show()
            utils.write_shapefile([predictions[-1]], './output/example.shp')

        except Exception as e:
             print("Error: " + e.__str__() + "\n")

#plt.show()
utils.write_shapefile([predictions[-1]], './output/example.shp')
print("Time range: " + str(tmin) + " - " + str(tmax))
print("Longitude range: " + str(lonmin) + " - " + str(lonmax))
print("Latitude range: " + str(latmin) + " - " + str(latmax))



#to_save = np.stack((time_deltas,errors,errors2))
#np.savetxt("highresml.csv",to_save,delimiter=",")

ind_21 = np.nonzero(abs(time_deltas-12)<1)
ind_22 = np.nonzero(abs(time_deltas-24)<1)
ind_23 = np.nonzero(abs(time_deltas-36)<1)

print(np.nanmean(errors[ind_21]/1000))
print(np.nanmean(errors[ind_22]/1000))
print(np.nanmean(errors[ind_23]/1000))

fig, ax = plt.subplots(1,2)

ax[0].plot(time_deltas, errors / 1000, 'o', markersize=3)
ax[0].plot([0,48],[0,20],'g')
ax[0].plot([0,48],[0,40],'r')
ax[0].set_xlim(left=0, right = 48)
ax[0].set_ylim(bottom=0, top = 50)
ax[0].set_xlabel('Time [h]')
ax[0].set_ylabel('Error [km]')
ax[0].grid()

ax[1].plot(time_deltas, errors2, 'o', markersize=3)
ax[1].set_xlim(left=0,right= 48)
ax[1].set_ylim(bottom=0,top = 5.0)
ax[1].set_xlabel('Time [h]')
ax[1].set_ylabel('$\epsilon$')

ax[1].grid()
plt.show()





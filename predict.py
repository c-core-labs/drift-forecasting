
import pydrift
import matplotlib.pyplot as plt
import utils
from glob import glob
import numpy as np

plt.style.use('default')
plt.rcParams['text.usetex'] = True

model = pydrift.LowResML()
context = pydrift.Context(model)

predictions = []

observations = utils.read_shapefile(fname)

print(fname + " " + str(np.size(observations)) + ' targets loaded')

for o in observations:
    try:
        t, lat, lon = context.forecast(o, o.time+np.timedelta64(48, 'h'))

        predictions.append((lat, lon))


        #plt.plot([o.lon, o_next.lon], [o.lat, o_next.lat], '-ro', markersize=2, linewidth=1)

        #if (abs(time_deltas[-1]-24)<1) & (errors[-1]>20000):
        #    plt.plot(lon, lat, '-k.', markersize=2)
        #    print(utils.get_error(o_next, t, lat, lon))

        #utils.write_shapefile([predictions[-1]], './output/example.shp')

    except Exception as e:
         print("Error: " + e.__str__() + "\n")





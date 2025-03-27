import pydrift
import matplotlib.pyplot as plt
import utils
import numpy as np
from observation import Observation

plt.rcParams['text.usetex'] = True

input_fname = "/home/renaty/memscale/Icebergs/JIP_2025/TargetsMerge_Drift/20250326_095635_S1_Targets_Merge.shp"
output_fname = './output/20250326_095635_forecast.shp'

observations = utils.read_product_shapefile(input_fname)
grounded = utils.check_groundings(observations)
print("Grounded: " + str(sum(grounded)) + " / " + str(len(observations)))

predictions = []

model = pydrift.HighResML()
context = pydrift.Context(model)

for o in observations[~grounded]:
    try:
        t, lat, lon = context.forecast(o, o.time + np.timedelta64(24, 'h'))
        tint, latint, lonint = utils.interpolate_track(t, lat, lon, 3*3600)
        plt.plot( latint, lonint, '-r.')
        predictions.append((tint, latint, lonint))
        utils.write_shapefile(predictions, output_fname)

    except Exception as e:
        estr = e.args[0]
        t,lat,lon = e.args[1]

        tint,latint,lonint = utils.interpolate_track(t,lat,lon, 3*3600)

        predictions.append((tint,latint,lonint))
        utils.write_shapefile(predictions, output_fname)

        print("Error: " + estr + "\n")

plt.show()

if len(predictions) > 0:
    utils.write_shapefile(predictions, output_fname)


import pydrift
import matplotlib.pyplot as plt
import utils
from glob import glob
import numpy as np
from observation import Observation

plt.rcParams['text.usetex'] = True

observations = utils.read_product_shapefile('./input/20240429_094938_RCM_Shapefiles/20240429_094938_RCM_Targets.shp')

model = pydrift.HighResML()
context = pydrift.Context(model)

predictions = []

for o in observations:
    try:
        t, lat, lon = context.forecast(o, o.time + np.timedelta64(48, 'h'))
        predictions.append((lat, lon))
    except Exception as e:
        # plt.clf()
        print("Error: " + e.__str__() + "\n")

utils.write_shapefile(predictions, './output/example.shp')

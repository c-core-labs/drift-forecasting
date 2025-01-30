import pydrift
import matplotlib.pyplot as plt
import utils
import numpy as np
from observation import Observation

plt.rcParams['text.usetex'] = True

input_fname = './input/20240417_094938_RCM_Targets/20240417_094938_RCM_Targets.shp'
output_fname = './output/example.shp'

observations = utils.read_product_shapefile(input_fname)
grounded = utils.check_groundings(observations)
print("Grounded: " + str(sum(grounded)) + " / " + str(len(grounded)))

predictions = []

model = pydrift.HighResML()
context = pydrift.Context(model)

for o in observations[~grounded]:
    try:
        t, lat, lon = context.forecast(o, o.time + np.timedelta64(48, 'h'))
        plt.plot(lon, lat, '-r.')
        predictions.append((lat, lon))
    except Exception as e:
        print("Error: " + e.__str__() + "\n")

plt.show()
utils.write_shapefile(predictions, output_fname)

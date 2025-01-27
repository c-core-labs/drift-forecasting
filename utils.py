import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import shapefile
from observation import Observation
from datetime import datetime
from catboost import MultiTargetCustomMetric, MultiTargetCustomObjective
import sys
import re

nHours = 48
R = 6381000

def get_limits(t, lat, lon):
    min_lat = np.min(lat) - 1.0
    max_lat = np.max(lat) + 1.0
    min_lon = np.min(lon) - 0.5
    max_lon = np.max(lon) + 0.5
    t1 = t[0] - np.timedelta64(3, 'h')
    t2 = t[-1] + np.timedelta64(3, 'h')

    return [t1,t2, min_lat,max_lat,min_lon,max_lon]

def get_error(o: Observation,t_pred, lat_pred, lon_pred):

    flon = interpolate.interp1d(t_pred.astype(float), lon_pred)
    flat = interpolate.interp1d(t_pred.astype(float), lat_pred)

    dlat = o.lat - flat(o.time.astype(float))
    dlon = o.lon - flon(o.time.astype(float))

    dx = dlon * np.pi / 180 * R * np.cos(o.lat * np.pi / 180)
    dy = dlat * np.pi / 180 * R

    return np.sqrt(dx**2+dy**2)

def get_dimensionless_error(o: Observation, o_next: Observation, t_pred, lat_pred, lon_pred):

    flon = interpolate.interp1d(t_pred.astype(float), lon_pred)
    flat = interpolate.interp1d(t_pred.astype(float), lat_pred)

    dlat = o_next.lat - flat(o_next.time.astype(float))
    dlon = o_next.lon - flon(o_next.time.astype(float))

    dx = dlon * np.pi / 180 * R * np.cos(o.lat * np.pi / 180)
    dy = dlat * np.pi / 180 * R

    dr = np.sqrt(dx**2+dy**2)

    dlat = o.lat - o_next.lat
    dlon = o.lon - o_next.lon

    dx = dlon * np.pi / 180 * R * np.cos(o.lat * np.pi / 180)
    dy = dlat * np.pi / 180 * R

    dl = np.sqrt(dx**2+dy**2)

    return dr/dl

def get_angle_error(o: Observation, o_next: Observation, t_pred, lat_pred, lon_pred):

    flon = interpolate.interp1d(t_pred.astype(float), lon_pred)
    flat = interpolate.interp1d(t_pred.astype(float), lat_pred)

    dlat = flat(o_next.time.astype(float)) - o.lat
    dlon = flon(o_next.time.astype(float)) - o.lon

    dx = dlon * np.pi / 180 * R * np.cos(o.lat * np.pi / 180)
    dy = dlat * np.pi / 180 * R

    dr = np.sqrt(dx**2+dy**2)

    dlat = o_next.lat - o.lat
    dlon = o_next.lon - o.lon

    dx = dlon * np.pi / 180 * R * np.cos(o.lat * np.pi / 180)
    dy = dlat * np.pi / 180 * R

    dl = np.sqrt(dx**2+dy**2)

    return dr/dl

def rms_time(observations, predictions):
    # print('compute error curve')

    n = np.shape(observations)[0]

    RMS_ERROR_CURVE = np.zeros((n, nHours + 1))
    RMS_ERROR_CURVE[:] = np.nan
    normalized_separation = np.zeros((n,))

    for i in range(0, n):
        x0 = observations[i][0]
        y0 = observations[i][1]
        x1 = predictions[i][0]
        y1 = predictions[i][1]
        e2 = (x1 - x0) ** 2 + (y1 - y0) ** 2
        dl = np.sqrt((x0[1:] - x0[0:-1]) ** 2 + (y0[1:] - y0[0:-1]) ** 2)
        e1 = np.sqrt(e2)
        normalized_separation[i] = np.sum(e1) / np.sum(np.cumsum(dl))
        RMS_ERROR_CURVE[i, 0:np.size(e2)] = e1

    # Error curve
    # E2=np.sqrt(np.nanmean(D,axis=0))
    E1 = np.nanmean(RMS_ERROR_CURVE, axis=0)
    E2 = np.nanmean(normalized_separation)

    plt.subplot(121)
    plt.plot(np.transpose(RMS_ERROR_CURVE) / 1000, 'k', alpha=0.05, linewidth=1)
    plt.plot(E1 / 1000, 'r', linewidth=1, alpha=1)
    plt.grid(True)
    plt.xlim([0, nHours])
    plt.ylim([0, 40])
    plt.xlabel('Time [h]')
    plt.ylabel('RMSE [km]')

    A = np.squeeze(RMS_ERROR_CURVE[:, nHours])
    plt.subplot(122)
    plt.hist(A[~np.isnan(A)] / 1000, 100)
    plt.xlabel('24h error [km]')
    plt.ylabel('Counts')
    plt.xlim([0, 70])
    plt.show()

    print("24h error: " + str(np.nanmax(E1)))
    print("24h error std: " + str(np.nanstd(RMS_ERROR_CURVE, axis=0)[-1]))
    print("24h error min: " + str(np.nanmin(RMS_ERROR_CURVE, axis=0)[-1]))
    print("24h error max: " + str(np.nanmax(RMS_ERROR_CURVE, axis=0)[-1]))

    # np.savetxt('imp_perf_xval.csv', E1, fmt='%.3e', delimiter=',')
    # return np.sum(E1)/24
    return np.nanmax(E1)

def sort_obs(observations):
    times = np.empty((0,),dtype='datetime64[s]')
    for o in observations:
        times = np.append(times, o.time)
    ind = np.argsort(times)

    return observations[ind]

def read_shapefile(fname):
    observations = np.empty(0, dtype=Observation)

    sf = shapefile.Reader(fname)

    for p in sf.shapeRecords():

        l = p.record['WtrLin']
        lat = p.shape.points[0][1]
        lon = p.shape.points[0][0]
        time = np.datetime64(datetime.strptime(p.record['acqDate'][0:15], "%Y%m%d_%H%M%S")).astype('datetime64[s]')
        if not(p.record['Grounded']=='grounded'):
            observations = np.append(observations,Observation(lat,lon,time,l, grounded=False))
        else:
            observations = np.append(observations, Observation(lat, lon, time, l, grounded=True))
            print("Grounded: " + p.record['Grounded'], end=" ")
            print("Notes: " + p.record['Notes'])

    return observations

def process_track(observations):
    observations = sort_obs(observations)
    pairs = zip(observations[:-1], observations[1:])

    clean_pairs = []

    for p in pairs:
        if p[0].grounded | p[1].grounded:
            ()
        else:
            clean_pairs.append(p)

    return clean_pairs


def write_shapefile(predictions,fname):
    sf = shapefile.Writer(fname)
    sf.field('name', 'C')
    sf.field('lon', 'F')
    sf.field('lat', 'F')

    for i in np.arange(0,len(predictions)):
        data = np.vstack((predictions[i][1], predictions[i][0])).T
        sf.multipoint(data)
        sf.record('track'+str(i))

    sf.close()

    with open("{}.prj".format(fname[:-4]), "w") as prj:
        wkt = 'GEOGCS["WGS 84",'
        wkt += 'DATUM["WGS_1984",'
        wkt += 'SPHEROID["WGS 84",6378137,298.257223563]]'
        wkt += ',PRIMEM["Greenwich",0],'
        wkt += 'UNIT["degree",0.0174532925199433]]'
        prj.write(wkt)

## User-defined loss-function
class DimensionlessErrorObjective(MultiTargetCustomObjective):

    def calc_ders_multi(self, approxes, target, weight):
        assert len(approxes) == len(target)
        min_float = sys.float_info.min
        dist0 = np.linalg.norm(target)+min_float
        dist1 =  ((target[0] - approxes[0]) ** 2 + (target[1] - approxes[1]) ** 2) ** 1.5 + min_float
        grad = [0.0, 0.0]
        hess = [[0.0, 0.0], [0.0, 0.0]]

        # gradients
        grad[0] = -(approxes[0] - target[0]) / dist0 / (np.sqrt(
            (target[0] - approxes[0]) ** 2 + (target[1] - approxes[1]) ** 2) + min_float)
        grad[1] = -(approxes[1] - target[1]) / dist0 / (np.sqrt(
            (target[0] - approxes[0]) ** 2 + (target[1] - approxes[1]) ** 2) + min_float)

        # Hessian
        hess[0][0] = -1 / dist0 / dist1 + (
                    approxes[0] - target[0]) / dist0 / dist1

        hess[1][1] = -1 / dist0 / dist1 + (
                    approxes[1] - target[1]) / dist0 / dist1

        hess[0][1] = (approxes[0] - target[0]) * (approxes[1] - target[1]) / dist0 / dist1
        hess[1][0] = (approxes[0] - target[0]) * (approxes[1] - target[1]) / dist0 / dist1

        return (grad,hess)


class DimensionlessErrorMetric(MultiTargetCustomMetric):
    def get_final_error(self, error, weight):
        return np.sqrt(error / (weight + 1e-38))

    def is_max_optimal(self):
        return False

    def evaluate(self, approxes, target, weight):
        assert len(target) == len(approxes)
        assert len(target[0]) == len(approxes[0])

        error_sum = 0.0
        weight_sum = 2.0

        for i in range(len(approxes)):
            error_sum += np.sqrt((approxes[i][0] - target[i][0]) ** 2 + (approxes[i][1] - target[i][1]) ** 2)/np.sqrt(target[i][0] ** 2 + target[i][1] ** 2)

        return error_sum, weight_sum


def read_product_shapefile(fname):
    observations = np.empty(0, dtype=Observation)

    sf = shapefile.Reader(fname)
    pattern = r'\d{8}_\d{6}'

    match = re.search(pattern, fname)
    if match:
        #time = np.datetime64(datetime.strptime(match.group(0), "%Y%m%d_%H%M%S")).astype('datetime64[s]')
        time = np.datetime64('now')
        print("Acquisition time: " + str(time))
    else:
        print('Can not read date from the shapefile name.')


    for p in sf.shapeRecords():
        if p.record['Class'] == "ICEBERG":
            l = p.record['WtrLin']
            lat = p.shape.points[0][1]
            lon = p.shape.points[0][0]
            observations = np.append(observations,Observation(lat,lon,time,l, grounded=False))

    print("Total iceberg targets: " + str(observations.size))
    return observations
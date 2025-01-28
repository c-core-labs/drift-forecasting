import numpy as np
from observation import Observation
from catboost import Pool, CatBoostRegressor
import pull_ora5, pull_era5
from scipy import integrate

model = CatBoostRegressor()
model.load_model("relative_explicit")

def get_alpha(phi_degrees):
    added_mass_coefficient = 0.5
    # this is a rough calculation
    return np.sin( phi_degrees * np.pi / 180.0) / (1.0 + added_mass_coefficient)

def get_new_velocity(unow, uold, psi, alpha):
    a = [[3.0,-alpha],[alpha, 3.0]]
    b = psi + 4 * unow - uold
    return np.linalg.solve(a, b)

# The High-res ML model
def forecast(obs: Observation, t1: np.datetime64) -> (np.array, np.array, np.array):
    lat0 = obs.lat
    lon0 = obs.lon
    t0 = obs.time

    tint = np.arange(t0,t1+np.timedelta64(3600,'s'),np.timedelta64(3600,'s'))
    latint = np.full((tint.size,), lat0)
    lonint = np.full((tint.size,), lon0)

    u = np.full((tint.size,2), [0.0,0.0]) # relative water
    uw = np.full((tint.size,2), [0.0,0.0])

    [fuw, fvw] = pull_ora5.get_interpolators(t0, t1, obs.depth)
    [fua, fva] = pull_era5.get_global_interpolators(t0)

    l = obs.length
    alpha = get_alpha(obs.lat)

    p0 = ((t0 - np.timedelta64(1, 'h')).astype(float), obs.lat, obs.lon)
    p1 = (t0.astype(float), obs.lat, obs.lon)

    leeway_coefficient = 0.014
    u0 = leeway_coefficient * fua(p0)
    v0 = leeway_coefficient * fva(p0)
    u1 = leeway_coefficient * fua(p1)
    v1 = leeway_coefficient * fva(p1)

    uold = np.squeeze([u0, v0])
    unow = np.squeeze([u1, v1])

    u[0] = unow
    uw[0] = np.squeeze([fuw(p1), fvw(p1)])

    i = 0
    while (i<tint.size-1):

        p = (tint[i+1], latint[i], lonint[i])       # this where it's not so implicit

        va = np.squeeze([fua(p), fva(p)])

        uw[i+1,:] = np.squeeze([fuw(p), fvw(p)])


        if np.isnan(fuw(p)):
            print('potential grounding')
            break
        else:
            x = Pool([np.hstack((uold, unow, va, uw[i+1,:], l))])
            psi = model.predict(x)[0]

            temp = get_new_velocity(unow, uold, psi, alpha)
            u[i+1] = temp

            uold = unow
            unow = temp

            displacement = integrate.trapezoid(u[:i+2] + uw[:i+2], dx=3600, axis=0)

            latint[i+1] = latint[0] + displacement[1] * 180.0 / np.pi / 6381000.0
            lonint[i+1] = lonint[0] + displacement[0] * 180.0 / np.pi / 6381000.0 / np.cos(latint[0] * np.pi / 180.0)

        i += 1


    #plt.plot(np.sqrt(uw[:,0]**2+uw[:,1]**2),'r.')
    #plt.plot(uw[:, 0], 'r.')
    #plt.plot(uw[:, 1], 'b.')
    #plt.show()
    #plt.plot(lonint,latint,'.')
    #plt.show()
    return tint, latint, lonint

def metadata():
    return "High-res ML model"

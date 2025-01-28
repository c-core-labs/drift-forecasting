import numpy as np
from numpy.ma.core import cumsum

from observation import Observation
import pull_ora5, pull_era5
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pickle

earthRadius = 6381000

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])  # Selecting the last output
        return out

def forecast(obs: Observation, t1: np.datetime64 ) -> (np.array, np.array, np.array):
    model = LSTMModel(input_dim=5, hidden_dim=10, layer_dim=1, output_dim=2)
    model.load_state_dict(torch.load(open('./coarse/coarse_model.pkl', 'rb'), weights_only=True))
    model.eval()

    lat0 = obs.lat
    lon0 = obs.lon
    t0 = obs.time

    [fuw, fvw] = pull_ora5.get_interpolators(t0, t1, obs.depth)
    [fua, fva] = pull_era5.get_global_interpolators(t0)

    lat = lat0
    lon = lon0
    t = t0

    predicted_lats=[lat]
    predicted_lons=[lon]
    times = [t]

    predictions = np.array([[0,0]])

    #try:
    while t < t1:
        tint = np.arange(t, t + 13 * np.timedelta64(3600, 's'), np.timedelta64(3600, 's'))
        latint = np.full((tint.size,), lat)
        lonint = np.full((tint.size,), lon)

        p = (tint.astype(float), latint, lonint)
        uw = fuw(p)
        vw = fvw(p)
        ua = fua(p)
        va = fva(p)
        l = obs.length * np.ones_like(uw)

        if np.any(np.isnan(ua)) | np.any(np.isnan(uw)):
            raise Exception("No current or wind data available")

        temp = (l / 200.0, uw, vw, ua, va)

        testX = torch.tensor(np.array([np.transpose(temp)]), dtype=torch.float32)
        predicted0 = model(testX)
        # print(predicted0)

        p12 = 10 * predicted0[0].detach().numpy()
        predictions=np.append(predictions,[p12],axis=0)

        lat = lat0 + 1000 * np.sum(predictions,axis=0)[1] * 180.0 / earthRadius / np.pi
        lon = lon0 + 1000 * np.sum(predictions,axis=0)[0] * 180.0 / earthRadius / np.pi / np.cos(lat * np.pi / 180.0)

        t = t + np.timedelta64(12, 'h')

        predicted_lats.append(lat)
        predicted_lons.append(lon)
        times.append(t)

#    except Exception as e:
#        print("Error: " + e.__str__())

    return (np.asarray(times),np.asarray(predicted_lats),np.array(predicted_lons))


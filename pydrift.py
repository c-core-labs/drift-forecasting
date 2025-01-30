from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
from observation import Observation

# Implementation of Strategy pattern
class Context():
    
    def __init__(self, strategy: Strategy) -> None:
        self._strategy = strategy
        print(strategy.metadata())

    @property
    def strategy(self) -> Strategy:
        return self._strategy

    @strategy.setter
    def strategy(self, strategy: Strategy) -> None:
        print(strategy.metadata())
        self._strategy = strategy

    def forecast(self, obs: Observation, t1: np.datetime64) -> (np.array, np.array, np.array):
        result = self._strategy.forecast(obs,t1)
        return result


class Strategy(ABC):

    @abstractmethod
    def forecast(self, observation: Observation):
        pass

    @abstractmethod
    def metadata(self):
        pass

# ML model
class HighResML(Strategy):
    import highResML

    def forecast(self, obs: Observation, t1: np.datetime64 ) -> (np.array, np.array, np.array):
       return self.highResML.forecast(obs, t1)

    def metadata(self):
        return "High-resolution ML forecasting algorithm"

# ML model
class LowResML(Strategy):
    import lowResML

    def forecast(self, obs: Observation, t1: np.datetime64 ) -> (np.array, np.array, np.array):
       return self.lowResML.forecast(obs, t1)

    def metadata(self):
        return "Low-resolution ML forecasting algorithm"


# The grounded model
class Grounded(Strategy):
    import grounded

    def forecast(self, obs: Observation, t1: np.datetime64) -> (np.array, np.array, np.array):
        return self.grounded.forecast(obs,t1)

    def metadata(self):
        return "Iceberg does not move"


# The leeway model
class Leeway(Strategy):
    import leeway

    def forecast(self, obs: Observation, t1: np.datetime64) -> (np.array, np.array, np.array):
        return self.leeway.forecast(obs,t1)

    def metadata(self):
        return "Leeway drift model"

# The dynamic model
class Dynamic(Strategy):
    import dynamic

    def forecast(self, obs: Observation, t1: np.datetime64) -> (np.array, np.array, np.array):
        return self.dynamic.forecast(obs,t1)

    def metadata(self):
        return "Dynamic drift model"


if __name__ == "__main__":
    ()
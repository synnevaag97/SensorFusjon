# %% Imports
from typing import Any, Dict
from dataclasses import dataclass
import numpy as np
from numpy import ndarray

import solution
# %% Measurement models interface declaration


@dataclass
class MeasurementModel:
    def h(self, x: ndarray, **kwargs) -> ndarray:
        """Calculate the noise free measurement location at x in sensor_state.
        Args:
            x (ndarray): state
        """
        raise NotImplementedError

    def H(self, x: ndarray, **kwargs) -> ndarray:
        """Calculate the measurement Jacobian matrix at x in sensor_state.
        Args:
            x (ndarray): state
        """
        raise NotImplementedError

    def R(self, x: ndarray, **kwargs) -> ndarray:
        """Calculate the measurement covariance matrix at x in sensor_state.
        Args:
            x (ndarray): state
        """
        raise NotImplementedError


@dataclass
class CartesianPosition2D(MeasurementModel):
    sigma_z: float

    def h(self, x: ndarray) -> ndarray:
        """Calculate the noise free measurement location at x in sensor_state.
        """

        # TODO replace this with your own code
        x_h = self.H(x)@x

        return x_h

    def H(self, x: ndarray) -> ndarray:
        """Calculate the measurement Jacobian matrix at x in sensor_state."""

        # TODO replace this with your own code
        H = np.array([[1,0,0,0], [0,1,0,0]])

        return H

    def R(self, x: ndarray) -> ndarray:
        """Calculate the measurement covariance matrix at x in sensor_state."""

        # TODO replace this with your own code
        R = np.array([[self.sigma_z**2,0],[0, self.sigma_z**2]])

        return R

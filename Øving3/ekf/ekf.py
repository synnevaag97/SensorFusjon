"""
Notation:
----------
x is generally used for either the state or the mean of a gaussian. It should be clear from context which it is.
P is used about the state covariance
z is a single measurement
Z are multiple measurements so that z = Z[k] at a given time step k
v is the innovation z - h(x)
S is the innovation covariance
"""
from typing import Optional
from dataclasses import dataclass, field
import numpy as np
import scipy.linalg as la

from config import DEBUG
from dynamicmodels import DynamicModel
from measurementmodels import MeasurementModel
from utils.gaussparams import MultiVarGaussian

import solution
# %% The EKF


@dataclass
class EKF:
    dynamic_model: DynamicModel
    sensor_model: MeasurementModel

    def predict(self,
                state_upd_prev_gauss: MultiVarGaussian,
                Ts: float,
                ) -> MultiVarGaussian:
        """Predict the EKF state Ts seconds ahead."""

        # TODO replace this with your own code
        F = self.dynamic_model.F(state_upd_prev_gauss.mean, Ts)
        Q = self.dynamic_model.Q(state_upd_prev_gauss.mean, Ts)
        x_pred = self.dynamic_model.f(state_upd_prev_gauss.mean, Ts) 
        
        P_pred = F@state_upd_prev_gauss.cov@F.T + Q
        state_pred_gauss = MultiVarGaussian(x_pred,P_pred)

        return state_pred_gauss

    def predict_measurement(self,
                            state_pred_gauss: MultiVarGaussian
                            ) -> MultiVarGaussian:
        """Predict measurement pdf from using state pdf and model."""
        x_bar, P = state_pred_gauss
        H = self.sensor_model.H(x_bar)
        R = self.sensor_model.R(x_bar)
        z_bar = self.sensor_model.h(x_bar)  # TODO
        S = H@P@H.T + R  # TODO
        measure_pred_gauss = MultiVarGaussian(z_bar, S)

        return measure_pred_gauss

    def update(self,
               z: np.ndarray,
               state_pred_gauss: MultiVarGaussian,
               measurement_gauss: Optional[MultiVarGaussian] = None,
               ) -> MultiVarGaussian:
        """Given the prediction and innovation, 
        find the updated state estimate."""
        x_pred, P = state_pred_gauss
        if measurement_gauss is None:
            measurement_gauss = self.predict_measurement(state_pred_gauss)

        z_bar, S = measurement_gauss
        H = self.sensor_model.H(x_pred)
        w = P@H.T@np.linalg.inv(S)
        v = z - z_bar
        x_upd = x_pred + w@v  # TODO

        P_upd = (np.eye(len(w)) - w@H)@P  # TODOy
    
        state_upd_gauss = MultiVarGaussian(x_upd, P_upd)

        return state_upd_gauss

    def step_with_info(self,
                       state_upd_prev_gauss: MultiVarGaussian,
                       z: np.ndarray,
                       Ts: float,
                       ) -> tuple[MultiVarGaussian,
                                  MultiVarGaussian,
                                  MultiVarGaussian]:
        """
        Predict ekfstate Ts units ahead and then update this prediction with z.

        Returns:
            state_pred_gauss: The state prediction
            measurement_pred_gauss: 
                The measurement prediction after state prediction
            state_upd_gauss: The predicted state updated with measurement
        """

        # TODO replace this with your own code
        state_pred_gauss = self.predict(state_upd_prev_gauss,Ts)
        measurement_pred_gauss = self.predict_measurement(state_pred_gauss)
        state_upd_gauss = self.update(z,state_pred_gauss)

        return state_pred_gauss, measurement_pred_gauss, state_upd_gauss

    def step(self,
             state_upd_prev_gauss: MultiVarGaussian,
             z: np.ndarray,
             Ts: float,
             ) -> MultiVarGaussian:

        _, _, state_upd_gauss = self.step_with_info(state_upd_prev_gauss,
                                                    z, Ts)
        return state_upd_gauss

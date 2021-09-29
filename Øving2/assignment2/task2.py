import numpy as np
import solution
from numpy import invert, ndarray
from scipy.stats import norm


def condition_mean(x: ndarray, P: ndarray,
                   z: ndarray, R: ndarray, H: ndarray) -> ndarray:
    """compute conditional mean

    Args:
        x (ndarray): initial state
        P (ndarray): initial state covariance
        z (ndarray): measurement
        R (ndarray): measurement covariance
        H (ndarray): measurement matrix i.e. z = H @ x + error

    Returns:
        cond_mean (ndarray): conditioned mean (state)
    """

    # TODO replace this with your own code
    # We use @ instead of *. * is elementwise multiplication and @ is matrix multiplication. 
    S = H@P@H.T+R
    S_inv = np.linalg.inv(S)
    cond_mean = x + P@H.T@S_inv@(z-H@x)

    #cond_mean = solution.task2.condition_mean(x, P, z, R, H)

    return cond_mean


def condition_cov(P: ndarray, R: ndarray, H: ndarray) -> ndarray:
    """compute conditional covariance

    Args:
        P (ndarray): covariance of state estimate
        R (ndarray): covariance of measurement
        H (ndarray): measurement matrix

    Returns:
        ndarray: the conditioned covariance
    """

    # TODO replace this with your own code
    S = H@P@H.T+R
    S_inv = np.linalg.inv(S)
    conditional_cov = P - P@H.T@S_inv@H@P
    #conditional_cov = solution.task2.condition_cov(P, R, H)

    return conditional_cov


def get_task_2f(x_bar: ndarray, P: ndarray,
                z_c: ndarray, R_c: ndarray, H_c: ndarray,
                z_r: ndarray, R_r: ndarray, H_r: ndarray
                ):
    """get state estimates after receiving measurement c or measurement r

    Args:
        x_bar (ndarray): initial state estimate
        P (ndarray): covariance of x_bar
        z_c (ndarray): measurement c
        R_c (ndarray): covariance of measurement c
        H_c (ndarray): measurement matrix i.e. z_c = H_c @ x + error
        z_r (ndarray): measurement r
        R_r (ndarray): covariance of measurement r
        H_r (ndarray): measurement matrix i.e. z_r + H_c @ x + error

    Returns:
        x_bar_c (ndarray): state estimate after measurement c
        P_c (ndarray): covariance of x_bar_c
        x_bar_r (ndarray): state estimate after measurement r
        P_r (ndarray): covariance of x_bar_r
    """

    # TODO replace this with your own code
    x_bar_c = condition_mean(x_bar,P,z_c,R_c,H_c)
    x_bar_r = condition_mean(x_bar, P, z_r, R_r, H_r)
    P_c = condition_cov(P,R_c,H_r)
    P_r = condition_cov(P,R_r, H_r)
    #x_bar_c, P_c, x_bar_r, P_r = solution.task2.get_task_2f(
        #x_bar, P, z_c, R_c, H_c, z_r, R_r, H_r)

    return x_bar_c, P_c, x_bar_r, P_r


def get_task_2g(x_bar_c: ndarray, P_c: ndarray,
                x_bar_r: ndarray, P_r: ndarray,
                z_c: ndarray, R_c: ndarray, H_c: ndarray,
                z_r: ndarray, R_r: ndarray, H_r: ndarray):
    """get state estimates after receiving measurement c and measurement r

    Args:
        x_bar_c (ndarray): state estimate after receiving measurement c
        P_c (ndarray): covariance of x_bar_c
        x_bar_r (ndarray): state estimate after receiving measurement r
        P_r (ndarray): covariance of x_bar_r
        z_c (ndarray): measurement c
        R_c (ndarray): covariance of measurement c
        H_c (ndarray): measurement matrix i.e. z_c = H_c @ x + error
        z_r (ndarray): measurement r
        R_r (ndarray): covariance of measurement r
        H_r (ndarray): measurement matrix i.e. z_r = H_r @ x + error

    Returns:
        x_bar_cr (ndarray): state estimate after receiving z_c then z_r
        P_cr (ndarray): covariance of x_bar_cr
        x_bar_rc (ndarray): state estimate after receiving z_r then z_c
        P_rc (ndarray): covariance of x_bar_rc
    """

    # TODO replace this with your own code
    # To solve we calculate the cr : We have the mean and covariance for p(x|z_c).
    # We then calculate the mean and covariance for p(x|z_c,z_r) from the mean and covariance for p(x|z_c).
    x_bar_cr = condition_mean(x_bar_c, P_c, z_r, R_r, H_r)
    P_cr = condition_cov(P_c, R_r, H_r)
    x_bar_rc = condition_mean(x_bar_r, P_r, z_c, R_c, H_c)
    P_rc = condition_cov(P_r, R_c, H_c)

    return x_bar_cr, P_cr, x_bar_rc, P_rc


def get_task_2h(x_bar_rc: ndarray, P_rc: ndarray):
    """get the probability that the boat is above the line

    Args:
        x_bar_rc (ndarray): state
        P_rc (ndarray): covariance

    Returns:
        prob_above_line: the probability that the boat is above the line
    """

    # TODO replace this with your own code
    x = np.array([[-1],[1]]).T
    prob_above_line = 1 - norm.cdf(5, x@x_bar_rc, np.sqrt(x@P_rc@x.T)) # This calculates the value of the cdf at the given value 5 with the given mean and covariance. 
    

    #prob_above_line = solution.task2.get_task_2h(x_bar_rc, P_rc)

    return prob_above_line

import numpy as np
from numpy import ndarray

import solution


def mixture_moments(weights: ndarray,
                    means: ndarray,
                    covs: ndarray,
                    ) -> tuple[ndarray, ndarray]:
    """Calculate the first two moments of a Gaussian mixture.

    Args:
        weights: shape = (N,)
        means: shape = (N, n)
        covs: shape = (N, n, n)

    Returns:
        mean: shape = (n,)
        cov: shape = (n, n)
    """
    #mean = np.zeros((len(means),len(means[0])))
    mean = np.zeros((len(means[0])))
    for i in range(len(weights)):
        mean = mean + weights[i]*means[i]  # TODO
    
    # internal covariance
    cov_internal = np.zeros((len(covs[0]),len(covs[0])))
    for i in range(len(weights)):
        cov_internal = cov_internal + weights[i]*covs[i]  # TODO

    # spread of means, aka. external covariance
    # If you vectorize: take care to make the operation order be symetric
    diffs = np.zeros((len(covs[0]),len(covs[0])))
    for i in range(len(weights)):
        diffs = diffs + weights[i]*means[i]*means[i].T  # TODO: optional intermediate
    cov_external = diffs - mean@mean.T # TODO: Hint loop, broadcast or np.einsum

    # total covariance
    cov = cov_internal + cov_external  # TODO

    # TODO replace this with your own code
    #mean, cov = solution.mixturereduction.mixture_moments(weights, means, covs)

    return mean, cov

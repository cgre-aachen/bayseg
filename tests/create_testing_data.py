import numpy as np
import scipy.stats
from scipy.stats import multivariate_normal


def create_1d_data():
    # create vectors
    coords = np.array([np.arange(500)]).T

    # define domain
    latent_1d = np.ones(25) * 2
    latent_1d = np.append(latent_1d, np.ones(75) * 0)
    latent_1d = np.append(latent_1d, np.ones(100) * 1)
    latent_1d = np.append(latent_1d, np.ones(50) * 2)
    latent_1d = np.append(latent_1d, np.ones(25) * 1)
    latent_1d = np.append(latent_1d, np.ones(25) * 2)
    latent_1d = np.append(latent_1d, np.ones(100) * 0)
    latent_1d = np.append(latent_1d, np.ones(100) * 2)

    # sample
    i = 4
    c1 = scipy.stats.multivariate_normal([7, 9, 10, 2.7], np.eye(i) * 0.35)
    c2 = scipy.stats.multivariate_normal([8, 8, 9.5, 2], np.eye(i) * 0.55)
    c3 = scipy.stats.multivariate_normal([10, 9.5, 9, 1.5], np.eye(i) * 10.25)

    obs = np.empty((len(coords[:, 0]), 4))

    for i, label in enumerate(latent_1d):
        # generate a 2-d random vector at each point
        if label == 0:
            obs[i, :] = c1.rvs()
        elif label == 1:
            obs[i, :] = c2.rvs()
        elif label == 2:
            obs[i, :] = c3.rvs()

    return obs, latent_1d


def func(x, y):
    return np.sin(y * x)


def create_2d_data(ny, nx):
    # **********************************************************************************************************
    # latent field
    xaxis = np.linspace(0, 4, ny)
    yaxis = np.linspace(0, 4, nx)
    a, b = np.meshgrid(yaxis, xaxis)
    result = func(a, b)

    latent_2d = np.zeros((ny, nx))
    latent_2d[result > -0.5] = 1
    latent_2d[result > 0.5] = 2

    # **********************************************************************************************************
    # sample
    f = 4
    c1 = multivariate_normal([7.5, 7.5, 10.5, 3.5], np.eye(f) * 0.35)
    c2 = multivariate_normal([8, 7.9, 10, 2.9], np.eye(f) * 0.55)
    c3 = multivariate_normal([8.5, 9, 9.5, 1], np.eye(f) * 0.75)

    obs = np.zeros((ny, nx, f))

    for y in range(ny):
        for x in range(nx):
            # generate a 2-d random vector at each point
            if latent_2d[y, x] == 0:
                obs[y, x, :] = c1.rvs()
            elif latent_2d[y, x] == 1:
                obs[y, x, :] = c2.rvs()
            elif latent_2d[y, x] == 2:
                obs[y, x, :] = c3.rvs()

    # feature_vector = np.array([obs[:, :, 0].flatten(), obs[:, :, 1].flatten(), obs[:, :, 2].flatten(), obs[:, :, 3].flatten()]).T

    # **********************************************************************************************************
    return obs, latent_2d
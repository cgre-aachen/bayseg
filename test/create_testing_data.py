import numpy as np
import scipy.stats


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
    c3 = scipy.stats.multivariate_normal([10, 9.5, 9, 1.5], np.eye(i) * 0.25)

    obs = np.empty((len(coords[:, 0]), 4))

    for i, label in enumerate(latent_1d):
        # generate a 2-d random vector at each point
        if label == 0:
            obs[i, :] = c1.rvs()
        elif label == 1:
            obs[i, :] = c2.rvs()
        elif label == 2:
            obs[i, :] = c3.rvs()

    return coords, obs, latent_1d

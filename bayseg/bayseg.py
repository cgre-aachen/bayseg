"""
BaySeg is a Python library for unsupervised clustering of n-dimensional data sets, designed for the segmentation of
one-, two- and three-dimensional data in the field of geological modeling and geophysics. The library is based on the
algorithm developed by Wang et al., 2017 and combines Hidden Markov Random Fields with Gaussian Mixture Models in a
Bayesian inference framework.

************************************************************************************************
References

[1] Wang, H., Wellmann, J. F., Li, Z., Wang, X., & Liang, R. Y. (2017). A Segmentation Approach for Stochastic
    Geological Modeling Using Hidden Markov Random Fields. Mathematical Geosciences, 49(2), 145-177.

************************************************************************************************
@authors: Alexander Schaaf, Hui Wang, Florian Wellmann
************************************************************************************************
BaySeg is licensed under the GNU Lesser General Public License v3.0
************************************************************************************************
"""

import numpy as np  # scientific computing library
from sklearn import mixture  # gaussian mixture model
from scipy.stats import multivariate_normal, norm  # normal distributions
from copy import copy
from itertools import combinations
import tqdm  # smart-ish progress bar
import matplotlib.pyplot as plt  # 2d plotting
from matplotlib import gridspec, rcParams  # plot arrangements
from .colors import cmap, cmap_norm  # custom colormap
from .ie import *

plt.style.use('bmh')  # plot style


class BaySeg:
    def __init__(self, data, n_labels, beta_init=1, stencil=None, normalize=True):
        """

        Args:
            data (:obj:`np.ndarray`): Multidimensional data array containing all observations (features) in the
                following shape:

                    1D = (Y, F)
                    2D = (Y, X, F)
                    3D = (Y, X, Z, F)

            n_labels (int): Number of labels representing the number of clusters to be segmented.
            beta_init (float): Initial penalty value for Gibbs energy calculation.
            stencil (int): Number specifying the stencil of the neighborhood system used in the Gibbs energy
                calculation.

        """
        # TODO: [DOCS] Main object description

        # store initial data
        self.data = data
        # get shape for physical and feature dimensions
        self.shape = np.shape(data)
        self.phys_shp = np.array(self.shape[:-1])

        # get number of features
        self.n_feat = self.shape[-1]

        # GRAPH COLORING
        self.stencil = stencil
        self.colors = pseudocolor(self.shape, self.stencil)

        # ************************************************************************************************
        # fetch dimensionality, coordinate and feature vector from input data

        # 1D
        if len(self.shape) == 2:
            # 1d case
            self.dim = 1
            # create coordinate vector
            # self.coords = np.array([np.arange(self.shape[0])]).T
            # feature vector
            self.feat = self.data

        # 2D
        elif len(self.shape) == 3:
            # 2d case
            self.dim = 2
            # create coordinate vector
            # y, x = np.indices(self.shape[:-1])
            # print(y, x)
            # self.coords = np.array([y.flatten(), x.flatten()]).T

            # feature vector
            self.feat = np.array([self.data[:, :, f].ravel() for f in range(self.n_feat)]).T

        # 3D
        elif len(self.shape) == 4:
            # 3d case
            raise Exception("3D segmentation not yet supported.")

        # mismatch
        else:
            raise Exception("Data format appears to be wrong (neither 1-, 2- or 3-D).")

        if normalize:
            self.normalize_feature_vectors()

        # ************************************************************************************************
        # INIT STORAGE ARRAYS

        # self.betas = [beta_init]  # initial beta
        # self.mus = np.array([], dtype=object)
        # self.covs = np.array([], dtype=object)
        # self.labels = np.array([], dtype=object)

        # ************************************************************************************************
        # INIT GAUSSIAN MIXTURE MODEL
        self.n_labels = n_labels
        self.gmm = mixture.GaussianMixture(n_components=n_labels, covariance_type="full")
        self.gmm.fit(self.feat)
        # do initial prediction based on fit and observations, store as first entry in labels

        # ************************************************************************************************
        # INIT LABELS, MU and COV based on GMM
        # TODO: [GENERAL] storage variables from lists to numpy ndarrays
        self.labels = [self.gmm.predict(self.feat)]
        # INIT MU (mean from initial GMM)
        self.mus = [self.gmm.means_]
        # INIT COV (covariances from initial GMM)
        self.covs = [self.gmm.covariances_]

        self.labels_probability = []
        self.storage_gibbs_e = []
        self.storage_like_e = []
        self.storage_te = []

        self.beta_acc_ratio = np.array([])
        self.cov_acc_ratio = np.array([])
        self.mu_acc_ratio = np.array([])

        # ************************************************************************************************
        # Initialize PRIOR distributions for beta, mu and covariance
        # BETA
        if self.dim == 1:
            self.prior_beta = norm(beta_init, np.eye(1) * 100)
            self.betas = [beta_init]
        elif self.dim == 2:
            if self.stencil == "4p":
                beta_dim = 2
            elif self.stencil == "8p" or self.stencil is None:
                beta_dim = 4

            self.betas = [[beta_init for i in range(beta_dim)]]
            self.prior_beta = multivariate_normal([beta_init for i in range(beta_dim)], np.eye(beta_dim) * 100)

        elif self.dim == 3:
            raise Exception("3D not yet supported.")

        # MU
        # generate distribution means for each label
        prior_mu_means = [self.mus[0][label] for label in range(self.n_labels)]
        # generate distribution covariances for each label
        prior_mu_stds = [np.eye(self.n_feat) * 100 for label in range(self.n_labels)]
        # use the above to generate multivariate normal distributions for each label
        self.priors_mu = [multivariate_normal(prior_mu_means[label], prior_mu_stds[label]) for label in
                          range(self.n_labels)]

        # COV
        # generate b_sigma
        self.b_sigma = np.zeros((self.n_labels, self.n_feat))
        for l in range(self.n_labels):
            self.b_sigma[l, :] = np.log(np.sqrt(np.diag(self.gmm.covariances_[l, :, :])))
        # generate kesi
        self.kesi = np.ones((self.n_labels, self.n_feat)) * 100
        # generate nu
        self.nu = self.n_feat + 1
        # ************************************************************************************************

    def fit(self, n, beta_jump_length=10, mu_jump_length=0.0005, cov_volume_jump_length=0.00005,
            theta_jump_length=0.0005, t=1., verbose=False, fix_beta=False):
        """Fit the segmentation parameters to the given data.

        Args:
            n (int): Number of iterations.
            beta_jump_length (float): Hyperparameter specifying the beta proposal jump length.
            mu_jump_length (float): Hyperparameter for the mean proposal jump length.
            cov_volume_jump_length (float):
            theta_jump_length (float):
            t (float):
            verbose (bool or :obj:`str`):
            fix_beta (bool):

        """
        for g in tqdm.trange(n):
            self.gibbs_sample(t, beta_jump_length, mu_jump_length, cov_volume_jump_length, theta_jump_length,
                              verbose, fix_beta)

    def gibbs_sample(self, t, beta_jump_length, mu_jump_length, cov_volume_jump_length, theta_jump_length, verbose,
                     fix_beta):
        """Takes care of the Gibbs sampling. This is the main function of the algorithm.

        Args:
            t: Hyperparameter
            beta_jump_length: Hyperparameter
            mu_jump_length: Hyperparameter
            cov_volume_jump_length: Hyperparameter
            theta_jump_length: Hyperparameter
            verbose (bool or :obj:`str`): Toggles verbosity.
            fix_beta (bool): Fixed beta to the inital value if True, else adaptive.

        Returns:
            The function updates directly on the object variables and appends new draws of labels and
            parameters to their respective storages.
        """
        # TODO: [GENERAL] In-depth description of the gibbs sampling function

        # ************************************************
        # CALCULATE TOTAL ENERGY
        # 1 - calculate energy likelihood for each element and label
        # way to avoid over-smoothing by the gibbs energy
        energy_like = self.calc_energy_like(self.mus[-1], self.covs[-1])
        if verbose == "energy":
            print("likelihood energy:", energy_like)
        # 2 - calculate gibbs/mrf energy
        gibbs_energy = self._calc_gibbs_energy_vect(self.labels[-1], self.betas[-1], verbose=verbose)
        if verbose == "energy":
            print("gibbs energy:", gibbs_energy)
        # 3 - self energy
        self_energy = np.zeros(self.n_labels)  # to fix specific labels to locations, theoretically
        # 5 - calculate total energy
        # total energy vector 2d: n_elements * n_labels
        total_energy = energy_like + gibbs_energy  # + self_energy
        if verbose == "energy":
            print("total_energy:", total_energy)
        # ************************************************************************************************
        # CALCULATE PROBABILITY OF LABELS
        labels_prob = _calc_labels_prob(total_energy, t)
        if verbose == "energy":
            print("Labels probability:", labels_prob)

        self.storage_te.append(total_energy)

        # make copy of previous labels
        new_labels = copy(self.labels[-1])
        # new_labels = np.empty_like(self.labels[-1])

        for i, color_f in enumerate(self.colors):
            # print(np.average(new_labels))
            new_labels[color_f] = draw_labels_vect(labels_prob[color_f])
            # print(np.average(new_labels))
            # now recalculate gibbs energy and other energies from the mixture of old and new labels
            gibbs_energy = self._calc_gibbs_energy_vect(new_labels, self.betas[-1], verbose=verbose)
            total_energy = energy_like + gibbs_energy  # + self_energy
            labels_prob = _calc_labels_prob(total_energy, t)

        self.labels_probability.append(labels_prob)
        self.labels.append(new_labels)

        # ************************************************************************************************
        # calculate energy for component coefficient
        energy_for_comp_coef = gibbs_energy  # + self_energy
        # print("ge shp:", gibbs_energy)
        # ************************************************************************************************
        # CALCULATE COMPONENT COEFFICIENT
        comp_coef = _calc_labels_prob(energy_for_comp_coef, t)
        # ************************************************************************************************
        # ************************************************************************************************
        # PROPOSAL STEP
        # make proposals for beta, mu and cov
        # beta depends on physical dimensions, for 1d its size 1
        beta_prop = self.propose_beta(self.betas[-1], beta_jump_length)
        # print("beta prop:", beta_prop)
        mu_prop = self.propose_mu(self.mus[-1], mu_jump_length)
        # print("mu prop:", mu_prop)
        cov_prop = _propose_cov(self.covs[-1], self.n_feat, self.n_labels, cov_volume_jump_length, theta_jump_length)
        # print("cov_prop:", cov_prop)

        # ************************************************************************************************
        # Compare mu, cov and beta proposals with previous, then decide which to keep for next iteration

        # prepare next ones
        mu_next = copy(self.mus[-1])
        cov_next = copy(self.covs[-1])

        # ************************************************************************************************
        # UPDATE MU
        for l in range(self.n_labels):
            # log-prob prior density for mu
            mu_temp = copy(mu_next)
            mu_temp[l, :] = mu_prop[l, :]

            lp_mu_prev = self.log_prior_density_mu(mu_next, l)
            lp_mu_prop = self.log_prior_density_mu(mu_temp, l)

            lmd_prev = self.calc_sum_log_mixture_density(comp_coef, mu_next, cov_next)
            # calculate log mixture density for proposed mu and cov
            lmd_prop = self.calc_sum_log_mixture_density(comp_coef, mu_temp, cov_next)

            # combine
            log_target_prev = lmd_prev + lp_mu_prev
            log_target_prop = lmd_prop + lp_mu_prop

            mu_eval = evaluate(log_target_prop, log_target_prev)
            if mu_eval[0]:
                mu_next[l, :] = mu_prop[l, :]
            else:
                pass
            self.mu_acc_ratio = np.append(self.mu_acc_ratio, mu_eval[1])

        self.mus.append(mu_next)

        # ************************************************************************************************
        # UPDATE COVARIANCE
        for l in range(self.n_labels):
            cov_temp = copy(cov_next)
            cov_temp[l, :, :] = cov_prop[l, :, :]

            # print("cov diff:", cov_next[l, :, :]-cov_temp[l, :, :])

            # log-prob prior density for covariance
            lp_cov_prev = self.log_prior_density_cov(cov_next, l)
            # print("lp_cov_prev:", lp_cov_prev)
            lp_cov_prop = self.log_prior_density_cov(cov_temp, l)
            # print("lp_cov_prop:", lp_cov_prop)

            lmd_prev = self.calc_sum_log_mixture_density(comp_coef, mu_next, cov_next)
            # print("lmd_prev:", lmd_prev)
            # calculate log mixture density for proposed mu and cov
            lmd_prop = self.calc_sum_log_mixture_density(comp_coef, mu_next, cov_temp)
            # print("lmd_prop:", lmd_prop)

            # combine
            log_target_prev = lmd_prev + lp_cov_prev
            log_target_prop = lmd_prop + lp_cov_prop

            mu_eval = evaluate(log_target_prop, log_target_prev)
            if mu_eval[0]:
                cov_next[l, :] = cov_prop[l, :]
            else:
                pass
            self.cov_acc_ratio = np.append(self.cov_acc_ratio, mu_eval[1])

        # append cov and mu
        self.covs.append(cov_next)
        self.storage_gibbs_e.append(gibbs_energy)
        self.storage_like_e.append(energy_like)

        if not fix_beta:
            # ************************************************************************************************
            # UPDATE BETA
            lp_beta_prev = self.log_prior_density_beta(self.betas[-1])
            lp_beta_prop = self.log_prior_density_beta(beta_prop)

            lmd_prev = self.calc_sum_log_mixture_density(comp_coef, self.mus[-1], self.covs[-1])

            # calculate gibbs energy with new labels and proposed beta
            gibbs_energy_prop = self._calc_gibbs_energy_vect(self.labels[-1], beta_prop, verbose=verbose)
            energy_for_comp_coef_prop = gibbs_energy_prop  # + self_energy
            comp_coef_prop = _calc_labels_prob(energy_for_comp_coef_prop, t)

            lmd_prop = self.calc_sum_log_mixture_density(comp_coef_prop, self.mus[-1], self.covs[-1])
            # print("lmd_prev:", lmd_prev)
            # print("lp_beta_prev:", lp_beta_prev)
            log_target_prev = lmd_prev + lp_beta_prev
            # print("lmd_prop:", lmd_prop)
            # print("lp_beta_prop:", lp_beta_prop)
            log_target_prop = lmd_prop + lp_beta_prop

            mu_eval = evaluate(log_target_prop, log_target_prev)
            if mu_eval[0]:
                self.betas.append(beta_prop)
            else:
                self.betas.append(self.betas[-1])
            self.beta_acc_ratio = np.append(self.beta_acc_ratio, mu_eval[1])  # store

            # acc_ratio = np.exp(log_target_prop - log_target_prev)
            # # print("beta acc_ratio:", acc_ratio)
            #
            # if verbose:
            #     print("BETA acceptance ratio:", acc_ratio)
            #
            # if (acc_ratio > 1) or (np.random.uniform() < acc_ratio):
            #     self.betas.append(beta_prop)
            # else:
            #     self.betas.append(self.betas[-1])

        else:
            self.betas.append(self.betas[-1])
            # ************************************************************************************************

    def log_prior_density_mu(self, mu, label):
        """Calculates the summed log prior density for a given mean and labels array."""
        with np.errstate(divide='ignore'):
            return np.sum(np.log(self.priors_mu[label].pdf(mu)))

    def log_prior_density_beta(self, beta):
        """Calculates the log prior density for a given beta array."""
        return np.log(self.prior_beta.pdf(beta))

    def log_prior_density_cov(self, cov, l):
        """Calculates the summed log prior density for the given covariance matrix and labels array."""
        lam = np.sqrt(np.diag(cov[l, :, :]))
        r = np.diag(1. / lam) @ cov[l, :, :] @ np.diag(1. / lam)
        logp_r = -0.5 * (self.nu + self.n_feat + 1) * np.log(np.linalg.det(r)) - self.nu / 2. * np.sum(
            np.log(np.diag(np.linalg.inv(r))))
        logp_lam = np.sum(np.log(multivariate_normal(mean=self.b_sigma[l, :], cov=self.kesi[l, :]).pdf(np.log(lam.T))))
        return logp_r + logp_lam

    def propose_beta(self, beta_prev, beta_jump_length):
        """Proposes a perturbed beta based on a jump length hyperparameter.

        Args:
            beta_prev:
            beta_jump_length:

        Returns:

        """
        # create proposal covariance depending on physical dimensionality
        # dim = [1, 4, 13]
        if self.dim == 1:
            beta_dim = 1

        elif self.dim == 2:
            if self.stencil == "4p":
                beta_dim = 2
            elif self.stencil == "8p" or self.stencil is None:
                beta_dim = 4

        elif self.dim == 3:
            raise Exception("3D not yet supported.")

        sigma_prop = np.eye(beta_dim) * beta_jump_length
        # draw from multivariate normal distribution and return
        # return np.exp(multivariate_normal(mean=np.log(beta_prev), cov=sigma_prop).rvs())
        return multivariate_normal(mean=beta_prev, cov=sigma_prop).rvs()

    def propose_mu(self, mu_prev, mu_jump_length):
        """Proposes a perturbed mu matrix using a jump length hyperparameter.

        Args:
            mu_prev (:obj:`np.ndarray`): Previous mean array for all labels and features
            mu_jump_length (float or int): Hyperparameter specifying the jump length for the new proposal mean array.

        Returns:
            :obj:`np.ndarray`: The newly proposed mean array.

        """
        # prepare matrix
        mu_prop = np.ones((self.n_labels, self.n_feat))
        # loop over labels
        for l in range(self.n_labels):
            mu_prop[l, :] = multivariate_normal(mean=mu_prev[l, :], cov=np.eye(self.n_feat) * mu_jump_length).rvs()
        return mu_prop

    def calc_sum_log_mixture_density(self, comp_coef, mu, cov):
        """Calculate sum of log mixture density with each observation at every element.

        Args:
            comp_coef (:obj:`np.ndarray`): Component coefficient for each element (row) and label (column).
            mu (:obj:`np.ndarray`): Mean value array for all labels and features.
            cov (:obj:`np.ndarray`): Covariance matrix.

        Returns:
            float: Summed log mixture density.

        """
        lmd = np.zeros((self.phys_shp.prod(), self.n_labels))

        for l in range(self.n_labels):
            draw = multivariate_normal(mean=mu[l, :], cov=cov[l, :, :]).pdf(self.feat)
            # print(np.shape(lmd[:,l]))
            multi = comp_coef[:, l] * np.array([draw])
            lmd[:, l] = multi
        lmd = np.sum(lmd, axis=1)
        with np.errstate(divide='ignore'):
            lmd = np.log(lmd)

        return np.sum(lmd)

    def calc_energy_like(self, mu, cov):
        """Calculates the energy likelihood for a given mean array and covariance matrix for the entire domain.

        Args:
            mu (:obj:`np.ndarray`):
            cov (:obj:`np.ndarray`):
            vect (bool, optional): Toggles the vectorized implementation. False activates the loop-based version if
                you really dig a loss of speed of about 350 times.

        Returns:
            :obj:`np.ndarray` : Energy likelihood for each label at each element.
        """
        energy_like_labels = np.zeros((self.phys_shp.prod(), self.n_labels))

        # uses flattened features array
        for l in range(self.n_labels):
            energy_like_labels[:, l] = np.einsum("...i,ji,...j",
                                                 0.5 * np.array([self.feat - mu[l, :]]),
                                                 np.linalg.inv(cov[l, :, :]),
                                                 np.array([self.feat - mu[l, :]])) + 0.5 * np.log(
                np.linalg.det(cov[l, :, :]))

        return energy_like_labels

    def _calc_gibbs_energy_vect(self, labels, beta, verbose=False):
        """Calculates the Gibbs energy for each element using the penalty factor(s) beta.

        Args:
            labels (:obj:`np.ndarray`):
            beta (:obj:`np.array` of float):
            verbose (bool):

        Returns:
            :obj:`np.ndarray` : Gibbs energy at every element for each label.
        """
        # ************************************************************************************************
        # 1D
        if self.dim == 1:
            # tile
            lt = np.tile(labels, (self.n_labels, 1)).T

            ge = np.arange(self.n_labels)  # elements x labels
            ge = np.tile(ge, (len(labels), 1)).astype(float)

            # first row
            top = np.expand_dims(np.not_equal(np.arange(self.n_labels), lt[1, :]) * beta, axis=0)
            # mid
            mid = (np.not_equal(ge[1:-1, :], lt[:-2, :]).astype(float) + np.not_equal(ge[1:-1, :], lt[2:, :]).astype(
                float)) * beta
            # last row
            bot = np.expand_dims(np.not_equal(np.arange(self.n_labels), lt[-2, :]) * beta, axis=0)
            # put back together and return gibbs energy
            return np.concatenate((top, mid, bot))

        # ************************************************************************************************
        # 2D
        elif self.dim == 2:

            # reshape the labels to 2D for "stencil-application"
            labels = labels.reshape(self.shape[0], self.shape[1])

            # prepare gibbs energy array (filled with zeros)
            ge = np.tile(np.zeros_like(labels).astype(float), (self.n_labels, 1, 1))

            # create comparison array containing the different labels
            comp = np.tile(np.zeros_like(labels), (self.n_labels, 1, 1)).astype(float)
            for i in range(self.n_labels):
                comp[i, :, :] = i

            # anisotropic beta directions
            #  3  1  2
            #   \ | /
            #   --+-- 0
            #   / | \

            # **********************************************************************************************************
            # direction 0 = 0° polar coord system
            ge[:, 1:-1, 1:-1] += (np.not_equal(comp[:, 1:-1, 1:-1], labels[:-2, 1:-1]).astype(float)  # compare with left
                                  + np.not_equal(comp[:, 1:-1, 1:-1], labels[2:, 1:-1]).astype(float)) * beta[0]  # compare with right

            # left column
            # right
            ge[:, :, 0] += np.not_equal(comp[:, :, 0], labels[:, 1]).astype(float) * beta[0]
            # right column
            # left
            ge[:, :, -1] += np.not_equal(comp[:, :, -1], labels[:, -2]).astype(float) * beta[0]
            # top row
            # right
            ge[:, 0, :-1] += np.not_equal(comp[:, 0, :-1], labels[0, 1:]).astype(float) * beta[0]
            # left
            ge[:, 0, 1:] += np.not_equal(comp[:, 0, 1:], labels[0, :-1]).astype(float) * beta[0]
            # bottom row
            # right
            ge[:, -1, :-1] += np.not_equal(comp[:, -1, :-1], labels[-1, 1:]).astype(float) * beta[0]
            # left
            ge[:, -1, 1:] += np.not_equal(comp[:, -1, 1:], labels[-1, :-1]).astype(float) * beta[0]

            # **********************************************************************************************************
            # direction 1 = 90° polar coord system
            ge[:, 1:-1, 1:-1] += (np.not_equal(comp[:, 1:-1, 1:-1], labels[1:-1, :-2]).astype(float)  # compare with above
                                  + np.not_equal(comp[:, 1:-1, 1:-1], labels[1:-1, 2:]).astype(float)) * beta[1]  # compare with below
            # left column
            # above
            ge[:, 1:, 0] += np.not_equal(comp[:, 1:, 0], labels[:-1, 0]).astype(float) * beta[1]
            # below
            ge[:, :-1, 0] += np.not_equal(comp[:, :-1, 0], labels[1:, 0]).astype(float) * beta[1]
            # right column
            # above
            ge[:, 1:, -1] += np.not_equal(comp[:, 1:, -1], labels[:-1, -1]).astype(float) * beta[1]
            # below
            ge[:, :-1, -1] += np.not_equal(comp[:, :-1, -1], labels[1:, -1]).astype(float) * beta[1]
            # top row
            # below
            ge[:, 0, :] += np.not_equal(comp[:, 0, :], labels[1, :]).astype(float) * beta[1]
            # bottom row
            # above
            ge[:, -1, :] += np.not_equal(comp[:, -1, :], labels[-2, :]).astype(float) * beta[1]

            # **********************************************************************************************************
            # direction 2 = 45° polar coord system
            if self.stencil is "8p":
                ge[:, 1:-1, 1:-1] += (np.not_equal(comp[:, 1:-1, 1:-1], labels[2:, :-2]).astype(float)  # compare with right up
                                      + np.not_equal(comp[:, 1:-1, 1:-1], labels[:-2, 2:]).astype(float)) * beta[2]  # compare with left down
                # left column
                # right up
                ge[:, 1:, 0] += np.not_equal(comp[:, 1:, 0], labels[:-1, 1]).astype(float) * beta[2]
                # right column
                # left down
                ge[:, :-1, -1] += np.not_equal(comp[:, :-1, -1], labels[1:, -2]).astype(float) * beta[2]
                # top row
                # below left
                ge[:, 0, 1:] += np.not_equal(comp[:, 0, 1:], labels[1, :-1]).astype(float) * beta[2]
                # bottom row
                # above right
                ge[:, -1, :-1] += np.not_equal(comp[:, -1, :-1], labels[-2, 1:]).astype(float) * beta[2]
            # **********************************************************************************************************
            # direction 3 = 135° polar coord system
            if self.stencil is "8p":
                ge[:, 1:-1, 1:-1] += (np.not_equal(comp[:, 1:-1, 1:-1], labels[:-2, :-2]).astype(float)  # compare with left up
                                      + np.not_equal(comp[:, 1:-1, 1:-1], labels[2:, 2:]).astype(float)) * beta[3]  # compare with right down
                # left column
                # right down
                ge[:, :-1, 0] += np.not_equal(comp[:, :-1, 0], labels[1:, 1]).astype(float) * beta[3]
                # right column
                # left up
                ge[:, 1:, -1] += np.not_equal(comp[:, 1:, -1], labels[:-1, -2]).astype(float) * beta[3]
                # top row
                # below right
                ge[:, 0, :-1] += np.not_equal(comp[:, 0, :-1], labels[1, 1:]).astype(float) * beta[3]
                # bottom row
                # above left
                ge[:, -1, 1:] += np.not_equal(comp[:, -1, 1:], labels[-2, :-1]).astype(float) * beta[3]

            # **********************************************************************************************************
            # overwrite corners
            # up left
            ge[:, 0, 0] = np.not_equal(comp[:, 0, 0], labels[1, 0]).astype(float) * beta[1] \
                          + np.not_equal(comp[:, 0, 0], labels[0, 1]).astype(float) * beta[0]
            if self.stencil is "8p":
                ge[:, 0, 0] += np.not_equal(comp[:, 0, 0], labels[1, 1]).astype(float) * beta[3]

            # low left
            ge[:, -1, 0] = np.not_equal(comp[:, -1, 0], labels[-1, 1]).astype(float) * beta[0] \
                           + np.not_equal(comp[:, -1, 0], labels[-2, 0]).astype(float) * beta[1]
            if self.stencil is "8p":
                ge[:, -1, 0] += np.not_equal(comp[:, -1, 0], labels[-2, 1]).astype(float) * beta[2]

            # up right
            ge[:, 0, -1] = np.not_equal(comp[:, 0, -1], labels[1, -1]).astype(float) * beta[1] \
                           + np.not_equal(comp[:, 0, -1], labels[0, -2]).astype(float) * beta[0]
            if self.stencil is "8p":
                ge[:, 0, -1] += np.not_equal(comp[:, 0, -1], labels[1, -2]).astype(float) * beta[2]

            # low right
            ge[:, -1, -1] = np.not_equal(comp[:, -1, -1], labels[-2, -1]).astype(float) * beta[1] \
                            + np.not_equal(comp[:, -1, -1], labels[-1, -2]).astype(float) * beta[0]
            if self.stencil is "8p":
                ge[:, -1, -1] += np.not_equal(comp[:, -1, -1], labels[-2, -2]).astype(float) * beta[3]

            # reshape and transpose gibbs energy, return
            return np.array([ge[l, :, :].ravel() for l in range(self.n_labels)]).T

        # ************************************************************************************************
        elif self.dim == 3:
            # TODO: [3D] implementation of gibbs energy
            raise Exception("3D not yet implemented.")

    def mcr(self, true_labels):
        """Compares classified with true labels for each iteration step (for synthetic data) to obtain a measure of
        mismatch/convergence."""
        mcr_vals = []
        n = len(true_labels)
        # TODO: [2D] implementation for MCR
        # TODO: [3D] implementation for MCR
        for label in self.labels:
            missclassified = np.count_nonzero(true_labels - label)
            mcr_vals.append(missclassified / n)
        return mcr_vals

    def get_std_from_cov(self, f, l):
        """
        Extracts standard deviation from covariance matrices for feature f and label l.
        :param f: feature (int)
        :param l: label (int)
        :return standard deviation from all covariance matrices for label/feature combination
        """
        stds = []
        for i in range(len(self.covs)):
            stds.append(np.sqrt(np.diag(self.covs[i][l])[f]))
        return stds

    def get_corr_coef_from_cov(self, l):
        """
        Extracts correlation coefficient from covariance matrix for label l.
        :param l: label (int)
        :retur: correlation coefficients from all covariance matrices for given label.
        """
        corr_coefs = []
        for i in range(len(self.covs)):
            corr_coef = self.covs[i][l, 0, 1]
            for f in [0, 1]:
                corr_coef = corr_coef / np.sqrt(np.diag(self.covs[i][l])[f])
            corr_coefs.append(corr_coef)
        return corr_coefs

    def plot_mu_stdev(self):
        """Plot mean and standard deviation over all iterations."""
        fig, ax = plt.subplots(nrows=self.n_feat, ncols=2, figsize=(15, 5 * self.n_feat))

        ax[0, 0].set_title(r"$\mu$")
        ax[0, 1].set_title(r"$\sigma$")

        for f in range(self.n_feat):
            for l in range(self.n_labels):
                if np.mean(np.array(self.mus)[:, :, f][:, l]) == -9999:
                    continue
                else:
                    ax[f, 0].plot(np.array(self.mus)[:, :, f][:, l], label="Label " + str(l))

            ax[f, 0].set_ylabel("Feature " + str(f))

            for l in range(self.n_labels):
                ax[f, 1].plot(self.get_std_from_cov(f, l), label="Label " + str(l))

        ax[f, 0].set_xlabel("Iterations")
        ax[f, 1].set_xlabel("Iterations")
        ax[f, 1].legend(loc=9, bbox_to_anchor=(0.5, -0.25), ncol=3)

        plt.show()

    def plot_acc_ratios(self, linewidth=1):
        """Plot acceptance ratios for beta, mu and covariance."""
        fig, ax = plt.subplots(ncols=3, figsize=(15, 4))

        ax[0].set_title(r"$\beta$")
        ax[0].plot(self.beta_acc_ratio, linewidth=linewidth, color="black")

        ax[1].set_title(r"$\mu$")
        ax[1].plot(self.mu_acc_ratio, linewidth=linewidth, color="red")

        ax[2].set_title("Covariance")
        ax[2].plot(self.cov_acc_ratio, linewidth=linewidth, color="indigo")

    def diagnostics_plot(self, true_labels=None, ie_range=None, transpose=False):
        """Diagnostic plots for analyzing convergence and segmentation results.


        Args:
            true_labels (:obj:`np.ndarray`):
            ie_range (:obj:`tuple` or :obj:`list`): Start and end point of iteration slice to used in the calculation
                of the information entropy.

        Returns:
            Plot
        """
        if true_labels is not None:
            fig = plt.figure(figsize=(15, 10))
            gs = gridspec.GridSpec(4, 2)
        else:
            fig = plt.figure(figsize=(15, 10))
            gs = gridspec.GridSpec(2, 2)

        rcParams.update({'font.size': 8})

        # plot beta
        ax1 = plt.subplot(gs[0, :-1])
        ax1.set_title(r"$\beta$")

        betas = np.array(self.betas)
        if self.dim == 1:
            ax1.plot(betas, label="beta", linewidth=1)
        else:
            for b in range(betas.shape[1]):
                ax1.plot(betas[:, b], label="beta "+str(b), linewidth=1)

        ax1.set_xlabel("Iterations")
        ax1.legend()

        # plot correlation coefficient
        ax2 = plt.subplot(gs[0, -1])
        ax2.set_title("Correlation coefficient")
        for l in range(self.n_labels):
            ax2.plot(self.get_corr_coef_from_cov(l), label="Label " + str(l), linewidth=1)
        ax2.legend()
        ax2.set_xlabel("Iterations")

        # 1D
        if self.dim == 1:
            # PLOT LABELS
            ax3 = plt.subplot(gs[1, :])
            ax3.imshow(np.array(self.labels), cmap=cmap, norm=cmap_norm, aspect='auto', interpolation='nearest')
            ax3.set_ylabel("Iterations")
            ax3.set_title("Labels")
            ax3.grid(False)  # disable grid

            if true_labels is not None:
                # plot the latent field
                ax4 = plt.subplot(gs[2, :])
                ax4.imshow(np.tile(np.expand_dims(true_labels, axis=1), 50).T,
                           cmap=cmap, norm=cmap_norm, aspect='auto', interpolation='nearest')
                ax4.set_title("Latent field")
                ax4.grid(False)

                # plot the mcr
                ax5 = plt.subplot(gs[3, :])
                ax5.plot(self.mcr(true_labels), color="black", linewidth=1)
                ax5.set_ylabel("MCR")
                ax5.set_xlabel("Iterations")

        # 2D
        elif self.dim == 2:
            if ie_range is None:  # use all
                a = 0
                b = -1
            else:  # use given range
                a = ie_range[0]
                b = ie_range[1]

            max_lp = labels_map(self.labels, r=(a, b))
            # print(max_lp)

            # PLOT LABELS
            ax3 = plt.subplot(gs[1, 0])
            ax3.set_title("Labels (MAP)")
            if transpose:
                max_lp_plot = np.array(max_lp.reshape(self.shape[0], self.shape[1])).T
            else:
                max_lp_plot = np.array(max_lp.reshape(self.shape[0], self.shape[1]))
            ax3.imshow(max_lp_plot, cmap=cmap, norm=cmap_norm, interpolation='nearest')
            ax3.grid(False)

            # PLOT INFORMATION ENTROPY
            ie = compute_ie(compute_labels_prob(np.array(self.labels[a:b])))  # calculate ie
            ax4 = plt.subplot(gs[1, 1])
            ax4.set_title("Information Entropy")
            if transpose:
                ie_plot = ie.reshape(self.shape[0], self.shape[1]).T
            else:
                ie_plot = ie.reshape(self.shape[0], self.shape[1])
            iep = ax4.imshow(ie_plot, cmap="viridis", interpolation='nearest')
            ax4.grid(False)
            plt.colorbar(iep)

        plt.show()

    def normalize_feature_vectors(self):
        return (self.feat - np.mean(self.feat, axis=0).T) / np.std(self.feat, axis=0)


def labels_map(labels, r=None):
    if r is None:
        r = (0, -1)

    lp = compute_labels_prob(np.array(labels[r[0]:r[1]]))
    return np.argmax(lp, axis=0)


def draw_labels_vect(labels_prob):
    """Vectorized draw of the label for each elements respective labels probability.

    Args:
        labels_prob (:obj:`np.ndarray`): (n_elements x n_labels) ndarray containing the element-specific labels
            probabilites for each element.

    Returns:
        :obj:`np.array` : Flat array containing the newly drawn labels for each element.

    """
    # draw a random number between 0 and 1 for each element
    r = np.random.rand(len(labels_prob))
    # cumsum labels probabilities for each element
    p = np.cumsum(labels_prob, axis=1)
    # calculate difference between random draw and cumsum probabilities
    d = (p.T - r).T
    # compare and count to get label
    return np.sum(np.greater_equal(0, d), axis=1)


def evaluate(log_target_prop, log_target_prev):

    ratio = np.exp(np.longfloat(log_target_prop - log_target_prev))

    if (ratio > 1) or (np.random.uniform() < ratio):
        return True, ratio  # if accepted

    else:
        return False, ratio  # if rejected


def _propose_cov(cov_prev, n_feat, n_labels, cov_jump_length, theta_jump_length):
    """Proposes a perturbed n-dimensional covariance matrix based on an existing one and a covariance jump length and
    theta jump length parameter.

    Args:
        cov_prev (:obj:`np.ndarray`): Covariance matrix.
        n_feat (int): Number of features.
        n_labels (int): Number of labels.
        cov_jump_length (float): Hyperparameter
        theta_jump_length (float): Hyperparameter

    Returns:
        :obj:`np.ndarray` : Perturbed covariance matrix.

    """
    # do svd on the previous covariance matrix
    comb = list(combinations(range(n_feat), 2))
    n_comb = len(comb)
    theta_jump = multivariate_normal(mean=[0 for i in range(n_comb)], cov=np.ones(n_comb) * theta_jump_length).rvs()

    if n_comb == 1:  # turn it into a list if there is only one combination (^= 2 features)
        theta_jump = [theta_jump]

    cov_prop = np.zeros_like(cov_prev)
    # print("cov_prev:", cov_prev)

    # loop over all labels (=layers of the covariance matrix)
    for l in range(n_labels):
        v_l, d_l, v_l_t = np.linalg.svd(cov_prev[l, :, :])
        # print("v_l:", v_l)
        # generate d jump
        log_d_jump = multivariate_normal(mean=[0 for i in range(n_feat)], cov=np.eye(n_feat) * cov_jump_length).rvs()
        # sum towards d proposal
        # if l == 0:
        d_prop = np.diag(np.exp(np.log(d_l) + log_d_jump))
        # else:
        #    d_prop = np.vstack((d_prop, np.exp(np.log(d_l) + np.log(d_jump))))
        # now tackle generating v jump
        a = np.eye(n_feat)
        # print("a init:", a)
        # print("shape a:", np.shape(a))
        for val in range(n_comb):
            rotation_matrix = _cov_proposal_rotation_matrix(v_l[:, comb[val][0]], v_l[:, comb[val][1]], theta_jump[val])
            # print("rot mat:", rotation_matrix)
            a = rotation_matrix @ a
            # print("a:", a)
        # print("v_l:", np.shape(v_l))
        v_prop = a @ v_l  # np.matmul(a, v_l)
        # print("d_prop:", d_prop)
        # print("v_prop:", np.shape(v_prop))
        cov_prop[l, :, :] = v_prop @ d_prop @ v_prop.T  # np.matmul(np.matmul(v_prop, d_prop), v_prop.T)
        # print("cov_prop:", cov_prop)

    return cov_prop


def _cov_proposal_rotation_matrix(x, y, theta):
    """Creates the rotation matrix needed for the covariance matrix proposal step.

    Args:
        x (:obj:`np.array`): First base vector.
        y (:obj:`np.array`): Second base vector.
        theta (float): Rotation angle.

    Returns:
        :obj:`np.ndarray` : Rotation matrix for covariance proposal step.

    """
    x = np.array([x]).T
    y = np.array([y]).T

    uu = x / np.linalg.norm(x)
    vv = y - uu.T @ y * uu
    vv = vv / np.linalg.norm(vv)
    # what is happening

    # rotation_matrix = np.eye(len(x)) - np.matmul(uu, uu.T) - np.matmul(np.matmul(vv, vv.T) + np.matmul(np.hstack((uu, vv)), np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])), np.hstack((uu, vv)).T)
    rotation_matrix = np.eye(len(x)) - uu @ uu.T - vv @ vv.T + np.hstack((uu, vv)) @ np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]) @ np.hstack((uu, vv)).T
    return rotation_matrix


def _calc_labels_prob(te, t):
    """"Calculate labels probability for array of total energies (te) and totally arbitrary skalar value t."""
    return (np.exp(-te / t).T / np.sum(np.exp(-te / t), axis=1)).T


def pseudocolor(shape, stencil=None):
    """Graph coloring based on the physical dimensions for independent labels draw.

    Args:
        extent (:obj:`tuple` of int): Data extent in (Y), (Y,X) or (Y,X,Z) for 1D, 2D or 3D respectively.
        stencil:

    Returns:

    """
    dim = len(shape) - 1
    # ************************************************************************************************
    # 1-DIMENSIONAL
    if dim == 1:
        i_w = np.arange(0, shape[0], step=2)
        i_b = np.arange(1, shape[0], step=2)

        return np.array([i_w, i_b]).T

    # ************************************************************************************************
    # 2-DIMENSIONAL
    elif dim == 2:
        if stencil is None or stencil == "8p":
            # use 8 stamp as default, resulting in 4 colors
            colors = 4
            # color image
            colored_image = np.tile(np.kron([[0, 1], [2, 3]] * int(shape[0] / 2), np.ones((1, 1))), int(shape[1] / 2))
            colored_flat = colored_image.reshape(shape[0] * shape[1])

            # initialize storage array
            ci = []
            for c in range(colors):
                x = np.where(colored_flat == c)[0]
                ci.append(x)
            return np.array(ci)

        elif stencil == "4p":
            # use 4 stamp, resulting in 2 colors (checkerboard)
            colors = 2
            # color image
            colored_image = np.tile(np.kron([[0, 1], [1, 0]] * int(shape[0] / 2), np.ones((1, 1))), int(shape[1] / 2))
            colored_flat = colored_image.reshape(shape[0] * shape[1])

            # initialize storage array
            ci = []
            for c in range(colors):
                x = np.where(colored_flat == c)[0]
                ci.append(x)
            return ci
        else:
            raise Exception(" In 2D space the stamp parameter needs to be either None (defaults to 8p), 4p or 8p.")

    # ************************************************************************************************
    # 3-DIMENSIONAL
    elif dim == 3:
        raise Exception("3D space not yet supported.")
        # TODO: 3d graph coloring


def bic(feat_vector, n_labels):
    """Plots the Bayesian Information Criterion of Gaussian Mixture Models for the given features and range of labels
    defined by the given upper boundary.

    Args:
        feat_vector (:obj:`np.ndarray`): Feature vector containing the data in a flattened format.
        n_labels (int): Sets the included upper bound for the number of features to be considered in the analysis.

    Returns:
        Plot

    """
    n_comp = np.arange(1, n_labels + 1)
    # create array of GMMs in range of components/labels and fit to observations
    gmms = np.array([mixture.GaussianMixture(n_components=n, covariance_type="full").fit(feat_vector) for n in n_comp])
    # calculate BIC for each GMM based on observartions
    bics = np.array([gmm.bic(feat_vector) for gmm in gmms])
    # take sequential difference
    # bic_diffs = np.ediff1d(bics)

    # find index of minimum BIC
    # bic_min = np.argmin(bics)
    # bic_diffs_min = np.argmin(np.abs(bic_diffs))

    # d = np.abs(bic_diffs[bic_diffs_min] * d_factor)
    bic_min = np.argmin(bics)

    # do a nice plot so the user knows intuitively whats happening
    fig = plt.figure()  # figsize=(10, 10)
    plt.plot(n_comp, bics, label="bic")
    plt.plot(n_comp[bic_min], bics[bic_min], "ko")
    plt.title("Bayesian Information Criterion")
    plt.xlabel("Number of Labels")
    plt.axvline(n_comp[bic_min], color="black", linestyle="dashed", linewidth=0.75)

    plt.show()
    print("global minimum: ", n_comp[bic_min])


def gibbs_comp_f(a, value):
    """Helper function for the Gibbs energy calculation using Scipy's generic filter function."""
    a = a[a != -999.]
    return np.count_nonzero(a != value)

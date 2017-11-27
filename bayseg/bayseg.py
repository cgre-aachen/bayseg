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
plt.style.use('bmh')  # plot style
from functools import partial
from scipy.ndimage import generic_filter


class BaySeg:
    def __init__(self, data, n_labels, beta_init=1, stamp=None):
        """

        :param coordinates: Physical coordinate system as numpy ndarray (n_coord, 1)
        :param observations: Observations collected at every coordinate as a numpy ndarray (n_coord, n_feat)
        :param n_labels: Number of labels to be used in the clustering (int)
        :param beta_init: Initial beta value (float)
        :param bic: (bool) for using Bayesian Information Criterion (Schwarz, 1978) for determining n_labels
        """
        # TODO: [DOCS] Main object description

        # store initial data
        self.data = data
        # get shape for physical and feature dimensions
        self.shape = np.shape(data)

        # get number of features
        self.n_feat = self.shape[-1]

        # ************************************************************************************************
        # fetch dimensionality, coordinate and feature vector from input data

        # 1D
        if len(self.shape) == 2:
            # 1d case
            self.dim = 1

            # create coordinate vector
            self.coords = np.array([np.arange(self.shape[0])]).T

            # feature vector
            self.feat = self.data

        # 2D
        elif len(self.shape) == 3:
            # 2d case
            self.dim = 2

            # create coordinate vector
            x, y = np.indices(self.shape[:-1])
            self.coords = np.array([x.flatten(), y.flatten()]).T

            # feature vector
            self.feat = np.array([self.data[:, :, f].flatten() for f in range(self.n_feat)]).T

        # 3D
        elif len(self.shape) == 4:
            # 3d case
            raise Exception("3D segmentation not yet supported.")

        # mismatch
        else:
            raise Exception("Data format appears to be wrong (neither 1-, 2- or 3-D).")

        # ************************************************************************************************
        # INIT STORAGE ARRAYS
        self.betas = [beta_init]  # initial beta
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

        # ************************************************************************************************
        # Initialize PRIOR distributions for beta, mu and covariance
        # BETA
        # beta_dim = [1, 4, 13]
        beta_dim = [1, 1, 1]
        self.prior_beta = norm(beta_init, np.eye(beta_dim[self.dim - 1]) * 100)

        # MU
        # generate distribution means for each label
        prior_mu_means = [self.mus[0][label] for label in range(self.n_labels)]
        # generate distribution covariances for each label
        prior_mu_stds = [np.eye(self.n_feat) * 100 for label in range(self.n_labels)]
        # use the above to generate multivariate normal distributions for each label
        self.priors_mu = [multivariate_normal(prior_mu_means[label], prior_mu_stds[label]) for label in range(self.n_labels)]

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

        # GRAPH COLORING
        self.stamp = stamp
        self.colors = _pseudocolor(self.coords, self.shape[:-1], self.stamp)

    def fit(self, n, beta_jump_length=10, mu_jump_length=0.0005, cov_volume_jump_length=0.00005,
            theta_jump_length=0.0005, t=1., verbose=False):
        """

        :param n:
        :param beta_jump_length:
        :param mu_jump_length:
        :param cov_volume_jump_length:
        :param theta_jump_length:
        :param t:
        :param verbose:
        :return:
        """
        for g in tqdm.trange(n):
            self.gibbs_sample(t, beta_jump_length, mu_jump_length, cov_volume_jump_length, theta_jump_length, verbose)

    def gibbs_sample(self, t, beta_jump_length, mu_jump_length, cov_volume_jump_length, theta_jump_length, verbose):
        """
        Gibbs sampler. This is the main function of the algorithm and needs an in-depth description

        :param t: Hyperparameter
        :param beta_jump_length: Hyperparameter
        :param mu_jump_length: Hyperparameter
        :param cov_volume_jump_length: Hyperparameter
        :param theta_jump_length: Hyperparameter
        :param verbose: bool or str specifying verbosity of the function

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
        gibbs_energy = self._calc_gibbs_energy_vect(self.labels[-1], self.betas[-1])
        if verbose == "energy":
            print("gibbs energy:", gibbs_energy)
        # 3 - self energy
        self_energy = np.zeros(self.n_labels)  # to fix specific labels to locations, theoretically
        # 5 - calculate total energy
        # total energy vector 2d: n_elements * n_labels
        total_energy = energy_like + self_energy + gibbs_energy
        if verbose == "energy":
            print("total_energy:", total_energy)
        # ************************************************************************************************
        # CALCULATE PROBABILITY OF LABELS
        labels_prob = _calc_labels_prob(total_energy, t)
        if verbose == "energy":
            print("Labels probability:", labels_prob)

        # make copy of previous labels
        new_labels = copy(self.labels[-1])

        for color_f in self.colors:
            new_labels[color_f] = draw_labels_vect(labels_prob[color_f])

            # now recalculate gibbs energy and other energies from the mixture of old and new labels
            gibbs_energy = self._calc_gibbs_energy_vect(new_labels, self.betas[-1])
            total_energy = energy_like + self_energy + gibbs_energy
            labels_prob = _calc_labels_prob(total_energy, t)

        # # now draw new labels for the other color
        # color_f = self.colors[:, 1]
        # new_labels[color_f] = draw_labels_vect(labels_prob[color_f])
        #
        # # recalculate gibbs energy
        # gibbs_energy = self._calc_gibbs_energy_vect(new_labels, self.betas[-1])
        # append labels to storage
        self.labels.append(new_labels)

        # ************************************************************************************************
        # calculate energy for component coefficient
        energy_for_comp_coef = gibbs_energy + self_energy
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

            # acceptance ratio
            acc_ratio = np.exp(log_target_prop - log_target_prev)
            if verbose:
                print("MU acceptance ratio:", acc_ratio)

            if (acc_ratio > 1) or (np.random.uniform() < acc_ratio):
                # replace old with new if accepted
                mu_next[l, :] = mu_prop[l, :]

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

            # acceptance ratio
            acc_ratio = np.exp(log_target_prop - log_target_prev)
            if verbose:
                print("COV acceptance ratio:", acc_ratio)
            # print("cov acc ratio", acc_ratio)
            random = np.random.uniform()
            # print("cov rand acc", random)
            if (acc_ratio > 1) or (random < acc_ratio):
                # replace old with new if accepted
                cov_next[l, :] = cov_prop[l, :]

        # append cov and mu
        self.covs.append(cov_next)

        # ************************************************************************************************
        # UPDATE BETA
        lp_beta_prev = self.log_prior_density_beta(self.betas[-1])
        lp_beta_prop = self.log_prior_density_beta(beta_prop)

        lmd_prev = self.calc_sum_log_mixture_density(comp_coef, self.mus[-1], self.covs[-1])

        # calculate gibbs energy with new labels and proposed beta
        gibbs_energy_prop = self._calc_gibbs_energy_vect(new_labels, beta_prop)
        energy_for_comp_coef_prop = gibbs_energy_prop + self_energy
        comp_coef_prop = _calc_labels_prob(energy_for_comp_coef_prop, t)

        lmd_prop = self.calc_sum_log_mixture_density(comp_coef_prop, self.mus[-1], self.covs[-1])
        # print("lmd_prev:", lmd_prev)
        # print("lp_beta_prev:", lp_beta_prev)
        log_target_prev = lmd_prev + lp_beta_prev
        # print("lmd_prop:", lmd_prop)
        # print("lp_beta_prop:", lp_beta_prop)
        log_target_prop = lmd_prop + lp_beta_prop

        acc_ratio = np.exp(log_target_prop - log_target_prev)
        # print("beta acc_ratio:", acc_ratio)

        if verbose:
            print("BETA acceptance ratio:", acc_ratio)

        if (acc_ratio > 1) or (np.random.uniform() < acc_ratio):
            self.betas.append(beta_prop)
        else:
            self.betas.append(self.betas[-1])
        # ************************************************************************************************

    def log_prior_density_mu(self, mu, label):
        """

        :param mu:
        :param label:
        :return:
        """
        return np.sum(np.log(self.priors_mu[label].pdf(mu)))

    def log_prior_density_beta(self, beta):
        """
        Calculates the log prior density of beta.
        :param beta: Beta value (float)
        :return: Log prior density of beta (float)
        """
        return np.log(self.prior_beta.pdf(beta))

    def log_prior_density_cov(self, cov, l):
        """
        Calculate the log prior density of the covariance matrix for a given label.
        :param cov: Covariance matrix.
        :param l: Label (int)
        :return: Log prior density of covariance for label (float)
        """
        lam = np.sqrt(np.diag(cov[l, :, :]))
        r = np.diag(1./lam) @ cov[l, :, :] @ np.diag(1. / lam)
        logp_r = -0.5 * (self.nu + self.n_feat + 1) * np.log(np.linalg.det(r)) - self.nu / 2. * np.sum(np.log(np.diag(np.linalg.inv(r))))
        logp_lam = np.sum(np.log(multivariate_normal(mean=self.b_sigma[l, :], cov=self.kesi[l, :]).pdf(np.log(lam.T))))
        return logp_r + logp_lam

    def propose_beta(self, beta_prev, beta_jump_length):
        """
        Proposes a perturbed beta based on a jump length hyperparameter.
        :param beta_prev: Beta value to be perturbed.
        :param beta_jump_length: Hyperparameter specifying the strength of the perturbation.
        :return: Perturbed beta.
        """
        # create proposal covariance depending on physical dimensionality
        # dim = [1, 4, 13]
        dim = [1, 1, 1]
        sigma_prop = np.eye(dim[self.dim - 1]) * beta_jump_length
        # draw from multivariate normal distribution and return
        # return np.exp(multivariate_normal(mean=np.log(beta_prev), cov=sigma_prop).rvs())
        return multivariate_normal(mean=beta_prev, cov=sigma_prop).rvs()

    def propose_mu(self, mu_prev, mu_jump_length):
        """
        Proposes a perturbed mu matrix using a jump length hyperparameter.
        :param mu_prev: The mu matrix to be perturbed.
        :param mu_jump_length: Hyperparameter specifying the strength of the perturbation.
        :return: Perturbed mu matrix.
        """
        # prepare matrix
        mu_prop = np.ones((self.n_labels, self.n_feat))
        # loop over labels
        for l in range(self.n_labels):
            mu_prop[l, :] = multivariate_normal(mean=mu_prev[l, :], cov=np.eye(self.n_feat) * mu_jump_length).rvs()
        return mu_prop

    def calc_sum_log_mixture_density(self, comp_coef, mu, cov):
        """
        Calculate sum of log mixture density with each observation at every element.
        :param comp_coef: Component coefficient.
        :param mu: Mean matrix
        :param cov: Covariance matrix
        :return: summed log mixture density of the system
        """
        lmd = np.zeros((len(self.coords), self.n_labels))

        for l in range(self.n_labels):
            draw = multivariate_normal(mean=mu[l, :], cov=cov[l, :, :]).pdf(self.feat)
            # print(np.shape(lmd[:,l]))
            multi = comp_coef[:, l] * draw
            lmd[:, l] = multi
        lmd = np.sum(lmd, axis=1)
        lmd = np.log(lmd)

        return np.sum(lmd)

    def calc_energy_like(self, mu, cov):
        """
        Calculates the energy likelihood of the system.
        Thanks a lot to Powei for helping with the Einstein summation notation!
        :param mu: Mean values
        :param cov: Covariance matrix
        :return:
        """

        energy_like_labels = np.zeros((len(self.coords), self.n_labels))

        # uses flattened features array
        for l in range(self.n_labels):
            energy_like_labels[:, l] = np.einsum("...i,ji,...j", 0.5 * np.array([self.feat - mu[l, :]]),
                                                 np.linalg.inv(cov[l, :, :]),
                                                 np.array([self.feat - mu[l, :]])) + 0.5 * np.log(np.linalg.det(cov[l, :, :]))

        return energy_like_labels

    def _calc_gibbs_energy_vect(self, labels, beta):
        """
        Calculates Gibbs energy for each element using a penalty factor beta.
        :param labels: flattened labels array!
        :param beta:
        :return: Gibbs energy matrix (n_obs times n_labels)
        """

        # ************************************************************************************************
        if self.dim == 1:
            # tile
            lt = np.tile(labels, (self.n_labels, 1)).T

            ge = np.arange(self.n_labels)  # elemnts x labels
            ge = np.tile(ge, (len(labels), 1))
            ge = ge.astype(float)

            # first row
            top = np.expand_dims(np.not_equal(np.arange(self.n_labels), lt[1, :]) * beta, axis=0)
            # mid
            mid = (np.not_equal(ge[1:-1, :], lt[:-2, :]).astype(float) + np.not_equal(ge[1:-1, :], lt[2:, :]).astype(
                float)) * beta
            # last row
            bot = np.expand_dims(np.not_equal(np.arange(self.n_labels), lt[-1, :]) * beta, axis=0)
            # put back together and return gibbs energy
            return np.concatenate((top, mid, bot))

        # ************************************************************************************************

        elif self.dim == 2:
            l = labels.reshape(self.shape[:-1])
            ge = np.tile(np.zeros_like(l).astype(float), (3, 1, 1))

            if self.stamp == 4:
                fp = np.array([[0, 1, 0],
                                   [1, 0, 1],
                                   [0, 1, 0]]).astype(bool)
            else:
                fp = np.array([[1, 1, 1],
                                   [1, 0, 1],
                                   [1, 1, 1]]).astype(bool)

            for i in range(self.n_labels):
                ge[i, :, :] = generic_filter(l, partial(gibbs_comp_f, value=i), footprint=fp, mode="constant",
                                                           cval=-999) * beta

            return ge.reshape(self.feat.shape[0], self.n_labels)

        # ************************************************************************************************
        elif self.dim == 3:
            # TODO: [3D] implementation of gibbs energy
            pass

    def mcr(self, true_labels):
        """Compares classified with true labels for each iteration step (for synthetic data)
        to obtain a measure of mismatch/convergence."""
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
        """
        Plot the mu and stdev for each label for each feature.
        :return: Fancy figures
        """
        fig, ax = plt.subplots(nrows=self.n_feat, ncols=2, figsize=(15, 5*self.n_feat))

        ax[0, 0].set_title(r"$\mu$")
        ax[0, 1].set_title(r"$\sigma$")
        # ax[0, 0].legend()

        for f in range(self.n_feat):

            # plot mus
            # ax[f, 0].set_title("MU, feature "+str(f))
            for l in range(self.n_labels):
                ax[f, 0].plot(np.array(self.mus)[:, :, f][:, l], label="Label " + str(l))

            # ax[f, 0].legend()
            ax[f, 0].set_ylabel("Feature " + str(f))

            # plot covs
            # ax[f, 1].set_title("STDEV, feature " + str(f))
            for l in range(self.n_labels):
                ax[f, 1].plot(self.get_std_from_cov(f, l), label="Label " + str(l))

        ax[f, 0].set_xlabel("Iterations")
        ax[f, 1].set_xlabel("Iterations")
        ax[f, 1].legend(loc=9, bbox_to_anchor=(0.5, -0.25), ncol=3)

        plt.show()

    def diagnostics_plot(self, true_labels=None):
        """
        Diagnostic plots for analyzing convergence: beta trace, correlation coefficient trace, labels trace and MCR.
        :param true_labels: If given calculates and plots MCR
        :return: Fancy figures
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
        ax1.plot(self.betas, label="beta", color="black", linewidth=1)
        ax1.set_xlabel("Iterations")

        # plot correlation coefficient
        ax2 = plt.subplot(gs[0, -1])
        ax2.set_title("Correlation coefficient")
        for l in range(self.n_labels):
            ax2.plot(self.get_corr_coef_from_cov(l), label="Label " + str(l), linewidth=1)
        ax2.legend()
        ax2.set_xlabel("Iterations")

        # plot labels
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

        plt.show()


def draw_labels_vect(labels_prob):
    """
    Vectorized draw of the label for each elements respective labels probability.
    :param labels_prob: (n_elements x n_labels) ndarray containing the element-specific labels probabilites
    :return: (n_elements)-long array of labels
    """
    # draw a random number between 0 and 1 for each element
    r = np.random.rand(len(labels_prob))
    # cumsum labels probabilities for each element
    p = np.cumsum(labels_prob, axis=1)
    # calculate difference between random draw and cumsum probabilities
    d = (p.T - r).T
    # compare and count to get label
    return np.count_nonzero(np.greater_equal(0, d), axis=1)


def _propose_cov(cov_prev, n_feat, n_labels, cov_jump_length, theta_jump_length):
    """
    Proposes a perturbed n-dimensional covariance matrix based on an existing one and a
    covariance jump length and theta jump length parameter.
    :param cov_prev: Covariance matrix to be perturbed.
    :param n_feat: Number of features represented by the covariance matrix.
    :param n_labels: Number of labels represented by the covariance matrix.
    :param cov_jump_length: Parameter which roughly determines the strength of the covariance perturbation
    :param theta_jump_length: Parameter which roughly determines the strength of the covariance perturbation
    :return: Perturbed covariance matrix.
    """
    # do svd on the previous covariance matrix
    comb = list(combinations(range(n_feat), 2))
    n_comb = len(comb)
    theta_jump = multivariate_normal(mean=[0 for i in range(n_comb)], cov=np.ones(n_comb) * theta_jump_length).rvs()
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
        for j in range(n_comb):
            rotation_matrix = _cov_proposal_rotation_matrix(v_l[:, comb[j][0]], v_l[:, comb[j][1]], theta_jump[j])
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
    """
    Creates the rotation matrix needed for the covariance matrix proposal step.
    :param x, y: two base vectors defining a plane
    :param theta: rotation angle in this plane
    :return: rotation matrix for covariance proposal step
    """
    x = np.array([x]).T
    y = np.array([y]).T

    uu = x / np.linalg.norm(x)
    vv = y - uu.T @ y * uu
    vv = vv / np.linalg.norm(vv)
    # what is happening

    # rotation_matrix = np.eye(len(x)) - np.matmul(uu, uu.T) - np.matmul(np.matmul(vv, vv.T) + np.matmul(np.hstack((uu, vv)), np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])), np.hstack((uu, vv)).T)
    rotation_matrix = np.eye(len(x)) - uu @ uu.T - vv @ vv.T + np.hstack((uu, vv)) @ np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]) @ np.hstack((uu, vv)).T
    return rotation_matrix


def _calc_labels_prob(te, t):
    """"Calculate labels probability for array of total energies (te) and totally arbitrary skalar value t."""
    return (np.exp(-te/t).T / np.sum(np.exp(-te/t), axis=1)).T


# def _calc_log_prior_density(self, mu, rv_mu):
#     """
#     :param mu:
#     :param rv_mu:
#     :return:
#     """
#     return np.log(rv_mu.pdf(mu))


def _pseudocolor(coords_vector, extent, stamp=None):
    """Graph coloring based on the physical dimensions for independent labels draw."""
    dim = np.shape(coords_vector)[1]
    # ************************************************************************************************
    # 1-DIMENSIONAL
    if dim == 1:
        i_w = np.arange(0, len(coords_vector), step=2)
        i_b = np.arange(1, len(coords_vector), step=2)

        return np.array([i_w, i_b]).T

    # ************************************************************************************************
    # 2-DIMENSIONAL
    elif dim == 2:
        if stamp is None or stamp == 8:
            # use 8 stamp as default, resulting in 4 colors
            colors = 4
            # color image
            colored_image = np.tile(np.kron([[0, 1], [2, 3]] * int(extent[0] / 2), np.ones((1, 1))), int(extent[1] / 2))
            colored_flat = colored_image.flatten()

            # initialize storage array
            ci = []
            for c in range(colors):
                x = np.where(colored_flat == c)[0]
                ci.append(x)
            return ci

        elif stamp == 4:
            # use 4 stamp, resulting in 2 colors (checkerboard)
            colors = 2
            # color image
            colored_image = np.tile(np.kron([[0, 1], [1, 0]] * int(extent[0] / 2), np.ones((1, 1))), int(extent[1] / 2))
            colored_flat = colored_image.flatten()

            # initialize storage array
            ci = []
            for c in range(colors):
                x = np.where(colored_flat == c)[0]
                ci.append(x)
            return ci
        else:
            raise Exception(" In 2D space the stamp parameter needs to be either None (defaults to 8), 4 or 8.")

    # ************************************************************************************************
    # 3-DIMENSIONAL
    elif dim == 3:
        raise Exception("3D space not yet supported.")
        # TODO: 3d graph coloring


def bic(feat_vector, n_labels):
    """
    Initializes GMM using either a single n_labels, or does BIC analysis and choses best n_labels basedon
    feature space.
    :param feat_vector: (np.ndarray) containing the observations/features
    :param n_labels: (int) maximum number of clusters considered in the BIC analysis
    """
    n_comp = np.arange(1, n_labels+1)
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
    a = a[a != -999.]
    return np.count_nonzero(a != value)
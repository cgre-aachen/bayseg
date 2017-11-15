"""Spatial segmentation with multiple features using Hidden Markov Random Fields and Finite Mixture Models

Approach based on Wang et al. 2016 paper

@author: Alexander Schaaf, Hui Wang
"""

import numpy as np
from sklearn import mixture
from scipy.stats import multivariate_normal, norm
from copy import copy
from itertools import combinations
import tqdm  # progress bar
import matplotlib.pyplot as plt
from matplotlib import gridspec  # plot arrangements


class HMRFGMM:
    def __init__(self, coordinates, observations, n_labels, beta_init=1):
        """

        :param coordinates: Physical coordinate system as numpy ndarray
        :param observations: Observations collected at every coordinate (numpy ndarray)
        :param n_labels: Number of labels to be used in the clustering.
        :param beta_init: Initial beta value.
        """
        # TODO: Main object description
        # store physical coordinates, set dimensionality
        self.coords = coordinates
        self.dim = np.shape(coordinates)[1]
        # store observations
        self.obs = observations
        self.n_feat = np.shape(observations)[1]
        # store number of labels
        self.n_labels = n_labels
        # ************************************************************************************************
        # INIT STORAGE ARRAYS
        self.betas = [beta_init]  # initial beta
        # self.mus = np.array([], dtype=object)
        # self.covs = np.array([], dtype=object)
        # self.labels = np.array([], dtype=object)
        # ************************************************************************************************
        # generate neighborhood system
        self.neighborhood = define_neighborhood_system(coordinates)
        # ************************************************************************************************
        # INIT GAUSSIAN MIXTURE MODEL
        self.gmm = mixture.GaussianMixture(n_components=self.n_labels, covariance_type="full")
        # fit it to features
        self.gmm.fit(self.obs)
        # do initial prediction based on fit and observations, store as first entry in labels
        # ************************************************************************************************
        # INIT LABELS based on GMM
        self.labels = [self.gmm.predict(self.obs)]
        # ************************************************************************************************
        # INIT MU (mean from initial GMM)
        self.mus = [self.gmm.means_]
        # ************************************************************************************************
        # INIT COV (covariances from initial GMM)
        self.covs = [self.gmm.covariances_]

        # ************************************************************************************************
        # Initialize PRIORS
        # beta
        if self.dim == 1:
            self.prior_beta = norm(beta_init, np.eye(1)*100)

        else:
            pass
            # TODO: 2d and 3d implementation of beta prior# mu

        # TODO: Clean up prior initialization
        # mu
        prior_mu_means = [self.mus[0][f] for f in range(self.n_labels)]
        prior_mu_stds = [np.eye(self.n_feat) * 100 for f in range(self.n_labels)]
        self.priors_mu = [multivariate_normal(prior_mu_means[f], prior_mu_stds[f]) for f in range(self.n_labels)]

        # cov
        # generate b_sigma
        self.b_sigma = np.zeros((self.n_labels, self.n_feat))
        for l in range(self.n_labels):
            self.b_sigma[l, :] = np.log(np.sqrt(np.diag(self.gmm.covariances_[l, :, :])))
        # generate kesi
        self.kesi = np.ones((self.n_labels, self.n_feat)) * 100
        # generate nu
        self.nu = self.n_feat + 1
        # ************************************************************************************************

    def fit(self, n, beta_jump_length=10, mu_jump_length=0.0005, cov_jump_length=0.00005, theta_jump_length=0.0005, t=1., verbose=False):
        """

        :param n:
        :param verbose:
        :return:
        """
        for g in tqdm.trange(n):
            self.gibbs_sample(t, beta_jump_length, mu_jump_length, cov_jump_length, theta_jump_length, verbose)

    def gibbs_sample(self, t, beta_jump_length, mu_jump_length, cov_jump_length, theta_jump_length, verbose):
        """
        Gibbs sampler. This is the main function of the algorithm and needs an in-depth description

        :param t: Hyperparameter
        :param beta_jump_length: Hyperparameter
        :param mu_jump_length: Hyperparameter
        :param cov_jump_length: Hyperparameter
        :param theta_jump_length: Hyperparameter
        :param verbose: bool or str specifying verbosity of the function

        The function updates directly on the object variables and appends new draws of labels and
        parameters to their respective storages.
        """
        # TODO: In-depth description of the gibbs sampling function

        i = -1
        # ************************************************
        # CALCULATE TOTAL ENERGY
        # 1 - calculate energy likelihood for each element and label
        energy_like = self.calc_energy_like(self.mus[-1], self.covs[-1])
        if verbose == "energy":
            print("likelihood energy:", energy_like)
        # 2 - calculate gibbs/mrf energy
        gibbs_energy = self.calc_gibbs_energy(self.labels[-1], self.betas[-1])
        if verbose == "energy":
            print("gibbs energy:", gibbs_energy)
        # 3 - self energy
        self_energy = np.zeros(self.n_labels)
        # 5 - calculate total energy
        total_energy = energy_like + self_energy + gibbs_energy
        if verbose == "energy":
            print("total_energy:", total_energy)

        # ************************************************************************************************
        # CALCULATE PROBABILITY OF LABELS
        labels_prob = calc_labels_prob(total_energy, t)
        if verbose == "energy":
            print("Labels probability:", labels_prob)

        # ************************************************************************************************
        # DRAW NEW LABELS FOR EVERY ELEMENT
        new_labels = np.array([np.random.choice(list(range(self.n_labels)), p=labels_prob[x, :]) for x in range(len(self.coords))])
        self.labels.append(new_labels)
        # ************************************************************************************************
        # RECALCULATE Gibbs energy with new labels
        gibbs_energy = self.calc_gibbs_energy(new_labels, self.betas[i])
        # calculate energy for component coefficient
        energy_for_comp_coef = gibbs_energy + self_energy

        # ************************************************************************************************
        # CALCULATE COMPONENT COEFFICIENT
        comp_coef = calc_labels_prob(energy_for_comp_coef, t)

        # ************************************************************************************************
        # ************************************************************************************************
        # PROPOSAL STEP
        # make proposals for beta, mu and cov
        beta_prop = self.propose_beta(self.betas[-1], beta_jump_length)
        # print("beta prop:", beta_prop)
        mu_prop = self.propose_mu(self.mus[-1], mu_jump_length)
        cov_prop = propose_cov(self.covs[-1], self.n_feat, self.n_labels, cov_jump_length, theta_jump_length)
        # print("cov_prop:", cov_prop)

        # ************************************************************************************************
        # COMPARE THE SHIT FOR EACH LABEL FOR EACH SHIT

        # prepare next ones
        mu_next = copy(self.mus[-1])
        cov_next = copy(self.covs[-1])

        for l in range(self.n_labels):
            # **************************************************************
            # UPDATE MU
            # logprob prior density for mu
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

        for l in range(self.n_labels):
            # **************************************************************
            # UPDATE COVARIANCE
            cov_temp = copy(cov_next)
            cov_temp[l, :, :] = cov_prop[l, :, :]

            # print("cov diff:", cov_next[l, :, :]-cov_temp[l, :, :])

            # logprob prior density for covariance
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

        # **************************************************************
        # UPDATE BETA
        lp_beta_prev = self.log_prior_density_beta(self.betas[-1])
        lp_beta_prop = self.log_prior_density_beta(beta_prop)

        lmd_prev = self.calc_sum_log_mixture_density(comp_coef, self.mus[-1], self.covs[-1])

        gibbs_energy_prop = self.calc_gibbs_energy(new_labels, beta_prop)  # calculate gibbs energy with new labels and proposed beta
        energy_for_comp_coef_prop = gibbs_energy_prop + self_energy
        comp_coef_prop = calc_labels_prob(energy_for_comp_coef_prop, t)

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

    def calc_log_prior_density(self, mu, rv_mu):
        """
        :param mu:
        :param rv_mu:
        :return:
        """
        return np.log(rv_mu.pdf(mu))

    def propose_beta(self, beta_prev, beta_jump_length):
        """
        Proposes a perturbed beta based on a jump length hyperparameter.
        :param beta_prev: Beta value to be perturbed.
        :param beta_jump_length: Hyperparameter specifying the strength of the perturbation.
        :return: Perturbed beta.
        """
        # create proposal covariance depending on physical dimensionality
        dim = [1, 4, 13]
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
        # create proposal covariance depending on observation dimensionality
        sigma_prop = np.eye(self.n_feat) * mu_jump_length
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
        if self.dim == 1:
            lmd = 0.

            for x in range(len(self.coords)):
                storage2 = []
                for l in range(self.n_labels):
                    a = comp_coef[x, l] * multivariate_normal(mean=mu[l, :], cov=cov[l, :, :]).pdf(self.obs[x])
                    # print(a)
                    storage2.append(a)

                lmd += (np.log(np.sum(storage2)))

        else:
            pass
        # TODO: 2-dimensional log mixture density
        # TODO: 3-dimensional log mixture density

        return lmd

    def calc_energy_like(self, mu, cov):
        """
        Calculates the energy likelihood of the system.
        :param mu: Mean values
        :param cov: Covariance matrix
        :return:
        """

        energy_like_labels = np.zeros((len(self.coords), self.n_labels))
        if self.dim == 1:
            for x in range(len(self.coords)):
                for l in  range(self.n_labels):
                    energy_like_labels[x, l] = 0.5 * np.array([self.obs[x] - mu[l, :]]) @ np.linalg.inv(cov[l, :, :]) @ np.array([self.obs[x] - mu[l, :]]).T + 0.5 * np.log(np.linalg.det(cov[l, :, :]))

        else:
            pass
        # TODO: 2-dimensional calculation of energy likelihood labels
        # TODO: 3-dimensional calculation of energy likelihood labels

        return energy_like_labels

    def calc_gibbs_energy(self, labels, beta):
        """
        Calculates Gibbs energy for each element using a penalty factor beta.
        :param labels: Array of labels at each element.
        :param beta: Energetic penalty parameter.
        :return: Gibbs energy for each element.
        """
        if self.dim == 1:
            # create ndarray for gibbs energy depending on element structure and n_labels
            gibbs_energy = np.zeros((len(self.coords), self.n_labels))
            for x, nl in enumerate(self.neighborhood):
                for n in nl:
                    for l in range(self.n_labels):
                        if l != labels[n]:
                            gibbs_energy[x, l] += beta

        elif self.dim == 2:
            pass
            # TODO: 2-dimensional calculation of gibbs energy
        elif self.dim == 3:
            pass
            # TODO: 3-dimensional calculation of gibbs energy

        # TODO: Optimize gibbs energy calculation
        return gibbs_energy

    def mcr(self, true_labels):
        """Compares classified with true labels for each iteration step (for synthetic data)
        to obtain a measure of mismatch/convergence."""
        mcr_vals = []
        n = len(true_labels)  # TODO: 2d and 3d implementation for MCR
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

    def diagnostics_plot(self, true_labels=None):
        """
        Diagnostic plots for analyzing convergence.
        :param true_labels: If given calculates and plots MCR
        :return: Fancy figures
        """
        fig = plt.figure(figsize=(15, 15))
        if true_labels is not None:
            gs = gridspec.GridSpec(5, 2)
        else:
            gs = gridspec.GridSpec(4, 2)

        # plot beta
        ax2 = plt.subplot(gs[0, :-1])
        ax2.set_title("beta")
        ax2.plot(self.betas, label="beta", color="black")

        # plot corr coef
        ax3 = plt.subplot(gs[0, -1])
        ax3.set_title("corr coef")
        for l in range(self.n_labels):
            ax3.plot(self.get_corr_coef_from_cov(l), label="Label " + str(l))
        ax3.legend()

        # plot labels
        ax1 = plt.subplot(gs[1, :])
        ax1.imshow(np.array(self.labels), cmap="YlOrRd")

        # plot mus
        ax4 = plt.subplot(gs[2, :-1])
        ax4.set_title("mu feature 0")
        for l in range(self.n_labels):
            ax4.plot(np.array(self.mus)[:, :, 0][:, l], label="Label " + str(l))
        ax4.legend()

        ax5 = plt.subplot(gs[2, -1])
        ax5.set_title("mu feature 0")
        for l in range(self.n_labels):
            ax5.plot(np.array(self.mus)[:, :, 1][:, l], label="Label " + str(l))
        ax5.legend()

        # plot
        ax6 = plt.subplot(gs[3, :-1])
        ax6.set_title("stdev feature 0")
        for l in range(self.n_labels):
            ax6.plot(self.get_std_from_cov(0, l), label="Label " + str(l))
        ax6.legend()

        ax7 = plt.subplot(gs[3, -1])
        ax7.set_title("stdev feature 1")
        for l in range(self.n_labels):
            ax7.plot(self.get_std_from_cov(1, l), label="Label " + str(l))
        ax7.legend()

        if true_labels is not None:
            ax8 = plt.subplot(gs[4, :])
            ax8.set_title("mcr")
            ax8.plot(self.mcr(true_labels), color="black")


def propose_cov(cov_prev, n_feat, n_labels, cov_jump_length, theta_jump_length):
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
            # print("rot mat:", rotation_matrix)
            a = rotation_matrix @ a
            # print("a:", a)
        # print("v_l:", np.shape(v_l))
        v_prop = a @ v_l  # np.matmul(a, v_l)
        # print("d_prop:", d_prop)
        # print("v_prop:", np.shape(v_prop))
        # TODO: Is this proposal covariance slicing correct?
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


def calc_labels_prob(te, t):
    """"Calculate labels probability for array of total energies (te) and totally arbitrary skalar value t."""
    return (np.exp(-te/t).T / np.sum(np.exp(-te/t), axis=1)).T


def define_neighborhood_system(coordinates):
    """

    :param coordinates:
    :return:
    """
    dim = np.shape(coordinates)[1]
    # TODO: neighborhood array creation for 2+ dimensions
    neighbors = [None for i in range(len(coordinates))]

    if dim == 1:
        for i, c in enumerate(coordinates):
            if i == 0:
                neighbors[i] = [i + 1]
            elif i == np.shape(coordinates)[0] - 1:
                neighbors[i] = [i - 1]
            else:
                neighbors[i] = [i - 1, i + 1]


    elif dim == 2:
        pass
        # TODO: neighborhood system for 2 dimensions

    elif dim == 3:
        pass
        # TODO: neighborhood system for 3 dimensions

    return neighbors


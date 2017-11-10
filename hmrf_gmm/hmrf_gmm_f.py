"""Spatial segmentation with multiple features using Hidden Markov Random Fields and Finite Mixture Models

Approach based on Wang et al. 2016 paper

@author: Alexander Schaaf, Hui Wang
"""

import sys
import numpy as np
from sklearn import mixture
from scipy.stats import multivariate_normal, norm
from copy import copy
from itertools import combinations
import tqdm


class HMRFGMM:
    def __init__(self, coordinates, observations, n_labels=2, beta_init=0.5):
        """

        :param coordinates:
        :param observations:
        :param n_labels:
        :param beta_init:
        """

        # store physical coordinates, set dimensionality
        self.coords = coordinates
        self.dim = np.shape(coordinates)[1]
        # store observations
        self.obs = observations
        self.n_obs = np.shape(observations)[1]
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
        self.gmm = mixture.GaussianMixture(n_components=self.n_obs, covariance_type="full")
        # fit it to features
        self.gmm.fit(self.obs)
        # do initial prediction based on fit and observations, store as first entry in labels
        # ************************************************************************************************
        # INIT LABELS based on GMM
        self.labels =[self.gmm.predict(self.obs)]
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
        prior_mu_means = [self.mus[0][l] for l in range(self.n_labels)]
        prior_mu_stds = [np.eye(self.n_obs)*100 for l in range(self.n_labels)]
        self.priors_mu = [multivariate_normal(prior_mu_means[l], prior_mu_stds[l]) for l in range(self.n_labels)]

        # cov
        # generate b_sigma
        self.b_sigma = np.zeros((self.n_labels, self.n_obs))
        for l in range(self.n_labels):
            self.b_sigma[l, :] = np.log(np.sqrt(np.diag(self.gmm.covariances_[l, :, :])))
        # generate kesi
        self.kesi = np.ones((self.n_labels, self.n_obs)) * 100
        # generate nu
        self.nu = self.n_obs + 1
        # ************************************************************************************************

    def fit(self, n, verbose=False):
        """

        :param n:
        :return:
        """
        for g in tqdm.trange(n):
            self.gibbs_sample()

    def gibbs_sample(self, verbose=False, t=1., beta_jump_length=0.1, mu_jump_length=0.0005,
                     cov_jump_length=0.00005, theta_jump_length=0.0005):
        """

        :param i:
        :param verbose:
        :param t:
        :param beta_jump_length:
        :param mu_jump_length:
        :param cov_jump_length:
        :param theta_jump_length:
        :return:
        """
        i = -1
        # ************************************************
        # CALCULATE TOTAL ENERGY
        # 1 - calculate energy likelihood for each element and label
        energy_like = self.calc_energy_like(self.mus[i], self.covs[i], self.labels[i])
        if verbose:
            print("likelihood energy:", energy_like)
        # 2 - calculate gibbs/mrf energy
        gibbs_energy = self.calc_gibbs_energy(self.labels[i], self.betas[i])
        if verbose:
            print("gibbs energy:", gibbs_energy)
        # 3 - self energy
        self_energy = np.zeros(self.n_labels)
        # 5 - calculate total energy
        total_energy = energy_like + self_energy + gibbs_energy
        if verbose:
            print("total_energy:", total_energy)

        # ************************************************************************************************
        # CALCULATE PROBABILITY OF LABELS
        labels_prob = calc_labels_prob(total_energy, t)
        if verbose:
            print("Labels probability:", labels_prob)

        # ************************************************************************************************
        # DRAW NEW LABELS FOR EVERY ELEMENT
        new_labels = np.array([np.random.choice(list(range(self.n_obs)), p=labels_prob[x, :]) for x in range(len(self.coords))])
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
        # CALCULATE LOG MIXTURE DENSITY for previous mu and cov
        lmd_prev = self.calc_sum_log_mixture_density(comp_coef, self.mus[i], self.covs[i])

        # ************************************************************************************************
        # ************************************************************************************************
        # PROPOSAL STEP
        # make proposals for beta, mu and cov
        beta_prop = self.propose_beta(self.betas[-1], beta_jump_length)
        mu_prop = self.propose_mu(self.mus[-1], mu_jump_length)
        cov_prop = self.propose_cov(self.covs[-1], cov_jump_length, theta_jump_length)

        # ************************************************************************************************
        # calculate proposal component coefficient
        # gibbs_energy_prop = self.calc_gibbs_energy(new_labels, beta_prop)  # calculate gibbs energy with new labels and proposed beta
        # energy_for_comp_coef_prop = gibbs_energy_prop + self_energy
        # comp_coef_prop = calc_labels_prob(energy_for_comp_coef_prop, t)

        # calculate log mixture density for proposed mu and cov
        lmd_prop = self.calc_sum_log_mixture_density(comp_coef, mu_prop, cov_prop)

        # ************************************************************************************************
        # COMPARE THE SHIT FOR EACH LABEL FOR EACH SHIT

        # prepare next ones
        mu_next = copy(self.mus[-1])
        cov_next = copy(self.covs[-1])

        # compare beta
        lp_beta_prev = self.log_prior_density_beta(self.betas[-1])
        lp_beta_prop = self.log_prior_density_beta(beta_prop)

        for l in range(self.n_labels):
            # logprob prior density for covariance
            lp_cov_prev = self.log_prior_density_cov(self.covs[-1], l)
            lp_cov_prop = self.log_prior_density_cov(cov_prop, l)
            if verbose:
                print("lp_cov_prev:", lp_cov_prev)
                print("lp_cov_prop:", lp_cov_prop)

            # logprob prior density for mu
            lp_mu_prev = self.log_prior_density_mu(self.mus[-1], l)
            lp_mu_prop = self.log_prior_density_mu(mu_prop, l)
            if verbose:
                print("lp_mu_prev:", lp_mu_prev)
                print("lp_mu_prop:", lp_mu_prop)

            # combine
            log_target_prev = lmd_prev + lp_cov_prev + lp_mu_prev + lp_beta_prev
            log_target_prop = lmd_prop + lp_cov_prop + lp_mu_prop + lp_beta_prev
            # acceptance ratio
            acc_ratio = log_target_prop / log_target_prev

            if (acc_ratio > 1) or (np.random.uniform() < acc_ratio):
                # replace old with new if accepted
                mu_next[l, :] = mu_prop[l, :]
                cov_next[l, :] = cov_prop[l, :]

        # append cov and mu
        self.mus.append(mu_next)
        self.covs.append(cov_next)

        lp_mu_prev = self.log_prior_density_mu(self.mus[i], l)
        lp_cov_prev = self.log_prior_density_cov(self.covs[i], l)
        lmd_prev = self.calc_sum_log_mixture_density(comp_coef, self.mus[i], self.covs[i])

        gibbs_energy_prop = self.calc_gibbs_energy(new_labels, beta_prop)  # calculate gibbs energy with new labels and proposed beta
        energy_for_comp_coef_prop = gibbs_energy_prop + self_energy
        comp_coef_prop = calc_labels_prob(energy_for_comp_coef_prop, t)

        lmd_prop = self.calc_sum_log_mixture_density(comp_coef_prop, self.mus[i], self.covs[i])

        log_target_prev = lmd_prev + lp_cov_prev + lp_mu_prev + lp_beta_prev
        log_target_prop = lmd_prop + lp_cov_prev + lp_mu_prev + lp_beta_prop

        acc_ratio = log_target_prop / log_target_prev
        if (acc_ratio > 1) or (np.random.uniform() < acc_ratio):
            self.betas.append(beta_prop)
        else:
            self.betas.append(self.betas[-1])

        if verbose:
            print("Gibbs sample completed.")

    def log_prior_density_mu(self, mu, label):
        """

        :param mu:
        :param label:
        :return:
        """
        return np.sum(np.log(self.priors_mu[label].pdf(mu)))

    def log_prior_density_beta(self, beta):
        """

        :param beta:
        :return:
        """
        return np.log(self.prior_beta.pdf(beta))

    def log_prior_density_cov(self, cov, label):
        """

        :param cov:
        :param label:
        :return:
        """
        lam = np.sqrt(np.diag(cov[label, :, :]))
        r = np.diag(1./lam) @ cov[label, :, :] @ np.diag(1./lam)
        logp_r = -0.5 * (self.nu + self.n_obs + 1) * np.log(np.linalg.det(r)) - self.nu/2. * np.sum(np.log(np.diag(np.linalg.inv(r))))
        # yeah obviously - does anyone in the world understand this?!
        logp_lam = np.sum(np.log(multivariate_normal(mean=self.b_sigma[label, :], cov=self.kesi[label, :]).pdf(np.log(lam.T))))
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

        :param beta_prev:
        :param beta_jump_length:
        :return:
        """
        # create proposal covariance depending on physical dimensionality
        dim = [1, 4, 13]
        sigma_prop = np.eye(dim[self.dim - 1]) * beta_jump_length
        # draw from multivariate normal distribution and return
        return multivariate_normal(mean=beta_prev, cov=sigma_prop).rvs()

    def propose_mu(self, mu_prev, mu_jump_length):
        """

        :param mu_prev:
        :param mu_jump_length:
        :return:
        """
        # create proposal covariance depending on observation dimensionality
        sigma_prop = np.eye(self.n_obs) * mu_jump_length
        # prepare matrix
        mu_prop = np.ones((self.n_obs, self.n_labels))
        # loop over labels
        for l in range(self.n_labels):
            mu_prop[l, :] = multivariate_normal(mean=mu_prev[l, :], cov=np.eye(self.n_obs) * mu_jump_length).rvs()
        return mu_prop

    def propose_cov(self, cov_prev, cov_jump_length, theta_jump_length):
        """

        :param cov_prev:
        :param cov_jump_length:
        :param theta_jump_length:
        :return:
        """
        # do svd on the previous covariance matrix
        comb = list(combinations(range(self.n_obs), 2))
        n_axis = len(comb)
        theta_jump = multivariate_normal(mean=[0 for i in range(n_axis)], cov=np.ones(n_axis) * theta_jump_length).rvs()
        cov_prop = np.zeros_like(cov_prev)

        # print("cov_prev:", cov_prev)

        for l in range(self.n_labels):

            v_l, d_l, v_l = np.linalg.svd(cov_prev[l, :, :])
            # print("v_l:", v_l)
            # generate d jump
            log_d_jump = multivariate_normal(mean=[0 for i in range(self.n_obs)], cov=np.eye(self.n_obs) * cov_jump_length).rvs()
            # sum towards d proposal
            # if l == 0:
            d_prop = np.diag(np.exp(np.log(d_l) + log_d_jump))
            # else:
            #    d_prop = np.vstack((d_prop, np.exp(np.log(d_l) + np.log(d_jump))))

            # now tackle generating v jump
            a = np.eye(self.n_obs)
            # print("a init:", a)
            # print("shape a:", np.shape(a))
            for j in range(n_axis):
                rotation_matrix = _cov_proposal_rotation_matrix(v_l[:, comb[j][0]], v_l[:, comb[j][1]], theta_jump)
                # print("rot mat:", rotation_matrix)
                # print("rot mat:", rotation_matrix)
                a = np.matmul(rotation_matrix, a)
                # print("a:", a)
            # print("v_l:", v_l)
            v_prop = np.matmul(a, v_l)
            # print("d_prop:", d_prop)
            # print("v_prop:", v_prop)
            # TODO: Is this proposal covariance slicing correct?
            cov_prop[l, :, :] = np.matmul(np.matmul(v_prop, d_prop), v_prop)
            # print("cov_prop:", cov_prop)

        return cov_prop

    def calc_sum_log_mixture_density(self, comp_coef, mu, cov):
        """
        Calculate sum of log mixture density.
        :param comp_coef:
        :param mu:
        :param cov:
        :return: lmd
        """
        if self.dim == 1:
            lmd = 0.
            for x in range(len(self.coords)):
                storage2 = []
                for l in range(self.n_labels):
                    storage2.append(comp_coef[x, l] * multivariate_normal(mean=mu[l, :], cov=cov[l, :]).pdf(self.obs[x]))
                lmd += (np.log(np.sum(storage2)))

        else:
            pass
        # TODO: 2-dimensional log mixture density
        # TODO: 3-dimensional log mixture density

        return lmd

    def calc_energy_like(self, mu, cov, labels):
        """

        :param mu:
        :param cov:
        :param labels:
        :return:
        """

        energy_like_labels = np.zeros((len(self.coords), self.n_labels))
        if self.dim == 1:
            for x in range(len(self.coords)):
                for l in  range(self.n_labels):
                    energy_like_labels[x, l] = 0.5 * np.array([self.obs[x] - mu[l, :]]) @ np.linalg.inv(cov[l, :, :]) @ np.array([self.obs[x] - mu[l, :]]).T + 0.5 * np.log(np.linalg.det(cov[l, :, :]))
                #print(energy_like_labels[x])
                    #energy_like_labels[x] = (0.5 * np.dot(np.dot((self.obs[x] - mu[l, :]), np.linalg.inv(cov[l, :, :])),
                    #                                   (self.obs[x] - mu[l, :]).transpose()) + 0.5 * np.log(
                    #    np.linalg.det(cov[l, :, :]))).flatten()
        else:
            pass
        # TODO: 2-dimensional calculation of energy likelihood labels
        # TODO: 3-dimensional calculation of energy likelihood labels

        return energy_like_labels

    def calc_gibbs_energy(self, labels, beta):
        """
        Calculate Gibbs energy for each element.
        :param labels:
        :param beta:
        :return:
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


def _cov_proposal_rotation_matrix(x, y, theta):
    """

    :param x, y: two base vectors defining a plane
    :param theta: rotation angle in this plane
    :return: rotation matrix for covariance proposal step
    """
    x = np.array([x]).T
    y = np.array([y]).T

    uu = x / np.linalg.norm(x)
    vv = y - np.matmul(uu.T, y) * uu
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


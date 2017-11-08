"""

"""

import numpy as np
import pandas as pd
from sklearn import mixture
from scipy.stats import multivariate_normal, norm
from copy import copy


class HMRFGMM2:
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
        # ************************************************
        # INIT STORAGE ARRAYS
        self.betas = np.array([beta_init])  # initial beta
        # self.mus = np.array([], dtype=object)
        # self.covs = np.array([], dtype=object)
        # self.labels = np.array([], dtype=object)
        # ************************************************
        # generate neighborhood system
        self.neighborhood = define_neighborhood_system(coordinates)
        # ************************************************
        # INIT GAUSSIAN MIXTURE MODEL
        self.gmm = mixture.GaussianMixture(n_components=self.n_obs, covariance_type="full")
        # fit it to features
        self.gmm.fit(self.obs)
        # do initial prediction based on fit and observations, store as first entry in labels
        # ************************************************
        # INIT LABELS based on GMM
        self.labels = np.array([self.gmm.predict(self.obs)])
        # ************************************************
        # INIT MU (mean from initial GMM)
        self.mus = np.array([self.gmm.means_])
        # ************************************************
        # INIT COV (covariances from initial GMM)
        self.covs = np.array([self.gmm.covariances_])

    def gibbs_sample(self, i=-1, verbose=True):
        # ************************************************
        # CALCULATE TOTAL ENERGY
        # 1 - calculate energy likelihood
        energy_like = self.calc_energy_like(self.mus[i], self.covs[i], self.labels[i])
        # 2 - calculate gibbs/mrf energy
        gibbs_energy = self.calc_gibbs_energy(self.labels[i], self.betas[i])
        # 3 - self energy
        self_energy = 0.
        # 4 - calculate energy likelihood labels
        energy_like_labels = self.calc_energy_like_labels(self.mus[i], self.covs[i], self.labels[i])
        # 5 - calculate total energy
        total_energy = energy_like + self_energy + gibbs_energy
        total_energy_labels = (energy_like_labels.T + total_energy).T
        # ************************************************
        # CALCULATE PROBABILITY OF LABELS
        labels_prob = calc_labels_prob(total_energy_labels, 1.)  # TODO: What does t mean? Why is it 1?
        # norm
        labels_prob = (labels_prob.T / np.sum(labels_prob, axis=1)).T  # TODO: Is this correct?
        if verbose:
            print("labels_prob:", labels_prob)
        # ************************************************
        # DRAW NEW LABELS FOR EVERY ELEMENT
        new_labels = np.array([np.random.choice(list(range(self.n_obs)), p=labels_prob[x, :]) for x in range(len(self.coords))])
        if verbose:
            print("new labels (sum):", np.sum(new_labels))

    def calc_energy_like(self, mu, cov, labels):
        """
        Calculate energy likelihood for each element based on Wang et al. 2016, Eq. 29.
        :param mu: mean
        :param cov: covariance
        :param labels: labels
        :return: 1-,2- or 3-dimensional array containing the energy likelihoods for each element.
        """
        energy_like = np.zeros_like(labels)
        if self.dim == 1:
            for x in range(len(self.coords)):
                l = labels[x]
                energy_like[x] = (0.5 * np.dot(np.dot((self.obs[x] - mu[l, :]), np.linalg.inv(cov[l, :, :])), (self.obs[x] - mu[l, :]).transpose()) + 0.5 * np.log(np.linalg.det(cov[l, :, :]))).flatten()
        elif self.dim == 2:
            pass
            # TODO: 2-dimensional calculation of energy likelihood
        elif self.dim == 3:
            pass
            # TODO: 3-dimensional calculation of energy likelihood

        # TODO: Optimize energy likelihood calculation
        return energy_like

    def calc_energy_like_labels(self, mu, cov, labels):
        """

        :param mu:
        :param cov:
        :param labels:
        :return:
        """
        energy_like_labels = copy(self.obs)
        if self.dim == 1:
            for x in range(len(self.coords)):
                l = labels[x]
                energy_like_labels[l] = (0.5 * np.dot(np.dot((self.obs[x] - mu[l, :]), np.linalg.inv(cov[l, :, :])),
                                                   (self.obs[x] - mu[l, :]).transpose()) + 0.5 * np.log(
                    np.linalg.det(cov[l, :, :]))).flatten()
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
        gibbs_energy = np.zeros_like(labels, dtype="float64")

        if self.dim == 1:
            for i, nl in enumerate(self.neighborhood):
                for n in nl:
                    if labels[i] != labels[n]:
                        gibbs_energy[i] += beta

        elif self.dim == 2:
            pass
            # TODO: 2-dimensional calculation of gibbs energy
        elif self.dim == 3:
            pass
            # TODO: 3-dimensional calculation of gibbs energy

        # TODO: Optimize gibbs energy calculation
        return gibbs_energy


def calc_labels_prob(te, t):
    """"Calculate labels probability for array of total energies (te) and totally arbitrary skalar value t."""
    return np.exp(-te/t) / np.sum(np.exp(-te/t))


def define_neighborhood_system(coordinates):
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

# ************************************************************************************************
# ************************************************************************************************
# ************************************************************************************************


class HMRFGMM:
    def __init__(self, coordinates, observations, n_labels=2, beta_init=0.5, phys_names=None, feat_names=None):
        """

        :param coordinates: 1-, 2- or 3-dimensional np.ndarray with the physical coordinates
        :param observations: n-dimensional np.ndarray with feature data
        """
        # create feature names if not given
        self.n_feat = np.shape(observations)[1]
        if not feat_names:
            self.feat_names = ["f" + str(i) for i in range(np.shape(observations)[1])]
        else:
            self.feat_names = feat_names

        # create dimension names if not given
        self.phys_dim = np.shape(coordinates)[1]
        if not phys_names:
            if self.phys_dim == 1:
                self.phys_names = ["x"]
            elif self.phys_dim == 2:
                self.phys_names = ["x", "y"]
            elif self.phys_dim == 3:
                self.phys_names = ["x", "y", "z"]
        else:
            self.phys_names = phys_names

        # data frame initialization
        self._cols = self.phys_names + self.feat_names
        self.data = pd.DataFrame(np.hstack((coordinates, observations)), columns=self._cols)

        # pseudocolor all elements for parallelization
        self.data["color"] = self.pseudocolor_elements()

        # ************************************************
        # STORAGE ARRAYS
        # ************************************************
        # initialize global output parameter storage arrays
        self.mu_all = []
        self.cov_all = []
        self.beta_all = []
        self.labels_all = []
        self.energy_mrf_all = []
        self.energy_like_all = []
        self.energy_self_all = []
        self.neighbors_all = []
        self.colors_all = []

        # define neighbors
        self.data["neighbors"] = None
        self.neighbors_define()

        # ************************************************
        # GAUSSIAN MIXTURE MODEL
        # ************************************************
        self.gmm_init()
        self.gmm = mixture.GaussianMixture(n_components=self.n_feat, covariance_type="full")
        # make initial fit
        self.gmm.fit(self.data[self.feat_names].values)
        # initial labeling according to GMM
        self.labels_all.append(self.gmm_predict_labels())
        self.data["label"] = self.gmm_predict_labels()

        # get initial mu mu: vector of mean values for features for all clusters (n x p matrix: n: number of
        # clusters, p: number of features)
        self.mu_init = self.gmm.means_
        # get initial covariance matrix, (p x p x n)
        self.cov_init = self.gmm.covariances_
        # set initial beta
        self.beta_init = beta_init

        self.mu_all.append(self.mu_init)
        self.cov_all.append(self.cov_init)

        # step length for mu and cov
        self.mu_step = 0.0005
        self.cov_step = 0.00005

        # number of gibbs iterations (max and current)
        self.n_gibbs_count = 0

        # number of labels
        self.n_labels = n_labels

    def calculate_gibbs_energy_element(self, i):
        """Calculates MRF energy for element i."""
        # reset mrf energy to 0
        self.data.set_value(i, "mrf_energy", 0)
        # loop over all neighbors
        # TODO: vectorize mrf energy calculation
        for n in self.data.iloc[i]["neighbors"]:
            # check if label of element and its neighbor are identical
            if self.data.iloc[i]["label"] != self.data.iloc[n]["label"]:
                # if not add beta to mrf energy
                # TODO: check with jack if this selection of beta is correct
                if self.n_gibbs_count == 0:
                    self.data.set_value(i, "mrf_energy", self.data.iloc[i]["mrf_energy"] + self.beta_init)
                else:
                    self.data.set_value(i, "mrf_energy", self.data.iloc[i]["mrf_energy"] + self.beta_all[-1])

    # def calculate_mrf_energy_all(self):
    #     """Calculates mrf energy for each element."""
    #     for i in self.data.index:
    #         self.calculate_mrf_energy_element(i)

    def calculate_likelihood_energy_element(self, i):
        vals = self.data.iloc[i][self.feat_names].values.astype("float32")

        mu = self.mu_all[-1]
        # print(np.shape(mu))
        cov = self.cov_all[-1]
        # print(np.shape(cov))
        l = self.data.iloc[i]["label"]
        like_energy = (0.5 * np.dot(np.dot((vals - mu[l, :]), np.linalg.inv(cov[l, :, :])),
                                    (vals - mu[l, :]).transpose()) + 0.5 * np.log(
            np.linalg.det(cov[self.data.iloc[i]["label"], :, :]))).flatten()
        # print(np.shape(like_energy))
        return like_energy

    def calculate_likelihood_energy_labels(self, i):
        vals = self.data.iloc[i][self.feat_names].values.astype("float32")

        mu = self.mu_all[-1]
        cov = self.cov_all[-1]

        like_energy_all = np.empty(self.n_labels)
        for l in range(self.n_labels):
            like_energy_all[l] = (0.5 * np.dot(np.dot((vals - mu[l, :]), np.linalg.inv(cov[l, :, :])), (vals - mu[l, :]).transpose()) + 0.5 * np.log(np.linalg.det(cov[l, :, :]))).flatten()
        return like_energy_all
        # TODO: maybe combine with calculate_likelihood_energy_element()

    def calculate_self_energy(self, i):
        # TODO: calculation of self energy
        return np.array([0.])

    def calculate_combined_energy(self, i):
        """Sum up all energies."""
        # print(np.shape(np.sum([self.calculate_likelihood_energy_element(i), self.calculate_likelihood_energy_labels(i), self.calculate_self_energy(i)])))
        return np.sum([self.calculate_likelihood_energy_element(i), self.calculate_likelihood_energy_labels(i), self.calculate_self_energy(i)])

    def calculate_p_labels(self, i):
        u = self.calculate_combined_energy(i)
        t = 1.  # FW: or simulated annealing TODO: T for p_labels calculation
        return np.exp(-u/t) / np.sum(np.exp(-u/t))

    def redraw_element_label(self, i):
        """Simple random draw to update label for element i."""
        self.data.set_value(i, "label", np.random.choice(list(range(self.n_labels)), p=self.calculate_p_labels(i)))

    def gibbs_sampling(self):
        """Gibbs sampling: directly update all the labels given their calculated energy."""
        for i in self.data.index:
            self.redraw_element_label(i)
        # TODO: make a more coherent gibbs sampling function

    def gmm_init(self):
        """Initialize GMM and fit it to features."""
        self.gmm = mixture.GaussianMixture(n_components=self.n_feat, covariance_type="full")
        self.gmm_fit()

    def gmm_fit(self):
        """Fit the Gaussian Mixture Model to all feature values."""
        self.gmm.fit(self.data[self.feat_names].values)

    def gmm_predict_labels(self):
        """Predict labels for each element using the Gaussian Mixture Model."""
        return self.gmm.predict(self.data[self.feat_names].values)

    def fit(self, n_gibbs):
        # 3 - Define "funky kesi" hyperparameters
        # TODO: what does this even mean
        b = np.log(np.sqrt(np.diag(self.cov_init[0, :])))
        kesi = n_gibbs * np.ones(self.n_feat)
        nu = self.n_feat + 1

        # 4 - Define hyperparameters of prior distribution of mu, separately for each cluster
        mu_clusters = [self.gmm.means_[feat] for feat in range(self.n_feat)]  # use output of GMM
        mu_std_clusters = [[[100, 0.], [0., 100]] for feat in range(self.n_feat)]  # define very wide std

        # 5 - Random variables for proposed new mu in feature space
        rvs_mu = [multivariate_normal(mu_clusters[feat], mu_std_clusters[feat]) for feat in range(self.n_feat)]

        # define random variables for warp jumps
        rv_jump_mu = multivariate_normal(mean=np.zeros(self.n_feat), cov=np.diag([self.mu_step, self.mu_step]))
        rv_jump_cov = multivariate_normal(mean=np.zeros(self.n_feat), cov=np.diag([self.cov_step, self.cov_step]))

        # loop
        for i in range(n_gibbs):
            # ************************************************
            # STEP 1: GIBBS SAMPLING (i.e.: update labels)
            # ************************************************
            self.gibbs_sampling()  # should automatically take latest mu, cov
            # ************************************************
            # STEP 2: HMRF
            # ************************************************
            cov_proposed = propose_cov(self.cov_all[-1], rv_jump_cov)
            mu_proposed = propose_mu(self.mu_all[-1], rv_jump_mu)

            mu = copy(self.mu_all[-1])
            cov = copy(self.cov_all[-1])

            # calculate likelihood
            # update alpha
            # apparently alpha seems to be the same as p_labels
            alpha = []
            for index in self.data.index:
                alpha.append(self.calculate_p_labels(index))
            # accept/reject for each cluster seperately
            for label in range(self.n_labels):
                # update mixture density - note: in a sense, this is another level of Gibbs sampling!
                lmd_prev = self.log_mixture_density(alpha, self.mu_all[-1], self.cov_all[-1])
                lmd_proposed = self.log_mixture_density(alpha, mu_proposed, cov_proposed)
                # calculate prior density for mean and cov matrices
                lp_cov_prev = self.logprob_cov(self.cov_all[-1][label, :], self.n_feat, b, kesi, nu)
                lp_cov_proposed = self.logprob_cov(cov_proposed[label, :], self.n_feat, b, kesi, nu)

                lp_mu_prev = self.logprob_mu(self.mu_all[-1][label, :], rvs_mu[label])
                lp_mu_proposed = self.logprob_mu(mu_proposed[label, :], rvs_mu[label])
                # combine likeihoold and priors:
                log_target_prev = lmd_prev + lp_cov_prev + lp_mu_prev
                log_target_proposed = lmd_proposed + lp_cov_proposed + lp_mu_proposed

                # determine acceptance ratio:
                acc_ratio = log_target_proposed / log_target_prev

                if (acc_ratio > 1) or (np.random.uniform() < acc_ratio):  # accept directly
                    mu[label, :] = mu_proposed[label, :]
                    cov[label, :] = cov_proposed[label, :]

            # ************************************************
            # Store newly generated field (or keep previous ones)
            # ************************************************
            self.mu_all.append(mu)  # TODO: this right here
            self.cov_all.append(cov)
            self.labels_all.append(self.data["label"].values)
            # print("iter:", i, "; labels sum:", self.labels_all[-1].sum())

            # +1 iteration counter
            # print("gibbs_count:", self.n_gibbs_count)
            self.n_gibbs_count += 1

    def neighbors_define(self):
        for i in self.data.index:
            if i == 0:
                self.data.set_value(i, "neighbors", [i + 1])
            elif i == self.data.index.max():
                self.data.set_value(i, "neighbors", [i - 1])
            else:
                self.data.set_value(i, "neighbors", [i - 1, i + 1])
                # so far just above and below for 1D example
                # TODO: define general methods to find neighbors based on distances

    def logprob_cov(self, cov_matrix, d, b, kesi, nu):
        """Caculate log probability of covariance matrix

            d: number of features
            parameter for kesi stuff (see Alvarez, 2014):
            b, kesi: shape and skew parameters (papers to be digged out by Jack)
            nu: another funky parameter from Alvarez, 2014
        """
        # TODO: parameter for kesi stuff (see Alvarez, 2014): b, kesi: shape and skew parameters (papers to be digged out by Jack)nu: another funky parameter from Alvarez, 2014
        # calculate lambda
        lam = np.sqrt(np.diag(cov_matrix))
        R = np.diag(1. / lam) @ cov_matrix @ np.diag(1. / lam)
        logP_R = -0.5 * (nu + d + 1) * np.log(np.linalg.det(R)) - nu / 2. * np.sum(np.log(np.diag(np.linalg.inv(R))))

        logP_lam = 0.
        for i in range(len(lam)):
            logP_lam += np.log(norm.pdf(lam[i], b[i], kesi[i]))

        return logP_R + logP_lam

    def logprob_mu(self, mu, rv_mu):
        """Log prob for mu for one cluster"""
        return np.log(rv_mu.pdf(mu))

    def log_mixture_density(self, alpha, mu, cov):
        """Calculate (log) mixture density for given mean and covariance matrices

        mu : mean matrix for all classes and features
        cov : dito
        """
        lmd = 0.
        # for j, e in enumerate(self.data):
        for j in self.data.index:
            for l in range(self.n_labels):
                lmd += alpha[j][l] * multivariate_normal(mean=mu[l, :], cov=cov[l, :]).pdf(self.data.iloc[j]["label"]) # TODO: is this correct to take the label value?

        return lmd

    def pseudocolor_elements(self):
        """Pseudocolor all elements for parallelization."""
        return np.mod(np.array(self.data.index), 2)
        # so far only for 1d data, so just binary switcharoo
        # TODO: implement 2- and 3-D coloring

def propose_mu(mu_prev, rv_jump_mu):
    """Propose new covariance function based on prev. cov and RV for cov. jump

    cov : prev. mu matrix
    rv_jump_mu : stats.multivariate_normal object for mu. matrix jump

    """
    jump_mu = rv_jump_mu.rvs()
    mu_proposed = mu_prev + jump_mu
    return mu_proposed


def propose_cov(cov_prev, rv_jump_cov):
    """Propose new covariance function based on prev. cov and RV for cov. jump

    cov : prev. covariance matrix
    rv_jump_cov : stats.multivariate_normal object for cov. matrix jump

    """
    jump_cov = rv_jump_cov.rvs()
    cov_proposed = np.empty_like(cov_prev)
    for l in [0, 1]:
        [eigenvals, eigenvec] = np.linalg.eig(cov_prev[l, :, :])
        D_star = np.diag(np.exp(np.log(eigenvals) + jump_cov))
        # rotation
        # define the jump for the rotation
        theta = np.random.normal(0, 0.005)
        c, s = np.cos(theta), np.sin(theta)
        R = np.matrix([[c, -s], [s, c]])
        V_star = eigenvec @ R
        cov_proposed[l, :] = V_star @ D_star @ V_star.transpose()

    return cov_proposed



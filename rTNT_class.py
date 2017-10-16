"""Module for classification function."""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from joblib import Parallel, delayed

from pyriemann.utils.mean import mean_covariance
from pyriemann.utils.distance import distance
from pyriemann.tangentspace import FGDA
from pyriemann.estimation import ERPCovariances
from pyriemann.utils.mean import mean_covariance
from pyriemann.utils.distance import distance
from eegdatabase_PIV import Database
import numpy as np
import matplotlib.pyplot as plt
from pyriemann.tangentspace import TangentSpace , FGDA



class Distribution_R_TNT():
    """Attributes
    classes: list of labels, with classes[0] = Target_Label and classes[1]=NonTarget_Label
    centroids: list of the class center covariance matrix, with centroids[0] is TARGET and centroids[1] is NONTARGET
    distribution: list of the r_TNT distribution, with distribution[0] is the r_TNT of the target train trials and distribution[1]is the r_TNT of the non-target train trials
    distribution_mean:list of the mean of the two r_TNT distribution with distribution_mean[0] the target distribution mean and distribution_mean[1] the non-target distribution mean
    distribution_sigma:list of the std of the two r_TNT distribution with distribution_sigma[0] the target distribution std and distribution_sigma[1] the non-target distribution mean
    """


    def __init__(self, metric='riemann'):
        """Init."""

        # store params for cloning purpose
        self.metric = metric

        if isinstance(metric, str):
            self.metric_mean = metric
            self.metric_dist = metric

        elif isinstance(metric, dict):
            # check keys
            for key in ['mean', 'distance']:
                if key not in metric.keys():
                    raise KeyError('metric must contain "mean" and "distance"')

            self.metric_mean = metric['mean']
            self.metric_dist = metric['distance']

        else:
            raise TypeError('metric must be dict or str')

    def fit(self, X, y, T = 0, NT = 1 , sample_weight=None):
        """Fit (estimates) the centroids and the parameters of the R_TNT distribution.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.
        y : ndarray shape (n_trials, 1)
            labels corresponding to each trial.
        T,NT : name of the label TARGET and the label NONTARGET given in y
        sample_weight : None | ndarray shape (n_trials, 1)
            the weights of each sample. if None, each sample is treated with
            equal weights.

        Returns
        -------
        self : Bayes_R_TNT instance

            The Bayes_R_TNT instance.
        """
        self.classes = [T,NT]

        # self.centroids = []

        if sample_weight is None:
            sample_weight = np.ones(X.shape[0])

        # for l in self.classes:
        #     self.centroids.append(
        #             mean_covariance(X[y == l], metric=self.metric_mean,
        #                             sample_weight=sample_weight[y == l]))
        self.centroids = [mean_covariance(X[y == l], metric=self.metric_mean,
                                    sample_weight=sample_weight[y == l]) for l in self.classes]
        distance_list = [np.array([distance(x, self.centroids[l]) for i, x in enumerate(X)]) for l in
                             self.classes]
        self.distribution = [
                [np.log(distance_list[0][index] / distance_list[1][index]) for index, value in enumerate(X) if
                 y[index] == l] for l in self.classes]
        self.distribution_mean = [np.mean(self.distribution[l]) for l in self.classes]
        self.distribution_sigma = [np.var(self.distribution[l]) for l in self.classes]

        if self.distribution_mean[0]>= self.distribution_mean[1]:
            print ('Target R_TNT Distribution mean should be smaller as Non Target R_TNT Distribution mean. Check the labels.')
        return self

    def _predict_R_TNT(self, X):
        """Helper to predict the r_TNT for a new set of trials.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels)

            """

        return predict_R_TNT(X, self.centroids, classes=['T', 'NT'])

    def predict(self, X):
        """get the predictions.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.

        Returns
        -------
        pred : ndarray of int, shape (n_trials, 1)
            the prediction for each trials according to the closest centroid. (without any inference)

        """
        R_TNT = self._predict_r_TNT(X)
        R_TNT = np.reshape(R_TNT, (R_TNT.shape[0]))

        prediction_TNT = []
        for l in self.classes:
            prediction_TNT.append(
                -0.5 * np.log(self.var_distribution[l]) - 0.5 * np.square(r_TNT - self.mean_distribution[l]) /
                self.var_distribution[l])

        # prediction_TNT = [[-0.5 * np.log(self.var_distribution[l]) - 0.5 * np.square(r_TNT - self.mean_distribution[l])/self.var_distribution[l]]
        #                   for l in self.classes]

        prediction_TNT = np.array(prediction_TNT)

        pred = [self.classes[i] for i in np.argmax(prediction_TNT, axis=0)]


        return pred

    def transform(self, X):
        """get the r_TNT to each distribution, given the centroids self.centroids. For the train data set

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.

        Returns
        -------
        dist : ndarray, shape (n_trials, n_classes)
            the r_TNT to each distribution according to the metric and the centroids.
        """

        return self._predict_R_TNT(X)

    def fit_predict(self, X, y):
        """Fit and predict in one function. For training data set. """
        self.fit(X, y)
        return self.predict(X)


class R_TNT_NaiveBayes():
    '''
    Attributes
    class_prior: (,36) array with the prior probabilities of each item at each flash
    mu_TNT: mean of the calibration r_TNT distribution (mu_TNT[0] is the target distribution, mu_TNT[1] is the non-target distribution)
    sigma_TNT: std of the calibration r_TNT distribution (sigma_TNT[0] is the target distribution, sigma_TNT[1] is the non-target distribution)
    targets: (,36) array wih the itels that are about to be flashed /
    logfeat: log of current likelyhood  of both Target and Non Target classes

    '''

    def __init__(self, targets_=np.ndarray(7), mu_TNT=None, sigma_TNT=None, class_prior=None):
        self.class_prior = class_prior
        self.mu_TNT = mu_TNT
        self.sigma_TNT = sigma_TNT
        self.targets_ = targets_


    def update_class_prior(self, r_TNT, flashed_item_index):

        n_classes = len(self.targets_)

        if self.class_prior is not None:
            if len(self.class_prior) != n_classes:
                raise ValueError("Number of priors must match number of classes.")
            self.class_prior = self.rte_PostProba_Dyn(r_TNT, self.class_prior, flashed_item_index)

        else:
            self.class_prior = (1 / float(n_classes)) * np.ones(n_classes)
            self.class_prior = self.rte_PostProba_Dyn(r_TNT, self.class_prior, flashed_item_index)

        return self.class_prior


    def rte_PostProba_Dyn(self,
            r_TNT,
            items_prior_array,
            flashed_item_index):

        # 0 is target, 1 is nontarget


        Vec0 = (r_TNT - self.mu_TNT[0]) ** 2
        Vec0 = Vec0 / self.sigma_TNT[0]

        Vec1 = (r_TNT - self.mu_TNT[1]) ** 2
        Vec1 = Vec1 / self.sigma_TNT[1]

        ld0 = np.log( 2 *np. pi *self.sigma_TNT[0])
        ld1 = np.log(2 * np.pi * self.sigma_TNT[1])

        lf0 = - 0.5 * (Vec0 + ld0)
        lf1 = - 0.5 * (Vec1 + ld1)

        flashed_item_posteriors_array = np.zeros_like(items_prior_array)

        for posterior_index, post in enumerate(flashed_item_posteriors_array):
            if (posterior_index) == flashed_item_index:
                flashed_item_posteriors_array[posterior_index] = np.log(items_prior_array[posterior_index]) + lf0
            else:
                flashed_item_posteriors_array[posterior_index] = np.log(items_prior_array[posterior_index]) + lf1

        # normalize posteriors array
        max_flashed_item_posteriors_array = flashed_item_posteriors_array.max()
        flashed_item_posteriors_array = flashed_item_posteriors_array - max_flashed_item_posteriors_array + 1
        exp_flashed_item_posteriors_array = np.exp(flashed_item_posteriors_array)

        posteriors = np.divide(exp_flashed_item_posteriors_array, exp_flashed_item_posteriors_array.sum())

        return posteriors

    def reset_bayes_prior(self):
        self.class_prior = None
        return self

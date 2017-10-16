from pyriemann.estimation import ERPCovariances
from pyriemann.utils.mean import mean_covariance
from pyriemann.utils.distance import distance
from eegdatabase_PIV import Database
import numpy as np
import matplotlib.pyplot as plt
from pyriemann.tangentspace import TangentSpace , FGDA
from pyriemann.utils.covariance import covariances, covariances_EP, cospectrum
import operator
from scipy.spatial.distance import euclidean
from pyriemann.utils.mean import mean_riemann
import pandas as pa
import scipy.io as sio

import random


def predict_ERP_centroids(x, y, metric = 'riemann' ,  ERP_bloc = None, T =0, NT =1):
    """Helper to predict the r_TNT for a new set of trials.

     Parameters
     ----------
     x : ndarray, shape (n_trials, n_channels, n_times)
     y : ndarray, shape (,n_trials)
     ERP_bloc : list with 0 or 1 for the class in the ERP

     Returns
    -------
    erp :  the ERPCovariance object with erp.P an ndarray, shape (n_channels*len(ERP_bloc), n_times)
    centroids : list of the two centers of classe which are both ndarray, shape (n_channels*len(ERP_bloc), n_channels*len(ERP_bloc))
    X : ndarray, shape (n_trials, n_channels*len(ERP_bloc), n_channels*len(ERP_bloc)), the set of super covariance matrices of set signals given in input
         """
    classes = [T,NT]
    erp = ERPCovariances(classes = ERP_bloc, estimator='cov')
    erp.fit(X = x, y = y)
    X = erp.transform(X = x )
    centroids = [mean_covariance(X[y == l,:,:], metric=metric) for l in classes]
    return erp, centroids, X

def apply_ERP(x, erp):
    """Helper to predict the r_TNT for a new set of trials.

         Parameters
         ----------
         x : ndarray, shape (n_trials, n_channels, n_times)
         erp : the ERPCovariance object with erp.P an ndarray, shape (n_channels*len(ERP_bloc), n_times)

         Returns
        -------
        X : ndarray, shape (n_trials, n_channels*len(ERP_bloc), n_channels*len(ERP_bloc)), the set of super covariance matrices of set signals given in input
             """
    return erp.transform(X = x )

def visualisation_ERP(erp, ERP_bloc):
    """Helper to visualize the ERP.

             Parameters
             ----------
             erp : the ERPCovariance object with erp.P an ndarray, shape (n_channels*len(ERP_bloc), n_times)
             ERP_bloc : list of the classes used to construct the erp
    """
    P = erp.P
    Nc, Nt = P.shape
    if len(ERP_bloc) == 1:
        ERP_fig = plt.figure()
        if ERP_bloc[0] == 0:
            col = 'g'
        if ERP_bloc[0]==1:
            col = 'r'
        plt.subplot(Nc + 1, 1, 1)
        plt.plot(np.mean(P, axis=0), c=col, label='mean above all channel')
        for i in range(Nc):
            plt.subplot(Nc + 1, 1, i+2)
            plt.plot(P[i,:], c = col)
    else:
        ERP_fig = plt.figure()
        plt.subplot(Nc + 1, 1, 1)
        plt.plot(np.mean(P[0:16,:], axis=0), c='g', label='mean above all channel for ERP-Target')
        plt.plot(np.mean(P[16:32,:], axis=0), c='r', label='mean above all channel for ERP-Non-Target')
        for i in range(Nc):
            plt.subplot(Nc + 1, 1, i + 2)
            plt.plot(P[i, :], c='g')
            plt.plot(P[i+16, :], c='r')

    plt.legend()
    plt.show()

    return ERP_fig

def predict_R_TNT(X, centroids_list):
    """Helper to predict the r_TNT for a new set of trials.

    Parameters
    ----------
    X : ndarray, shape (n_trials, n_channels, n_channels)

        """
    T = 0
    NT = 1
    dist = [distance(X, centroids_list[m])
            for m in [T,NT]]

    r_TNT = np.log(dist[0] / dist[1])

    r_TNT = np.array(r_TNT)

    return r_TNT

def visualisation_R_TNT(r_TNT, y, T = 0, NT = 1 , xlim = [-0.06, 0.06]):
    """Helper to visualize the R_TNT distribution

                 Parameters
                 ----------
                 R_TNT :  ndarray, shape (,n_trials) with the r_TNT
                 y : ndarray, shape (,n_trials) with the labels
        """

    R_TNT_fig = plt.figure()
    plt.hist(r_TNT[y == NT], 50, normed=1, facecolor='r', alpha=0.75,
             label='Classe NT')
    plt.hist(r_TNT[y == T], 50, normed=1, facecolor='g', alpha=0.75,
             label='Classe T')
    plt.xlim(xlim)
    plt.legend()
    plt.title('Distribution des ERP')
    plt.show()


    return R_TNT_fig







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

    def fit(self, X, y, centroids = None, T = 0, NT = 1 , sample_weight=None):
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
        if centroids == None:
            self.centroids = [mean_covariance(X[y == l], metric=self.metric_mean,
                                    sample_weight=sample_weight[y == l]) for l in self.classes]
        else:
            self.centroids = centroids
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


    def visualisation(self, xlim, T = 0, NT = 1):
        distribution_fig = plt.figure()
        plt.hist(self.distribution[NT], 50, normed=1, facecolor='r', alpha=0.75,
                 label='Classe NT')
        plt.hist(self.distribution[T], 50, normed=1, facecolor='g', alpha=0.75,
                 label='Classe T')
        plt.xlim(xlim)
        plt.legend()
        plt.title('Distribution des R_TNT')
        return distribution_fig


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


    def __init__(self, targets= None, mu_TNT=None, sigma_TNT=None, class_prior=None):
        self.class_prior = class_prior
        self.mu_TNT = mu_TNT
        self.sigma_TNT = sigma_TNT
        self.targets = targets

        self.list_lf0 = []
        self.list_lf1 = []
        self.list_vec0 = []
        self.list_vec1 = []

    def fit(self, train_distribution_R_TNT):
        self.mu_TNT = train_distribution_R_TNT.distribution_mean
        self.sigma_TNT = train_distribution_R_TNT.distribution_sigma


    def update_class_prior(self, r_TNT, flashed_item_index):

        n_classes = len(self.targets)

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

        # lf0 = - 0.5 * (Vec0 + ld0)
        # lf1 = - 0.5 * (Vec1 + ld1)

        lf0 = - 0.5 * (Vec0)
        lf1 = - 0.5 * (Vec1)

        self.list_lf0.append(lf0)
        self.list_lf1.append(lf1)
        self.list_vec0.append(Vec0)
        self.list_vec1.append(Vec1)

        flashed_item_posteriors_array = np.zeros_like(items_prior_array)

        for posterior_index, post in enumerate(flashed_item_posteriors_array):
            if (posterior_index) in flashed_item_index:
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

def rtp_indexSplotch2Targets(SplotchMatrixNumber, Indexplotch,  items_vec ):

    Splotchfilename = '/dycog/Jeremie/DATA_P3S_OptStop/MATLAB/MatrixSplotch/MatSplotch' + "%02d" % SplotchMatrixNumber + '.txt'
    SPM = np.loadtxt(Splotchfilename)
    index_Target = SPM[: ,Indexplotch - 1 ]
    index_Target = index_Target.astype(int) - 1
    item_Target = items_vec[index_Target]

    return item_Target

def rtp_indexColumn2Targets(IndexColumn, Screen):
    if IndexColumn in [1,2,3,4,5,6]:
        item_Target = Screen[IndexColumn -1 ,: ]

    if IndexColumn in [7,8,9,10,11,12]:
        item_Target = Screen[:, IndexColumn-7]
    return item_Target


def test_loop_PIV(r_TNT_test, y_test, e_test, t_test, train_NaiveBayes,items_list, nb_repetitions, T=0, NT=1, visu=False):


    item_prediction = []

    Max_indice = (r_TNT_test).shape[0]

    J = len(items_list) * nb_repetitions

    indice_flash = 0
    indice_char = 0
    nb_rep = 0
    item_selected_list = []

    if visu:
            # Visualisation de la distribution des ERP tests
        visualisation_R_TNT(r_TNT = r_TNT_test, y = y_test, T=T, NT=NT, xlim=[-0.06, 0.06])

        # Debut de l'inference

    while indice_flash < Max_indice:

            #         Get the r_TNT of the current trial

        r_TNT = r_TNT_test[indice_flash]
            #         Update the bayes_prior (vector of length 36)

        flashed_item = e_test[indice_flash]

        flashed_item_index = items_list.index(flashed_item)

        items_posterior_array = train_NaiveBayes.update_class_prior(r_TNT, flashed_item_index)

        item_selected = items_list[np.argmax(items_posterior_array)]

        item_selected_list.append(item_selected)

            #   Ask if it's flashed_items_total_number is enough to reset class_prior or to take a decision and change of target
        indice_flash += 1

        if not indice_flash % len(items_list):
            nb_rep += 1
                # a = plt.figure('Posteriors at each repetition_ColumnTarget_' + str(indice_char) )
                # plt.plot(items_posterior_array, label = str(nb_rep))
                # plt.legend()
                # plt.show()
                #
                # a.savefig('Posteriors_each_rep_CT_sess2_'+ str(indice_char) + '.png')
            print item_selected

        if not indice_flash % J:
            indice_char += 1
                # b = plt.figure('Posteriors at each decision')
                # plt.plot(items_posterior_array, label = str(indice_char))
                # plt.legend()
                # plt.show()
                # b.savefig('Posteriors_each_decision_sess2.png')
            item_prediction.append(item_selected)
            train_NaiveBayes.reset_bayes_prior()
                # print item_selected

    temp = np.matrix(item_selected_list)

    I = temp.shape[1] / J

    item_selected_mat = temp.reshape([I, J])

    w = t_test
    a = np.matrix(w.tolist())
    target_mat = a.reshape([I, J])
    accuracy_list = []

    k = 0
    for i in range(I):
        item_comparison = target_mat[i, :] == item_selected_mat[i, :]
        item_comparison = item_comparison.astype('float')

        if k == 0:
            A = item_comparison
            k = 1
        else:
            A = np.vstack([A, item_comparison])

    mean_acc_vector = np.mean(A, axis=0)
    mean_acc_vector = np.asarray(mean_acc_vector)[0]
    std_acc_vector = np.std(A, axis=0)
    std_acc_vector = np.asarray(std_acc_vector)[0]
    return mean_acc_vector, std_acc_vector

def find_target_word(e, y, nb_repetitions, items_vec, flashmode ):

    T = 0
    if flashmode in ['Splotch', 'RoCo']:
        column_number = 12
    else:
        column_number = 7

    target_list = []
    indice_char = 0
    for indiceflash, event in enumerate(e):
        if not indiceflash % np.int(column_number* nb_repetitions):
            flashed_item = []
            temp_e = e[indiceflash:np.int(indiceflash+2*column_number)]
            temp_y = y[indiceflash:np.int(indiceflash+2*column_number)]

            for k, temp_event in enumerate (temp_e):
                if temp_y[k] == T:
                    # The target has been flashed
                    if flashmode == 'RoCo':
                        screen = np.reshape(items_vec, [6, 6])
                        flashed_item.append(rtp_indexColumn2Targets(Screen=screen, IndexColumn= temp_event))
                    if flashmode == 'Splotch':
                        SplotchMatrixNumber = np.floor((indiceflash - indice_char * column_number * nb_repetitions) / column_number) + 1
                        flashed_item.append(rtp_indexSplotch2Targets(SplotchMatrixNumber=SplotchMatrixNumber,
                                                    Indexplotch=temp_event, items_vec=items_vec))
                    if flashmode == 'EIV':
                        flashed_item.append([items_vec[temp_event-1]])
            for item in flashed_item[0]:
                if item in flashed_item[1]:
                    target_list.append(item)
                    indice_char += 1
    return np.array(target_list)


def apply_bandpass_filter(signals, lowf, hif, filtre_order, chnames):
    """
    Apply bandpass filter to the signals.

    signals -- signals to filter
    lowf -- low cut-frequency
    hif -- high cut-frequency
    """
    sampling_rate = signals['SamplingRate'][0]

    from scipy.signal import butter, lfilter
    B, A = butter(filtre_order, np.array([lowf, hif]) / (sampling_rate / 2.0), btype='bandpass')

    X = np.array(signals[chnames])
    X = lfilter(B, A, X, axis=0)

    for i, ch in enumerate(chnames):
        signals[ch] = X[:, i]

    return signals



def extract_ERP_from_Dict(ERP_Dict, subject_key):
    return ERP_Dict[subject_key]['ERP Sum'] / ERP_Dict[subject_key]['ERP number trials']

def generic_test_loop(data, labels, event, ERP, Centroids_list, mu_TNT, sigma_TNT, nb_repetitions,  column_number = 12   , items_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
                'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '1', '2', '3', '4', '5', '6', '7', '8', '9', '_'], visu = False, flashmode = 'Splotch'):

    # nb_repetitions = data.shape[0]/(column_number*nb_targets)
    T = 0
    NT = 1
    items_vec = np.array(items_list)

    item_prediction = []

    Max_indice = data.shape[0]

    J = column_number * nb_repetitions

    indice_flash = 0
    indice_char = 0
    nb_rep = 0
    item_selected_list = []
    r_TNT_test = []

    train_NaiveBayes = R_TNT_NaiveBayes(targets= items_list, mu_TNT=mu_TNT, sigma_TNT=sigma_TNT, class_prior=None)


        # Debut de l'inference

    while indice_flash < Max_indice:

            #         Get the r_TNT of the current trial
        x_test = data[indice_flash,:,:]
        X_test = np.cov(np.concatenate([ERP, x_test], axis = 0))
        r_TNT = np.log(distance(X_test, Centroids_list[T])) - np.log(distance(X_test, Centroids_list[NT]))
        r_TNT_test.append(r_TNT)
            #         Update the bayes_prior (vector of length 36)
        if flashmode == 'RoCo':
            screen = np.reshape(items_vec, [6, 6])
            flashed_item = rtp_indexColumn2Targets(Screen=screen, IndexColumn=event[indice_flash])
            flashed_item_index = [i for i, e in enumerate(items_list) if e in list(flashed_item)]
        if flashmode == 'Splotch':
            SplotchMatrixNumber = int((indice_flash - indice_char * 12*nb_repetitions) / 12) + 1
            flashed_item = rtp_indexSplotch2Targets(SplotchMatrixNumber = SplotchMatrixNumber, Indexplotch = event[indice_flash],  items_vec  = items_vec)
            flashed_item_index = [i for i, e in enumerate(items_list) if e in list(flashed_item)]
        if flashmode == 'EIV':
            flashed_item_index = [event[indice_flash]-1]



        items_posterior_array = train_NaiveBayes.update_class_prior(r_TNT, flashed_item_index)

        item_selected = items_list[np.argmax(items_posterior_array)]

        item_selected_list.append(item_selected)

            #   Ask if it's flashed_items_total_number is enough to reset class_prior or to take a decision and change of target
        indice_flash += 1

        if not indice_flash % column_number:
            nb_rep += 1
            if visu:
                print item_selected

        if not indice_flash % J:
            indice_char += 1
            item_prediction.append(item_selected)
            train_NaiveBayes.reset_bayes_prior()


    if flashmode in ['Splotch', 'RoCo']:
        temp = np.matrix(item_selected_list)
        I = temp.shape[1] / J
    else:
        temp = np.array(item_selected_list)
        I = temp.shape[0] / J


    item_selected_mat = temp.reshape([I, J])

    w = find_target_word(e = event, y = labels, nb_repetitions = nb_repetitions, items_vec = items_vec, flashmode = flashmode)

    if flashmode in ['Splotch', 'RoCo']:
        target_vec = np.matrix(w.tolist())
    else:
        target_vec = w



    k = 0
    for j in range(J):
        item_comparison = np.transpose(target_vec) == item_selected_mat[:, j]
        item_comparison = item_comparison.astype('float')

        if k == 0:
            A = item_comparison
            k = 1
        else:
            A = np.vstack([A, item_comparison])

    # A = A.reshape((I,J))
    mean_accuracy = np.mean(A, axis=1)
    var_accuracy = np.std(A, axis=1)

    if visu:
    # Visualisation de la distribution des ERP tests
        visualisation_R_TNT(r_TNT= np.array(r_TNT_test), y=labels, T=1, NT=0, xlim=[-0.06, 0.06])
        plt.figure()
        x = np.linspace(0, J, J)
        plt.plot(mean_accuracy)
        plt.fill_between(x, mean_accuracy - 0.5 * var_accuracy, mean_accuracy + 0.5 * var_accuracy, alpha=0.1)
        plt.ylim(0,1)
        plt.xlim(0,J,J)
        plt.xlabel('#Flash')
        plt.ylabel('Mean accuracy')
        plt.title('Accuracy evolution with flashs ')
        plt.show()

    return mean_accuracy, var_accuracy


def test_loop_P300(r_TNT_test, y_test, e_test, train_NaiveBayes, T=0, NT=1, flashmode = 'RoCo', visu=False, nb_targets = 60 ):


    nb_repetitions = r_TNT_test.shape[0]/(12*nb_targets*2)

    items_list = train_NaiveBayes.targets
    items_vec = np.array(items_list)
    screen = np.reshape(items_vec, [6, 6])

    item_prediction = []

    Max_indice = (r_TNT_test).shape[0]

    J = 12 * nb_repetitions

    indice_flash = 0
    indice_char = 0
    nb_rep = 0
    item_selected_list = []

    w = find_target_word(e=e_test, y=y_test, nb_repetitions=nb_repetitions, items_vec=items_vec, flashmode=flashmode)

    target_vec = np.matrix(w.tolist())


    if visu:
            # Visualisation de la distribution des ERP tests
        visualisation_R_TNT(r_TNT = r_TNT_test, y = y_test, T=T, NT=NT, xlim=[-0.06, 0.06])

        # Debut de l'inference

    while indice_flash < Max_indice:

            #         Get the r_TNT of the current trial

        r_TNT = r_TNT_test[indice_flash]
            #         Update the bayes_prior (vector of length 36)
        if flashmode == 'RoCo':
            flashed_item = rtp_indexColumn2Targets(Screen=screen, IndexColumn=e_test[indice_flash])
        if flashmode == 'Splotch':
            SplotchMatrixNumber = floor((indice_flash - indice_char * 12*nb_repetitions) / 12) + 1
            flashed_item = rtp_indexSplotch2Targets(SplotchMatrixNumber = SplotchMatrixNumber, Indexplotch = e_test[indice_flash],  items_vec  = items_vec)

        flashed_item_index = [i for i, e in enumerate(items_list) if e in list(flashed_item)]

        items_posterior_array = train_NaiveBayes.update_class_prior(r_TNT, flashed_item_index)

        item_selected = items_list[np.argmax(items_posterior_array)]

        item_selected_list.append(item_selected)

            #   Ask if it's flashed_items_total_number is enough to reset class_prior or to take a decision and change of target
        indice_flash += 1

        if not indice_flash % 12:
            nb_rep += 1
                # a = plt.figure('Posteriors at each repetition_ColumnTarget_' + str(indice_char) )
                # plt.plot(items_posterior_array, label = str(nb_rep))
                # plt.legend()
                # plt.show()
                #
                # a.savefig('Posteriors_each_rep_CT_sess2_'+ str(indice_char) + '.png')
            print item_selected

        if not indice_flash % J:
            indice_char += 1
                # b = plt.figure('Posteriors at each decision')
                # plt.plot(items_posterior_array, label = str(indice_char))
                # plt.legend()
                # plt.show()
                # b.savefig('Posteriors_each_decision_sess2.png')
            item_prediction.append(item_selected)
            train_NaiveBayes.reset_bayes_prior()
                # print item_selected

    temp = np.matrix(item_selected_list)

    I = temp.shape[1] / J

    item_selected_mat = temp.reshape([I, J])


    # target_mat = a.reshape([I, J])
    accuracy_list = []

    k = 0
    for j in range(J):
        item_comparison = np.transpose(target_vec) == item_selected_mat[:, j]
        item_comparison = item_comparison.astype('float')

        if k == 0:
            A = item_comparison
            k = 1
        else:
            A = np.hstack([A, item_comparison])

    mean_acc_vector = np.mean(A, axis=0)
    std_acc_vector = np.std(A, axis=0)
    if visu:
        plt.figure()
        x = np.linspace(0,J,J)
        plt.plot(mean_acc_vector)
        plt.fill_between(x,mean_acc_vector-0.5*std_acc_vector , mean_acc_vector+0.5*std_acc_vector, alpha = 0.1)
        plt.xlabel('#Flash')
        plt.ylabel('Mean accuracy')
        plt.title('Accuracy evolution with flashs ')




    return mean_acc_vector, std_acc_vector



def learn_ts_fgda(X_train, y_train, T = 0, NT = 1):

    classes = [T,NT]
    train_fgda = FGDA()

    train_fgda.fit(X =X_train , y = y_train )


    X_train_fgda = train_fgda.transform(X = X_train)

    centroids_train_fgda = [mean_covariance(X_train_fgda[y_train == l, :, :], metric='riemann') for l in
                                  classes]


    return X_train_fgda, centroids_train_fgda, train_fgda

def apply_ts_fgda(X_test, train_fgda, centroids_train_fgda, T = 0, NT = 1):

    classes = [T,NT]

    ts_test = train_fgda._ts.transform(X = X_test)

    X_test_fgda = train_fgda.transform(X = X_test)


    dist = [distance(X_test_fgda, centroids_train_fgda[l])
            for l in classes]

    r_TNT_fgda = np.log(dist[0] / dist[1])
    r_TNT_fgda = np.array(r_TNT_fgda)

    return X_test_fgda, r_TNT_fgda, ts_test


def find_soulmate_iter(population_signals , stranger_signals ):
    dist_dict = {}

    stranger_covariance = covariances(X=stranger_signals)
    stranger_reference = mean_covariance(covmats=stranger_covariance, metric='riemann')

    for subject, subject_signal in population_signals.iteritems():
        subject_covariance = covariances(X=subject_signal)
        subject_reference = mean_covariance(covmats=subject_covariance, metric='riemann')

        dist_dict[subject] = distance(stranger_reference, subject_reference)

    sorted_dist_dict = sorted(dist_dict.items(), key=operator.itemgetter(1))

    return dict(sorted_dist_dict)

def reduce_population_signals(population_signals, len_timeline = 500):
    reduced_population_signals = {}
    for subject, subject_signal in population_signals.iteritems():

        Ns = subject_signal.shape[0]
        rand_indice_choice = random.sample(k=len_timeline, population=range(Ns))
        reduced_population_signals[subject] = subject_signal[rand_indice_choice,:,:]
    return reduced_population_signals

def dist_dict_loop_on_sig(population_signals , stranger_subject , nb_iter = 5,len_timeline = 500 ):
    mega_dist_dict = {}
    for k in range(nb_iter):
        temp_population_signals = reduce_population_signals(population_signals = population_signals , len_timeline = len_timeline)
        temp_sorted_dist_dict = find_soulmate_iter(population_signals = temp_population_signals , stranger_signals = temp_population_signals[stranger_subject] )
        for subject, dist in temp_sorted_dist_dict.iteritems():
            if k == 0:
                mega_dist_dict[subject] = [dist]
            else :
                mega_dist_dict[subject].append(dist)
    return mega_dist_dict

def select_soul_list(population_signals , stranger_subject , nb_exclusion = 0,len_timeline = 500):
    mega_dist_dict = {}
    soul_group = []
    temp_population_signals = reduce_population_signals(population_signals = population_signals , len_timeline = len_timeline)
    temp_sorted_dist_dict = find_soulmate_iter(population_signals = temp_population_signals , stranger_signals = temp_population_signals[stranger_subject] )
    for subject, dist in temp_sorted_dist_dict.iteritems():
        mega_dist_dict[subject] = dist
    sorted_mega_dist_dict = sorted(mega_dist_dict.items(), key=operator.itemgetter(1))
    L = len(sorted_mega_dist_dict) - int(nb_exclusion)
    for i,d in enumerate(sorted_mega_dist_dict[0:L+1]):
        soul_group.append(sorted_mega_dist_dict[i][0])
    return soul_group

def dist_dict_loop_on_sub(population_signals , nb_iter = 5,len_timeline = 500 ):
    list_mega_dist_dict= []
    for stranger_subject in population_signals.keys():
        list_mega_dist_dict.append(dist_dict_loop_on_sig(population_signals , stranger_subject , nb_iter = nb_iter,len_timeline = len_timeline ) )
    return list_mega_dist_dict






def test_loop_P300_dynamic(x_test, y_test, e_test, x_train, y_train, targets, T=0, NT=1, flashmode='RoCo', visu=False,
                       nb_targets=60):
    classes = [T,NT]

    x_train_nontar = x_train[y_train == NT, :, :]
    y_train_nontar = y_train[y_train == NT]

    x_train_tar = x_train[y_train == T, :,:]
    y_train_tar = y_train[y_train == T]

    Max_indice = x_test.shape[0]
    nb_repetitions = Max_indice / (12 * nb_targets)
    J = 12 * nb_repetitions

    items_list = targets
    items_vec = np.array(items_list)
    screen = np.reshape(items_vec, [6, 6])

    item_prediction = []

    indice_flash = 0
    indice_char = 0
    nb_rep = 0
    item_selected_list = []

    Nt, Ne, Ns = x_train.shape

    virtual_subject_P = np.mean(x_train[y_train == T, :, :], axis=0)
    virtual_real_subject_P = virtual_subject_P

    # real_subject_P = np.mean(x_pipe, axis = 0)

    t = 0



    # real_subject_P = virtual_subject_P

    covmats_train = np.zeros((Nt, 2*Ne, 2*Ne))

    for i in range(Nt):
        covmats_train[i, :, :] = np.cov(np.concatenate((virtual_subject_P , x_train[i, :, :]), axis=0))

    train_centroids =  [mean_covariance(covmats_train[y_train == l,:,:], metric='riemann') for l in classes]

    # For the first loop, take the aruthmetic mean
    # train_centroids = [np.mean(covmats_train[y_train == l, :, :] , axis = 0 ) for l in classes]


    train_r_TNT = [predict_R_TNT(X = x , centroids = train_centroids, classes=['T', 'NT'])for i, x in enumerate(covmats_train)]
    train_r_TNT_vec = np.array(train_r_TNT)
    train_r_TNT_nontar = train_r_TNT_vec[y_train == NT]
    train_r_TNT_tar = train_r_TNT_vec[y_train == T]

    train_distribution = Distribution_R_TNT()

    train_distribution.fit(X =covmats_train , y = y_train )

    train_NaiveBayes = R_TNT_NaiveBayes(targets= targets)

    train_NaiveBayes.fit(train_distribution)

    if visu == True:
        visualisation_R_TNT(r_TNT = train_r_TNT, y = y_train)


    while indice_flash < Max_indice:

            #         Get the r_TNT of the current trial
        x = x_test[indice_flash,:,:]

        # X = np.cov(np.concatenate([virtual_subject_P, real_subject_P , x], axis=0))
        X = np.cov(np.concatenate([virtual_real_subject_P, x], axis=0))
        r_TNT = predict_R_TNT(X = X , centroids = train_distribution.centroids, classes=['T', 'NT'])

        if r_TNT <= 0:
            if t == 0:
                x_dynamic = np.zeros((1 , Ne, Ns))
                x_dynamic[0,:,:] = x
                y_dynamic = np.array([T])
                t = 1
            else:
                xx = np.zeros((1, Ne, Ns))
                xx[0, :, :] = x
                x_dynamic = np.concatenate([x_dynamic, xx], axis = 0)
                y_dynamic = np.concatenate([y_dynamic, np.array([T])], axis=0)
        else:
            if t == 0:
                x_dynamic = np.zeros((1, Ne, Ns))
                x_dynamic[0, :, :] = x
                y_dynamic = np.array([NT])
                t = 1
            else:
                xx = np.zeros((1, Ne, Ns))
                xx[0, :, :] = x
                x_dynamic = np.concatenate([x_dynamic, xx], axis=0)
                y_dynamic = np.concatenate([y_dynamic, np.array([NT])], axis=0)



            #         Update the bayes_prior (vector of length 36)
        if flashmode == 'RoCo':
            flashed_item = rtp_indexColumn2Targets(Screen=screen, IndexColumn=e_test[indice_flash])
        if flashmode == 'Splotch':
            SplotchMatrixNumber = floor((indice_flash - indice_char * 12 * nb_repetitions) / 12) + 1
            flashed_item = rtp_indexSplotch2Targets(SplotchMatrixNumber=SplotchMatrixNumber,
                                                        Indexplotch=e_test[indice_flash], items_vec=items_vec)

        flashed_item_index = [i for i, e in enumerate(items_list) if e in list(flashed_item)]

        items_posterior_array = train_NaiveBayes.update_class_prior(r_TNT, flashed_item_index)

        item_selected = items_list[np.argmax(items_posterior_array)]

        item_selected_list.append(item_selected)

            #   Ask if it's flashed_items_total_number is enough to reset class_prior or to take a decision and change of target
        indice_flash += 1

        if not indice_flash % 12:
            nb_rep += 1



        if not indice_flash % J:
            indice_char += 1
            # real_subject_P = np.mean(x_dynamic[y_dynamic == T, :, :], axis=0)


            n_dyn_T = (np.array(np.where(y_dynamic == T ))).shape[1]
            n_dyn_NT = (np.array(np.where(y_dynamic == NT ))).shape[1]
            n_dyn = len(y_dynamic)


            indices_tar  = clean_r_TNT(train_r_TNT_tar, n_dyn_T)
            indices_nontar  = clean_r_TNT(train_r_TNT_nontar, n_dyn_NT)

            x_train_tar_bis = x_train_tar[indices_tar, :, :]
            y_train_tar_bis = y_train_tar[indices_tar]

            x_train_nontar_bis = x_train_nontar[indices_nontar, :, :]
            y_train_nontar_bis = y_train_nontar[indices_nontar]

            x_train_dynamic = np.concatenate([x_train_tar_bis,x_train_nontar_bis, x_dynamic], axis = 0)
            y_train_dynamic = np.concatenate([y_train_tar_bis,y_train_nontar_bis, y_dynamic], axis = 0)

            virtual_real_subject_P = np.mean(x_train_dynamic[y_train_dynamic == T, :, :], axis=0)

            Nt, Ne, Ns = x_train_dynamic.shape
            covmats_train_dynamic = np.zeros((Nt, 2 * Ne, 2 * Ne))


            for i in range(Nt):
                covmats_train_dynamic[i, :, :] = np.cov(
                    np.concatenate((virtual_real_subject_P, x_train_dynamic[i, :, :]), axis=0))

            mT = mean_riemann(covmats=covmats_train_dynamic[y_train_dynamic == T, :, :], tol=10e-9, maxiter=10,
                              init=None,
                              sample_weight=None)
            mNT = mean_riemann(covmats=covmats_train_dynamic[y_train_dynamic == NT, :, :], tol=10e-9, maxiter=10,
                               init=None,
                               sample_weight=None)

            train_centroids = [mT, mNT]

            train_distribution.fit(X=covmats_train_dynamic, y=y_train_dynamic , centroids = train_centroids)
            train_NaiveBayes.fit(train_distribution)

                # b = plt.figure('Posteriors at each decision')
                # plt.plot(items_posterior_array, label = str(indice_char))
                # plt.legend()
                # plt.show()
                # b.savefig('Posteriors_each_decision_sess2.png')
            item_prediction.append(item_selected)
            train_NaiveBayes.reset_bayes_prior()
                # print item_selected

    temp = np.matrix(item_selected_list)

    I = temp.shape[1] / J

    item_selected_mat = temp.reshape([I, J])

    w = find_target_word(e=e_test, y=y_test, nb_repetitions=nb_repetitions, items_vec=items_vec,
                             flashmode=flashmode, T=T, NT=NT)

    target_vec = np.matrix(w.tolist())
        # target_mat = a.reshape([I, J])
    # accuracy_list = []

    k = 0
    for j in range(J):
        item_comparison = np.transpose(target_vec) == item_selected_mat[:, j]
        item_comparison = item_comparison.astype('float')

        if k == 0:
            A = item_comparison
            k = 1
        else:
            A = np.hstack([A, item_comparison])

    mean_acc_vector = np.mean(A, axis=0)
    std_acc_vector = np.std(A, axis=0)

    return mean_acc_vector, std_acc_vector

def clean_r_TNT(the_list, first_n):
    # 1) Associer les indices aux valeurs:

    list_2 = [(i, v) for i, v in enumerate(the_list)]

    # 2) Trier selon valeur v:

    list_3 = sorted(list_2, key=lambda x: x[1])

    # 3) Prendre les N derniers (les plus grands)

    list_4 = list_3[first_n:]

    # 4) Prendre que les indices:

    list_indices = [i for i, v in list_4]

    return list_indices



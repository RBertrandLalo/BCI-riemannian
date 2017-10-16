import numpy as np
import matplotlib.pyplot as plt
from generate_template_utils import create_centroids_Dict, create_rTNT_Dict, create_ERP_Dict, create_covmats_Dict
from eegdatabase_P300 import Database as Database_P300
from eegdatabase_single_template import Database as Database_single




def visualize_distribution(distribution , title = 'Distribution des r_TNT '):
    rTNT_list = distribution[0] + distribution[1]
    label_list = [0]*len(distribution[0] ) + [1]*len(distribution[1] )
    return visualize_R_TNT(r_TNT = rTNT_list, y= label_list)



def visualize_R_TNT(r_TNT, y, T = 0, NT = 1 , xlim = [-0.06, 0.06], title = 'Distribution des rTNT'):
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
    plt.title(title)
    plt.show()

    return R_TNT_fig


def visualize_ERP(ERP, chnames = None):
    n = ERP.shape[0]
    plt.figure()
    for i in range(n):
        a = plt.subplot(n,1,i+1)
        plt.plot(ERP[i,:])
        if chnames is not None:
            a.set_title(chnames[i])


def visualize_list_ERP(ERP_list , ERP_names , chnames = None):
    n = ERP_list[0].shape[0]
    plt.figure()
    for i in range(n):
        a = plt.subplot(n,1,i+1)
        for k, ERP in enumerate(ERP_list):
            if n!=ERP.shape[0]:
              print('The 2 ERP must have the same size')
            f = plt.plot(ERP[i,:], label = ERP_names[k])
            if i == 0:
                plt.legend()
        if chnames is not None:
            a.set_title(chnames[i])


def visualize_mean_ERP(ERP, chindices = None, visu = True):
    if chindices is None:
        chindices = range(0,ERP.shape[0])
    n = len(chindices)
    if visu:
        plt.figure()
        plt.plot(np.mean(ERP[chindices,:], axis = 0))
    return np.mean(ERP[chindices,:], axis = 0)


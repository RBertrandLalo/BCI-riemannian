from pandas import DataFrame , concat
from rTNTutils import R_TNT_NaiveBayes, test_loop_P300
from pyriemann.utils.distance import distance
from pyriemann.estimation import ERPCovariances

from pyriemann.utils.mean import mean_covariance

from pyriemann.utils.distance import distance

from eegdatabase_PIV import Database
from eegdatabase_P300 import Database as Database_P300
from Visualization_Utils import visualize_R_TNT

import random

import numpy as np

from pyriemann.estimation import ERPCovariances

import matplotlib.pyplot as plt

from rTNTutils import predict_ERP_centroids , predict_R_TNT, generic_test_loop
from rTNT_class import R_TNT_NaiveBayes

from rTNT_class import R_TNT_NaiveBayes
from generate_template_utils import extract_ERP_from_Dict
from Get_Data_Utils import get_data_from_csv_P300, get_data_from_csv_EIV



def distribution_single_from_template(test_sub, test_sess, template_path, database ):

    x_test,y_test,e_test,t_test,_,_ = database.subjects[test_sub].get_data([test_sess])


    ERP_train = np.load(template_path + 'ERP_Array.npy')
    erp_train = ERPCovariances(estimator='cov')
    erp_train.P = ERP_train
    X_test = erp_train.transform(x_test)

    centroids_train = np.load(template_path + 'Centroids_List.npy')

    erp_train = ERPCovariances(estimator = 'cov')
    erp_train.P = ERP_train

    r_TNT_test = predict_R_TNT(X=X_test, centroids_list=centroids_train)

    return r_TNT_test, y_test

def main_EIV_single_from_template(myfilename,  template_path ):
    # x_test,y_test,e_test,t_test = make_epochs_EIV(myfilename, chnames, bandpass = (1.0,20.0), filtre_order = 2 ,delta=0.6, target=[2], nontarget=[1])

    centroids_train = np.load(template_path + 'Centroids_List.npy')
    Covmats_Dict = np.load(template_path + 'Covmats_Dict.npy')
    Covmats_Dict = Covmats_Dict.item()
    chnames = Covmats_Dict['channel names']

    ERP_train = np.load(template_path + 'ERP_Array.npy')
    erp_train = ERPCovariances(estimator='cov')
    erp_train.P = ERP_train
    # X_test = erp_train.transform(x_test)

    r_TNT_mu_List = np.load(template_path + 'rTNT_mu.npy')
    r_TNT_var_List = np.load(template_path + 'rTNT_var.npy')


    data, labels, event ,target = get_data_from_csv_EIV(myfilename =myfilename, chnames = chnames)

    mean, var = generic_test_loop(data, labels, event, ERP_train, centroids_train, r_TNT_mu_List, r_TNT_var_List, column_number=7, nb_repetitions=4,
                                  items_list=[1, 2, 3, 4, 5, 6, 7], visu=False, flashmode='EIV')


    return mean, var



def test_from_cross_template_P300(template_path , subject_path, test_chnames, flashmode = 'RoCo', nb_targets = 180, visu = False):
    T = 0
    NT = 1


    ERP = np.load(template_path +'ERP_Array.npy' )

    Centroids_List = np.load(template_path +'Centroids_List.npy' )

    mu_TNT = np.load(template_path +'rTNT_mu.npy' )
    sigma_TNT = np.load(template_path +'rTNT_var.npy' )

    data, labels, event = get_data_from_csv_EIV(myfilename = subject_path + '-signals.csv', markersfile=subject_path + 'markers.csv', chnames = test_chnames)


    erp = ERPCovariances()
    erp.P = ERP
    erp.estimator = 'cov'
    X = erp.transform(data)
    train_NaiveBayes = R_TNT_NaiveBayes(targets= ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
                'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '1', '2', '3', '4', '5', '6', '7', '8', '9', '_'], mu_TNT=mu_TNT, sigma_TNT=sigma_TNT, class_prior=None)


    dist = [np.array([distance(x, Centroids_List[l]) for i, x in enumerate(X)])
            for l in [T,NT]]

    r_TNT = np.array(np.log(dist[0] / dist[1]))


    mean, var = test_loop_P300(r_TNT_test = r_TNT , y_test = labels, e_test = event, train_NaiveBayes = train_NaiveBayes, T=0, NT=1, flashmode=flashmode, visu=visu,
                               nb_targets=nb_targets)

    return  mean, var

def test_from_single_template_P300(template_path , subject_path, subject_name, flashmode, nb_targets, test_chnames, visu = False):
    T = 0
    NT = 1

    ERP_Dict = np.load(template_path +'ERP_Dict.npy' )
    ERP_Dict = ERP_Dict.item()


    RTNT_Dict = np.load(template_path +'rTNT_Dict.npy' )
    RTNT_Dict = RTNT_Dict.item()


    Centroids_Dict = np.load(template_path +'Centroids_Dict.npy' )
    Centroids_Dict = Centroids_Dict.item()

    ERP = extract_ERP_from_Dict(ERP_Dict, subject_name)

    RTNT_list = RTNT_Dict[subject_name]
    mu_TNT = [np.mean(np.array(RTNT_list[m])) for m in [T,NT]]
    sigma_TNT = [np.std(np.array(RTNT_list[m])) for m in [T,NT]]

    Centroids_list = Centroids_Dict[subject_name]

    data, labels, event = get_data_from_csv_P300(signalfile=subject_path + '-signals.csv', markersfile=subject_path + '-markers.csv',chnames = test_chnames )

    mean, var = generic_test_loop(data, labels, event, ERP, Centroids_list, mu_TNT, sigma_TNT,nb_targets=nb_targets,visu = False, flashmode=flashmode)

    return mean, var

def test_loop(r_TNT_test, y_test, e_test, t_test, train_NaiveBayes,items_list, nb_repetitions, T=0, NT=1, visu=False):


    item_prediction = []

    Max_indice = (r_TNT_test).shape[0]

    J = len(items_list) * nb_repetitions

    indice_flash = 0
    indice_char = 0
    nb_rep = 0
    item_selected_list = []

    if visu:
            # Visualisation of the distribution of ERP to be tested 
        visualize_R_TNT(r_TNT = r_TNT_test, y = y_test, T=T, NT=NT, xlim=[-0.06, 0.06], title = 'r_TNT TEST')

        # Start of the inference 

    while indice_flash < Max_indice:

            #  Get the r_TNT of the current trial

        r_TNT = r_TNT_test[indice_flash]
            #  Update the bayes_prior (vector of length 36)

        flashed_item = e_test[indice_flash]

        flashed_item_index = items_list.index(flashed_item)

        items_posterior_array = train_NaiveBayes.update_class_prior(r_TNT, flashed_item_index)

        item_selected = items_list[np.argmax(items_posterior_array)]

        item_selected_list.append(item_selected)

            #   Ask if it's flashed_items_total_number is enough to reset class_prior or to take a decision and change of target
        indice_flash += 1

        if not indice_flash % len(items_list):
            nb_rep += 1

        if not indice_flash % J:
            indice_char += 1
            item_prediction.append(item_selected)
            train_NaiveBayes.reset_bayes_prior()

    temp = np.matrix(item_selected_list)

    I = temp.shape[1] / J

    item_selected_mat = temp.reshape([I, J])

    w = t_test
    a = np.matrix(w.tolist())
    a = a[0,:]
    target_mat = a.reshape([I, J])

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
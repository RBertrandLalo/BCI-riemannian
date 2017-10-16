# from rTNTutils import make_database, get_part_of_data_equal, get_data, predict_ERP_centroids, apply_ERP, visualisation_ERP, visualisation_R_TNT, predict_R_TNT, Distribution_R_TNT, R_TNT_NaiveBayes,test_loop_PIV, test_loop_P300, find_target_word, learn_ts_fgda,apply_ts_fgda

from pyriemann.estimation import ERPCovariances
from pyriemann.utils.mean import mean_covariance
from pyriemann.utils.distance import distance
from eegdatabase_PIV import Database as Database_PIV
# from eegdatabase_OpStop import Database as Database_OpStop
from eegdatabase_P300 import Database as Database_P300
import numpy as np
# from classification_bar_faf import Bayes_R_TNT , predict_r_TNT
import matplotlib.pyplot as plt
from pyriemann.tangentspace import TangentSpace , FGDA
import scipy.io as sio

T = 0
NT = 1

ERP_bloc = [T]
P300_items_vector = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
                'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '1', '2', '3', '4', '5', '6', '7', '8', '9', '_']





def create_Cross_ERP(ERP_Dict):
    Nc = len(ERP_Dict['channel names'])
    Nt = int(600 * ERP_Dict['Sampling Rate'] / 1000.0)

    ERP_sum = np.zeros([Nc,Nt])
    number_trials = 0
    for subject in list(set(ERP_Dict.keys()) - set(['Sampling Rate', 'channel names'])):
        ERP_sum += ERP_Dict[subject]['ERP Sum']
        number_trials += ERP_Dict[subject]['ERP number trials']
    cross_ERP = ERP_sum/float(number_trials)
    return cross_ERP

def create_ERP_Dict(database_P300_ERP):
    ERP_Dict = {}
    for subject in database_P300_ERP.get_subject_names():
        d_subject = {}
        for session in database_P300_ERP.subjects[subject].get_session_names():
            ERP_Dict['Sampling Rate'] = database_P300_ERP.subjects[subject].sessions[session].SamplingRate
            d_session = {'ERP Sum' : database_P300_ERP.subjects[subject].sessions[session].T_sum ,  'ERP number trials' : database_P300_ERP.subjects[subject].sessions[session].T_trials}

        ERP_Dict[subject] = d_session

    ERP_Dict['channel names'] = database_P300_ERP.chnames
    return ERP_Dict

def create_covmats_Dict(database_P300_covmats):
    covmats_Dict = {}
    for subject in database_P300_covmats.get_subject_names():
        d_subject = {}
        for session in database_P300_covmats.subjects[subject].get_session_names():
            covmats_Dict['Sampling Rate'] = database_P300_covmats.subjects[subject].sessions[session].SamplingRate
            d_session = {'covmats': database_P300_covmats.subjects[subject].sessions[session].covmats, 'y': database_P300_covmats.subjects[subject].sessions[session].labels}
            d_subject[session] = d_session
        covmats_Dict[subject] = d_subject
    # covmats_Dict['ERP'] = database_P300_covmats.ERP
    covmats_Dict['channel names'] = database_P300_covmats.chnames
    return covmats_Dict

def create_centroids_Dict(covmats_Dict, save_path):
    T = 0
    NT = 1
    classes = [T,NT]
    centroids_Dict = {}
    sample_weight_Dict = {}
    for subject in list(set(covmats_Dict.keys()) - set(['Sampling Rate', 'channel names'])):
        print 'Make centroids of ' + subject
        covmats_list = []
        labels_list = []
        for session, session_dict in covmats_Dict[subject].iteritems():
            covmats_list.append(session_dict['covmats'])
            labels_list.append(session_dict['y'])
        X = np.array(covmats_list)
        XX = np.concatenate([X[k] for k in range(X.shape[0])])
        y = np.array(labels_list)
        yy = np.concatenate([y[k] for k in range(y.shape[0])])
        centroids_Dict[subject] = [mean_covariance(XX[yy == l, :, :], metric='riemann') for l in classes]
        sample_weight_Dict[subject] = [len(np.where(np.array(y)==l)[0]) for l in classes]
        np.save(save_path + 'partial_Centroids_Dict', centroids_Dict)
        np.save(save_path + 'partial_Weight_Dict', sample_weight_Dict)
    return centroids_Dict, sample_weight_Dict


def create_centroids_List(centroids_Dict,sample_weight_Dict ):
    target_centroids_list = []
    target_weight_list = []
    non_target_centroids_list = []
    non_target_weight_list = []
    for subject, centroids in centroids_Dict.iteritems():
        subject_weight_list = sample_weight_Dict[subject]

        target_centroids_list.append(centroids[0])
        target_weight_list.append(subject_weight_list[0])

        non_target_centroids_list.append(centroids[1])
        non_target_weight_list.append(subject_weight_list[1])

    CT = np.array(target_centroids_list)
    CNT = np.array(non_target_centroids_list)
    return [mean_covariance(CT, metric='riemann', sample_weight=np.array(target_weight_list)), mean_covariance(CNT, metric='riemann', sample_weight=np.array(non_target_weight_list))]

def create_rTNT_Dict(database_P300_rTNT):
    rTNT_Dict = {}
    T = 0
    NT = 1
    popu_T_rTNT_list = []
    popu_NT_rTNT_list = []
    for subject in database_P300_rTNT.get_subject_names():
        T_rTNT_list = []
        NT_rTNT_list = []
        for session in database_P300_rTNT.subjects[subject].get_session_names():
            rTNT_Dict['Sampling Rate'] = database_P300_rTNT.subjects[subject].sessions[session].SamplingRate

            r = database_P300_rTNT.subjects[subject].sessions[session].rTNT
            y = database_P300_rTNT.subjects[subject].sessions[session].labels
            T_rTNT_list += list(r[y == T])
            NT_rTNT_list += list(r[y == NT])
            popu_T_rTNT_list += list(r[y == T])
            popu_NT_rTNT_list += list(r[y == NT])
        rTNT_Dict[subject] = [T_rTNT_list , NT_rTNT_list]

    rTNT_Dict['rTNT mean'] = [np.mean(np.array(popu_T_rTNT_list)) , np.mean(np.array(popu_NT_rTNT_list)) ]
    rTNT_Dict['rTNT std'] = [np.std(np.array(popu_T_rTNT_list)) , np.std(np.array(popu_NT_rTNT_list)) ]
    rTNT_Dict['rTNT Target'] = np.array(popu_T_rTNT_list)
    rTNT_Dict['rTNT Non Target'] = np.array(popu_NT_rTNT_list)
    rTNT_Dict['channel names'] = database_P300_rTNT.chnames
    rTNT_Dict['Centroids'] = database_P300_rTNT.centroids
    rTNT_Dict['ERP'] = database_P300_rTNT.ERP

    return rTNT_Dict

def create_rTNT_List(rTNT_Dict):
    return [rTNT_Dict['rTNT Target'], rTNT_Dict['rTNT Non Target']]

def extract_ERP_from_Dict(ERP_Dict, subject_key):
    return ERP_Dict[subject_key]['ERP Sum'] / ERP_Dict[subject_key]['ERP number trials']
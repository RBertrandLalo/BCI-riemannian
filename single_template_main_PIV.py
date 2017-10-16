import numpy as np
import matplotlib.pyplot as plt
from generate_template_utils import create_centroids_Dict, create_rTNT_Dict, create_ERP_Dict, create_covmats_Dict, create_Cross_ERP, create_centroids_List, create_rTNT_List, create_centroids_Dict, create_rTNT_Dict, create_ERP_Dict, create_covmats_Dict, create_Cross_ERP, create_centroids_List, create_rTNT_List
from eegdatabase_P300 import Database as Database_P300
from eegdatabase_single_template import Database as Database_single
from eegdatabase_cross_template import Database as Database_cross
from rTNTutils import predict_R_TNT, generic_test_loop, extract_ERP_from_Dict,visualisation_R_TNT
from eegdatabase_PIV import Database as Database_PIV
from pyriemann.estimation import ERPCovariances
from pyriemann.utils.mean import mean_covariance, mean_riemann
from pyriemann.utils.distance import distance
from eegdatabase_PIV import Database
import numpy as np
import matplotlib.pyplot as plt
from pyriemann.tangentspace import TangentSpace , FGDA
from pyriemann.utils.covariance import covariances, covariances_EP, cospectrum
import operator
from scipy.spatial.distance import euclidean
from read_csv_data import make_epochs_EIV, make_ERP_EIV, make_rTNT_EIV, make_covmats_EIV
from Test_Template_utils import main_EIV_single_from_template

save_path = '/root/PycharmProjects/Create_Template/Template_PIV/Single/'

chnames_manu_err = ['Fz', 'Cz', 'CP5', 'CP1', 'CP2', 'CP6', 'P7', 'P3', 'Pz', 'P4', 'P8', 'PO9', 'O1', 'Oz', 'O2', 'PO10']
subject_list = ['ELSHE', 'BOURO', 'BARQU','COLSY','GAVMA']

# T_tot = np.zeros([16, 600])
# NT_tot = np.zeros([16, 600])
# T_trials_tot = 0
# NT_trials_tot = 0
# subject_list = ['ELSHE', 'BOURO', 'BARQU','COLSY','GAVMA']
#
#
# # ERP
# ERP_Dict = {}
# for subject in subject_list:
#     print('load ' + subject + ' ERP')
#     T_sum, NT_sum, T_trials, NT_trials = make_ERP_EIV(myfilename = '/root/PycharmProjects/EIV/DATA/' + subject + '/Calib1.csv', chnames = chnames_manu_err)
#     ERP_Dict[subject] = T_sum/T_trials
#     np.save(save_path + 'ERP_Dict', ERP_Dict)
#
#
#
# # T_ERP = np.load(save_path + 'ERP_Array.npy')
#
#
# #  COVMATS
# covmats_Dict = {}
# covmats_Dict['Sampling Rate'] = 1000.0
# covmats_Dict['channel names'] = chnames_manu_err
#
# for subject in subject_list:
#     print('load ' + subject + ' covmats')
#     covmats_Dict[subject] = {}
#     covmats_Dict[subject]['Calib1'] = {}
#     covmats, labels, event, targets = make_covmats_EIV(myfilename = '/root/PycharmProjects/EIV/DATA/' + subject + '/Calib1.csv', chnames = chnames_manu_err, ERP =  ERP_Dict[subject])
#     covmats_Dict[subject]['Calib1'] = {'covmats' : covmats, 'y':labels }
#     np.save(save_path + 'Covmats_Dict', covmats_Dict)
#
# centroids_Dict, sample_weight_Dict = create_centroids_Dict(covmats_Dict, save_path)
# np.save(save_path + 'Centroids_Dict', centroids_Dict)
#
#
# # rTNT
#
# rTNT_Dict = {}
# for subject in subject_list:
#     print('load ' + subject + ' rTNT')
#     rTNT, labels, event, targets = make_rTNT_EIV(myfilename = '/root/PycharmProjects/EIV/DATA/' + subject + '/Calib1.csv', chnames = chnames_manu_err, ERP =  ERP_Dict[subject], centroids_List=centroids_Dict[subject])
#     rTNT_Dict[subject] = {'rTNT' : rTNT, 'y':labels }
#     np.save(save_path + 'rTNT_Dict', rTNT_Dict)
# #
rTNT_Dict = np.load(save_path + 'rTNT_Dict.npy')
rTNT_Dict = rTNT_Dict.item()
#
# for subject in subject_list:
#     visualisation_R_TNT(r_TNT=rTNT_Dict[subject]['rTNT'], y=rTNT_Dict[subject]['y'])
#
#
#
# # Bayes Parameters
# rTNT_mu_Dict = {}
rTNT_var_Dict = {}
for subject in subject_list:
    rTNT_List = rTNT_Dict[subject]['rTNT']
    rTNT_Label = rTNT_Dict[subject]['y']
#
#     rTNT_mu_Dict[subject] = [np.mean(rTNT_List[rTNT_Label == 0]),np.mean(rTNT_List[rTNT_Label == 1]) ]
#     np.save(save_path + 'rTNT_mu_Dict', rTNT_mu_Dict)
#
    rTNT_var_Dict[subject] = [np.std(rTNT_List[rTNT_Label == 0]),np.std(rTNT_List[rTNT_Label == 1]) ]
    np.save(save_path + 'rTNT_var_Dict', rTNT_var_Dict)



ERP_Dict = np.load(save_path + 'ERP_Dict.npy')
ERP_Dict = ERP_Dict.item()

Centroids_Dict= np.load(save_path + 'Centroids_Dict.npy')
Centroids_Dict = Centroids_Dict.item()

rTNT_mu_Dict = np.load(save_path + 'rTNT_mu_Dict.npy')
rTNT_mu_Dict = rTNT_mu_Dict.item()

rTNT_var_Dict = np.load(save_path + 'rTNT_var_Dict.npy')
rTNT_var_Dict = rTNT_var_Dict.item()

for subject in subject_list:
    path = save_path + subject + '/'
    np.save(path + 'ERP_Array', ERP_Dict[subject])
    np.save(path + 'Centroids_List', Centroids_Dict[subject] )
    np.save(path + 'rTNT_mu' , rTNT_mu_Dict[subject])
    np.save(path + 'rTNT_var' , rTNT_var_Dict[subject])

a = 1









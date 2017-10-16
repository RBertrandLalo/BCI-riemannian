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

save_path = '/root/PycharmProjects/Create_Template/Template_PIV/Cross/'

chnames_manu_err = ['Fz', 'Cz', 'CP5', 'CP1', 'CP2', 'CP6', 'P7', 'P3', 'Pz', 'P4', 'P8', 'PO9', 'O1', 'Oz', 'O2', 'PO10']

T_tot = np.zeros([16, 600])
NT_tot = np.zeros([16, 600])
T_trials_tot = 0
NT_trials_tot = 0
subject_list = ['ELSHE', 'BOURO', 'BARQU','COLSY','GAVMA']


# ERP
for subject in subject_list:
    print('load ' + subject + ' ERP')
    T_sum, NT_sum, T_trials, NT_trials = make_ERP_EIV(myfilename = '/root/PycharmProjects/EIV/DATA/' + subject + '/Calib1.csv', chnames = chnames_manu_err)
    T_tot+=T_sum
    NT_tot += NT_sum
    T_trials_tot += T_trials
    NT_trials_tot += NT_trials
    Cross_ERP_T = T_tot/T_trials_tot
    Cross_ERP_NT = T_tot / NT_trials_tot
    np.save(save_path + 'ERP_Array' , Cross_ERP_T)
    np.save(save_path + 'Cross_ERP_NT', Cross_ERP_NT)


# T_ERP = np.load(save_path + 'ERP_Array.npy')

#  COVMATS
covmats_Dict = {}
covmats_Dict['Sampling Rate'] = 1000.0
covmats_Dict['channel names'] = chnames_manu_err

for subject in subject_list:
    print('load ' + subject + ' covmats')
    covmats_Dict[subject] = {}
    covmats_Dict[subject]['Calib1'] = {}
    covmats, labels, event, targets = make_covmats_EIV(myfilename = '/root/PycharmProjects/EIV/DATA/' + subject + '/Calib1.csv', chnames = chnames_manu_err, ERP =  Cross_ERP_T)
    covmats_Dict[subject]['Calib1'] = {'covmats' : covmats, 'y':labels }
    np.save(save_path + 'Covmats_Dict', covmats_Dict)

centroids_Dict, sample_weight_Dict = create_centroids_Dict(covmats_Dict, save_path)
centroids_List = create_centroids_List(centroids_Dict,sample_weight_Dict )
np.save(save_path + 'Centroids_List', centroids_List)

centroids_List = np.load(save_path + 'Centroids_List.npy')

# rTNT

rTNT_Dict = {}
for subject in subject_list:
    print('load ' + subject + ' rTNT')
    rTNT_Dict[subject] = {}
    rTNT_Dict[subject]['Calib1'] = {}
    rTNT, labels, event, targets = make_rTNT_EIV(myfilename = '/root/PycharmProjects/EIV/DATA/' + subject + '/Calib1.csv', chnames = chnames_manu_err, ERP =  Cross_ERP_T, centroids_List=centroids_List)
    rTNT_Dict[subject]['Calib1'] = {'rTNT' : rTNT, 'y':labels }
    np.save(save_path + 'rTNT_Dict', rTNT_Dict)

rTNT_Dict = np.load(save_path + 'rTNT_Dict.npy')
rTNT_Dict = rTNT_Dict.item()
for subject in ['ELSHE', 'BOURO', 'BARQU','COLSY','GAVMA']:
    visualisation_R_TNT(r_TNT=rTNT_Dict[subject]['Calib1']['rTNT'], y=rTNT_Dict[subject]['Calib1']['y'])

rTNT_Target1 = []
rTNT_NonTarget1 = []
for subject in ['ELSHE', 'BOURO', 'BARQU','COLSY','GAVMA']:
    rTNT = rTNT_Dict[subject]['Calib1']['rTNT']
    label = rTNT_Dict[subject]['Calib1']['y']
    rTNT_Target1 += list(rTNT[label == 0])
    rTNT_NonTarget1 += list(rTNT[label == 1])


# Bayes Parameters
rTNT_List = [rTNT_Target1,rTNT_NonTarget1]
np.save(save_path + 'rTNT_List', rTNT_List )

rTNT_mu = [np.mean(rTNT_List[0]),np.mean(rTNT_List[1]) ]
np.save(save_path + 'rTNT_mu', rTNT_mu)

rTNT_var = [np.std(rTNT_List[0]),np.std(rTNT_List[1]) ]
np.save(save_path + 'rTNT_var', rTNT_var)





# mean, var, _,_ = main_EIV_single_from_template(myfilename = '/root/PycharmProjects/EIV/DATA/' + subject + '/Test1.csv',chnames = chnames_manu_err,  template_path = save_path )
#
# data, labels, event, targets = make_epochs_EIV(myfilename ='/root/PycharmProjects/EIV/DATA/BOURO/Test1.csv' , chnames= chnames_manu_err, bandpass = (1.0,20.0), filtre_order = 2 ,delta=0.6, target=[2], nontarget=[1])
#
# erp_cross = ERPCovariances(estimator='cov')
# erp_cross.P = T_ERP
#
# romain_covmats = erp_cross.transform(data)
#
# Romain_RTNT_cross = predict_R_TNT(X = romain_covmats , centroids_list = centroids_List)
# visualisation_R_TNT(r_TNT = Romain_RTNT_cross, y=labels)
#
#
# main_EIV_single_from_template
#
# visualize_2_ERP(T_ERP,NT_ERP,chnames = chnames_manu_err)
#
#
# database_EIV = Database_PIV(path = '/root/PycharmProjects/EIV/DATA/Test/' , delta = 0.6, target = [2], nontarget = [1],
# chnames = chnames_manu_err, bandpass = (1.0, 20.0), filtre_order = 2)
# database_EIV.fill_database(verbose = True)
#
# xp6,yp6,ep6,tp6,_ = database_EIV.subjects['P06'].get_data(['S01'])
# xp4,yp4,ep4,tp4,_ = database_EIV.subjects['P04'].get_data(['S01'])
#
# temp_path = '/root/PycharmProjects/Create_Template/Template1a/Cross/'
# cross_ERP = np.load(temp_path + 'ERP_Array.npy')
#
# erp_cross = ERPCovariances(estimator='cov')
# erp_cross.P = cross_ERP
#
# romain_covmats = erp_cross.transform(xp6)
# quentin_covmats = erp_cross.transform(xp4)
#
# romain_centroids_List = [mean_riemann(romain_covmats[yp6 ==0,:,:]) ,mean_riemann(romain_covmats[yp6 ==1,:,:])]
#
# quentin_centroids_list = [mean_riemann(romain_covmats[yp4 ==0,:,:]) ,mean_riemann(romain_covmats[yp4 ==1,:,:])]
#
# cross_crentroids_list = np.load(temp_path + 'Centroids_List.npy')
#
#
# Romain_RTNT_quentin = predict_R_TNT(X = romain_covmats , centroids_list = quentin_centroids_list)
#
# visualisation_R_TNT(r_TNT = Romain_RTNT_quentin, y=yp6)

a = 1









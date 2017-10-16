import numpy as np
import matplotlib.pyplot as plt
from generate_template_utils import create_centroids_Dict, create_rTNT_Dict, create_ERP_Dict, create_covmats_Dict, create_Cross_ERP, create_centroids_List, create_rTNT_List
from eegdatabase_P300 import Database as Database_P300
from eegdatabase_single_template import Database as Database_single
from eegdatabase_cross_template import Database as Database_cross
from rTNTutils import predict_R_TNT, get_data_from_csv, generic_test_loop, extract_ERP_from_Dict
from Test_Template_utils import test_from_cross_template

P300_Err_path ='/root/PycharmProjects/Err/DATA/'

P300_path ='/root/PycharmProjects/P300/DATA/test/'

P300_Splotch_path ='/root/PycharmProjects/Splotch/DATA/test/'

T = 0
NT = 1

chnames_manu_err = ['Fz', 'Cz', 'CP5', 'CP1', 'CP2', 'CP6', 'P7', 'P3', 'Pz', 'P4', 'P8', 'PO9', 'O1', 'Oz', 'O2', 'PO10']
chnames_manu_splotch = ['Fz', 'Cz', 'CP5', 'CP1', 'CP2', 'CP6', 'P7', 'P3', 'Pz', 'P4', 'P8', 'PO7', 'O1', 'POz', 'O2', 'P08']

chnames_barachant_splotch = ['Fp1', 'Fp2', 'F7', 'Fz', 'F8', 'T7', 'Cz', 'T8', 'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'POz','O2']
chnames_barachant_err = ['Fp1', 'Fp2', 'F7', 'Fz', 'F8', 'T7', 'Cz', 'T8', 'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'Oz','O2']

chnames_P300 = ['Fz', 'C3', 'Cz', 'C4', 'CP5', 'CP1', 'CP2', 'CP6', 'P7', 'P3', 'Pz', 'P4', 'P8',  'O1', 'O2']



chosen_path = P300_Err_path
chosen_chnames = chnames_manu_err
temp_path = '/root/PycharmProjects/Create_Template/Template1a/Cross/'

# ERP

database_ERP_cross = Database_cross(path=chosen_path, delta=0.6, target=[1], nontarget=[2],
              chnames=chosen_chnames, bandpass=(1.0, 20.0),
              filtre_order=2)

database_ERP_cross.create_ERP(verbose = True)

ERP_Dict = create_ERP_Dict(database_ERP_cross)
np.save(temp_path + 'ERP_Dict', ERP_Dict)

cross_ERP = create_Cross_ERP(ERP_Dict)
np.save(temp_path + 'ERP_Array', cross_ERP)

# COVMATS

database_cross_covmats = Database_cross(path=chosen_path, delta=0.6, target=[1], nontarget=[2],
                                      chnames=chosen_chnames, bandpass=(1.0,20.0),
                                      filtre_order=2)
database_cross_covmats.create_covmats(ERP = cross_ERP, verbose= True , savepath = temp_path)

covmats_Dict = create_covmats_Dict(database_cross_covmats)

np.save(temp_path + 'Covmats_Dict', covmats_Dict)

ERP_Dict = np.load(temp_path +'ERP_Dict.npy')
ERP_Dict = ERP_Dict.item()
covmats_Dict = np.load(temp_path +'Covmats_Dict.npy')
covmats_Dict = np.load(temp_path +'partial_Covmats_Dict.npy')
covmats_Dict = covmats_Dict.item()

# CENTROIDS & WEIGHT

centroids_Dict, weight_Dict = create_centroids_Dict(covmats_Dict,  savepath = temp_path)
np.save(temp_path + 'Centroids_Dict', centroids_Dict)
np.save(temp_path + 'Weight_Dict', weight_Dict)

centroids_Dict = np.load(temp_path +'Centroids_Dict.npy')
centroids_Dict = centroids_Dict.item()
weight_Dict = np.load(temp_path +'Weight_Dict.npy')
weight_Dict = weight_Dict.item()

centroids_List = create_centroids_List(centroids_Dict = centroids_Dict, sample_weight_Dict = weight_Dict)
np.save(temp_path + 'Centroids_List', centroids_List)
centroids_List = np.load(temp_path + 'Centroids_List.npy')

# rTNT
cross_ERP = np.load(temp_path + 'ERP_Array.npy')
database_P300_rTNT = Database_cross(path=chosen_path, delta=0.6, target=[1], nontarget=[2],
                                  chnames=chosen_chnames, bandpass=(1.0,20.0),
                                    filtre_order=2)

database_P300_rTNT.create_rTNT(ERP = cross_ERP , centroids_List = centroids_List, verbose= True , savepath = temp_path)

rTNT_Dict = create_rTNT_Dict(database_P300_rTNT)
np.save(temp_path + 'rTNT_Dict', rTNT_Dict)


# Bayes Parameters

rTNT_mu = [np.mean(rTNT_Dict['rTNT Target']), np.mean(rTNT_Dict['rTNT Non Target'])]
rTNT_var = [np.std(rTNT_Dict['rTNT Target']), np.std(rTNT_Dict['rTNT Non Target'])]
np.save(temp_path + 'rTNT_mu', rTNT_mu)
np.save(temp_path + 'rTNT_var', rTNT_var)



a = 1
























# #
# # covmats_File= np.load('/root/PycharmProjects/Create_Template/Template4b/Covmats_Splotch_Manu_T_Dict.npy')
# # covmats_Dict = covmats_File.item()
#
#
#
# # centroids_File= np.load('/root/PycharmProjects/Create_Template/Template4b/Centroids_Splotch_Manu_T_Dict.npy')
# # centroids_Dict = centroids_File.item()
#
# centroids_List = centroids_File= np.load('/root/PycharmProjects/Create_Template/Template4b/Centroids_Cross_Splotch_Manu_T_List.npy')
#
# ERP_tar = np.load('/root/PycharmProjects/Create_Template/Template4b/ERP_Splotch_Manu_T_Array.npy')
# covmats_list = []
# labels_list = []
# r_TNT_Dict = {}
#
# for subject in list(set(covmats_Dict.keys()) - set(['Sampling Rate', 'channel names'])):
#     for session ,session_dict in covmats_Dict[subject].iteritems():
#         covmats_list.append(session_dict['covmats'])
#         labels_list.append(session_dict['y'])
#     X = np.array(covmats_list)
#     XX = np.concatenate([X[k] for k in range(X.shape[0])])
#     y = np.array(labels_list)
#     yy = np.concatenate([y[k] for k in range(y.shape[0])])
#     r_TNT_Dict[subject] = {'rTNT': predict_R_TNT(X = XX, centroids = centroids_List), 'label' : yy }
#     np.save(temp_path + 'rTNT_Cross_Splotch_Manu_T_Dict' ,r_TNT_Dict)
#
#
#
#
#
# centroids_Dict = create_centroids_Dict(covmats_Dict)
#
# if save_centroids:
#     np.save(temp_path + 'Centroids_Splotch_Manu_T_List', centroids_list)
#
#
# database_P300_rTNT = Database_P300(path=chosen_path, delta=0.6, target=[1], nontarget=[2],
#                               chnames=chosen_chnames, bandpass=(1.0, 48.0),
#                               filtre_order=2)
#
# database_P300_rTNT.create_rTNT(ERP = ERP_tar, centroids_list= centroids_list)
#
# rTNT_Dict = create_rTNT_Dict(database_P300_rTNT)
#
# if save_rTNT:
#     np.save(temp_path + 'rTNT_Splotch_Manu_T_Dict', rTNT_Dict)
#
#
#
# visualize_ERP(ERP_Target = ERP_Target, ERP_Non_Target =ERP_Non_Target,  chnames  = Err_Target_Dict['channel names'])
# # visualize_ERP(ERP_Target =Err_Target_Dict['P07']['ERP Sum']/float(Err_Target_Dict['P07']['ERP number trials']), ERP_Non_Target =Err_Non_Target_Dict['P07']['ERP Sum']/float(Err_Non_Target_Dict['P07']['ERP number trials']),  chnames  = Err_Target_Dict['channel names'])
# a = 1
#



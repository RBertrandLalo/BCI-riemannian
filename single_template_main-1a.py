import numpy as np
import matplotlib.pyplot as plt
from generate_template_utils import create_centroids_Dict, create_rTNT_Dict, create_ERP_Dict, create_covmats_Dict
from eegdatabase_P300 import Database as Database_P300
from eegdatabase_single_template import Database as Database_single
from rTNTutils import predict_R_TNT, get_data_from_csv, generic_test_loop, extract_ERP_from_Dict
from Test_Template_utils import test_from_single_template

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
temp_path = '/root/PycharmProjects/Create_Template/Template1a/Single/'


# ERP

database_ERP_single = Database_single(path=chosen_path, delta=0.6, target=[1], nontarget=[2],
              chnames=chosen_chnames, bandpass=(1.0, 20.0),
              filtre_order=2)

database_ERP_single.create_ERP(verbose = True)

ERP_Dict = create_ERP_Dict(database_ERP_single)
np.save(temp_path + 'ERP_Dict', ERP_Dict)
# ERP_Dict = np.load(temp_path +'ERP_Dict.npy')
# ERP_Dict = ERP_Dict.item()


# COVMATS

database_P300_covmats = Database_single(path=chosen_path, delta=0.6, target=[1], nontarget=[2],
                                      chnames=chosen_chnames, bandpass=(1.0,20.0),
                                      filtre_order=2)
database_P300_covmats.create_covmats(ERP_Dict = ERP_Dict, save_path = temp_path,  verbose= True)

covmats_Dict = create_covmats_Dict(database_P300_covmats)
np.save(temp_path + 'Covmats_Dict', covmats_Dict)

ERP_Dict = np.load(temp_path +'ERP_Single_Dict.npy')
ERP_Dict = ERP_Dict.item()
covmats_Dict = np.load(temp_path +'Covmats_Dict.npy' )
covmats_Dict = covmats_Dict.item()
# centroids_Dict = np.load(temp_path +'Centroids_Dict.npy' )
# centroids_Dict = centroids_Dict.item()

covmats_Dict_bis = {key:covmats_Dict[key] for key in list(set(covmats_Dict.keys()) - set(['P06']))}

# CENTROIDS

centroids_Dict = create_centroids_Dict(covmats_Dict_bis, save_path = temp_path)



# R_TNT
database_P300_rTNT = Database_single(path=chosen_path, delta=0.6, target=[1], nontarget=[2],
                                  chnames=chosen_chnames, bandpass=(1.0,20.0),
                                    filtre_order=2)

database_P300_rTNT.create_rTNT(ERP_Dict= ERP_Dict , centroids_Dict = centroids_Dict, save_path = temp_path, verbose= True )

rTNT_Dict = create_rTNT_Dict(database_P300_rTNT)
np.save(temp_path + 'rTNT_Dict', rTNT_Dict)
























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



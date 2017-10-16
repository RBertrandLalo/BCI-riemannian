from Test_Template_utils import main_EIV_single_from_template, distribution_single_from_template
import matplotlib.pyplot as plt
import numpy  as np

chnames_manu_err = ['Fz', 'Cz', 'CP5', 'CP1', 'CP2', 'CP6', 'P7', 'P3', 'Pz', 'P4', 'P8', 'PO9', 'O1', 'Oz', 'O2', 'PO10']
chnames_manu_splotch = ['Fz', 'Cz', 'CP5', 'CP1', 'CP2', 'CP6', 'P7', 'P3', 'Pz', 'P4', 'P8', 'PO7', 'O1', 'POz', 'O2', 'P08']

chnames_barachant_splotch = ['Fp1', 'Fp2', 'F7', 'Fz', 'F8', 'T7', 'Cz', 'T8', 'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'POz','O2']
chnames_barachant_err = ['Fp1', 'Fp2', 'F7', 'Fz', 'F8', 'T7', 'Cz', 'T8', 'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'Oz','O2']

subject_with_test_session = ['BOURO', 'COLSY','ELSHE']

subject_without_test_session = ['BARQU','GAVMA']

template_path_1a = '/root/PycharmProjects/Create_Template/Template1a/Cross/'
template_path_PIV = '/root/PycharmProjects/Create_Template/Template_PIV/'

k = 0

for subject in subject_with_test_session:
    mean0, var0 = main_EIV_single_from_template(myfilename='/root/PycharmProjects/EIV/DATA/' + subject + '/Calib1.csv', template_path=template_path_PIV + 'Single/'+ subject + '/')


    mean1, var1 = main_EIV_single_from_template(myfilename = '/root/PycharmProjects/EIV/DATA/' + subject  + '/Test1.csv' , template_path = template_path_PIV + 'Single/'+ subject + '/' )


    mean2, var2 = main_EIV_single_from_template(myfilename='/root/PycharmProjects/EIV/DATA/' + subject + '/Test2.csv',
                                                template_path=template_path_PIV + 'Single/'+ subject + '/')

    mean3, var3 = main_EIV_single_from_template(myfilename='/root/PycharmProjects/EIV/DATA/' + subject + '/Calib2.csv',
                                                template_path=template_path_PIV + 'Single/'+ subject + '/')

    if k == 0:
        mean_mat = np.vstack([mean1, mean2, mean3])
        k+=1
    else:
        mean_mat = np.vstack([mean_mat, mean1, mean2,mean3])
    # plt.figure()
    # plt.title(subject + ' tested using EIV_Cross_Template')
    # plt.plot(mean0, label='Calib O1', color='k')
    # plt.xlim(0, 28)
    # plt.plot(mean1, label = 'Test 01', color = 'c')
    # plt.fill_between(np.linspace(0,28,28), mean1-0.5*var1, mean1 +0.5*var1 , alpha = 0.3, color = 'c')
    # plt.plot(mean2, label='Test 02', color='darkorange')
    # plt.fill_between(np.linspace(0, 28, 28), mean2 - 0.5 * var2, mean2 + 0.5 * var2, alpha=0.3, color='darkorange')
    # plt.plot(mean3, label='Calib O2', color='hotpink')
    # plt.fill_between(np.linspace(0, 28, 28), mean3 - 0.5 * var3, mean3 + 0.5 * var3, alpha=0.3, color='hotpink')
    # plt.xlim(0,28)
    # plt.xlabel('# flashs')
    # plt.ylim(0,1)
    # plt.ylabel('Accuracy')
    # plt.legend(loc = 3)
    # plt.savefig(subject + '_accuracy_template_1a_inverse')

for subject in subject_without_test_session:

    mean1, var1 = main_EIV_single_from_template(myfilename = '/root/PycharmProjects/EIV/DATA/' + subject  + '/Calib2.csv' , template_path = template_path_PIV + 'Single/'+ subject + '/' )
    mean0, var0 = main_EIV_single_from_template(myfilename='/root/PycharmProjects/EIV/DATA/' + subject + '/Calib1.csv', template_path=template_path_PIV + 'Single/'+ subject + '/')
    mean_mat = np.vstack([mean_mat, mean1, mean0])
    # plt.figure()
    # plt.title(subject + ' tested using EIV_Cross_Template')
    # plt.plot(mean1, label = 'Calib 02', color = 'hotpink')
    # plt.fill_between(np.linspace(0,28,28), mean1-0.5*var1, mean1 +0.5*var1 , alpha = 0.3, color = 'hotpink')
    # plt.plot(mean0, label='Calib 01', color='k')
    # plt.xlim(0,28)
    # plt.xlabel('# flashs')
    # plt.ylim(0,1)
    # plt.ylabel('Accuracy')
    # plt.legend()
    # plt.savefig(subject + '_accuracy_template_1a_inverse')

a = 1



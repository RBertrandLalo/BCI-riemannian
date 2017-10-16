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
import matplotlib.pyplot as plt

import random

def make_database(data_directory, filtre_order, chnames_list, bandpass, type, delta):
    database = Database(data_directory, delta, target=[2], nontarget=[1],
                        chnames=chnames_list, bandpass=bandpass, filtre_order=filtre_order, type=type)
    database.fill_database()

    return database

def get_data(database, subject_names, session_names):
    x, y, e, _, _, _ = database.get_data(subject_names=subject_names, session_names=session_names)
    return x,y,e

def get_part_of_data(database, subject_names, session_names , ratio = 1 ):
    x, y, e, _, _, _ = database.get_data(subject_names=subject_names, session_names=session_names)
    Ns = x.shape[0]
    if ratio <= 1:
        len_timeline = int(ratio * Ns)
    else:
        if ratio <= Ns:
            len_timeline = ratio
        else:
            print('Not enough data')

    rand_indice_choice = random.sample(k=len_timeline, population=range(Ns))

    return x[rand_indice_choice,:,:], y[rand_indice_choice], e[rand_indice_choice]



def get_part_of_data_equal(database, subject_names, session_names , Ns, T = 0 , NT = 1 ):
    k = 0

    for subject in subject_names:

        x, y, _, _, _ = database.get_data(subject_names=[subject], session_names=session_names)


        xT = x[y == T, : ]
        yT = y[y == T]
        nT = xT.shape[0]
        xNT = x[y == NT, : ]
        yNT = y[y == NT]
        nNT = xNT.shape[0]

        if Ns > nT:
            Ns = nT
            print('sample larger than population')
        rand_indice_choice_T = random.sample(k=Ns, population=range(nT))
        rand_indice_choice_NT = random.sample(k=Ns, population=range(nNT))

        if k == 0:
            extract_x = np.concatenate([xT[rand_indice_choice_T, :, :], xNT[rand_indice_choice_NT, :, :]], axis=0)
            extract_y = np.concatenate([yT[rand_indice_choice_T], yNT[rand_indice_choice_NT]], axis=0)
            k = 1

        else:
            extract_x = np.concatenate([extract_x, xT[rand_indice_choice_T,:,:], xNT[rand_indice_choice_NT,:,:]], axis = 0)
            extract_y = np.concatenate([extract_y, yT[rand_indice_choice_T], yNT[rand_indice_choice_NT]], axis = 0)

    return extract_x , extract_y


def get_data_from_csv_P300(signalfile, markersfile,  chnames, bandpass = (1.0,20.0), filtre_order = 2, delta = 0.6, target = [1], nontarget = [2]):
    signals = pa.read_csv(signalfile, sep=',')
    markers = pa.read_csv(markersfile, sep=',')
    sampling_rate = signals['SamplingRate'][0]
    if bandpass is not None:
        lowf, hif = bandpass
        signals = apply_bandpass_filter_P300(signals, lowf, hif,sampling_rate, filtre_order, chnames)

    s = signals.set_index('Time(s)')[chnames]
    N = int(delta * sampling_rate)

    all_epochs = []
    all_labels = []
    all_event = []

    # for t in markers[markers['Identifier'] in [self.target, self.nontarget]]['Time(s)']:
    for i, t in enumerate(markers['Time(s)']):
        if markers['Identifier'][i] in target + nontarget:
            tmp = np.asarray(s.loc[t:t + delta]).T
            if tmp.shape[1] >= N:
                all_epochs.append(tmp[:, :N])
                all_event.append(markers['Event'][i])
                if markers['Identifier'][i] in target:
                    all_labels.append(0)
                if markers['Identifier'][i] in nontarget:
                    all_labels.append(1)
    data = np.array(all_epochs)
    labels = np.array(all_labels)
    event = np.array(all_event)

    return data, labels, event


def apply_bandpass_filter(myfile, lowf, hif, sampling_rate, filtre_order, chnames):
    """
    Apply bandpass filter to the signals.

    signals -- signals to filter
    lowf -- low cut-frequency
    hif -- high cut-frequency
    """

    from scipy.signal import butter, lfilter
    B, A = butter(filtre_order, np.array([lowf, hif]) / (sampling_rate / 2.0), btype='bandpass')

    X = np.array(myfile[chnames])
    X = lfilter(B, A, X, axis=0)

    signals = pa.DataFrame()
    for i, ch in enumerate(chnames):
        signals[ch] = X[:, i]

    signals['Time'] = myfile['Time']

    return signals

def apply_bandpass_filter_P300(myfile, lowf, hif, sampling_rate, filtre_order, chnames):
    """
    Apply bandpass filter to the signals.

    signals -- signals to filter
    lowf -- low cut-frequency
    hif -- high cut-frequency
    """

    from scipy.signal import butter, lfilter
    B, A = butter(filtre_order, np.array([lowf, hif]) / (sampling_rate / 2.0), btype='bandpass')

    X = np.array(myfile[chnames])
    X = lfilter(B, A, X, axis=0)

    signals = pa.DataFrame()
    for i, ch in enumerate(chnames):
        signals[ch] = X[:, i]

    signals['Time(s)'] = myfile['Time(s)']

    return signals

def apply_bandpass_filter(myfile, lowf, hif, sampling_rate, filtre_order, chnames):
    """
    Apply bandpass filter to the signals.

    signals -- signals to filter
    lowf -- low cut-frequency
    hif -- high cut-frequency
    """

    from scipy.signal import butter, lfilter
    B, A = butter(filtre_order, np.array([lowf, hif]) / (sampling_rate / 2.0), btype='bandpass')

    X = np.array(myfile[chnames])
    X = lfilter(B, A, X, axis=0)

    signals = pa.DataFrame()
    for i, ch in enumerate(chnames):
        signals[ch] = X[:, i]

    signals['Time'] = myfile['Time']

    return signals

def get_data_from_csv_EIV(myfilename, chnames, bandpass = (1.0,20.0), filtre_order = 2 ,delta=0.6, target=[2], nontarget=[1]):
    """
    Segment the signals in epochs of length delta after a marker.
    """
    myfile = pa.read_csv(myfilename, sep=',')
    if chnames is None:
        chnames = list(
            set(list(myfile.keys())) - set(['EVT', 'SamplingRate', 'Time', 'Target', 'NumTargetColumn']))


    SamplingRate = myfile['SamplingRate'][0]
    if bandpass is not None:
        lowf, hif = bandpass
        signals = apply_bandpass_filter(myfile, lowf, hif, sampling_rate=SamplingRate, filtre_order = filtre_order, chnames = chnames)

    s = signals.set_index('Time')[chnames]
    N = int(delta * SamplingRate)
    all_epochs = []
    all_labels = []
    all_event = []
    all_targets = []
    # for t in markers[markers['Identifier'] in [self.target, self.nontarget]]['Time(s)']:
    for i, t in enumerate(myfile['Time']):
        if myfile['Target'][i] in target + nontarget:
            tmp = np.asarray(s.loc[t:t + delta]).T
            if tmp.shape[1] >= N:
                all_epochs.append(tmp[:, :N])
                all_event.append(myfile['EVT'][i])

                if myfile['Target'][i] in target:
                    all_labels.append(0)
                    all_targets.append(myfile['NumTargetColumn'][i])
                if myfile['Target'][i] in nontarget:
                    all_labels.append(1)
                    all_targets.append(myfile['NumTargetColumn'][i])

    data = np.array(all_epochs)
    labels = np.array(all_labels)
    event = np.array(all_event)
    targets = np.array(all_targets)

    return data, labels, event, targets

def make_ERP_EIV(myfilename, chnames, bandpass = (1.0,20.0), filtre_order = 2 ,delta=0.6, target=[2], nontarget=[1]):
    """
    Segment the signals in epochs of length delta after a marker.
    """
    myfile = pa.read_csv(myfilename, sep=',')
    if chnames is None:
        chnames = list(
            set(list(myfile.keys())) - set(['EVT', 'SamplingRate', 'Time', 'Target', 'NumTargetColumn']))


    SamplingRate = myfile['SamplingRate'][0]
    if bandpass is not None:
        lowf, hif = bandpass
        signals = apply_bandpass_filter(myfile, lowf, hif, sampling_rate=SamplingRate, filtre_order = filtre_order, chnames = chnames)

    s = signals.set_index('Time')[chnames]
    N = int(delta * SamplingRate)
    Nc = len(chnames)
    T_sum = np.zeros([Nc, N])
    T_trials = 0
    NT_sum = np.zeros([Nc, N])
    NT_trials = 0
    for i, t in enumerate(myfile['Time']):
        if myfile['Target'][i] in target + nontarget:
            tmp = np.asarray(s.loc[t:t + delta]).T
            if tmp.shape[1] >= N:
                if myfile['Target'][i] in target:
                    T_sum += tmp[:, :N]
                    T_trials += 1
                if myfile['Target'][i] in nontarget:
                    NT_sum += tmp[:, :N]
                    NT_trials += 1
    return T_sum, NT_sum, T_trials, NT_trials


def make_covmats_EIV(myfilename, chnames, ERP, bandpass = (1.0,20.0), filtre_order = 2 ,delta=0.6, target=[2], nontarget=[1]):
    """
    Segment the signals in epochs of length delta after a marker.
    """
    myfile = pa.read_csv(myfilename, sep=',')
    if chnames is None:
        chnames = list(
            set(list(myfile.keys())) - set(['EVT', 'SamplingRate', 'Time', 'Target', 'NumTargetColumn']))


    SamplingRate = myfile['SamplingRate'][0]
    if bandpass is not None:
        lowf, hif = bandpass
        signals = apply_bandpass_filter(myfile, lowf, hif, sampling_rate=SamplingRate, filtre_order = filtre_order, chnames = chnames)

    s = signals.set_index('Time')[chnames]
    # N = int(delta * myfile['SamplingRate'][0])
    N = int(delta * SamplingRate)
    all_covmats = []
    all_labels = []
    all_event = []
    all_targets = []
    # for t in markers[markers['Identifier'] in [self.target, self.nontarget]]['Time(s)']:
    for i, t in enumerate(myfile['Time']):
        if myfile['Target'][i] in target + nontarget:
            tmp = np.asarray(s.loc[t:t + delta]).T
            if tmp.shape[1] >= N:
                trial_cov = np.cov(np.concatenate([ERP, tmp[:, :N]], axis=0))
                all_covmats.append(trial_cov)
                all_event.append(myfile['EVT'][i])

                if myfile['Target'][i] in target:
                    all_labels.append(0)
                    all_targets.append(myfile['NumTargetColumn'][i])
                if myfile['Target'][i] in nontarget:
                    all_labels.append(1)
                    all_targets.append(myfile['NumTargetColumn'][i])

    covmats = np.array(all_covmats)
    labels = np.array(all_labels)
    event = np.array(all_event)
    targets = np.array(all_targets)

    return covmats, labels, event, targets

def make_rTNT_EIV(myfilename, chnames, ERP, centroids_List, bandpass = (1.0,20.0), filtre_order = 2 ,delta=0.6, target=[2], nontarget=[1]):
    """
    Segment the signals in epochs of length delta after a marker.
    """
    T = 0
    NT = 1
    myfile = pa.read_csv(myfilename, sep=',')
    if chnames is None:
        chnames = list(
            set(list(myfile.keys())) - set(['EVT', 'SamplingRate', 'Time', 'Target', 'NumTargetColumn']))


    SamplingRate = myfile['SamplingRate'][0]
    if bandpass is not None:
        lowf, hif = bandpass
        signals = apply_bandpass_filter(myfile, lowf, hif, sampling_rate=SamplingRate, filtre_order = filtre_order, chnames = chnames)

    s = signals.set_index('Time')[chnames]
    # N = int(delta * myfile['SamplingRate'][0])
    N = int(delta * SamplingRate)
    all_rTNT = []
    all_labels = []
    all_event = []
    all_targets = []
    # for t in markers[markers['Identifier'] in [self.target, self.nontarget]]['Time(s)']:
    for i, t in enumerate(myfile['Time']):
        if myfile['Target'][i] in target + nontarget:
            tmp = np.asarray(s.loc[t:t + delta]).T
            if tmp.shape[1] >= N:
                trial_cov = np.cov(np.concatenate([ERP, tmp[:, :N]], axis=0))
                dist = [distance(trial_cov, centroids_List[m]) for m in [T, NT]]
                all_rTNT.append(np.log(dist[0] / dist[1]))
                all_event.append(myfile['EVT'][i])

                if myfile['Target'][i] in target:
                    all_labels.append(0)
                    all_targets.append(myfile['NumTargetColumn'][i])
                if myfile['Target'][i] in nontarget:
                    all_labels.append(1)
                    all_targets.append(myfile['NumTargetColumn'][i])

    rTNT = np.array(all_rTNT)
    labels = np.array(all_labels)
    event = np.array(all_event)
    targets = np.array(all_targets)

    return rTNT, labels, event, targets


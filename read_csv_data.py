import numpy as np
import pandas as pa
from pyriemann.utils.distance import distance


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

def make_epochs_EIV(myfilename, chnames, bandpass = (1.0,20.0), filtre_order = 2 ,delta=0.6, target=[2], nontarget=[1]):
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


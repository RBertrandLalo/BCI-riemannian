# coding: utf-8

# Nathalie: Am I building a skyscraper with mud?

import numpy as np
from numpy.linalg import norm
import pandas as pa
from os import listdir
from os.path import isfile, join
from pyriemann.utils.distance import distance
from generate_template_utils import create_centroids_Dict, create_rTNT_Dict, create_ERP_Dict, create_covmats_Dict

# Database of subjects and sessions

class Database:
    """ 
    Database object allows to load signals from csv files and build a dataset based on subsets of users and sessions.
    """

    def __init__(self, path, delta=0.6, target=[1], nontarget=[2],
                 chnames=['P7', 'P3', 'Pz', 'P4', 'P8', 'PO9', 'O1', 'O2', 'PO10'], bandpass=(1.0, 48.0), filtre_order = 2):
        """
        path -- path to the directory containing csv files.
        delta -- epoch duration
        target -- event identifier for target
        nontarget -- event identifier for nontarget
        chnames -- names of channels
        bandpass -- a tuple (low_freq, high_freq) for the bandpass filter, or none is no filter needs to be applied
        """
        self.subjects = {}  # Mapping of subject names and subject objects
        self.path = path
        self.delta = delta
        self.target = target
        self.nontarget = nontarget
        self.chnames = chnames
        self.bandpass = bandpass
        self.filtre_order = filtre_order

    def fill_database(self, verbose=False):
        """
        Load the csv files.
        """

        file_status = []

        # feed each filename to the subject creator
        for filename in listdir(self.path):

            # check that filename refers to aa file and 'signal' is in the filename, or skip the file
            if 'signal' not in filename or not isfile(join(self.path, filename)):
                continue

            signals = pa.read_csv(join(self.path, filename), sep=',')
            if self.chnames is None:
                self.chnames = list(set(list(signals.keys())) - set(['SamplingRate', 'Unnamed: 0', 'Time(s)']))
            if not (set(self.chnames) < set(list(signals.keys()))):
                print filename + ' does not have all channels'
                continue

            self.add_data(filename, verbose)


    def create_ERP(self, verbose=False):
        """
        Load the csv files.
        """

        file_status = []

        # feed each filename to the subject creator
        for filename in listdir(self.path):

            # check that filename refers to aa file and 'signal' is in the filename, or skip the file
            if 'signal' not in filename or not isfile(join(self.path, filename)):
                continue

            signals = pa.read_csv(join(self.path, filename), sep=',')
            if self.chnames is None:
                self.chnames = list(set(list(signals.keys())) - set(['SamplingRate', 'Unnamed: 0', 'Time(s)']))
            if not(set(self.chnames) < set(list(signals.keys()))):
                print filename + ' does not have all channels'
                continue


            self.add_ERP(filename, verbose)

    def create_covmats(self, ERP_Dict, save_path, verbose=False):
        """
        Load the csv files.
        """

        file_status = []

        # feed each filename to the subject creator
        self.ERP_Dict = ERP_Dict
        for filename in listdir(self.path):

            # check that filename refers to aa file and 'signal' is in the filename, or skip the file
            if 'signal' not in filename or not isfile(join(self.path, filename)):
                continue

            signals = pa.read_csv(join(self.path, filename), sep=',')
            if self.chnames is None:
                self.chnames = list(set(list(signals.keys())) - set(['SamplingRate', 'Unnamed: 0', 'Time(s)']))
            if not (set(self.chnames) < set(list(signals.keys()))):
                print filename + ' does not have all channels'
                continue

            self.add_covmats(filename, ERP_Dict, verbose)

            covmats_Dict = create_covmats_Dict(self)
            np.save(save_path + 'Temporary_Covmats_Dict', covmats_Dict)


    def create_rTNT(self, ERP_Dict, centroids_Dict , save_path, verbose = False):
        """
        Load the csv files.

        Returns
        -------
        object
        """
        self.ERP = ERP_Dict
        self.centroids = centroids_Dict
        file_status = []

        # feed each filename to the subject creator
        for filename in listdir(self.path):

            # check that filename refers to aa file and 'signal' is in the filename, or skip the file
            if 'signal' not in filename or not isfile(join(self.path, filename)):
                continue

            signals = pa.read_csv(join(self.path, filename), sep=',')
            if self.chnames is None:
                self.chnames = list(set(list(signals.keys())) - set(['SamplingRate', 'Unnamed: 0', 'Time(s)']))
            if not (set(self.chnames) < set(list(signals.keys()))):
                print filename + ' does not have all channels'
                continue

            self.add_rTNT(filename, ERP_Dict, centroids_Dict, verbose)

            rTNT_Dict = create_rTNT_Dict(self)
            np.save(save_path + 'Temporary_rTNT_Dict', rTNT_Dict)





    def get_subject_names(self):
        """
        Return subjects
        """
        return list(self.subjects.keys())

    def add_data(self, filename, verbose):
        """
        Load files for a given session.
        
        filename -- name of the file to load.
        """

        # subject id is the number that follows P in the filename
        subject_name = filename[filename.find('P'):filename.find('S')]

        # session id is the number that follows S in the filename
        session_name = filename[filename.find('S'):filename.find('-')]

        if verbose:
            print("Loading signals" + subject_name + session_name)

        if subject_name in self.subjects.keys():
            subject = self.subjects[subject_name]

        else:
            subject = Subject(subject_name, self.path, self.delta, self.target, self.nontarget, self.chnames,
                              self.bandpass, self.filtre_order)
            self.subjects[subject_name] = subject

            self.chnames = subject.chnames

        return subject.add_session(session_name) if session_name not in subject.sessions else False

    def add_ERP(self, filename, verbose):
        """
        Load files for a given session.

        filename -- name of the file to load.
        """

        # subject id is the number that follows P in the filename
        subject_name = filename[filename.find('P'):filename.find('S')]

        # session id is the number that follows S in the filename
        session_name = filename[filename.find('S'):filename.find('-')]

        if verbose:
            print("Loading ERP " + subject_name + session_name)

        if subject_name in self.subjects.keys():
            subject = self.subjects[subject_name]

        else:
            subject = Subject(subject_name, self.path, self.delta, self.target, self.nontarget, self.chnames,
                              self.bandpass, self.filtre_order)
            self.subjects[subject_name] = subject

        return subject.add_session_ERP(session_name) if session_name not in subject.sessions else False

    def add_covmats(self, filename, ERP_Dict, verbose):
        """
        Load files for a given session.

        filename -- name of the file to load.
        """

        # subject id is the number that follows P in the filename
        subject_name = filename[filename.find('P'):filename.find('S')]

        # session id is the number that follows S in the filename
        session_name = filename[filename.find('S'):filename.find('-')]

        if verbose:
            print("Loading covmats " + subject_name + session_name)

        if subject_name in self.subjects.keys():
            subject = self.subjects[subject_name]

        else:
            subject = Subject(subject_name, self.path, self.delta, self.target, self.nontarget, self.chnames,
                              self.bandpass, self.filtre_order)
            self.subjects[subject_name] = subject

        return subject.add_session_covmats(session_name, ERP_Dict) if session_name not in subject.sessions else False

    def add_rTNT(self, filename, ERP_Dict, centroids_Dict,  verbose):
        """
        Load files for a given session.

        filename -- name of the file to load.
        """

        # subject id is the number that follows P in the filename
        subject_name = filename[filename.find('P'):filename.find('S')]

        # session id is the number that follows S in the filename
        session_name = filename[filename.find('S'):filename.find('-')]

        if verbose:
            print("Loading rTNT " + subject_name + session_name)

        if subject_name in self.subjects.keys():
            subject = self.subjects[subject_name]

        else:
            subject = Subject(subject_name, self.path, self.delta, self.target, self.nontarget, self.chnames,
                              self.bandpass, self.filtre_order)
            self.subjects[subject_name] = subject
        ERP = ERP_Dict[subject_name]['ERP Sum']/ERP_Dict[subject_name]['ERP number trials']
        centroids_list = centroids_Dict[subject_name]

        return subject.add_session_rTNT(session_name, ERP, centroids_list) if session_name not in subject.sessions else False

    def get_data(self, subject_names=None, session_names=None):
        """
        Get all data that belong to a set of subjects for all of his available sessions. The set can be given by the user.
        
        subject_names -- (sub)set of subjects to include in the dataset. Default is None. 
        
        If subject_names is None then the data for all subjects will be loaded.
        The function returns epochs , labels (eg. target/nontarget), subject names and session names. 
        """

        X_list = []
        y_list = []
        e_list = []
        subject_name_list = []
        session_list = []

        subject_names = self.subjects.keys() if subject_names is None else subject_names

        for subject_name in subject_names:

            if subject_name not in self.subjects.keys():
                print(subject_name + ' is not in the list of subjects')
                continue

            tmpX, tmpy, tmpe, tmpsession = self.subjects[subject_name].get_data(session_names)

            X_list.append(tmpX)
            y_list.append(tmpy)
            e_list.append(tmpe)
            subject_name_list += [subject_name] * len(tmpy)
            session_list.extend(tmpsession)

        return (
        np.vstack(X_list), np.hstack(y_list), np.hstack(e_list), subject_name_list, session_list)


class Subject:
    """
    This class allows to load and manipulate sessions for a given subject
    """

    def __init__(self, name, path, delta=0.6, target=[1], nontarget=[2],
                 chnames=None, bandpass=(1.0, 48.0), filtre_order = 2):
        """
        name -- name of subject
        path -- path to csv file
        delta -- epoch duration
        target -- event identifier for target
        nontarget -- event identifier for nontarget
        chnames -- names of channels
        bandpass -- a tuple (low_freq, high_freq) for the bandpass filter, or none is no filter needs to be applied
        filtre_order -- int for degree of the bandpass filter
        """
        self.sessions = {}
        self.name = name
        self.path = path
        self.delta = delta
        self.target = target
        self.nontarget = nontarget
        self.chnames = chnames
        self.bandpass = bandpass
        self.filtre_order = filtre_order

    def get_name(self):
        return self.name

    def get_session_names(self):
        return list(self.sessions.keys())

    def add_session(self, session_name):
        """
        Load csv files for one particular session.
        """
        session = Session(session_name, self.name, self.path, self.delta, self.target, self.nontarget,
                          self.chnames, self.bandpass, self.filtre_order)
        session.read_files()

        self.chnames = session.chnames

        self.sessions[session_name] = session

    def add_session_ERP(self, session_name):
        """
        Load csv files for one particular session.
        """
        session = Session(session_name, self.name, self.path, self.delta, self.target, self.nontarget,
                          self.chnames, self.bandpass, self.filtre_order)
        session.read_files_ERP()

        self.chnames = session.chnames

        self.sessions[session_name] = session

    def add_session_covmats(self, session_name, ERP_Dict):
        """
        Load csv files for one particular session.
        """
        session = Session(session_name, self.name, self.path, self.delta, self.target, self.nontarget,
                          self.chnames, self.bandpass, self.filtre_order)

        self.ERP = extract_ERP_from_Dict(ERP_Dict= ERP_Dict, subject_key = self.get_name())

        session.read_files_covmats(self.ERP)

        self.sessions[session_name] = session

    def add_session_rTNT(self, session_name, ERP, centroids_list):
        """
        Load csv files for one particular session.
        """
        session = Session(session_name, self.name, self.path, self.delta, self.target, self.nontarget,
                          self.chnames, self.bandpass, self.filtre_order)
        session.read_files_rTNT(ERP, centroids_list)

        self.sessions[session_name] = session

    def get_data(self, session_names=None):
        """
        Get all data that belong to a set of sessions for this subject. The set can be given by the user.
        
        session_names -- (sub)set of sessions to include in the dataset
        """

        X_list = []
        y_list = []
        e_list = []
        session_list = []

        session_names = self.sessions.keys() if session_names is None else session_names

        for session_name in session_names:
            X_list.append(self.sessions[session_name].data)
            y_list.append(self.sessions[session_name].labels)
            e_list.append(self.sessions[session_name].event)
            session_list.extend([session_name] * len(self.sessions[session_name].labels))

        return (np.vstack(X_list), np.hstack(y_list), np.hstack(e_list), session_list)


class Session:
    """
    This class allows to manipulate data for a single session 
    """

    def __init__(self, session_name, subject_name, path, delta=0.6, target=[1], nontarget=[2],
                 chnames=['P7', 'P3', 'Pz', 'P4', 'P8', 'PO9', 'O1', 'O2', 'PO10'], bandpass=(1.0, 48.0),filtre_order = 2):
        """
        session_name -- id for a session 'S1', 'S2'...
        subject_name -- id for a subject 'P01', 'P11'
        path -- path to the files
        delta -- epoch duration
        target -- event identifier for target
        nontarget -- event identifier for nontarget
        chnames -- names of channels
        bandpass -- a tuple (low_freq, high_freq) for the bandpass filter, or none is no filter needs to be applied
        """
        self.session_name = session_name
        self.subject_name = subject_name
        self.path = path
        self.delta = delta
        self.target = target
        self.nontarget = nontarget
        self.chnames = chnames
        self.bandpass = bandpass
        self.filtre_order = filtre_order

    def get_name(self):
        return self.subject_name + self.session_name

    def get_epoch_duration(self):
        return self.delta

    def count_label(self, label):
        if label in self.target:
            return np.where(self.labels == 0)[0].shape[0]
        elif label in self.nontarget:
            return np.where(self.labels == 1)[0].shape[0]

    def apply_bandpass_filter(self, signals, lowf, hif):
        """
        Apply bandpass filter to the signals.

        signals -- signals to filter
        lowf -- low cut-frequency
        hif -- high cut-frequency
        """
        sampling_rate = signals['SamplingRate'][0]
        self.SamplingRate = sampling_rate

        from scipy.signal import butter, lfilter
        B, A = butter(self.filtre_order, np.array([lowf, hif]) / (sampling_rate / 2.0), btype='bandpass')

        X = np.array(signals[self.chnames])
        X = lfilter(B, A, X, axis=0)

        for i, ch in enumerate(self.chnames):
            signals[ch] = X[:, i]

        return signals

    def make_epochs(self, signals, markers, delta):
        """
        Segment the signals in epochs of length delta after a marker.
        """
        if self.bandpass is not None:
            lowf, hif = self.bandpass
            signals = self.apply_bandpass_filter(signals, lowf, hif)

        s = signals.set_index('Time(s)')[self.chnames]
        N = int(delta * signals['SamplingRate'][0])

        all_epochs = []
        all_labels = []
        all_event = []

        # for t in markers[markers['Identifier'] in [self.target, self.nontarget]]['Time(s)']:
        for i, t in enumerate(markers['Time(s)']):
            if markers['Identifier'][i] in self.target + self.nontarget:
                tmp = np.asarray(s.loc[t:t + delta]).T
                if tmp.shape[1] >= N:
                    all_epochs.append(tmp[:, :N])
                    all_event.append(markers['Event'][i])
                    if markers['Identifier'][i] in self.target:
                        all_labels.append(0)
                    if markers['Identifier'][i] in self.nontarget:
                        all_labels.append(1)

        self.data = np.array(all_epochs)
        self.labels = np.array(all_labels)
        self.event = np.array(all_event)


    def create_covmats(self, signals, markers, delta, ERP):
        """
        Segment the signals in epochs of length delta after a marker.
        """
        if self.bandpass is not None:
            lowf, hif = self.bandpass
            signals = self.apply_bandpass_filter(signals, lowf, hif)

        s = signals.set_index('Time(s)')[self.chnames]
        Nt = int(delta * signals['SamplingRate'][0])
        Nc = len(self.chnames)

        if ERP.shape[0] != Nc:
            raise ValueError ('ERP channels number does not fit data')
        if ERP.shape[1] != Nt:
            raise ValueError('ERP times number does not fit data ')


        all_labels = []
        all_covs =[]

        # for t in markers[markers['Identifier'] in [self.target, self.nontarget]]['Time(s)']:
        for i, t in enumerate(markers['Time(s)']):
            if markers['Identifier'][i] in self.target + self.nontarget:
                tmp = np.asarray(s.loc[t:t + delta]).T
                if tmp.shape[1] >= Nt:
                    trial_cov = np.cov(np.concatenate((ERP, tmp[:, :Nt]), axis = 0))
                    all_covs.append(trial_cov)
                    if markers['Identifier'][i] in self.target:
                        all_labels.append(0)
                    if markers['Identifier'][i] in self.nontarget:
                        all_labels.append(1)

        my_covmats = np.array(all_covs)
        self.covmats = np.array(all_covs)
        self.labels = np.array(all_labels)

    def create_rTNT(self, signals, markers, delta, ERP, centroids_list):
        """
        Segment the signals in epochs of length delta after a marker.
        """

        T = 0
        NT = 1

        if self.bandpass is not None:
            lowf, hif = self.bandpass
            signals = self.apply_bandpass_filter(signals, lowf, hif)

        s = signals.set_index('Time(s)')[self.chnames]
        Nt = int(delta * signals['SamplingRate'][0])
        Nc = len(self.chnames)

        if ERP.shape[0] != Nc:
            raise ValueError('ERP channels number does not fit data')
        if ERP.shape[1] != Nt:
            raise ValueError('ERP times number does not fit data ')

        all_rTNT = []
        all_labels = []

        # for t in markers[markers['Identifier'] in [self.target, self.nontarget]]['Time(s)']:
        for i, t in enumerate(markers['Time(s)']):
            if markers['Identifier'][i] in self.target + self.nontarget:
                tmp = np.asarray(s.loc[t:t + delta]).T
                if tmp.shape[1] >= Nt:
                    trial_cov = np.cov(np.concatenate([ERP, tmp[:, :Nt]], axis=0))
                    dist = [distance(trial_cov, centroids_list[m]) for m in [T,NT]]
                    all_rTNT.append(np.log(dist[0] / dist[1]))
                    if markers['Identifier'][i] in self.target:
                        all_labels.append(0)
                    if markers['Identifier'][i] in self.nontarget:
                        all_labels.append(1)

        self.rTNT = np.array(all_rTNT)
        self.labels = np.array(all_labels)



    def create_ERP(self, signals, markers,delta):
        """
        Segment the signals in epochs of length delta after a marker.
        """

        if self.bandpass is not None:
            lowf, hif = self.bandpass
            signals = self.apply_bandpass_filter(signals, lowf, hif)

        s = signals.set_index('Time(s)')[self.chnames]
        Nt = int(delta * signals['SamplingRate'][0])
        Nc = len(self.chnames)

        T_sum = np.zeros([Nc,Nt])
        T_trials = 0
        NT_sum = np.zeros([Nc,Nt])
        NT_trials = 0
        all_epochs_amplitude = []

        # for t in markers[markers['Identifier'] in [self.target, self.nontarget]]['Time(s)']:
        for i, t in enumerate(markers['Time(s)']):
            if markers['Identifier'][i] in self.target + self.nontarget:
                tmp = np.asarray(s.loc[t:t + delta]).T
                if tmp.shape[1] >= Nt:
                    if markers['Identifier'][i] in self.target:
                        T_sum += tmp[:, :Nt]
                        T_trials += 1
                    if markers['Identifier'][i] in self.nontarget:
                        NT_sum += tmp[:, :Nt]
                        NT_trials += 1
        self.T_trials = T_trials
        self.NT_trials = NT_trials
        self.T_sum = T_sum
        self.NT_sum  = NT_sum

        return self.T_sum, self.NT_sum, self.T_trials, self.NT_trials








    def read_files(self):
        """
        Read files and load signals for a given session.
        """
        SIGNALSFILE = self.subject_name + self.session_name + '-signals.csv'
        MARKERSFILE = self.subject_name + self.session_name + '-markers.csv'


        signals = pa.read_csv(join(self.path, SIGNALSFILE), sep=',')
        markers = pa.read_csv(join(self.path, MARKERSFILE), sep=',')

        if self.chnames is None:
            self.chnames = list(set(list(signals.keys())) - set(['SamplingRate' , 'Unnamed: 0' , 'Time(s)']))

        self.make_epochs(signals, markers, self.delta)

    def read_files_ERP(self):
        """
        Read files and load signals for a given session.
        """
        SIGNALSFILE = self.subject_name + self.session_name + '-signals.csv'
        MARKERSFILE = self.subject_name + self.session_name + '-clean2_markers.csv'

        signals = pa.read_csv(join(self.path, SIGNALSFILE), sep=',')
        markers = pa.read_csv(join(self.path, MARKERSFILE), sep=',')



        self.create_ERP(signals, markers, self.delta)

    def read_files_covmats(self, ERP):
        """
        Read files and load signals for a given session.
        """
        SIGNALSFILE = self.subject_name + self.session_name + '-signals.csv'
        MARKERSFILE = self.subject_name + self.session_name + '-clean2_markers.csv'

        signals = pa.read_csv(join(self.path, SIGNALSFILE), sep=',')
        markers = pa.read_csv(join(self.path, MARKERSFILE), sep=',')


        self.create_covmats(signals, markers, self.delta, ERP)

    def read_files_rTNT(self, ERP, centroids_list):
        """
        Read files and load signals for a given session.
        """
        SIGNALSFILE = self.subject_name + self.session_name + '-signals.csv'
        MARKERSFILE = self.subject_name + self.session_name + '-clean2_markers.csv'

        signals = pa.read_csv(join(self.path, SIGNALSFILE), sep=',')
        markers = pa.read_csv(join(self.path, MARKERSFILE), sep=',')

        self.create_rTNT(signals, markers, self.delta, ERP, centroids_list)

def extract_ERP_from_Dict(ERP_Dict, subject_key):
    return ERP_Dict[subject_key]['ERP Sum'] / ERP_Dict[subject_key]['ERP number trials']






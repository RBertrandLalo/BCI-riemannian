# coding: utf-8

# Nathalie: Am I building a skyscraper with mud?

import numpy as np
import pandas as pa
from os import listdir
from os.path import isfile, join


# Database of subjects and sessions

class Database:
    """ 
    Database object allows to load signals from csv files and build a dataset based on subsets of users and sessions.
    """

    def __init__(self, path, delta=0.6, target=[2], nontarget=[1],
                 chnames=['Fz','Cz','CP5','CP1','CP2','CP6','P7', 'P3', 'Pz', 'P4', 'P8', 'PO9', 'O1', 'O2', 'PO10'], bandpass=(1.0, 48.0), filtre_order = 2):
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
            if 'EIV' not in filename or not isfile(join(self.path, filename)):
                continue

            self.add_data(filename, verbose)

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
        session_name = filename[filename.find('S'):filename.find('.')]

        if verbose:
            print("Loading " + subject_name + session_name)

        if subject_name in self.subjects.keys():
            subject = self.subjects[subject_name]

        else:
            subject = Subject(subject_name, self.path, self.delta, self.target, self.nontarget, self.chnames,
                              self.bandpass)
            self.subjects[subject_name] = subject
            self.chnames = subject.chnames

        return subject.add_session(session_name) if session_name not in subject.sessions else False

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
        w_list = []
        subject_name_list = []
        session_list = []

        subject_names = self.subjects.keys() if subject_names is None else subject_names

        for subject_name in subject_names:

            if subject_name not in self.subjects.keys():
                print(subject_name + ' is not in the list of subjects')
                continue

            tmpX, tmpy, tmpe, tmpw, tmpsession = self.subjects[subject_name].get_data(session_names)

            X_list.append(tmpX)
            y_list.append(tmpy)
            e_list.append(tmpe)
            w_list.append(tmpw)
            subject_name_list += [subject_name] * len(tmpy)
            session_list.extend(tmpsession)

        return (
        np.vstack(X_list), np.hstack(y_list), np.hstack(e_list), np.hstack(w_list), subject_name_list, session_list)


class Subject:
    """
    This class allows to load and manipulate sessions for a given subject
    """

    def __init__(self, name, path, delta=0.6, target=1, nontarget=2,
                 chnames=['Fz', 'Cz', 'CP5', 'CP1', 'CP2', 'CP6', 'P7', 'P3', 'Pz', 'P4', 'P8', 'PO9', 'O1', 'O2',
                          'PO10'], bandpass=(1.0, 48.0)):
        """
        name -- name of subject
        path -- path to csv file
        delta -- epoch duration
        target -- event identifier for target
        nontarget -- event identifier for nontarget
        chnames -- names of channels
        bandpass -- a tuple (low_freq, high_freq) for the bandpass filter, or none is no filter needs to be applied
        """
        self.sessions = {}
        self.name = name
        self.path = path
        self.delta = delta
        self.target = target
        self.nontarget = nontarget
        self.chnames = chnames
        self.bandpass = bandpass

    def get_name(self):
        return self.name

    def get_session_names(self):
        return list(self.sessions.keys())

    def add_session(self, session_name):
        """
        Load csv files for one particular session.
        """
        session = Session(session_name, self.name, self.path, self.delta, self.target, self.nontarget,
                          self.chnames, self.bandpass)

        session.read_files_PIV()

        self.sessions[session_name] = session
        self.chnames = session.chnames

    def get_data(self, session_names=None):
        """
        Get all data that belong to a set of sessions for this subject. The set can be given by the user.
        
        session_names -- (sub)set of sessions to include in the dataset
        """

        X_list = []
        y_list = []
        e_list = []
        w_list = []
        session_list = []

        session_names = self.sessions.keys() if session_names is None else session_names

        for session_name in session_names:
            X_list.append(self.sessions[session_name].data)
            y_list.append(self.sessions[session_name].labels)
            e_list.append(self.sessions[session_name].event)
            w_list.append(self.sessions[session_name].targets)
            session_list.extend([session_name] * len(self.sessions[session_name].labels))

        return (np.vstack(X_list), np.hstack(y_list), np.hstack(e_list), np.hstack(w_list), session_list)


class Session:
    """
    This class allows to manipulate data for a single session 
    """

    def __init__(self, session_name, subject_name, path, delta=0.6, target=[2], nontarget=[1],
                 chnames=['Fz', 'Cz', 'CP5', 'CP1', 'CP2', 'CP6', 'P7', 'P3', 'Pz', 'P4', 'P8', 'PO9', 'O1', 'O2',
                          'PO10'], bandpass=(1.0, 48.0), filtre_order = 2):
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

    def apply_bandpass_filter(self, myfile, lowf, hif , sampling_rate):
        """
        Apply bandpass filter to the signals.
        
        signals -- signals to filter
        lowf -- low cut-frequency
        hif -- high cut-frequency
        """



        from scipy.signal import butter, lfilter
        B, A = butter(self.filtre_order, np.array([lowf, hif]) / (sampling_rate / 2.0), btype='bandpass')

        X = np.array(myfile[self.chnames])
        X = lfilter(B, A, X, axis=0)

        signals = pa.DataFrame()
        for i, ch in enumerate(self.chnames):
            signals[ch] = X[:, i]

        signals['Time'] = myfile['Time']

        return signals

    def make_epochs_EIV(self, myfile, delta):
        """
        Segment the signals in epochs of length delta after a marker.
        """


        if self.bandpass is not None:
            lowf, hif = self.bandpass
            signals = self.apply_bandpass_filter(myfile, lowf, hif , sampling_rate = myfile['SamplingRate'][0])

        s = signals.set_index('Time')[self.chnames]
        # N = int(delta * myfile['SamplingRate'][0])
        N = int(delta * 1000.0)
        all_epochs = []
        all_labels = []
        all_event = []
        all_targets = []
        newtarget = 1
        # for t in markers[markers['Identifier'] in [self.target, self.nontarget]]['Time(s)']:
        for i, t in enumerate(myfile['Time']):
            if myfile['Target'][i] in self.target + self.nontarget:
                tmp = np.asarray(s.loc[t:t + delta]).T
                if tmp.shape[1] >= N:
                    all_epochs.append(tmp[:, :N])
                    all_event.append(myfile['EVT'][i])

                    if myfile['Target'][i] in self.target:
                        all_labels.append(0)
                        all_targets.append(myfile['NumTargetColumn'][i])
                    if myfile['Target'][i] in self.nontarget:
                        all_labels.append(1)
                        all_targets.append(myfile['NumTargetColumn'][i])

        self.data = np.array(all_epochs)
        self.labels = np.array(all_labels)
        self.event = np.array(all_event)
        self.targets = np.array(all_targets)
        return self.data, self.labels, self.event, self.targets

    def create_ERP(self, myfile, delta):
        """
        Segment the signals in epochs of length delta after a marker.
        """

        if self.bandpass is not None:
            lowf, hif = self.bandpass
            signals = self.apply_bandpass_filter(myfile, lowf, hif, sampling_rate=myfile['SamplingRate'][0])

        s = signals.set_index('Time')[self.chnames]
        # N = int(delta * myfile['SamplingRate'][0])
        N = int(delta * 1000.0)
        T_sum = np.zeros(N)
        NT_sum = np.zeros(N)
        all_labels = []
        all_event = []
        all_targets = []
        newtarget = 1
        # for t in markers[markers['Identifier'] in [self.target, self.nontarget]]['Time(s)']:
        for i, t in enumerate(myfile['Time']):
            if myfile['Target'][i] in self.target + self.nontarget:
                tmp = np.asarray(s.loc[t:t + delta]).T
                if tmp.shape[1] >= N:
                    if myfile['Target'][i] in self.target:
                        T_sum += tmp[:, :N]
                    if myfile['Target'][i] in self.nontarget:
                        NT_sum += tmp[:, :N]

        self.data = np.array(all_epochs)
        self.labels = np.array(all_labels)
        self.event = np.array(all_event)
        self.targets = np.array(all_targets)
        return self.data, self.labels, self.event, self.targets

    def make_words(self, words):
        self.words = words[self.session_name]
        return self.words

    def read_files_PIV(self):
        """
        Read files and load signals for a given session.
        """
        MYFILE = 'EIV_' + self.subject_name + self.session_name + '.csv'
        myfile = pa.read_csv(join(self.path, MYFILE), sep=',')
        if self.chnames is None:
            self.chnames = list(
                set(list(myfile.keys())) - set(['EVT', 'SamplingRate', 'Time', 'Target', 'NumTargetColumn']))

        self.make_epochs_EIV(myfile, self.delta)







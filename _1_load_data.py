import matplotlib.pyplot as plt
import pickle
import numpy as np
from scipy.signal import find_peaks
import matplotlib.gridspec as gridspec
from biosppy import plotting, utils
from biosppy.signals import tools
import pandas as pd
from collections import OrderedDict
from sklearn.preprocessing import minmax_scale
import joana
swim_style = ['butterfly', 'breaststroke', 'frontcrawl', 'backstroke']

from sklearn.preprocessing import minmax_scale

import biosppy as bp
window = 20
sampling_rate = 100
overlap = 0.5

plot_titles = ["Time", "Yaw", "Pitch", "Roll", "Heading", "Ax", "Ay", "Az", "gx", "gy", "gz", "mx", "my", "mz"];
pool_distance = 25  # metros
periodoamostragem = 10  # ms
freq_amostragem = 100  # Hz
no_colunas = 14
fs = 100  # Hz
tempo_paragem = 3  # seconds
window_turn_stop = 0.5  # seconds
from collections import OrderedDict
import os
import csv
no_amostras_paragem = tempo_paragem * 1000 / periodoamostragem
no_amostras_window = window_turn_stop * 1000 / periodoamostragem

def findFiles(folder=None, extensao='.csv'):
    files = []
    path = os.path.abspath(folder)
    l_files = []
    for pastaAtual, subPastas, arquivos in os.walk(path):
        for arquivo in arquivos:
            if arquivo.endswith(extensao):
                # print(arquivo)
                l_files.append(arquivo)
        files.extend([os.path.join(pastaAtual, arquivo) for arquivo in arquivos if arquivo.endswith(extensao)])
    return files, l_files

def get_data():
    list_ = findFiles(folder='.\data')
    list__ = list_[0]
    list_names = list_[1]

    print(list_names)
    user = []

    for l in range(0, len(list__)):
        print(l)
        # OPEN and READING .csv data files
        filename = list_names[l]
        print(filename)
        file = list__[l]
        with open(file, 'r') as ficheiro:
            all_sensors = OrderedDict()
            reader = csv.reader(ficheiro, delimiter=',', quoting=csv.QUOTE_NONE)
            for linha in reader:
                for l in range(13):
                    if plot_titles[l] not in all_sensors.keys():
                        all_sensors[plot_titles[l]] = []
                    try:
                        all_sensors[plot_titles[l]].append(float(linha[l]))
                    except:
                        all_sensors[plot_titles[l]].append(110000)
            user.append(all_sensors)

    return pd.DataFrame.from_dict(user)

def data_segmentation(data_path, show=False):
    """

    :param data_path: file name to extract
    :param show: show segmentation marks to see if it was well segmented
    :return: data structured and segmented by swimming laps
    """

    data = get_data() #or open pickle with data already structured
    print('-- Data was successfully extracted --')

    metadata = pd.read_excel('Metadados.xlsx') #open Metadata
    new_data = OrderedDict() #create new dict to place segmented data
    file = open('perfect_segment_idx.txt', 'r') #open txt with the segmentation indexes
    line_file = file.readlines() #each line will be one user (all his activities indexes)

    for user in range(len(data)):
        new_data[user] = {} # new data dict is divided by users/swimmers

        segment_idx = line_file[user][1:-2].split(' ') #line_file is of type String, [1:-2] removes '[]\n'
        segment_idx = [int(seg) for seg in segment_idx ] #transform each number from string to integers

        idx_meta = metadata['User'].values.tolist().index(user) #the idx in metadata corresponding to this user
        sequence = metadata['Sequence'][idx_meta].split(', ') #get that user's sequence of activities (labels)

        if show:

            plt.figure(figsize=(10, 5)) #create figure
            data_ = data['Pitch'][user]
            plt.plot(data_) # plot pitch
            plt.vlines(segment_idx, np.max(data_), np.min(data_)) #plot segmentation indexes as vertical lines

            plt.title(sequence)
            for s in range(0, len(segment_idx)):
                plt.text(segment_idx[s] + 500, 50, str(s)) # plot 0 to 16 if the user has 16 swimming laps
            plt.show()

        new_data[user]['Label'] = [] # create list of labels in new_data[user] dict

        for seg in range(1, len(segment_idx)): # run through segment_idx list starting in 1
            start = segment_idx[seg-1] # the start will be the previous idx
            end = segment_idx[seg] # the end will be the current idx
            style = sequence[seg-1] # the swimming style is the previous label
            new_data[user]['Label'].append(style.capitalize()) # capitalize style to ensure all labels are equal

            for cl in data.iloc[[user]].columns: # Iterave over columns ['Time', 'Yaw', 'Pitch', ...]
                if cl not in new_data[user].keys(): # if cl is not in new_data[user] create it
                    new_data[user][cl] = []

                #dump data inside new_data[user][cl] list, segmented by start and end
                new_data[user][cl].append(np.array(data.iloc[[user]][cl].values[0][start:end]))

        #dump new_data into a pickle file or continue to further steps
        pickle.dump(new_data, open('data\swim_pitch_segmented_', 'wb'))
    file.close()
    print('-- Data was successfully segmented --')
    return new_data



def standarize(sig):

    return (sig-np.mean(sig))/np.std(sig)


data_path = 'data\swim_data_'

def load_swim_samples(sensor_list=['Pitch', 'Roll', 'Yaw', 'Az', 'Ay', 'Ax']):

    #data = pickle.load(open('data\swim_pitch_segmented','rb'))
    data = data_segmentation(data_path) #get segmented data

    user_feat = [] #create a list for dumping features
    user_label = [] #create a list for dumping labels
    for user in range(len(data)):
        user_df = pd.DataFrame.from_dict(data[user]) # transform from dict to DataFrame

        for cl in sensor_list: # extract only columns expressed in sensor_list

            for line in range(len(user_df[cl])): # go over each segment

                feat_row = bp.signals.tools.signal_stats(user_df[cl][line]) # compute statistical features

                feat_names = [cl + '_' + feat for feat in feat_row._names] # get features names with column prefix
                feat_row = pd.DataFrame([feat_row._values], columns=feat_names)
                #join every feats into a dataframe
                if line == 0:
                    feat_rows = feat_row
                else:
                    feat_rows = pd.concat([feat_rows, feat_row], axis=0, sort=False, ignore_index=True)

            label_ = user_df['Label'].values # get the label

            if cl == sensor_list[0]:
                feats = feat_rows.copy()

            else:
                feats = pd.concat([feats, feat_rows], axis=1, sort=False) #join every row of features

        user_feat += [pd.DataFrame(standarize(feats), columns=feats.columns)] #features are standarize and saved

        user_label += [label_] #save labels

    #pickle.dump(user_feat, open('data\swim_pitch_feats_pool','wb'))
    #pickle.dump(user_label, open('data\swim_pitch_labels_pool', 'wb'))
    print('-- Features Extracted! -- ')
    return user_feat, user_label


#user_feat, user_label = load_swim_samples()
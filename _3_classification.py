best = 0
best_classifier = None
print("<START Classification>")

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pickle
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import seaborn as sns
import _1_load_data


def plot_confusion_matrix(y_true, y_pred, true_labels=None, normalize=False):
    """

    :param y_true: type(array), contains true labels
    :param y_pred: type(array), contains predicted labels
    :param true_labels: list of unique labels
    :param normalize: boolean
    :return:
    """
    if true_labels == []:
        true_labels = np.unique(y_true)
    cm = confusion_matrix(y_true, y_pred, labels=true_labels)
    if normalize:
        cm = np.round(cm / np.max(cm), 2)
    ax = plt.subplot(1, 1, 1)
    ax.set_title(n)
    sns.heatmap(cm, annot=True, ax=ax, fmt='g', cmap='Blues')  # annot=True to annotate cells
    # labels, title and ticks
    ax.set_xlabel('Predicted', fontsize=20)
    ax.xaxis.set_label_position('top')
    ax.xaxis.set_ticklabels(true_labels, fontsize=15)
    ax.xaxis.tick_top()
    ax.set_ylabel('True', fontsize=20)
    ax.yaxis.set_ticklabels(true_labels, fontsize=15)


def cross_validation_leave_user_out(classifier, X_train, y_train, feat_idx=None, random_num=42, test_size=0.25, show=False):
    """
    This function account for data divided by users, it performs cross validation with leave-one-user-out
    where the test set may contain one user or more, according to the test_size chosen.
    It returns accuracy
    :param classifier: classifier in sklearn format
    :param X_train: type(array), X_train is divided by users, contains the features
    :param y_train: type(array), y_train contains the labels, in the same format as X_train
    :param feat_idx: indexes of specific features
    :param random_num: type(int), it will determine random_state
    :param test_size: type(float), between 0 and 1
    :param show: boolean decision upon confusion matrix plot
    :return: accuracy, y_pred, x_test, y_true
    """
    kf = KFold(n_splits=int(len(X_train)*test_size), random_state=random_num, shuffle=False)
    acc = []
    y_pred, y_true, x_test = [], [], []
    for train_index, test_index in kf.split(X_train):

        Ux_train, Ux_test = np.concatenate(X_train[train_index]), np.concatenate(X_train[test_index])
        Uy_train, Uy_test = np.concatenate(y_train[train_index]), np.concatenate(y_train[test_index])
        if feat_idx is not None:
            Ux_train, Ux_test = Ux_train[:, feat_idx], Ux_test[:,feat_idx]

            if type(feat_idx) == int:
                Ux_train = Ux_train.reshape(-1,1)
                Ux_test = Ux_test.reshape(-1, 1)
        else:
            print('None ' + str(feat_idx))
        try:
            classifier.fit(Ux_train, Uy_train)
        except:
            aaa

        U_pred = classifier.predict(Ux_test)

        acc_ = np.round(accuracy_score(Uy_test, U_pred)*100,2)
        y_pred.append(U_pred)
        y_true.append(Uy_test)
        acc.append(acc_)
        x_test.append(Ux_test)

    if show:
        y_pred = np.concatenate(y_pred)
        y_true = np.concatenate(y_true)
        true_labels = np.unique(y_true)
        plot_confusion_matrix(y_true, y_pred, true_labels)

    return acc


plt.figure(figsize=(18,9))
sns.set(font_scale=2)
i = 1
show = False

# for each classifier we have the list of the features that achieved the best result in 10 repetitions of feature selection

all_best_feats = {'Decision Tree': ['Az_mean', 'Az_abs_dev', 'Ay_var', 'Pitch_max', 'Pitch_median', 'Az_min', 'Pitch_min'],
                  'Random Forest': ['Az_mean', 'Az_abs_dev', 'Ay_var', 'Az_skewness', 'Roll_kurtosis','Pitch_max', 'Ay_kurtosis',
                                    'Pitch_mean','Pitch_median', 'Az_min', 'Roll_max_amp', 'Roll_var', 'Ax_kurtosis','Ax_var', 'Ay_max',
                                    'Roll_mean','Az_min','Pitch_max_amp', 'Yaw_mean', 'Yaw_max_amp'],
                  'ExtraTree':['Az_mean', 'Az_abds_dev', 'Ay_var', 'Az_skewness', 'Roll_kurtosis', 'Pitch_max','Ay_kurtosis', 'Roll_var',
                               'Ax_kurtosis', 'Az_kurtosis', 'Az_min','Pitch_skewness','Yaw_skewness','Pitch_kurtosis','Yaw_max','Yaw_var'],
                  'Logistic Regression':['Az_mean','Az_abs_dev','Ay_var','Az_skewness','Pitch_mean','Pitch_median','Roll_max_amp'],
                  'Bernoulli NB':['Ay_var', 'Roll_var', 'Ax_var'],
                  'Linear Discriminant Analysis':['Az_mean','Az_abs_dev','Pitch_max','Roll_max_amp','Pitch_skewness'],
                  'Quadratic Discriminant Analysis':['Az_mean','Az_abs_dev','Ay_var','Ax_max','Pitch_max_amp'],
                  'MLP':['Az_mean','Az_abs_dev','Ay_var','Az_skewness','Roll_kurtosis','Pitch_max','Ay_kurtosis','Pitch_median','Roll_max_amp',
                         'Ax_max','Roll_var','Ax_kurtosis','Yaw_max','Yaw_min','Pitch_abs_dev','Yaw_kurtosis','Ay_max'],
                  'SGD':['Az_mean','Az_abs_dev','Ay_var','Az_skewness','Roll_kurtosis','Pitch_max','Roll_max_amp','Az_var','Ax_max',
                         'Roll_var','Roll_min','Az_kurtosis','Pitch_min','Yaw_skewness','Yaw_mean','Roll_skewness','Ay_max_amp'],
                  'GradientBoosting':['Az_mean','Az_abs_dev','Ay_var','Roll_kurtosis','Roll_mean','Roll_min','Az_kurtosis'],
                  'Gaussian Process':['Az_mean', 'Az_abs_dev', 'Az_skewness', 'Ax_kurtosis', 'Ay_kurtosis', 'Pitch_max_amp'],
                  'GaussianNB': ['Az_mean', 'Az_abs_dev', 'Az_skewness', 'Ax_kurtosis', 'Pitch_max', 'Yaw_max']
                  }

names = [ "Decision Tree", "Random Forest","GradientBoosting",
          "Linear Discriminant Analysis","Quadratic Discriminant Analysis",
          "MLP", "Bernoulli NB", "Gaussian Process", "GaussianNB",
          "SGD", "Logistic Regression", "ExtraTree"]


classifiers = [
        DecisionTreeClassifier(),
        RandomForestClassifier(),
        GradientBoostingClassifier(),
        LinearDiscriminantAnalysis(),
        QuadraticDiscriminantAnalysis(),
        MLPClassifier(),
        BernoulliNB(),
        GaussianProcessClassifier(),
    GaussianNB(), SGDClassifier(), LogisticRegression(), ExtraTreesClassifier()
]


for n, c in zip(names, classifiers): # run all desired classifiers

    #load data
    # swim_feats = pickle.load(open('data\swim_pitch_feats_pool', 'rb'))
    # swim_labels = pickle.load(open('data\swim_pitch_labels_pool', 'rb'))

    swim_feats, swim_labels = _1_load_data.load_swim_samples()

    # features_names
    feats_names = list(swim_feats[0].columns)

    # get best features for each classifier
    best_feats = all_best_feats[n]

    # get removable indexes
    remove_idx = [idx for idx in range(len(feats_names)) if feats_names[idx] not in best_feats]
    remove_columns = [swim_feats[0].columns[idx] for idx in remove_idx]
    for user in range(len(swim_feats)):
        swim_feats[user] = swim_feats[user].drop(columns=remove_columns) # drop all unwanted features

    # transform from features and labels from dataframes to arrays
    array_swim_feats = []
    array_swim_labels = []
    for sf in range(len(swim_feats)):
        array_swim_feats.append(swim_feats[sf].values)
        array_swim_labels.append(np.array(swim_labels[sf]))
    array_swim_feats = np.array(array_swim_feats)
    array_swim_labels = np.array(array_swim_labels)

    # do cross validation with leave one user out
    acc = cross_validation_leave_user_out(c, array_swim_feats, array_swim_labels, show=False)

    print('Classifier: ' + n + ' with accuracy ' + str(np.round(np.mean(acc),2)) + ' +- ' + str(np.round(np.std(acc),2)))

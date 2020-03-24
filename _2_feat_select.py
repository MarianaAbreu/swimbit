import pickle
from scipy.stats import pearsonr
import pandas as pd
import numpy as np

import _1_load_data

from collections import Counter

from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
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


def remove_redundant(data, type = 'flat'):
    data = np.vstack(data)

    if type == 'flat':
        remove_idx = [idx for idx in range(len(data.T)) if len(np.unique(data[:,idx])) <= 1]
    elif type == 'corr':
        remove_idx = [idx for idx in range(1, len(data.T))
                      if pearsonr(data.T[idx-1],data.T[idx])[0] >= 0.9]

    return remove_idx


def remove_features(swim_feats):

    remove_idx = remove_redundant(swim_feats)
    remove_corr = remove_redundant(swim_feats, 'corr')
    remove_idx = sorted(np.hstack([remove_idx,remove_corr]))

    feats_names = list(swim_feats[0].columns)

    remove_columns = [swim_feats[0].columns[idx] for idx in remove_idx]

    for user in range(len(swim_feats)):
        swim_feats[user] = swim_feats[user].drop(columns=remove_columns)

    feats_names = list(swim_feats[0].columns)
    return swim_feats, feats_names


def leave_one_user_out(classifier, X_train, y_train, feat_idx, random_num):
    """

    :param classifier: sklearn type supervised learning classifier
    :param X_train: array type features
    :param y_train: array type labels
    :param feat_idx: indexes of features to use
    :param random_num: to split kfold randomly
    :return: list of accuracies
    """

    list_idx = np.arange(len(X_train))

    kf = KFold(n_splits=4, random_state=random_num, shuffle=False)
    acc = []

    for train_index, test_index in kf.split(X_train):

        Ux_train, Ux_test = np.concatenate(X_train[train_index]), np.concatenate(X_train[test_index])
        Uy_train, Uy_test = np.concatenate(y_train[train_index]), np.concatenate(y_train[test_index])

        Ux_train, Ux_test = Ux_train[:, feat_idx], Ux_test[:,feat_idx]

        if type(feat_idx) == int:
            Ux_train = Ux_train.reshape(-1,1)
            Ux_test = Ux_test.reshape(-1, 1)

        classifier.fit(Ux_train, Uy_train)

        U_pred = classifier.predict(Ux_test)

        acc_ = np.round(accuracy_score(Uy_test, U_pred)*100,2)
        acc.append(acc_)

    return acc


def fse_leave_one_user_out(X_train, y_train, features_descrition, classifier, random_num):
    """ Performs a sequential forward feature selection.
    Parameters
    ----------
    X_train : array
        Training set feature-vector.

    y_train : array
        Training set class-labels groundtruth.

    features_descrition : array
        Features labels.

    classifier : object
        Classifier.

    Returns
    -------
    FS_idx : array
        Selected set of best features indexes.

    FS_lab : array
        Label of the selected best set of features.

    FS_X_train : array
        Transformed feature-vector with the best feature set.

    References
    ----------
    TSFEL library: https://github.com/fraunhoferportugal/tsfel
    """
    total_acc, total_std, FS_lab, acc_list, acc_std, FS_idx = [], [], [], [], [], []
    X_train = np.array(X_train)

    print("*** Feature selection started ***")
    for feat_idx, feat_name in enumerate(features_descrition):

        cv_result = leave_one_user_out(classifier, X_train, y_train, feat_idx, random_num)
        acc_list.append((np.array(cv_result).prod()**(1.0/len(cv_result))))
        acc_std.append(np.std(cv_result))

    curr_acc_idx = np.argmax(acc_list)
    FS_lab.append(features_descrition[curr_acc_idx])
    last_acc = acc_list[curr_acc_idx]
    total_acc.append(last_acc)
    total_std.append(acc_std[curr_acc_idx])
    FS_idx.append(curr_acc_idx)
    while 1:
        acc_list = []
        print(FS_lab)
        for feat_idx, feat_name in enumerate(features_descrition):
            if feat_name not in FS_lab:
                feats_idx = FS_idx[:]
                feats_idx.append(feat_idx)
                cv_result = leave_one_user_out(classifier, X_train, y_train, feats_idx, random_num)
                acc_list.append(np.array(cv_result).prod()**(1.0/len(cv_result)))
                acc_std.append(np.std(cv_result))

            else:
                acc_list.append(0)
        curr_acc_idx = np.argmax(acc_list)
        if last_acc < acc_list[curr_acc_idx]:
            FS_lab.append(features_descrition[curr_acc_idx])
            last_acc = acc_list[curr_acc_idx]
            total_acc.append(last_acc)
            total_std.append(acc_std[curr_acc_idx])
            FS_idx.append(curr_acc_idx)
        else:
            print("FINAL Features: " + str(FS_lab))
            print("Number of selected features", len(FS_lab))
            print("Features idx: ", FS_idx)
            print("Acc: ", str(total_acc))
            print(curr_acc_idx)
            print('Acc std ', str(total_std))
            print("From ", str(X_train[0].shape[1]), " features to ", str(len(FS_lab)))
            break
    print("*** Feature selection finished ***")
    FS_X_train = []


    return np.array(FS_idx), np.array(FS_lab), np.array([total_acc[-1], total_std[-1]]), FS_X_train


def swim_to_array(swim_feats, swim_labels):
    array_swim_feats = []
    array_swim_labels = []
    for sf in range(len(swim_feats)):
        array_swim_feats.append(swim_feats[sf].values)
        array_swim_labels.append(np.array(swim_labels[sf]))
    array_swim_feats = np.array(array_swim_feats)
    array_swim_labels = np.array(array_swim_labels)
    return array_swim_feats, array_swim_labels


# load data
swim_feats, swim_labels = _1_load_data.load_swim_samples()

# remove redundant features
swim_feats, feats_names = remove_features(swim_feats)

# tranform from dataframes to arrays
array_swim_feats, array_swim_labels = swim_to_array(swim_feats, swim_labels)


def find_best_classifier(X, Y, features_names, classifiers, names_cl, iter_number= 10):
    """
    Finds best features for each classifier
    :param: X: array of users of features
    :param: Y: array of users of labels
    :param: features_names: list of strings with the names of each feature ex - 'Pitch_mean'
    :param classifiers: list of sklearn supervised classifiers
    :param names: list of the classifiers names
    :param iter_number: number of iterations to perform feature selection
    :return:
    """

    c_acc = {}
    for n,classifier in zip(names_cl,classifiers):
        c_acc[n] = {}

        print('--------------------------------------------------------')
        all_feats = []
        all_acc = []
        for ki in range(iter_number):
            idx_feat, name_feat, acc, _ = fse_leave_one_user_out(X, Y, features_names, classifier, ki)
            all_feats.append(name_feat)

            all_acc.append(acc)

        all_feats = Counter(np.concatenate(all_feats))
        c_acc[n]['Mean Score'] = np.mean(all_acc, axis=0) # mean accuracy and standard deviation
        c_acc[n]['Features'] = all_feats
        print('Feature Selection for classifier --' + n)

        print(all_feats)

    return pd.DataFrame.from_dict(c_acc)


all_results = find_best_classifier(array_swim_feats, array_swim_labels, feats_names, classifiers[-2:], names[-2:], 2)

all_results.to_csv('CL_RESULTS')
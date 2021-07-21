# -*- coding: utf-8 -*-
import numpy as np
from sklearn import metrics
from sklearn.metrics import accuracy_score


def select_feature(train, label, size_feature, selected_feature=[]):
    """input is the training matrix(list, 20*Feature_size), training label(list with 0 and 1),
    size of the selected feature(integer from 1 to Feature_size),
    output is the selected feature(list with integer from 1 to Feature_size),
    corresponding AUC(0 to 1), std of training matrix by column(np.array 1*Feature_size),
    mean vector of vG(np.array 1*Feature_size), and threshold of distance(a real number)."""
    train = np.array(train)
    label = np.array(label)
    Feature_size = train.shape[1]
    # normalize data
    std_train = train.std(axis=0)
    mean_train = train.mean(axis = 0)
    norm_train = (train-mean_train) / std_train
    norm_N = norm_train[label == 0, :]
    norm_T = norm_train[label == 1, :]

    # mean vector of vG, i.e. vG center
    center_true = norm_T.mean(axis=0)

    auc_select = []
    # selected_feature = []
    while len(selected_feature) <= size_feature:
        index_auc_pairs = []
        unselected_feature = [i for i in range(Feature_size) if i not in selected_feature]  # Feature subset we choose
        for F in unselected_feature:
            # feature we choose
            temp_feature = selected_feature + [F]
            # choose some features
            N = norm_N[:, temp_feature]
            T = norm_T[:, temp_feature]

            # mean vector of vG, i.e. vG center
            center_true_temp = center_true[temp_feature]

            # distance between painting and vG center (差的二范数)
            dist_N = (((N - center_true_temp) ** 2).sum(axis=1)) ** 0.5
            dist_T = (((T - center_true_temp) ** 2).sum(axis=1)) ** 0.5

            temp_label = np.append(np.ones(T.shape[0]), np.zeros(N.shape[0]))
            dist = np.append(dist_T, dist_N)

            fpr, tpr, thresholds = metrics.roc_curve(temp_label, dist, pos_label=0, drop_intermediate=False)  # mark
            index_auc_pairs += [[F, metrics.auc(fpr, tpr)]]

        index_auc_pairs = sorted(index_auc_pairs, key=lambda x: x[1])
        feature_select = index_auc_pairs[-1][0]
        auc_select = index_auc_pairs[-1][1]
        selected_feature += [feature_select]

    # repeat operation in the loop
    N = norm_N[:, selected_feature]
    T = norm_T[:, selected_feature]
    center_true_temp = center_true[selected_feature]
    dist_N = (((N - center_true_temp) ** 2).sum(axis=1)) ** 0.5
    dist_T = (((T - center_true_temp) ** 2).sum(axis=1)) ** 0.5
    temp_label = np.append(np.ones(T.shape[0]), np.zeros(N.shape[0]))
    dist = np.append(dist_T, dist_N)
    accuracy = []
    for j in range(len(dist)):
        thre = dist[j]
        predict = dist.copy()
        predict[dist > thre] = 0
        predict[dist <= thre] = 1
        accuracy += [accuracy_score(temp_label, predict)]

    tempt = np.argsort(accuracy)
    thres = np.mean(dist[tempt[-2:]])

    return selected_feature, auc_select, mean_train,std_train, center_true, thres


def predict(test, selected_feature, mean_train,std_train, center_true, thres):
    if (type(test[0] == np.float)):
        test = [test]
    test = np.array(test)
    prediction = []
    for i in range(test.shape[0]):
        ite = test[i]
        norm_test = (ite-mean_train) / std_train
        if type(norm_test[0]) == np.float64:
            T = norm_test[selected_feature]
        else:
            T = norm_test[:, selected_feature]
        center_true_temp = center_true[selected_feature]
        dist = (((T - center_true_temp) ** 2).sum(axis=1)) ** 0.5
        prediction += [dist < thres]
    return prediction


def gabor_fn(sigma, theta, Lambda, psi, gamma):
    sigma_x = sigma
    sigma_y = float(sigma) / gamma

    # Bounding box
    nstds = 3  # Number of standard deviation sigma
    xmax = max(abs(nstds * sigma_x * np.cos(theta)), abs(nstds * sigma_y * np.sin(theta)))
    xmax = np.ceil(max(1, xmax))
    ymax = max(abs(nstds * sigma_x * np.sin(theta)), abs(nstds * sigma_y * np.cos(theta)))
    ymax = np.ceil(max(1, ymax))
    xmin = -xmax
    ymin = -ymax
    (y, x) = np.meshgrid(np.arange(ymin, ymax + 1), np.arange(xmin, xmax + 1))

    # Rotation
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)

    gb = np.exp(-.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2)) * np.cos(
        2 * np.pi / Lambda * x_theta + psi)
    return gb


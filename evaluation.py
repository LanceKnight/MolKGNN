import numpy as np
from sklearn.metrics import confusion_matrix, auc, roc_curve, f1_score, \
    roc_auc_score


def sigmoid(x):
    z = 1 / (1 + np.exp(-x))
    return z


def calculate_logAUC(true_y, predicted_score, FPR_range=(0.001, 0.1)):
    """
    Author: Yunchao "Lance" Liu (lanceknight26@gmail.com)
    Calculate logAUC in a certain FPR range (default range: [0.001, 0.1]).
    This was used by previous methods [1] and the reason is that only a
    small percentage of samples can be selected for experimental tests in
    consideration of cost. This means only molecules with very high
    predicted activity values can be worth testing, i.e., the decision
    threshold is high. And the high decision threshold corresponds to the
    left side of the ROC curve, i.e., those FPRs with small values. Also,
    because the threshold cannot be predetermined, the area under the curve
    is used to consolidate all possible thresholds within a certain FPR
    range. Finally, the logarithm is used to bias smaller FPRs. The higher
    the logAUC[0.001, 0.1], the better the performance.

    A perfect classifer gets a logAUC[0.001, 0.1] ) of 1, while a random
    classifer gets a logAUC[0.001, 0.1] ) of around 0.0215 (See [2])

    References:
    [1] Mysinger, M.M. and B.K. Shoichet, Rapid Context-Dependent Ligand
    Desolvation in Molecular Docking. Journal of Chemical Information and
    Modeling, 2010. 50(9): p. 1561-1573.
    [2] Mendenhall, J. and J. Meiler, Improving quantitative
    structure–activity relationship models using Artificial Neural Networks
    trained with dropout. Journal of computer-aided molecular design,
    2016. 30(2): p. 177-189.
    :param true_y: numpy array of the ground truth. Values are either 0 (
    inactive) or 1(active).
    :param predicted_score: numpy array of the predicted score (The
    score does not have to be between 0 and 1)
    :param FPR_range: the range for calculating the logAUC formated in
    (x, y) with x being the lower bound and y being the upper bound
    :return: a numpy array of logAUC of size [1,1]
    """

    np.seterr(divide='ignore')
    # FPR range validity check
    if FPR_range == None:
        raise Exception('FPR range cannot be None')
    lower_bound = FPR_range[0]
    upper_bound = FPR_range[1]
    if (lower_bound >= upper_bound):
        raise Exception('FPR upper_bound must be greater than lower_bound')

    fpr, tpr, thresholds = roc_curve(true_y, predicted_score, pos_label=1)

    tpr = np.append(tpr, np.interp([lower_bound, upper_bound], fpr, tpr))
    fpr = np.append(fpr, [lower_bound, upper_bound])

    # Sort both x-, y-coordinates array
    tpr = np.sort(tpr)
    fpr = np.sort(fpr)

    # Get the data points' coordinates. log_fpr is the x coordinate, tpr is the y coordinate.
    log_fpr = np.log10(fpr)
    x = log_fpr
    y = tpr
    lower_bound = np.log10(lower_bound)
    upper_bound = np.log10(upper_bound)

    # Get the index of the lower and upper bounds
    lower_bound_idx = np.where(x == lower_bound)[-1][-1]
    upper_bound_idx = np.where(x == upper_bound)[-1][-1]

    # Create a new array trimmed at the lower and upper bound
    trim_x = x[lower_bound_idx:upper_bound_idx + 1]
    trim_y = y[lower_bound_idx:upper_bound_idx + 1]

    area = auc(trim_x, trim_y) / (upper_bound-lower_bound)
    return area

def calculate_auc(true_y, predicted_score):
    try:
        res = roc_auc_score(true_y, predicted_score)
    except:
        res = -1
    return res


def calculate_ppv(true_y, predicted_score, cutoff = 0.5):
    '''
    Calculate positve predictive value (PPV)
    :param true_y: numpy array of the ground truth
    :param predicted_score: numpy array of the predicted score (The
    score does not have to be between 0 and 1)
    :return: a numpy array of PPV of size [1,1]
    '''
    predicted_prob = sigmoid(predicted_score) # Convert to range [0,1]
    predicted_y = np.where(predicted_prob > cutoff, 1, 0) # Convert to binary

    tn, fp, fn, tp = confusion_matrix(
        true_y, predicted_y, labels=[0, 1]).ravel()

    if (tp + fp) != 0:
        ppv = (tp / (tp + fp))
    else:
        ppv = np.NAN
    return ppv

def calculate_accuracy(true_y, predicted_score):
    predicted_prob = sigmoid(predicted_score) # Convert to range [0,1]
    predicted_y = np.where(predicted_prob > 0.5, 1, 0) # Convert to binary

    tn, fp, fn, tp = confusion_matrix(
        true_y, predicted_y, labels=[0, 1]).ravel()

    if (tp + fp + tn + fn) != 0:
        accuracy = ((tp + tn) / (tp + fp + tn + fn))
    else:
        accuracy = np.NAN
    return accuracy

def calculate_f1_score(true_y, predicted_score):
    predicted_prob = sigmoid(predicted_score) # Convert to range [0,1]
    predicted_y = np.where(predicted_prob > 0.5, 1, 0) # Convert to binary
    f1_sc = f1_score(true_y, predicted_y)
    return f1_sc

import math
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, auc, roc_curve


def sigmoid(x):
    z = 1 / (1 + np.exp(-x))
    return z


def calculate_logAUC(true_y, predicted_score, FPR_range=(0.001, 0.1)):
    """
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
    :param true_y: numpy array of the ground truth
    :param predicted_score: numpy array of the predicted score (The
    score does not have to be between 0 and 1)
    :param FPR_range: the range for calculating the logAUC formated in
    (x, y) with x being the lower bound and y being the upper bound
    :return: a numpy array of logAUC of size [1,1]
    """

    if FPR_range is not None:
        range1 = np.log10(FPR_range[0])
        range2 = np.log10(FPR_range[1])
        if (range1 >= range2):
            raise Exception('FPR range2 must be greater than range1')
    # print(f'true_y:{true_y}, predicted_score:{predicted_score}')
    fpr, tpr, thresholds = roc_curve(true_y, predicted_score, pos_label=1)
    x = fpr
    y = tpr
    x = np.log10(x)

    y1 = np.append(y, np.interp(range1, x, y))
    y = np.append(y1, np.interp(range2, x, y))
    x = np.append(x, range1)
    x = np.append(x, range2)

    x = np.sort(x)
    # print(f'x:{x}')
    y = np.sort(y)
    # print(f'y:{y}')

    range1_idx = np.where(x == range1)[-1][-1]
    range2_idx = np.where(x == range2)[-1][-1]

    trim_x = x[range1_idx:range2_idx + 1]
    trim_y = y[range1_idx:range2_idx + 1]

    area = auc(trim_x, trim_y) / 2
    return area


def calculate_ppv(true_y, predicted_score):
    '''
    Calculate positve predictive value (PPV)
    :param true_y: numpy array of the ground truth
    :param predicted_score: numpy array of the predicted score (The
    score does not have to be between 0 and 1)
    :return: a numpy array of PPV of size [1,1]
    '''
    predicted_prob = sigmoid(predicted_score) # Convert to range [0,1]
    predicted_y = np.where(predicted_prob > 0.5, 1, 0) # Convert to binary

    tn, fp, fn, tp = confusion_matrix(
        true_y, predicted_y, labels=[0, 1]).ravel()
    print(f'\ntn:{tn}, fp:{fp}, fn:{fn}, tp:{tp}, tp+fp:{tp + fp}')
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
        print(f'\ntn:{tn}, fp:{fp}, fn:{fn}, tp:{tp}, all:'
              f'{tp + fp + tn + fn}, accuracy:{accuracy}')
    else:
        accuracy = np.NAN
        print(f'\ntn:{tn}, fp:{fp}, fn:{fn}, tp:{tp}, all:'
              f'{tp + fp + tn + fn}, accuracy:NAN')
    return accuracy
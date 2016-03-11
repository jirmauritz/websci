import json
import numpy as np
from sklearn import linear_model
from sklearn.cross_validation import KFold
from math import sqrt, inf


DATA_DIR = '../data/'
MAX_ALPHA = 700
MAX_ITER = 7**5

def test():
    """
    Computes RMSE for various linear regression models.
    """
    # load matrix
    hpv_matrix = np.loadtxt(DATA_DIR + 'hpv_data.txt', dtype=(int), delimiter=',')
    mfr_matrix = np.loadtxt(DATA_DIR + 'mfr_data.txt', dtype=(int), delimiter=',')
    # load clinical data
    hpv_clinical_data = np.array(json.load(open(DATA_DIR + 'clinical/HPV-1.json'))[1])
    mfr_clinical_data = np.array(json.load(open(DATA_DIR + 'clinical/MMR-1.json'))[1])
    # avg baseline
    avg(hpv_clinical_data, 'HPV')
    avg(mfr_clinical_data, 'MFR')
    # ordinary least squares
    ols(hpv_matrix, hpv_clinical_data, 'HPV')
    ols(mfr_matrix, mfr_clinical_data, 'MFR')
    # lasso
    lasso(hpv_matrix, hpv_clinical_data, 'HPV')
    lasso(mfr_matrix, mfr_clinical_data, 'MFR')

def avg(clinical_data, name):
    """
    Baseline for testing, predict average value for any input.
    """
    avg = np.mean(clinical_data)
    print('method: AVG, data: %s, RMSE: %.2f' %
            (name, RMSE([avg]*len(clinical_data), clinical_data)))

    print()

def ols(matrix, clinical_data, name):
    """
    Testing with ordinary least squares model.
    """
    clf = linear_model.LinearRegression()
    clf.fit(matrix, clinical_data)
    # rmse on training data just for check
    predicted = clf.predict(matrix)
    print('method: OLS, data: %s training, RMSE: %.2f' %
            (name, round(RMSE(predicted, clinical_data), 2)))
    # cross validation
    kf = KFold(len(clinical_data), n_folds=5, shuffle=True)
    summ = 0
    for train, test in kf:
        clf.fit(matrix[train], clinical_data[train])
        predicted = clf.predict(matrix[test])
        summ += RMSE(predicted, clinical_data[test])
    print('method: OLS, data: %s testing, RMSE: %.2f' %
        (name, round(summ/len(kf), 2)))
    print()

def lasso(matrix, clinical_data, name):
    """
    Testing with LASSO model. Optimizing parameter alpha.
    """
    kf = KFold(len(clinical_data), n_folds=5, shuffle=True)
    # optimizing parameter alpha
    alpha = 1
    best_alpha = 1
    best_rmse = inf
    for alpha in range(1,MAX_ALPHA):
        clf = linear_model.Lasso(alpha=alpha, max_iter=MAX_ITER)
        summ = 0
        for train, test in kf:
            clf.fit(matrix[train], clinical_data[train])
            predicted = clf.predict(matrix[test])
            summ += RMSE(predicted, clinical_data[test])
        if summ/len(kf) < best_rmse:
            best_rmse = summ/len(kf)
            best_alpha = alpha
    # training data
    clf = linear_model.Lasso(alpha=best_alpha, max_iter=MAX_ITER)
    clf.fit(matrix, clinical_data)
    predicted = clf.predict(matrix)
    print('method: LASSO, data: %s training, RMSE: %.2f, for alpha=%i' %
            (name, round(RMSE(predicted, clinical_data), 2), round(best_alpha, 2)))
    # testing data
    print('method: LASSO, data: %s testing, RMSE: %.2f for alpha=%i' %
            (name, round(best_rmse, 2), round(best_alpha, 2)))
    print()


# --------- help functions --------------

def RMSE(predicted, original):
    summ = 0
    for p, o in zip(predicted, original):
        summ += (p - o)**2
    return sqrt(summ/len(predicted))



if __name__ == '__main__':
    test()

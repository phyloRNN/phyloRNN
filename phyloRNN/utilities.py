import sys
import os
import copy
import numpy as np
import pickle
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends import backend_pdf  # saves pdfs
import pandas as pd
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Align import MultipleSeqAlignment
import scipy.stats
from numpy.random import MT19937, RandomState, SeedSequence

def get_rnd_gen(seed=None):
    return RandomState(MT19937(SeedSequence(seed)))

def print_update(s):
    sys.stdout.write('\r')
    sys.stdout.write(s)
    sys.stdout.flush()

def unique_unsorted(a_tmp):
    a = copy.deepcopy(a_tmp)
    indx = np.sort(np.unique(a, return_index=True)[1])
    u = a_tmp[indx]
    return u

def load_pkl(file_name):
    try:
        with open(file_name, 'rb') as f:
            return pickle.load(f)
    except:
        print("Didn't work!")
        # import pickle5
        # with open(file_name, 'rb') as f:
        #     return pickle.load(f)

def save_pkl(obj, out_file):
    with open(out_file, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


"""
functions from: https://github.com/jmenglund/pandas-charm
"""

def frame_as_categorical(frame, include_categories=None):
    """
    Return a pandas DataFrame with each column treated as a
    categorical with unordered categories. The same categories
    are applied to all columns.
    Parameters
    ----------
    frame : pandas.DataFrame
    include_categories : list (default: None)
        Categories to add unless they are already present
        in `frame`.
    """
    include_categories = include_categories if include_categories else []
    current_categories = pd.unique(frame.values.ravel())
    current_categories_notnull = (
        current_categories[pd.notnull(current_categories)])
    categories = set(current_categories_notnull).union(include_categories)
    categorical = frame.apply(lambda x: pd.Series(x.astype('category')))
    unified_categorical = categorical.apply(
        lambda x: x.cat.set_categories(new_categories=categories))
    return unified_categorical


def df_from_charmatrix(charmatrix, categorical=True):
    """
    Convert a DendroPy CharacterMatrix to a pandas DataFrame.
    Parameters
    ----------
    charmatrix : dendropy.CharacterMatrix
    categorical : bool (default: True)
        If True, the result will be returned as a categorical frame.
    """
    frame = pd.DataFrame()
    for taxon, seq in charmatrix.items():
        s = pd.Series(
            seq.symbols_as_list(), name=taxon.label)
        frame = pd.concat([frame, s], axis=1)
    if categorical:
        state_alphabet = charmatrix.state_alphabets[0].symbols
        new_frame = frame_as_categorical(
            frame, include_categories=state_alphabet)
    else:
        new_frame = frame
    return new_frame.transpose()


def biopython_msa_from_charmatrix(charmatrix):
    s = charmatrix.as_string('phylip').split()
    l = []
    for i in range(2, len(s), 2):  # skip first 2 rows
        l.append(SeqRecord(Seq(s[i + 1]), id=s[i]))

    aln = MultipleSeqAlignment(l)
    return aln


def calc_confusion_matrix(y,lab):
    prediction = np.argmax(y, axis=1)
    y_actu = pd.Categorical(lab, categories=np.unique(lab))
    y_pred = pd.Categorical(prediction, categories=np.unique(lab))
    df_confusion = pd.crosstab(y_actu, y_pred, margins=False, rownames=['True'], colnames=['Predicted'],dropna=False)
    return df_confusion

def calc_accuracy(y,lab):
    if len(y.shape) == 3: # if the posterior softmax array is used, return array of accuracies
        acc = np.array([np.sum(i==lab)/len(i) for i in np.argmax(y,axis=2)])
    else:
        prediction = np.argmax(y, axis=1)
        acc = np.sum(prediction==lab)/len(prediction)
    return acc

def calc_label_accuracy(y,lab):
    prediction = np.argmax(y, axis=1)
    label_accs = []
    for label in np.unique(lab):
        cat_lab = lab[lab==label]
        cat_prediction = prediction[lab==label]
        acc = np.sum(cat_prediction==cat_lab)/len(cat_prediction)
        label_accs.append(acc)
    return np.array(label_accs)


def get_r2(x, y, return_mean=False, return_median=True, indx=None):
    if indx is None:
        R = range(len(x))
    else:
        R = indx
    if len(x.shape) == 1:
        if indx is not None:
            x = x[indx]
            y = y[indx]
        x = 0 + x.reshape((1, x.shape[0]))
        y = 0 + y.reshape((1, y.shape[0]))
        R = [0]

    r2s = []
    for i in R:
        x_i = x[i]
        y_i = y[i]
        try:
            _, _, r_value, _, _ = scipy.stats.linregress(x_i[np.isfinite(x_i)], y_i[np.isfinite(x_i)])
            r2s.append(r_value ** 2)
        except:
            pass

    if return_median:
        return np.nanmedian(r2s)
    elif return_mean:
        return np.nanmean(r2s)
    else:
        return np.array(r2s)

def get_mse(x, y, return_mean=True, return_median=False, indx=None):
    if indx is None:
        se = (x - y)**2
    else:
        se = (x[indx] - y[indx]) ** 2

    if return_median:
        return np.nanmedian(se)
    elif return_mean:
        return np.nanmean(se)
    else:
        return np.array(se)

def get_nrmse(x, y,
              return_mean=True,
              return_median=False,
              range_normalized=True,
              mean_normalized=False,
              indx=None):
    if indx is None:
        R = range(len(x))
    else:
        R = indx
    if len(x.shape) == 1:
        if indx is not None:
            x = x[indx]
            y = y[indx]
        x = 0 + x.reshape((1, x.shape[0]))
        y = 0 + y.reshape((1, y.shape[0]))
        R = [0]

    mse = []
    val_range = []
    val_mean = []
    for i in R:
        mse.append(np.nanmean((x[i] - y[i])**2))
        val_range.append( np.max(y[i]) - np.min(y[i]) )
        val_mean.append(np.mean(y[i]))

    mse = np.array(mse)
    rmse = np.sqrt(mse)
    r_nrmse = rmse / np.array(val_range)
    m_rmse = rmse / np.array(val_mean)

    if range_normalized:
        res = r_nrmse
    elif mean_normalized:
        res = m_rmse
    else:
        res = rmse

    if return_median:
        return np.nanmedian(res)
    elif return_mean:
        return np.nanmean(res)
    else:
        return res


def get_avg_r2(Ytrue, Ypred):
    r2 = []
    if len(Ypred.shape) == 3:
        Ypred = Ypred[:, :, 0]

    for i in range(Ytrue.shape[0]):
        x = Ytrue[i]
        y = Ypred[i, :]
        r2.append(get_r2(x[x > 0], y[x > 0]))
    res = {'mean r2': np.nanmean(r2),   # change to nanmean? check if this works for sqs data.
           'min r2': np.nanmin(r2),
           'max r2': np.nanmax(r2),
           'std r2': np.nanstd(r2)}
    return res


def print_msa_compare(mle_fr, mle_g, dle, true, indx=None, digits=3):
    g = [round(get_mse(mle_g, true, indx=indx), digits), round(get_r2(mle_g, true, indx=indx), digits)]
    print("Gamma Rate MSE:", g[0], "R2:", g[1])
    fr = [round(get_mse(mle_fr, true, indx=indx), digits), round(get_r2(mle_fr, true, indx=indx), digits)]
    print("Free Rate MSE:", fr[0], "R2:", fr[1])
    dl = [round(get_mse(dle, true, indx=indx), digits), round(get_r2(dle, true, indx=indx), digits)]
    print("RNN Rate MSE:", dl[0], "R2:", dl[1])
    res = {
        'mse': [g[0], fr[0], dl[0]],
        'R2': [g[1], fr[1], dl[1]]
    }
    return res
    #
    #
    #
    #
    # print("Gamma Rate MSE:", round(get_mse(mle_g[np.isfinite(mle_g)], true[np.isfinite(mle_g)], indx=indx), digits),
    #       "R2:", round(get_r2(mle_g[np.isfinite(mle_g)], true[np.isfinite(mle_g)], indx=indx), digits))
    # print("Free Rate MSE:", round(get_mse(mle_fr[np.isfinite(mle_fr)], true[np.isfinite(mle_fr)], indx=indx), digits),
    #       "R2:", round(get_r2(mle_fr[np.isfinite(mle_fr)], true[np.isfinite(mle_fr)], indx=indx), digits))
    # print("RNN Rate MSE:", round(get_mse(dle, true, indx=indx), digits),
    #       "R2:", round(get_r2(dle, true, indx=indx), digits))


def get_mape(x, y, return_mean=True, return_median=False, indx=None):
    y[y == 0] = 1e-10
    if indx is None:
        se = np.abs(x - y) / y
    else:
        se = np.abs(x[indx] - y[indx]) / y[indx]

    if return_median:
        return np.nanmedian(se)
    elif return_mean:
        return np.nanmean(se)
    else:
        return np.array(se)
def print_mape_compare(mle_fr, mle_g, dle, true, indx=None, digits=3):
    g = [round(get_mape(mle_g, true, indx=indx), digits), round(get_r2(mle_g, true, indx=indx), digits)]
    print("Gamma Rate MAPE:", g[0], "R2:", g[1])
    fr = [round(get_mape(mle_fr, true, indx=indx), digits), round(get_r2(mle_fr, true, indx=indx), digits)]
    print("Free Rate MAPE:", fr[0], "R2:", fr[1])
    dl = [round(get_mape(dle, true, indx=indx), digits), round(get_r2(dle, true, indx=indx), digits)]
    print("RNN Rate MAPE:", dl[0], "R2:", dl[1])
    res = {
        'mape': [g[0], fr[0], dl[0]],
        'R2': [g[1], fr[1], dl[1]]
    }
    return res


def calcCI(data, level=0.95):
    assert (0 < level < 1)
    d = list(data)
    d.sort()
    nData = len(data)
    nIn = int(round(level * nData))
    if nIn < 2 :
        sys.exit('\n\nToo little data to calculate marginal parameters.')
    i = 0
    r = d[i+nIn-1] - d[i]
    for k in range(len(d) - (nIn - 1)):
            rk = d[k+nIn-1] - d[k]
            if rk < r :
                r = rk
                i = k
    assert 0 <= i <= i+nIn-1 < len(d)
    return (d[i], d[i+nIn-1])

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
from .utilities import get_r2, get_mse
import seaborn as sn

def plot_confusion_matrix(cm, show=True, wd='', filename=""):
    fig = plt.figure(figsize = (10,7))
    sn.heatmap(cm, annot=True, fmt='d')
    if show:
        plt.show()
    else:
        file_name = os.path.join(wd, 'confusion_matrix' + filename + '.pdf')
        plot_div = matplotlib.backends.backend_pdf.PdfPages(file_name)
        fig.subplots_adjust(top=0.92)
        plot_div.savefig(fig)
        plot_div.close()
        print("Plot saved as:", file_name)



def plot_training_history(history, criterion='val_loss', b=0,
                          show=True, wd='', filename="", digits=3,
                          log=True):
    try:
        h = history.history
    except:
        h = history

    stopping_point = np.argmin(h[criterion])
    fig = plt.figure(figsize=(10, 5))
    if len(h.keys()) == 2:
        plt.plot(h['loss'][b:], label='Training loss (%s)' % np.round(h['loss'][stopping_point], digits))
        plt.plot(h['val_loss'][b:], label='Validation loss (%s)' % np.round(h['val_loss'][stopping_point], digits))
    else:
        out_dict = {'epochs': stopping_point}
        for i in h.keys():
            if log:
                plt.plot(np.log(h[i][b:]), label='%s (%s)' % (i, np.round(h[i][stopping_point], digits)))
            else:
                plt.plot(h[i][b:], label='%s (%s)' % (i, np.round(h[i][stopping_point], digits)))
            out_dict[i] = h[i][stopping_point]
    plt.axvline(stopping_point, linestyle='--', color='red', label='Early stopping point')
    plt.grid(axis='y', linestyle='dashed', which='major', zorder=0)
    plt.xlabel('Training epoch')
    if log:
        plt.ylabel('Loss (Log)')
    else:
        plt.ylabel('Loss')

    plt.legend() #loc='upper right')
    plt.tight_layout()
    if show:
        fig.show()
    else:
        file_name = os.path.join(wd, 'history_plot' + filename + '.pdf')
        plot_div = matplotlib.backends.backend_pdf.PdfPages(file_name)
        fig.subplots_adjust(top=0.92)
        plot_div.savefig(fig)
        plot_div.close()
        print("Plot saved as:", file_name)
    return out_dict



def plot_rate_prediction(feature_set=None, label_set=None, model=None,
                         true_lab=None, pred_lab=None,
                         log=True,
                         rescale=1,
                         n_sim_plot = 20,
                         alpha = 0.05,
                         rate_indices= None,
                         sim_indx=None,
                         rm_outliers = True,
                         show=True, wd='', filename="",
                         title=None,
                         save_pdf=False):

    if model is not None:
        n_sims = feature_set.shape[0]
        if sim_indx is None:
            sim_indx = np.random.randint(0, n_sims, n_sim_plot)
        pred = model.predict(feature_set[sim_indx])
        labels_flat = label_set[sim_indx].flatten()
        pred_flat = pred.flatten()

    else:
        n_sims = true_lab.shape[0]
        if sim_indx is None:
            sim_indx = np.random.randint(0, n_sims, n_sim_plot)
        labels_flat = true_lab[sim_indx].flatten()
        pred_flat = pred_lab[sim_indx].flatten()

    print(sim_indx)

    if rate_indices is None:
        rate_indices = np.random.randint(0,n_sims,3)

    if rm_outliers:
        tmp = np.sort(labels_flat)
        x_min = tmp[int(0.025 * (len(pred_flat) - 1))]
        x_max = tmp[int(0.975 * (len(pred_flat) - 1))]
        indx = np.intersect1d(np.where(pred_flat > x_min)[0],
                              np.where(pred_flat < x_max)[0])
        pred_flat = 0 + pred_flat[indx]
        labels_flat = 0 + labels_flat[indx]

    fig = plt.figure(figsize=(14, 9))


    if log:
        y_true = np.log(labels_flat)
        y_pred = np.log(pred_flat * rescale)
        fig.add_subplot(231)
        plt.plot(y_true, y_true, color="red")
        plt.scatter(y_true, y_pred, alpha=alpha)
        plt.xlabel('True per-site rate (log)')
        plt.ylabel('Estimated per-site rate (log)')
        m = np.min([np.min(y_pred), np.min(y_true)])
        M = np.max([np.max(y_pred), np.max(y_true)])
        plt.xlim(m - 1, 1 + M)
        plt.ylim(m - 1, 1 + M)
        if title is None:
            title = "MSE: %s" % np.round(np.mean((y_true - y_pred) ** 2), 3)
        plt.gca().set_title(title, fontweight="bold", fontsize=12)
        fig.add_subplot(232)
        plt.hist(np.exp(y_true),
                 bins=np.linspace(0, np.max(np.exp(y_true))))
        plt.yticks([])
        plt.xlabel('True per-site rate')
        plt.ylabel('Frequency')
        fig.add_subplot(233)
        plt.hist(np.exp(y_pred),
                 bins=np.linspace(0, np.max(np.exp(y_true))))
        plt.yticks([])
        plt.xlabel('Estimated per-site rate')
        plt.ylabel('Frequency')

    else:
        y_true = labels_flat
        y_pred = pred_flat
        fig.add_subplot(231)
        plt.plot(y_true, y_true, color="red")
        plt.scatter(y_true, y_pred, alpha=alpha)
        plt.xlabel('True per-site rate')
        plt.ylabel('Estimated per-site rate')
        m = np.min([np.nanmin(y_pred), np.nanmin(y_true)])
        M = np.max([np.nanmax(y_pred), np.nanmax(y_true)])
        plt.xlim(m - 1, 1 + M)
        plt.ylim(m - 1, 1 + M)
        if title is None:
            title = "MSE: %s" % np.round(np.mean((y_true - y_pred) ** 2), 3)
        plt.gca().set_title(title, fontweight="bold", fontsize=12)
        fig.add_subplot(232)
        plt.hist(labels_flat, bins=np.linspace(0, np.max(labels_flat)))
        plt.yticks([])
        plt.xlabel('True per-site rate')
        plt.ylabel('Frequency')
        fig.add_subplot(233)
        plt.hist(pred_flat, bins=np.linspace(0, np.max(labels_flat)))
        plt.yticks([])
        plt.xlabel('Estimated per-site rate')
        plt.ylabel('Frequency')

    if len(rate_indices):
        fig_indx = [234, 235, 236]
        for i in range(len(rate_indices)):
            fig.add_subplot(fig_indx[i])
            indx = rate_indices[i]
            ttl = "MSE: %s, R2: %s" % (round(get_mse(true_lab[indx, :], pred_lab[indx, :]), 2),
                                       round(get_r2(true_lab[indx, :], pred_lab[indx, :]), 2))
            plt.plot(pred_lab[indx, :], label='Predicted rates', color="#969696")
            plt.plot(true_lab[indx, :], label='True rates',color="#d7191c", linewidth=1, linestyle='dashed')
            plt.xlabel('Site (sim: %s)' % indx)
            plt.ylabel('Rate')
            plt.gca().set_title(ttl, fontsize=10) #, fontweight="bold", fontsize=12)
            plt.legend()

    if show:
        fig.show()
    else:
        # fig.subplots_adjust(top=0.95)
        if save_pdf:
            file_name = os.path.join(wd, 'predictions' + filename + '.pdf')
            plot_div = matplotlib.backends.backend_pdf.PdfPages(file_name)
            plot_div.savefig(fig)
            plot_div.close()
        else:
            file_name = os.path.join(wd, 'predictions' + filename + '.png')
            plt.savefig(file_name)
        print("Plot saved as:", file_name)


def plot_prediction(feature_set=None, label_set=None, model=None,
                    true_lab=None, pred_lab=None,
                    log=True,
                    rescale=1,
                    n_sim_plot = 20,
                    alpha = 0.05,
                    rate_indices= None,
                    rm_outliers = True,
                    show=True, wd='', filename="predictions",
                    save_pdf=False):

    if model is not None:
        pred = model.predict(feature_set[-n_sim_plot:])
        labels_flat = label_set[-n_sim_plot:].flatten()
        pred_flat = pred.flatten()

    else:
        labels_flat = true_lab[-n_sim_plot:].flatten()
        pred_flat = pred_lab[-n_sim_plot:].flatten()

    if rate_indices is None:
        rate_indices = [0,1,2]

    if rm_outliers:
        tmp = np.sort(labels_flat)
        x_min = tmp[int(0.01 * (len(labels_flat) - 1))]
        x_max = tmp[int(0.99 * (len(labels_flat) - 1))]
        indx = np.intersect1d(np.where(labels_flat > x_min)[0],
                              np.where(labels_flat < x_max)[0])
        pred_flat = pred_flat[indx]
        labels_flat = labels_flat[indx]

    fig = plt.figure(figsize=(14, 8))

    if log:
        y_true = np.log(labels_flat)
        y_pred = pred_flat * rescale
        fig.add_subplot(231)
        plt.plot(y_true, y_true, color="red")
        plt.scatter(y_true, y_pred, alpha=alpha)
        plt.xlabel('True per-site rate (log)')
        plt.ylabel('Estimated per-site rate (log)')
        m = np.min([np.min(y_pred), np.min(y_true)])
        M = np.max([np.max(y_pred), np.max(y_true)])
        plt.xlim(m - 1, 1 + M)
        plt.ylim(m - 1, 1 + M)
        fig.add_subplot(232)
        plt.hist(np.exp(y_true),
                 bins=np.linspace(0, np.max(np.exp(y_true))))
        plt.yticks([])
        plt.xlabel('True per-site rate')
        plt.ylabel('Frequency')
        fig.add_subplot(233)
        plt.hist(np.exp(y_pred),
                 bins=np.linspace(0, np.max(np.exp(y_true))))
        plt.yticks([])
        plt.xlabel('Estimated per-site rate')
        plt.ylabel('Frequency')

    else:
        y_true = np.log(labels_flat)
        y_pred = np.log(pred_flat)
        fig.add_subplot(231)
        plt.plot(y_true, y_true, color="red")
        plt.scatter(y_true, y_pred, alpha=alpha)
        plt.xlabel('True per-site rate')
        plt.ylabel('Estimated per-site rate')
        m = np.min([np.min(y_pred), np.min(y_true)])
        M = np.max([np.max(y_pred), np.max(y_true)])
        plt.xlim(m - 1, 1 + M)
        plt.ylim(m - 1, 1 + M)
        fig.add_subplot(232)
        plt.hist(labels_flat, bins=np.linspace(0, np.max(labels_flat)))
        plt.yticks([])
        plt.xlabel('True per-site rate')
        plt.ylabel('Frequency')
        fig.add_subplot(233)
        plt.hist(pred_flat, bins=np.linspace(0, np.max(labels_flat)))
        plt.yticks([])
        plt.xlabel('Estimated per-site rate')
        plt.ylabel('Frequency')

    if len(rate_indices):
        fig_indx = [234, 235, 236]
        for i in range(len(rate_indices)):
            fig.add_subplot(fig_indx[i])
            indx = rate_indices[i]
            plt.plot(pred_lab[indx, :], label='Predicted rates', color="#969696")
            plt.plot(true_lab[indx, :], label='True rates',color="#d7191c", linewidth=2)
            plt.xlabel('Site')
            plt.ylabel('Rate')
            plt.legend()

    if show:
        fig.show()
    else:
        fig.subplots_adjust(top=0.92)
        if save_pdf:
            file_name = os.path.join(wd, filename + '.pdf')
            plot_div = matplotlib.backends.backend_pdf.PdfPages(file_name)
            plot_div.savefig(fig)
            plot_div.close()
        else:
            file_name = os.path.join(wd, filename + '.png')
            plt.savefig(file_name)
        print("Plot saved as:", file_name)

"""
functions from: https://github.com/jmenglund/pandas-charm
"""



# pn.get_r2(mle_fr, y_true)

def plot_sample_rate_results(dle, mle_g, mle_fr, y_true,
                             n_sim_plot,
                             sim_indx,
                             rate_indices=None,
                             show=False, wd="", outname="res",
                             plot_log_rates=False,
                             rm_outliers=False,
                             alpha=0.005,
                             mse_values=None):

    if mse_values is None:
        dle_mse = np.round(np.mean((dle - y_true) ** 2), 2)
        fr_mse = np.round(np.mean((mle_fr - y_true) ** 2), 2)
        g_mse = np.round(np.mean((mle_g - y_true) ** 2), 2)
    else:
        (g_mse, fr_mse, dle_mse) = mse_values


    plot_rate_prediction(true_lab=y_true,
                            pred_lab=dle,
                            log=plot_log_rates,
                            n_sim_plot=n_sim_plot,
                            sim_indx=sim_indx,
                            alpha=alpha,
                            rate_indices=rate_indices,
                            wd=wd, show=show, rm_outliers=rm_outliers,
                            filename=outname + "_dle",
                            title="RNN Rate model - MSE: %s" % dle_mse)
    
    plot_rate_prediction(true_lab=y_true,
                            pred_lab=mle_fr,
                            log=plot_log_rates,
                            n_sim_plot=n_sim_plot,
                            sim_indx=sim_indx,
                            alpha=alpha,
                            rate_indices=rate_indices,
                            wd=wd, show=show, rm_outliers=rm_outliers,
                            filename=outname + "_mle_fr",
                            title="Free Rate model - MSE: %s" % fr_mse)
    
    plot_rate_prediction(true_lab=y_true,
                            pred_lab=mle_g,
                            log=plot_log_rates,
                            n_sim_plot=n_sim_plot,
                            sim_indx=sim_indx,
                            alpha=alpha,
                            rate_indices=rate_indices,
                            wd=wd, show=show, rm_outliers=rm_outliers,
                            filename=outname + "_mle_g",
                            title="Gamma Rate model - MSE: %s" % g_mse)


def plot_treelength(mle_fr, mle_g, dle, y_true,
                    indx=None, alpha = 1,
                    show=True, wd='', outname="tl_predictions"):
    if indx is not None:
        dle = dle [indx]
        mle_fr = mle_fr[indx]
        mle_g = mle_g[indx]
        y_true = y_true[indx]


    m = np.min([np.nanmin(mle_g), np.nanmin(y_true)])
    M = np.max([np.nanmax(mle_g), np.nanmax(y_true)])

    fig = plt.figure(figsize=(14, 8))

    fig.add_subplot(231)
    plt.plot(y_true, y_true, color="red")
    plt.scatter(y_true, mle_g, alpha=alpha)
    plt.xlabel('True tree length')
    plt.ylabel('Estimated tree length')
    plt.gca().set_title("Gamma rate model", fontweight="bold", fontsize=12)
    plt.xlim(m - 1, 1 + M)
    plt.ylim(m - 1, 1 + M)

    fig.add_subplot(232)
    plt.plot(y_true, y_true, color="red")
    plt.scatter(y_true, mle_fr, alpha=alpha)
    plt.xlabel('True tree length')
    plt.ylabel('Estimated tree length')
    plt.gca().set_title("Free rate model", fontweight="bold", fontsize=12)
    plt.xlim(m - 1, 1 + M)
    plt.ylim(m - 1, 1 + M)

    fig.add_subplot(233)
    plt.plot(y_true, y_true, color="red")
    plt.scatter(y_true, dle, alpha=alpha)
    plt.xlabel('True tree length')
    plt.ylabel('Estimated tree length')
    plt.gca().set_title("RNN model", fontweight="bold", fontsize=12)
    plt.xlim(m - 1, 1 + M)
    plt.ylim(m - 1, 1 + M)

    # log space
    fig.add_subplot(234)
    y_true = np.log(y_true)
    mle_g = np.log(mle_g)
    mle_fr = np.log(mle_fr)
    dle = np.log(dle)
    m = np.log(m)
    M = np.log(M)

    plt.plot(y_true, y_true, color="red")
    plt.scatter(y_true, mle_g, alpha=alpha)
    plt.xlabel('True tree length')
    plt.ylabel('Estimated tree length')
    plt.gca().set_title("Gamma rate model", fontweight="bold", fontsize=12)
    plt.xlim(m - .1, .1 + M)
    plt.ylim(m - .1, .1 + M)

    fig.add_subplot(235)
    plt.plot(y_true, y_true, color="red")
    plt.scatter(y_true, mle_fr, alpha=alpha)
    plt.xlabel('True tree length')
    plt.ylabel('Estimated tree length')
    plt.gca().set_title("Free rate model", fontweight="bold", fontsize=12)
    plt.xlim(m - .1, .1 + M)
    plt.ylim(m - .1, .1 + M)

    fig.add_subplot(236)
    plt.plot(y_true, y_true, color="red")
    plt.scatter(y_true, dle, alpha=alpha)
    plt.xlabel('True tree length')
    plt.ylabel('Estimated tree length')
    plt.gca().set_title("RNN model", fontweight="bold", fontsize=12)
    plt.xlim(m - .1, .1 + M)
    plt.ylim(m - .1, .1 + M)

    if show:
        fig.show()
    else:
        file_name = os.path.join(wd, outname + '.pdf')
        plot_div = matplotlib.backends.backend_pdf.PdfPages(file_name)
        fig.subplots_adjust(top=0.92)
        plot_div.savefig(fig)
        plot_div.close()
        print("Plot saved as:", file_name)





















import os
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


# load pickle file function
def load_pickle(path):
    """
    input
    path   : str
             path to input data
    return
    output : ndarray shape(100, 100)
             output data
    a      : ndarray shape(5, 3, 3) or (5, 5, 5)
             transition probability matrix
    b      : ndarray shape(5, 3, 5) or (5, 5, 5)
             output probability
    pi     : ndarray shape(5, 3) or (5, 5)
             initial probability
    ans    : ndarray shape(100, )
             answer models
    """

    with open(path, 'rb') as f:
        data = pickle.load(f)

    output = np.array(data["output"])
    a = np.array(data["models"]["A"])
    b = np.array(data["models"]["B"])
    pi = np.array(data["models"]["PI"]).squeeze()
    ans = np.array(data["answer_models"])

    return output, a, b, pi, ans


# forward and viterbi algorithms
def predict(output, a, b, pi):
    """
    input
    output : ndarray shape(100, 100)
             output data
    a      : ndarray shape(5, 3, 3) or (5, 5, 5)
             transition probability matrix
    b      : ndarray shape(5, 3, 5) or (5, 5, 5)
             output probability
    pi     : ndarray shape(5, 3) or (5, 5)
             initial probability
    return
    pred_f : ndarray shape(100, )
             predicted models by forward algorithm
    pred_v : ndarray shape(100, )
             predicted models by viterbi algorithm
    """

    n, tr = output.shape
    pred_f = np.zeros(n)
    pred_v = np.zeros(n)

    for i in range(n):
        alpha_f = pi * b[:, :, output[i, 0]]
        alpha_v = pi * b[:, :, output[i, 0]]
        for j in range(1, tr):
            alpha_f = b[:, :, output[i, j]] * np.sum(a.T * alpha_f.T, axis=1).T
            alpha_v = b[:, :, output[i, j]] * np.max(a.T * alpha_v.T, axis=1).T
        pred_f[i] = np.argmax(np.sum(alpha_f, axis=1))
        pred_v[i] = np.argmax(np.max(alpha_v, axis=1))

    return pred_f, pred_v


# plot confusion matrix
def plot_cm(ans, pred_f, pred_v, path):
    """
    input
    ans    : ndarray shape(100, )
             answer models
    pred_f : ndarray shape(100, )
             predicted models by forward algorithm
    pred_v : ndarray shape(100, )
             predicted models by viterbi algorithm
    path   : str
             use to get figure name to save
    """

    plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
    plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True

    # calc confusion matrix
    labels = list(set(ans))
    cm_f = confusion_matrix(ans, pred_f, labels)
    cm_v = confusion_matrix(ans, pred_v, labels)
    acc_f = np.sum(ans == pred_f) / len(ans)
    acc_v = np.sum(ans == pred_v) / len(ans)

    # plot
    fig, ax = plt.subplots(1, 2, figsize=(8, 4), tight_layout=True)
    ax[0].imshow(cm_f, interpolation="nearest", cmap='binary')
    ax[1].imshow(cm_v, interpolation="nearest", cmap='binary')
    for i in range(cm_f.shape[1]):
        for j in range(cm_f.shape[0]):
            if cm_f[i, j] > 9:
                cf = 'white'
            else:
                cf = 'black'
            if cm_v[i, j] > 9:
                cv = 'white'
            else:
                cv = 'black'
            ax[0].text(x=j, y=i, s=cm_f[i, j], va="center", ha="center", color=cf)
            ax[1].text(x=j, y=i, s=cm_v[i, j], va="center", ha="center", color=cv)

    ax[0].set_title(f'Forward algorithm\n(Acc. {acc_f:.1%})\nPredicted model', fontsize=16)
    ax[0].set_ylabel("Actual model", fontsize=16)
    ax[0].tick_params(top=False, left=False)

    ax[1].set_title(f'Viterbi algorithm\n(Acc. {acc_v:.1%})\nPredicted model', fontsize=16)
    ax[1].set_ylabel("Actual model", fontsize=16)
    ax[1].tick_params(top=False, left=False)

    # show and save fig
    plt.show(block=True)
    name = path.split('/')[1].split('.')[0]
    fig.savefig(f"./out/{name}.png")


def main():
    # argument setting
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help='テストデータの正解ファイルCSVのパス')
    args = parser.parse_args()

    # check output dir exist
    if not os.path.exists('./out'):
        os.makedirs('./out')

    # load data
    output, a, b, pi, ans = load_pickle(args.input)
    # predict by two algorithms
    pred_f, pred_v = predict(output, a, b, pi)
    # plot confusion matrix
    plot_cm(ans, pred_f, pred_v, args.input)


if __name__ == "__main__":
    main()

import numpy as np


def compute_labels_prob(labels):
    """Blocks must be just the lith blocks!"""
    l = np.unique(labels)
    label_count = np.zeros((len(l), labels.shape[1]))
    for i, l_id in enumerate(l):

        label_count[i,:] = np.sum(labels == l_id, axis=0)
    label_prob = label_count / len(labels)
    return label_prob


def compute_ie(labels_prob):
    ie = np.zeros_like(labels_prob[0])
    for l in labels_prob:
        pm = np.ma.masked_equal(l, 0)  # mask where layer prob is 0
        ie -= (pm * np.ma.log2(pm)).filled(0)
    return ie


def compute_ie_total(ie, absolute=False):
    if absolute:
        return np.sum(ie)
    else:
        return np.sum(ie) / np.size(ie)
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances

from pycor.load_annotations.load_anno2 import lemma_groups as ANNO
from pycor.utils.preprocess import clean_wcl
from pycor.utils.vectors import vectorize


def get_representations_from_lemma(lemma, wcl, lemma_groups=ANNO, info_types=['def']):
    group = lemma_groups.get_group((lemma, clean_wcl(wcl)))

    data = []
    labels = []
    for row in group.itertuples():
        vector = vectorize(row, infotypes=info_types)
        if type(vector) != float:
            data.append(vector)
            labels.append(row.ddo_bet)

    data = np.array(data)
    labels = np.array(labels)
    inds = labels.argsort()
    return data[inds], labels[inds]


def get_cosine_matrix(list_of_senses, labels):
    matrix = pairwise_distances(list_of_senses, metric="cosine")

    return pd.DataFrame(matrix, columns=labels, index=labels)

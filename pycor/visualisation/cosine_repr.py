import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine

from pycor.load_annotations.load_anno2 import lemma_groups as ANNO
from pycor.visualisation.utils import clean_wcl, vectorize

def get_representations(lemma_groups, lemma, wcl):
    group = lemma_groups.get_group((lemma, clean_wcl(wcl)))

    data = []
    labels = []
    for row in group.itertuples():
        vector = vectorize(row)
        if type(vector) != float:
            data.append(vector)
            labels.append(row.ddo_bet)
    return np.array(data), labels

def get_cosine_matrix(list_of_senses, labels):
    matrix = pairwise_distances(list_of_senses, metric="cosine")

    print(matrix)
    return matrix

senses, labels = get_representations(ANNO, 'stor', 'adj.')

matrix = get_cosine_matrix(senses, labels)




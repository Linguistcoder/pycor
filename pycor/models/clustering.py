import pandas as pd
import numpy as np
from collections import namedtuple

Select = 'all-4'
#Select = 'Gård-So'
#Select = 'A-B'
#Select = 'C-D-E'
#Select = 'F'
#Select = 'G-H'
#Select = 'I-Sp'
#Select = 'St'
#Select = 'T-Ø'

dataset = pd.read_csv(f'../../var/log_proba_{Select}.tsv', sep='\t', encoding='utf8')
dataset = dataset[dataset['ordklasse']=='sb.']
Pair = namedtuple('Pair', ['bets', 'distance', 'label'])

def get_lab_type(df):
    if df['label'].nunique() == 1:
        if df['label'].all():
            return 1
        else:
            return 0
    else:
        return 2


def get_group_means(proba: pd.Series):
    minimum = proba.min()
    maximum = proba.max()

    def get_closest(val):
        return min([minimum, maximum], key=lambda x: abs(x - val))

    min_group = [prob for i, prob in proba.iteritems()
                        if get_closest(prob) == minimum]

    max_group = [prob for i, prob in proba.iteritems()
                        if get_closest(prob) == maximum]

    return np.mean(min_group), np.mean(max_group), min_group, max_group


all_clusters = {}
uncertain = {}

devel = [['lemma', 'mean', 'std_dv', 'min_mean', 'max_mean', 'type']]


for name, group in dataset.groupby('lemma'):
    clusters = {}
    pairs = [Pair(bets=(row.bet_1, row.bet_2),
                  distance=row.proba,
                  label=row.label) for row in group.itertuples()
             ]

    if len(pairs) == 1:
        pair = pairs[0]
        if pair.distance <= 0.48:
            clusters[1] = [pair.bets[0]]
            clusters[2] = [pair.bets[1]]
        elif pair.distance >= 0.52:
            clusters[1] = list(pair.bets)
        else:
            uncertain[name] = (pair.bets[0], pair.bets[1], pair.distance, pair.label)

    else:
        bets = set([bet for pair in pairs for bet in pair.bets])
        std_dv = group['proba'].std()
        mean = group['proba'].mean()

        min_mean, max_mean, minimums, maximums = get_group_means(group['proba'])
        lab_type = get_lab_type(group)

        devel.append([name, mean, std_dv, min_mean, max_mean, lab_type])

        if (max_mean-min_mean) > 0.42:
            for pair in pairs:
                k = 1
                smallest = True if pair.distance in minimums else False
                inv_clusters = {bet: key for key, val in clusters.items() for bet in val}

                if smallest:
                    if pair.bets[0] not in inv_clusters:
                        clusters[k] = [pair.bets[0]]
                        k += 1
                    if pair.bets[1] not in inv_clusters:
                        clusters[k] = [pair.bets[1]]
                        k += 1
                else:
                    if pair.bets[0] in inv_clusters:
                        clusters[inv_clusters[pair.bets[0]]] += [pair.bets[1]]
                    elif pair.bets[1] in inv_clusters:
                        clusters[inv_clusters[pair.bets[1]]] += [pair.bets[0]]
                    else:
                        clusters[k] = list(pair.bets)
                        k += 1

        elif std_dv < 0.2 or (max_mean-min_mean) < 0.42:
            if mean > 0.3:
                clusters[1] = list(bets)
            else:
                for index, bet in enumerate(bets):
                    clusters[index+1] = [bet]

        else:
            for pair in pairs:
                uncertain[name] = (pair.bets[0], pair.bets[1], pair.distance, pair.label)

    all_clusters[name] = clusters

final_clusters = []

for lemma, clus in all_clusters.items():
    for key, value in clus.items():
        for ddo in value:
            final_clusters.append({'lemma': lemma,
                                   'cor': key,
                                   'DDO': ddo})

final_clusters = pd.DataFrame(final_clusters)
devel = pd.DataFrame(devel)
final_clusters.to_csv(f'../../var/clusters_{Select}.tsv', sep='\t', encoding='utf8')
devel.to_csv(f'../../var/devel_{Select}.tsv', sep='\t', encoding='utf8')
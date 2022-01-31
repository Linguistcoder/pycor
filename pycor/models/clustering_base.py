import pandas as pd
import numpy as np
from collections import namedtuple
import plotly.express as px
from sklearn.metrics import f1_score

from sklearn.preprocessing import MinMaxScaler

Select = 'pred_10_2'

dataset = pd.read_csv(f'../../var/feature_dataset_{Select}.tsv', sep='\t', encoding='utf8')

dataset = dataset[dataset['ordklasse'] == 'sb.']

#dataset['pred'] = dataset.apply(lambda x: 1 if x.score==1 else 0, axis=1)
dataset['acc'] = dataset.apply(lambda x: 1 if x.pred == x.label else 0, axis=1)
print(f1_score(dataset['label'], dataset['pred']))
print(dataset['acc'].mean())

# dataset.score[dataset.score < 0] = 0

fig = px.histogram(dataset, x='pred', color='label',
                   marginal="box",  # or violin, rug
                   hover_data=dataset.columns
                   )
#fig.show()

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


def merge(clusters, inv_clusters, pair, uniq_pairs):
    bet_1, bet_2 = pair.bets
    COR = max(clusters.keys(), default=0) + 1

    # Merge
    if bet_1 in inv_clusters and bet_2 not in inv_clusters:
        clus = inv_clusters[bet_1]
        clusters[clus] += [bet_2]
    elif bet_2 in inv_clusters and bet_1 not in inv_clusters:
        clus = inv_clusters[bet_2]
        clusters[clus] += [bet_1]
    elif bet_1 not in inv_clusters and bet_2 not in inv_clusters:
        clusters[COR] = list(pair.bets)
        COR += 1
    elif set(pair.bets) in uniq_pairs and pair.distance > 0.87:
        if pair.label != 1:
            print('NOT merge:', pair.distance)

        clus1 = inv_clusters[bet_1]
        clus2 = inv_clusters[bet_2]

        if clus1 != clus2:
            merged = clusters[clus1] + clusters[clus2]
            clusters.pop(clus1, None)
            clusters.pop(clus2, None)
            clusters = {i + 1: value for i, (key, value) in enumerate(clusters.items())}
            COR = len(clusters) + 1
            clusters[COR] = merged
            COR += 1
    return clusters


def split(clusters, inv_clusters, pair, uniq_pairs):
    bet_1, bet_2 = pair.bets
    COR = max(clusters.keys(), default=0) + 1

    if bet_1 in inv_clusters and bet_2 not in inv_clusters:
        clusters[COR] = [bet_2]
    elif bet_2 in inv_clusters and bet_1 not in inv_clusters:
        clusters[COR] = [bet_1]
    elif bet_1 not in inv_clusters and bet_2 not in inv_clusters:
        clusters[COR] = [bet_1]
        clusters[COR + 1] = [bet_2]
    elif set(pair.bets) in uniq_pairs and pair.distance <= 0.05:
        if pair.label != 0:
            print('NOT split:', pair.distance)
        if inv_clusters[bet_1] == inv_clusters[bet_2]:
            clus = inv_clusters[bet_1]
            newest = bet_1 if clusters[clus].index(bet_1) > clusters[clus].index(bet_2) else bet_2
            clusters[clus].remove(newest)
            clusters[COR] = [newest]

    return clusters


def do_clustering(data: pd.DataFrame, wcl='[]'):
    all_clusters = {}
    devel = [['lemma', 'mean', 'std_dv', 'min_mean', 'max_mean', 'type']]

    for name, group in data.groupby(['lemma', 'ordklasse', 'homnr']):
        lemma = name[0].lower()
        wcl = name[1].lower()

        clusters = {}

        pairs = [Pair(bets=(row.bet_1, row.bet_2),
                      distance=row.pred,
                      label=row.label) for row in group.itertuples()
                 ]

        uniq_pairs = set([frozenset(pair.bets) for pair in pairs])

        if len(uniq_pairs) == 1:
            distance = np.mean([pair.distance for pair in pairs])
            if distance == 0:
                if pairs[0].label != 0:
                    print('should not split', lemma, distance)
                clusters[1] = [pairs[0].bets[0]]
                clusters[2] = [pairs[0].bets[1]]
            elif distance == 1:
                if pairs[0].label != 1:
                    print('should not merge', lemma, distance)
                clusters[1] = list(pairs[0].bets)

            else:
                print('no cat:', lemma, distance)

        else:
            std_dv = group['pred'].std()
            mean = group['pred'].mean()

            min_mean, max_mean, minimums, maximums = get_group_means(group['pred'])

            if (max_mean - min_mean) <= 0.2 and max_mean >= 0.8:
                print('removes a CATEGORY', lemma)
                maximums += minimums
                minimums = []

            if (max_mean - min_mean) <= 0.2 and max_mean <= 0.1:
                minimums += maximums
                maximums = []

            for pair in pairs:
                if set(pair.bets) not in uniq_pairs:
                    continue

                inv_clusters = {bet: key for key, val in clusters.items() for bet in val}

                if pair.distance == 1:
                    clusters = merge(clusters, inv_clusters, pair, uniq_pairs)

                elif pair.distance == 0: #0.6
                    clusters = split(clusters, inv_clusters, pair, uniq_pairs)

                uniq_pairs.discard(frozenset(pair.bets))

            lab_type = get_lab_type(group)

        all_clusters[name] = clusters

    final_clusters = []

    for (lemma, wcl, hom), clus in all_clusters.items():
        for key, value in clus.items():
            for ddo in value:
                final_clusters.append({'lemma': lemma,
                                       'wcl': wcl,
                                       'homnr': hom,
                                       'cor': key,
                                       'DDO': ddo})

    return pd.DataFrame(final_clusters), pd.DataFrame(devel)
    # return pd.DataFrame(pairs)


all_clusters, devel = do_clustering(dataset)

all_clusters.to_csv(f'../../var/clusters_base{Select}.tsv', sep='\t', encoding='utf8')


# devel.to_csv(f'../../var/devel_{Select}.tsv', sep='\t', encoding='utf8')
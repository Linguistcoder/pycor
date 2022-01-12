import pandas as pd
import numpy as np
from collections import namedtuple
import plotly.express as px

from sklearn.preprocessing import MinMaxScaler

Select = "10_2"

dataset = pd.read_csv(f'../../data/word2vec/reduction_score_{Select}.tsv', sep='\t', encoding='utf8')

dataset = dataset[dataset['ordklasse'] == 'sb.']

score = dataset.score.values.reshape(-1, 1)
# normalize data
#dataset.score = MinMaxScaler().fit_transform(score)

fig = px.histogram(dataset, x='score', color='label',
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


def do_clustering(data: pd.DataFrame, wcl='[]'):
    all_clusters = {}
    uncertain = {}
    # pairs = []
    devel = [['lemma', 'mean', 'std_dv', 'min_mean', 'max_mean', 'type']]

    for name, group in data.groupby(['lemma', 'ordklasse', 'homnr']):
        lemma = name[0].lower()
        wcl = name[1].lower()
        COR = 1
        clusters = {}
        pairs = [Pair(bets=(row.bet_1, row.bet_2),
                      distance=row.score,
                      label=row.label) for row in group.itertuples()#for name, group in group.groupby(['bet_1', 'bet_2'])
                 ]
        uniq_pairs = set([frozenset(pair.bets) for pair in pairs])

        #group = group.groupby(['bet_1', 'bet_2']).aggregate({'homnr': 'mean',
        #                                                     'label': 'mean',
        #                                                     'score': lambda x: x.iloc[0]})

        if len(uniq_pairs) == 1:
            distance = np.mean([pair.distance for pair in pairs])
            if distance > 0.4:
                #if pairs[0].label != 0:
                #    print('should not split', lemma, distance)
                clusters[1] = [pairs[0].bets[0]]
                clusters[2] = [pairs[0].bets[1]]
            elif distance <= 0.4:
                #if pairs[0].label != 1:
                #    print('should not merge', lemma, distance)
                clusters[1] = list(pairs[0].bets)

            # else:
        #             uncertain[name] = (pair.bets[0], pair.bets[1], pair.distance, pair.label)

        else:
            std_dv = group['score'].std()
            mean = group['score'].mean()


            min_mean, max_mean, minimums, maximums = get_group_means(group['score'])
            pairs.sort(key=lambda x: 1 + x.distance
            if x.distance >= 0.6 else 1 - x.distance + 0.5
            if x.distance <= 0.3 else x.distance,
                       reverse=True)
            #pairs.sort(key=lambda x: x.distance)
            if group['label'].mean() == 1:
                print()

            if (max_mean - min_mean) <= 0.2 and max_mean >= 0.49 or min_mean > 0.55:
                #print('removes a CATEGORY', lemma)
                maximums += minimums
                minimums = []

            if (max_mean - min_mean) <= 0.2 and max_mean <= 0.35:
                minimums += maximums
                maximums = []

            #if np.mean(maximums) <= 0.1:
             #   minimums += maximums
              #  maximums = []

            for pair in pairs:
                if pair.label == 1:
                    print(pair.distance)
                if set(pair.bets) not in uniq_pairs:
                    continue

                bet_1, bet_2 = pair.bets
                inv_clusters = {bet: key for key, val in clusters.items() for bet in val}

                if pair.distance in minimums:
                    if pair.label != 1:
                        pass
                        #print('should not MERGE', lemma, pair.distance)
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
                    elif set(pair.bets) in uniq_pairs and pair.distance <=0.4:
                        clus1 = inv_clusters[bet_1]
                        clus2 = inv_clusters[bet_2]

                        if clus1 != clus2:
                            merged = clusters[clus1] + clusters[clus2]
                            clusters.pop(clus1, None)
                            clusters.pop(clus2, None)
                            clusters = {i+1: value for i, (key, value) in enumerate(clusters.items())}
                            COR = len(clusters) + 1
                            clusters[COR] = merged
                            COR += 1

                    else:
                        if lemma not in uncertain:
                            uncertain[lemma] = [pair]
                        else:
                            uncertain[lemma] += pair

                if pair.distance in maximums:
                    if pair.label != 0:
                        pass
                        #print('should not split', lemma, pair.distance)

                    # Split
                    if bet_1 in inv_clusters and bet_2 not in inv_clusters:
                        clusters[COR] = [bet_2]
                        COR += 1
                    elif bet_2 in inv_clusters and bet_1 not in inv_clusters:
                        clusters[COR] = [bet_1]
                        COR += 1
                    elif bet_1 not in inv_clusters and bet_2 not in inv_clusters:
                         clusters[COR] = [bet_1]
                         clusters[COR + 1] = [bet_2]
                         COR += 2
                    elif set(pair.bets) in uniq_pairs and pair.distance >= 0.8:
                        if inv_clusters[bet_1] == inv_clusters[bet_2]:
                            clus = inv_clusters[bet_1]
                            newest = bet_1 if clusters[clus].index(bet_1) > clusters[clus].index(bet_2) else bet_2
                            clusters[clus].remove(newest)
                            clusters[COR] = [newest]
                            COR += 1
                    else:
                        if inv_clusters.get(bet_1, 'n') == inv_clusters.get(bet_2, 'n'):
                            if lemma not in uncertain:
                                uncertain[lemma] = [pair]
                            else:

                                uncertain[lemma] += pair

                uniq_pairs.discard(frozenset(pair.bets))
            # steps
            # 1 - sort pairs
            # 2 - assign

            lab_type = get_lab_type(group)

            devel.append([name, mean, std_dv, min_mean, max_mean, lab_type])

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

all_clusters.to_csv(f'../../var/clusters_w2v{Select}.tsv', sep='\t', encoding='utf8')
#devel.to_csv(f'../../var/devel_{Select}.tsv', sep='\t', encoding='utf8')

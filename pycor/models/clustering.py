import pandas as pd
import math
dataset = pd.read_csv('../../var/log_proba_test_semi.tsv', sep='\t', encoding='utf8')

class Pair(object):
    def __init__(self, bets, distance, label):
        self.bets = bets
        self.distance = distance
        self.label = label
all_clusters = {}
uncertain = {}

for name, group in dataset.groupby('lemma'):
    clusters = {}
    pairs = [Pair(bets=(row.bet_1, row.bet_2),
                  distance=row.proba,
                  label=row.label) for row in group.itertuples()
             ]

    if len(pairs) == 1:
        pair = pairs[0]
        if pair.distance <= 0.4:
            clusters[1] = [pair.bets[0]]
            clusters[2] = [pair.bets[1]]
        elif pair.distance >= 0.6:
            clusters[1] = list(pair.bets)
        else:
            uncertain[name] = (pair.bets[0], pair.bets[1], pair.distance, pair.label)

    else:
        bets = set([bet for pair in pairs for bet in pair.bets])
        k = 1

        std_dv = group['proba'].std()
        mean = group['proba'].mean()

        if std_dv < 0.2:
            if mean >= 0.5:
                clusters[1] = list(bets)
            else:
                for index, bet in enumerate(bets):
                    clusters[index+1] = [bet]

        else:
            k = 1
            minimum = group['proba'].min()
            maximum = group['proba'].max()

            for pair in pairs:
                closest = min([minimum, maximum], key=lambda x: abs(x-pair.distance))

                inv_clusters = {bet: key for key, val in clusters.items() for bet in val}

                if closest == minimum:
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


    all_clusters[name] = clusters

final_clusters = []

for lemma, clus in all_clusters.items():
    for key, value in clus.items():
        for ddo in value:
            final_clusters.append({'lemma': lemma,
                                   'cor': key,
                                   'DDO': ddo})

final_clusters = pd.DataFrame(final_clusters)
final_clusters.to_csv('../../var/clusters2.tsv', sep='\t', encoding='utf8')
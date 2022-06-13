from collections import namedtuple

import pandas as pd
import numpy as np


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


def get_lab_type(df):
    if df['label'].nunique() == 1:
        if df['label'].all():
            return 1
        else:
            return 0
    else:
        return 2


class Cluster(dict):
    def __init__(self, cluster):
        super().__init__(cluster)
        self.inverse = {bet: key for key, val in self.items() for bet in val}

    def merge(self, pair, uniq_pairs, dis=0):
        bet_1, bet_2 = pair.bets
        COR = max(self.keys(), default=0) + 1

        # Merge
        if bet_1 in self.inverse and bet_2 not in self.inverse:
            clus = self.inverse[bet_1]
            self[clus] += [bet_2]
        elif bet_2 in self.inverse and bet_1 not in self.inverse:
            clus = self.inverse[bet_2]
            self[clus] += [bet_1]
        elif bet_1 not in self.inverse and bet_2 not in self.inverse:
            self[COR] = list(pair.bets)
            COR += 1
        elif set(pair.bets) in uniq_pairs and dis > 0 and pair.distance < dis:
            #if pair.label == 0:
                #print(str(pair.distance).replace(".", ","))
                #print('NOT merge:', pair.distance)

            clus1 = self.inverse[bet_1]
            clus2 = self.inverse[bet_2]

            if clus1 != clus2:
                merged = self[clus1] + self[clus2]
                self.pop(clus1, None)
                self.pop(clus2, None)
                new = {i + 1: value for i, (key, value) in enumerate(self.items())}
                COR = len(new) + 1
                new[COR] = merged
                self = new
                COR += 1
        self.__init__(self)
        return self

    def split(self, pair, uniq_pairs, dis=0):
        bet_1, bet_2 = pair.bets
        COR = max(self.keys(), default=0) + 1

        if bet_1 in self.inverse and bet_2 not in self.inverse:
            self[COR] = [bet_2]
        elif bet_2 in self.inverse and bet_1 not in self.inverse:
            self[COR] = [bet_1]
        elif bet_1 not in self.inverse and bet_2 not in self.inverse:
            self[COR] = [bet_1]
            self[COR + 1] = [bet_2]
        elif set(pair.bets) in uniq_pairs and dis > 0 and pair.distance >= dis:
            #if pair.label == 1:
                #print(str(pair.distance).replace(".", ","))
               # print('NOT split:', pair.distance)
            if self.inverse[bet_1] == self.inverse[bet_2]:
                clus = self.inverse[bet_1]
                newest = bet_1 if self[clus].index(bet_1) > self[clus].index(bet_2) else bet_2
                self[clus].remove(newest)
                self[COR] = [newest]
        self.__init__(self)
        return self


class ClusterAlgorithm(object):
    def __init__(self, config):
        self.input_model = config.model_name
        self.bh = config.bh  # binary threshold
        self.h0 = config.h0  # lower threshold
        self.h1 = config.h1  # upper threshold
        self.h_mean1 = config.h_mean1  # cluster upper threshold
        self.hm_mean = config.hm_mean  # cluster median threshold
        self.h_mean0 = config.h_mean0 # cluster median lower threshold
        self.absmin = config.absmin
        self.absmax = config.absmax
        self.bias = config.bias
        self.all_clusters = {}

    def clustering2(self, data: pd.DataFrame):
        self.h0 = 0.76
        self.h1 = 0.5
        self.h_mean0 = 0.3


        for name, group in data.groupby(['lemma', 'ordklasse', 'homnr']):
            lemma = name[0].lower()
            wcl = name[1].lower()

            clusters = Cluster({})
            k = 1
            unique_bet = set(group['bet_1']).union(set(group['bet_2']))

            mat = np.zeros((len(unique_bet), len(unique_bet)))
            mat = pd.DataFrame(mat, columns=unique_bet, index=unique_bet)

            for row in group.itertuples():
                bet_1, bet_2 = (row.bet_1, row.bet1_id), (row.bet_2, row.bet2_id)
                mat.loc[bet_1, bet_2] = 1 - row.score
                mat.loc[bet_2, bet_1] = 1 - row.score

            #print(mat)

            for row in mat.iterrows():
                bet, distances = row
                distances.pop(bet)
                prob = distances.product()
                #print(prob)
                if prob < self.h1:
                    if bet in clusters.inverse:
                        clus = clusters.inverse[bet]
                        if clusters[clus] > 1:
                            clusters[clus].remove(bet)
                            clusters = Cluster(clusters)
                        else:
                            continue
                    else:
                        clusters[k] = [bet]
                        k += 1
                        clusters = Cluster(clusters)
                else:
                    for edge, dis in distances.iteritems():
                        if dis >= self.h0:
                            if bet in clusters.inverse or edge in clusters.inverse:
                                clus = clusters.inverse.get(bet, clusters.inverse[edge])
                                other_bets = clusters[clus]
                                prob = np.product([mat.loc[bet, b] for b in other_bets])
                                #print(prob)
                                if prob >= self.h_mean0:
                                    clusters[clus] += [name, edge]
                                    clusters[clus] = list(set(clusters[clus]))
                                    clusters = Cluster(clusters)
                        else:
                            clusters[k] = [bet, edge]
                            clusters = Cluster(clusters)
                            k += 1
            for bet in unique_bet:
                if bet not in clusters.inverse:
                    clusters[k] = [bet]
                    clusters = Cluster(clusters)
                    k += 1

            self.all_clusters[name] = clusters

        return self

    def clustering(self, data: pd.DataFrame):
        Pair = namedtuple('Pair', ['bets', 'distance', 'label'])

        for name, group in data.groupby(['lemma', 'ordklasse', 'homnr']):
            lemma = name[0].lower()
            wcl = name[1].lower()
            clusters = {}
            pairs = [Pair(bets=((row.bet_1, row.bet1_id), (row.bet_2, row.bet2_id)),
                          distance=row.score,
                          label=row.label) for row in group.itertuples()
                     # for name, group in group.groupby(['bet_1', 'bet_2'])
                     ]
            uniq_pairs = set([frozenset(pair.bets) for pair in pairs])

            # group = group.groupby(['bet_1', 'bet_2']).aggregate({'homnr': 'mean',
            #                                                     'label': 'mean',
            #                                                     'score': lambda x: x.iloc[0]})

            if len(uniq_pairs) == 1:
                distance = np.mean([pair.distance for pair in pairs])
            #     if pairs[0].label == 1:
            #        print(str(distance).replace(".", ","))
                if distance > self.bh:
                    #if pairs[0].label != 0:
                        #print('should not split', lemma, distance)
                    clusters[1] = [pairs[0].bets[0]]
                    clusters[2] = [pairs[0].bets[1]]
                elif distance <= self.bh:
                    #if pairs[0].label != 1:
                        #print('should not merge', lemma, distance)
                    clusters[1] = list(pairs[0].bets)

            else:
                std_dv = group['score'].std()
                mean = group['score'].mean()

                min_mean, max_mean, minimums, maximums = get_group_means(group['score'])
                pairs.sort(key=lambda x: 1 + x.distance
                if x.distance >= self.h_mean1 else 1 - x.distance + self.bias
                if x.distance <= self.h_mean0 else x.distance,
                           reverse=True
                           )
                # pairs.sort(key=lambda x: x.distance)
                #if group['label'].mean() == 1:
                    #print()

               # if group['label'].nunique() == 1:
                    #print(str(max_mean - min_mean).replace(".",","))
                     # if pairs[0].label == 1:
                     #     print(str(max_mean).replace(".", ","),
                     #           str(min_mean).replace(".", ","),
                     #           str(max_mean-min_mean).replace(".", ","),
                     #           sep='\t')


                if (max_mean - min_mean) <= self.hm_mean and max_mean >= self.h_mean1 and min_mean >= self.h_mean0:
                    maximums += minimums
                    minimums = []

                    # clusters = Cluster(clusters)
                    # for pair in pairs:
                    #     if set(pair.bets) not in uniq_pairs:
                    #         continue
                    #     clusters = clusters.split(pair, uniq_pairs)
                    #     uniq_pairs.discard(frozenset(pair.bets))

                elif (max_mean - min_mean) <= self.hm_mean and max_mean <= self.h_mean1 and min_mean <= self.h_mean0:
                    minimums += maximums
                    maximums = []

                for pair in pairs:
                    if set(pair.bets) not in uniq_pairs:
                        continue

                    clusters = Cluster(clusters)
                    #if pair.label == 0: #and self.h0 >= pair.distance or pair.distance >= self.h1:
                        #print(str(pair.distance).replace(".", ","))
                    if pair.distance <= self.h0:
                        clusters = clusters.merge(pair, uniq_pairs, self.absmin)

                    elif pair.distance >= self.h1:  # 0.6
                        clusters = clusters.split(pair, uniq_pairs, self.absmax)

                    else:
                        if pair.distance in minimums:
                            clusters = clusters.merge(pair, uniq_pairs, self.absmin)

                        elif pair.distance in maximums:
                            clusters = clusters.split(pair, uniq_pairs, self.absmax)

                    uniq_pairs.discard(frozenset(pair.bets))

            self.all_clusters[name] = clusters

        return self

    def to_dataframe(self):

        final_clusters = []

        for (lemma, wcl, hom), clus in self.all_clusters.items():
            #if len(clus) >= 10:
            #    print(lemma)
            #    print(clus.values())
            for key, value in clus.items():
                for ddo, semid in value:
                    final_clusters.append({'lemma': lemma,
                                           'wcl': wcl,
                                           'homnr': hom,
                                           'cor': key,
                                           'DDO': ddo,
                                           'ddo_dannetsemid': semid})

        return pd.DataFrame(final_clusters)  # , pd.DataFrame(devel)

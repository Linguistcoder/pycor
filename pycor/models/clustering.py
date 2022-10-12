from collections import namedtuple
import pandas as pd
import numpy as np


def get_group_means(proba: pd.Series):
    """Splits the pairs into groups depeding on whether their distance is closest to
    the maximum or minimum of all pairs"""
    minimum = proba.min()
    maximum = proba.max()

    def get_closest(val):
        return min([minimum, maximum], key=lambda x: abs(x - val))

    min_group = [prob for i, prob in proba.iteritems()
                 if get_closest(prob) == minimum]

    max_group = [prob for i, prob in proba.iteritems()
                 if get_closest(prob) == maximum]

    return np.mean(min_group), np.mean(max_group), min_group, max_group


class Cluster(dict):
    """
    Custom dict class for updating clusters

    Attributes
    ----------
     :attr inverse: mapping senses to clusters (self maps clusters to senses)

    Methods
    -------
    merge(self, pair, uniq_pairs, dis=0)
        :returns: updated self with added (merged) pair

    split(self, pair, uniq_pair, dis=0)
        :returns: updated self with added (split) pair
    """

    def __init__(self, cluster):
        super().__init__(cluster)
        self.inverse = {bet: key for key, val in self.items() for bet in val}

    def merge(self, pair, uniq_pairs, dis=0):
        """
        merge pair if possible
        :param pair: sense pair
        :param uniq_pairs: unprocessed pairs
        :param dis: threshold for allowing overruling of previous cluster assignment
        :return: self
        """
        bet_1, bet_2 = pair.bets
        COR = max(self.keys(), default=0) + 1

        # check if sense 1 already in a cluster (and sense 2 is not)
        if bet_1 in self.inverse and bet_2 not in self.inverse:
            # get sense 1s cluster
            clus = self.inverse[bet_1]
            # add sense 2 to that cluster
            self[clus] += [bet_2]

        # check if sense 2 is in a cluster (and sense 1 is not)
        elif bet_2 in self.inverse and bet_1 not in self.inverse:
            # get sense 2s cluster
            clus = self.inverse[bet_2]
            # add sense 1 to that cluster
            self[clus] += [bet_1]

        # if neither senses are in a cluster, then create new cluster
        elif bet_1 not in self.inverse and bet_2 not in self.inverse:
            self[COR] = list(pair.bets)
            COR += 1

        # if both senses are in different clusters and pair distance is below a threshold (dis)
        # then we merge the two clusters together
        # this is risky --> therefore the extra threshold
        elif set(pair.bets) in uniq_pairs and dis >= 0 and pair.distance <= dis:
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
        """
        split pair if possible
        :param pair: sense pair
        :param uniq_pairs: unprocessed pairs
        :param dis: threshold for allowing overruling of previous cluster assignment
        :return: self
        """
        bet_1, bet_2 = pair.bets
        COR = max(self.keys(), default=0) + 1

        # check if sense 1 already in a cluster (and sense 2 is not)
        if bet_1 in self.inverse and bet_2 not in self.inverse:
            # we create a new cluster for sense 2
            self[COR] = [bet_2]
        # check if sense 2 is in a cluster (and sense 1 is not)
        elif bet_2 in self.inverse and bet_1 not in self.inverse:
            # we create a new cluster for sense 1
            self[COR] = [bet_1]

        # if neither senses are in a cluster, then create two new cluster
        elif bet_1 not in self.inverse and bet_2 not in self.inverse:
            self[COR] = [bet_1]
            self[COR + 1] = [bet_2]

        # if both senses are in the same clusters and pair distance is above a threshold (dis)
        # then we remove the sense that was added most recently (and therefore with the worse score)
        # and create a new cluster for that
        # this is risky --> therefore the extra threshold
        elif set(pair.bets) in uniq_pairs and 0 < dis <= pair.distance:
            if self.inverse[bet_1] == self.inverse[bet_2]:
                clus = self.inverse[bet_1]
                # find the sense that was added most recently
                newest = bet_1 if self[clus].index(bet_1) > self[clus].index(bet_2) else bet_2
                self[clus].remove(newest)
                self[COR] = [newest]
        self.__init__(self)
        return self


class ClusterAlgorithm(object):
    """
    Clustering algorithm class for clustering sense pairs based on their distance score

    Attributes
    ----------
     :attr input_model: name of model used for computing distance score
     :attr bh: binary threshold (when there is only a single sense pair for a lemma)
     :attr h0: lower threshold (merge below)
     :attr h1: upper threshold (split above)
     :attr h_mean1: cluster upper threshold (for grouping pairs in maximums and minimums)
     :attr hm_mean: cluster median threshold (for grouping pairs in maximums and minimums)
     :attr h_mean0: cluster lower threshold (for grouping pairs in maximums and minimums)
     :attr absmin: highest distance that allows to overrule previous splitting
     :attr absmax: lowest distance that allows to overrule previous merging
     :attr bias: bias that decides distance sorting
     :all_clusters: dict of clusters

    Methods
    -------
    clustering(self, data: pd.DataFrame)
        :returns: self with updated self.all_clusters

    clustering2(self, data: pd.DataFrame) NOT FINISHED!

    to_dataframe(self)
        :returns: pd.Dataframe with restructured clusters

    """

    def __init__(self, config):
        self.input_model = config.model_name
        self.bh = config.bh  # binary threshold
        self.h0 = config.h0  # lower threshold
        self.h1 = config.h1  # upper threshold
        self.h_mean1 = config.h_mean1  # cluster upper threshold
        self.hm_mean = config.hm_mean  # cluster median threshold
        self.h_mean0 = config.h_mean0  # cluster median lower threshold
        self.absmin = config.absmin  # highest distance that allows to overrule previous splitting
        self.absmax = config.absmax  # lowest distance that allows to overrule previous merging
        self.bias = config.bias
        self.all_clusters = {}

    def clustering(self, data: pd.DataFrame):
        """
        Clustering algorithm version 1

        :param data: reduction (pd.DataFrame) dataset where senses are structured in pairs with distance scores
        :return: self
        """
        # clustering is based on whether or not to split pairs
        Pair = namedtuple('Pair', ['bets', 'distance', 'label'])

        # move through data lemma for lemma
        for name, group in data.groupby(['lemma', 'homnr']):

            # clusters will be a dict of cluster labels (key) with a list of assigned senses (value)
            clusters = {}
            # each pair consists of:
            #  1) a tuple of a tuple (sense pairs: ((sense, sense_id), (sense, sense_id))
            #  2) distance calculated by model
            #  3) label all merged (0) all split (1) or different groups (2)
            pairs = [Pair(bets=((row.bet_1, row.bet1_id), (row.bet_2, row.bet2_id)),
                          distance=row.score,
                          label=row.label) for row in group.itertuples()
                     ]

            # a pair may be represented multiple times. Therefore, we keep track on the ones we need to process
            uniq_pairs = set([frozenset(pair.bets) for pair in pairs])

            # if there is only one pair
            if len(uniq_pairs) == 1:
                distance = np.mean([pair.distance for pair in pairs])
                if distance > self.bh:  # split
                    clusters[1] = [pairs[0].bets[0]]
                    clusters[2] = [pairs[0].bets[1]]
                elif distance <= self.bh:  # merge
                    clusters[1] = list(pairs[0].bets)

            else:
                # if there is more than one pair, when we split the pairs into groups depending on the distance
                # min_mean == average of the group with the lowest distances
                # max_mean == average of the group with the highest distances
                min_mean, max_mean, minimums, maximums = get_group_means(group['score'])

                # we sort the pairs so the most probable (highest or lowest scores) are processed first
                pairs.sort(key=lambda x: 1 + x.distance
                if x.distance >= self.h_mean1 else 1 - x.distance + self.bias
                if x.distance <= self.h_mean0 else x.distance,
                           reverse=True
                           )

                # if the difference between max_mean and min_mean are lower (or equal) than a threshold (hm_mean)
                # and max_mean is higher (or equal to) than the cluster upper threshold
                # and min_mean is higher (or equal to) than cluster median lower threshold
                # Then all the distances for all the pairs are close together and high
                # and we make one big maximum group
                if (max_mean - min_mean) <= self.hm_mean and max_mean >= self.h_mean1 and min_mean >= self.h_mean0:
                    maximums += minimums
                    minimums = []

                # if the difference between max_mean and min_mean are lower (or equal) than a threshold (hm_mean)
                # and max_mean is lower (or equal to) than the cluster upper threshold
                # and min_mean is lower (or equal to) than cluster median lower threshold
                # Then all the distances for all the pairs are close together and  low
                # and we make one big minimum group
                elif (max_mean - min_mean) <= self.hm_mean and max_mean <= self.h_mean1 and min_mean <= self.h_mean0:
                    minimums += maximums
                    maximums = []

                for pair in pairs:
                    # if we already processed the pair
                    if set(pair.bets) not in uniq_pairs:
                        continue

                    # by transforming clusters into a Clusters object, the inverse cluster mapping is updated
                    clusters = Cluster(clusters)

                    if pair.distance <= self.h0:  # merge if below threshold
                        clusters = clusters.merge(pair, uniq_pairs, self.absmin)

                    elif pair.distance >= self.h1:  # split if above threshold
                        clusters = clusters.split(pair, uniq_pairs, self.absmax)

                    else:  # if distance is between the thresholds, then it depends on its group
                        if pair.distance in minimums:  # minimums are merged
                            clusters = clusters.merge(pair, uniq_pairs, self.absmin)

                        elif pair.distance in maximums:  # maximums are split
                            clusters = clusters.split(pair, uniq_pairs, self.absmax)

                    # remove pair when processed
                    uniq_pairs.discard(frozenset(pair.bets))

            self.all_clusters[name] = clusters

        return self

    def clustering2(self, data: pd.DataFrame):
        """This algorithm is not finished..."""
        self.h0 = 0.76
        self.h1 = 0.5
        self.h_mean0 = 0.3

        for name, group in data.groupby(['lemma', 'homnr']):

            clusters = Cluster({})
            k = 1
            unique_bet = set(group['bet_1']).union(set(group['bet_2']))

            mat = np.zeros((len(unique_bet), len(unique_bet)))
            mat = pd.DataFrame(mat, columns=unique_bet, index=unique_bet)

            for row in group.itertuples():
                bet_1, bet_2 = (row.bet_1, row.bet1_id), (row.bet_2, row.bet2_id)
                mat.loc[bet_1, bet_2] = 1 - row.score
                mat.loc[bet_2, bet_1] = 1 - row.score

            for row in mat.iterrows():
                bet, distances = row
                distances.pop(bet)
                prob = distances.product()

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

    def to_dataframe(self):
        """restructure clusters to a dataframe stucture
        with columns: ['lemma', 'wcl', 'homnr', 'cor', 'DDO', 'ddo_dannetsemid']"""
        final_clusters = []

        for (lemma, wcl, hom), clus in self.all_clusters.items():
            for key, value in clus.items():
                for ddo, semid in value:
                    final_clusters.append({'lemma': lemma,
                                           'wcl': wcl,
                                           'homnr': hom,
                                           'cor': key,
                                           'DDO': ddo,
                                           'ddo_dannetsemid': semid})

        return pd.DataFrame(final_clusters)

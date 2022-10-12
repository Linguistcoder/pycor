import sys
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.metrics import f1_score, rand_score
from sklearn.preprocessing import MinMaxScaler

from configs.config_clustering import ClusteringConfig
from pycor.models.clustering import ClusterAlgorithm


def do_clustering(subset_name, model_name, tune_name, binary_thres, select_wcl=None, path='../data'):
    file = f'{path}/{model_name}/reduction_score_{subset_name}.tsv'
    tune_file = f'{path}/{model_name}/reduction_score_{tune_name}.tsv'
    dataset = pd.read_csv(file, sep='\t', encoding='utf8')
    tuneset = pd.read_csv(tune_file, sep='\t', encoding='utf8')

    if select_wcl:
        dataset = dataset[dataset['ordklasse'] == select_wcl]

    # normalize data
    score = tuneset.score.values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaler.fit(score)
    dataset['score'] = scaler.transform(dataset.score.values.reshape(-1, 1))

    dataset['score'] = dataset.apply(lambda x: 1 - x.score, axis=1)
    dataset['pred'] = dataset.apply(lambda x: 1 if x.score <= binary_thres else 0, axis=1)
    dataset['acc'] = dataset.apply(lambda x: 1 if x.pred == x.label else 0, axis=1)

    print('the binary classification F1 is:', f1_score(dataset['label'], dataset['pred']))

    fig = px.histogram(dataset, x='score', color='label',
                       marginal="box",  # or violin, rug
                       hover_data=dataset.columns
                       )
    fig.show()

    cluster_config = ClusteringConfig(model_name=model_name)
    cluster_algo = ClusterAlgorithm(cluster_config)

    all_clusters = cluster_algo.clustering(dataset)
    all_clusters = all_clusters.to_dataframe()
    all_clusters.to_csv(f'../var/clusters_{model_name}_{subset_name}.tsv', sep='\t', encoding='utf8')

    return all_clusters


def evaluate_clusters(annotated_with_clusters, model, subset):
    correct_n = 0
    total_n = 0

    merge_bias = 0
    split_bias = 0
    merge_bias_n = 0
    split_bias_n = 0
    rand_scores = []

    for name, group in annotated_with_clusters.groupby(['lemma', 'homnr']):
        rand_scores.append(rand_score(group['COR'], group['cor']))

        pred_groups = group.groupby('cor').groups
        lab_groups = group.groupby('COR').groups

        # is the number of clusters the same?
        if len(pred_groups) == len(lab_groups):
            correct_n += 1
        else:
            bias = len(pred_groups) - len(lab_groups)
            if bias > 0:
                merge_bias += bias
                merge_bias_n += 1
            else:
                split_bias += abs(bias)
                split_bias_n += 1
        total_n += 1

    print(f'________________{model.upper()}_{subset.upper()}___________________')
    print('rand', np.mean(rand_scores))
    print('reduce:', correct_n / total_n)
    print('merge_bias', merge_bias_n / total_n)
    print('split_bias:', split_bias_n / total_n)


# ones = dataset[dataset['label'] == 1]
# zeros = dataset[dataset['label'] == 0]
#
# Q1_1, Q1_2, Q1_3 = np.quantile(ones.score, 0.25), np.quantile(ones.score, 0.5), np.quantile(ones.score, 0.75)
# Q0_1, Q0_2, Q0_3 = np.quantile(zeros.score, 0.25), np.quantile(zeros.score, 0.5), np.quantile(zeros.score, 0.75)
#
# print(f'{ones.score.min()} {Q1_1} {Q1_2} {Q1_3} {ones.score.max()}'.replace('.', ',').replace(' ', '\t'))
# print(f'{zeros.score.min()} {Q0_1} {Q0_2} {Q0_3} {zeros.score.max()}'.replace('.', ',').replace(' ', '\t'))

# devel.to_csv(f'../../var/devel_{Select}.tsv', sep='\t', encoding='utf8')

if __name__ == "__main__":
    model_name = 'base'  # 'word2vec'
    subset_name = 'cbc_test'  # 'cbc_devel3'
    tune_name = 'cbc_test'  # 'cbc_devel3'
    binary_thres = 0.31

    model_name = sys.argv[1]
    subset_name = sys.argv[2]

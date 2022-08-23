import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.metrics import f1_score

from sklearn.preprocessing import MinMaxScaler

from configs.config_clustering import ClusteringConfig
from pycor.models.clustering import ClusterAlgorithm

model_name = 'base' #'word2vec'
subset_name = 'cbc_test' #'cbc_devel3'
tune_name = 'cbc_test' #'cbc_devel3'
binary_thres = 0.31

file = f'../data/{model_name}/reduction_score_{subset_name}.tsv'
tune_file = f'../data/{model_name}/reduction_score_{tune_name}.tsv'
dataset = pd.read_csv(file, sep='\t', encoding='utf8')
tuneset = pd.read_csv(tune_file, sep='\t', encoding='utf8')


#dataset = dataset[dataset['ordklasse'] == 'sb.']

#print(dataset['acc'].mean())


# normalize data
score = tuneset.score.values.reshape(-1, 1)
scaler = MinMaxScaler()
scaler.fit(score)
dataset['score'] = scaler.transform(dataset.score.values.reshape(-1, 1))
#dataset['score'] = dataset.apply(lambda x: 1 if x.score == 0 else 0, axis=1)
dataset['score'] = dataset.apply(lambda x: 1-x.score, axis=1)

dataset['pred'] = dataset.apply(lambda x: 1 if x.score <= binary_thres else 0, axis=1)
dataset['acc'] = dataset.apply(lambda x: 1 if x.pred == x.label else 0, axis=1)

print(f1_score(dataset['label'], dataset['pred']))


fig = px.histogram(dataset, x='score', color='label',
                   marginal="box",  # or violin, rug
                   hover_data=dataset.columns
                   )
fig.show()

ones = dataset[dataset['label'] == 1]
zeros = dataset[dataset['label'] == 0]

Q1_1, Q1_2, Q1_3 = np.quantile(ones.score, 0.25), np.quantile(ones.score, 0.5), np.quantile(ones.score, 0.75)
Q0_1, Q0_2, Q0_3 = np.quantile(zeros.score, 0.25), np.quantile(zeros.score, 0.5), np.quantile(zeros.score, 0.75)

print(f'{ones.score.min()} {Q1_1} {Q1_2} {Q1_3} {ones.score.max()}'.replace('.', ',').replace(' ', '\t'))
print(f'{zeros.score.min()} {Q0_1} {Q0_2} {Q0_3} {zeros.score.max()}'.replace('.', ',').replace(' ', '\t'))

cluster_config = ClusteringConfig(model_name=model_name)
cluster_algo = ClusterAlgorithm(cluster_config)

all_clusters = cluster_algo.clustering(dataset)
all_clusters = all_clusters.to_dataframe()

all_clusters.to_csv(f'../var/clusters_{model_name}_{subset_name}.tsv', sep='\t', encoding='utf8')
# devel.to_csv(f'../../var/devel_{Select}.tsv', sep='\t', encoding='utf8')
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import plotly.express as px

RANDOM_STATE = 1
np.random.seed(RANDOM_STATE)

subset = 'keywords'
split = 'train'

subset2 = 'keywords'
split2 = 'test'

wcl = 'sb.'
#n_s = 3
# lemma ordklasse homnr bet_1 bet1_id bet_2 bet2_id cosine bert onto main_sense figurative label

dataset = pd.read_csv(f'../../data/{subset}_{split}_feature_dataset.tsv', '\t', encoding='utf8', header=0)
dataset['n_sense'] = dataset.groupby(by=['lemma', 'ordklasse', 'homnr'])['homnr'].transform(lambda x: x.shape[0])

dataset = dataset.dropna()
dataset = dataset[dataset['ordklasse'] == wcl]
#dataset = dataset[dataset['n_sense'] == n_s]

group = dataset.groupby('label')
balanced_data = group.apply(lambda x:
                            x.sample(group.size().min()).reset_index(drop=True)
                            ).sort_values(by='lemma').droplevel(0)

dataset2 = pd.read_csv(f'../../data/{subset2}_{split2}_feature_dataset.tsv', '\t', encoding='utf8', header=0)
dataset2['n_sense'] = dataset2.groupby(by=['lemma', 'ordklasse', 'homnr'])['homnr'].transform(lambda x: x.shape[0])
dataset2 = dataset2.dropna()
dataset2 = dataset2[dataset2['ordklasse'] == wcl]
#dataset2 = dataset2[dataset2['n_sense'] == n_s]


#train, test = train_test_split(balanced_data, test_size=0.2, random_state=RANDOM_STATE)
# train, test = train_test_split(dataset, test_size=0.5)
# X_train, X_test, y_train, y_test = train_test_split(dataset, test_size=0.2)

train = balanced_data.copy()
test = dataset2.copy()

X = train.loc[:, ['cosine',
                  'bert',
                  'onto',
                  'main_sense',
                  'figurative'
                  ]
    ]

y = train.loc[:, ['label']]

teX = test.loc[:, ['cosine',
                   'bert',
                   'onto',
                   'main_sense',
                   'figurative'
                   ]
      ]
tey = test.loc[:, ['label']]

# normalize data
norm = MinMaxScaler().fit(X)
# transform training data
X = norm.transform(X)
# transform testing dataabs
teX = norm.transform(teX)

classifiers = [
    RandomForestClassifier(max_depth=7),
    #LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
    #KNeighborsClassifier(n_neighbors=5),
    #GaussianNB()
]

for classifier in classifiers:
    classifier.fit(X, y)
    pred_train = classifier.predict(X)
    train.loc[:, 'proba'] = classifier.predict_proba(X)[:, 1]

    train = train.sort_values(by='lemma')
    # train.to_csv('../../var/log_proba_train_semi.tsv', sep='\t', encoding='utf8')
    print(accuracy_score(y, pred_train))
    print(f1_score(y, pred_train))
    train['pred'] = pred_train.tolist()

    cm = confusion_matrix(y, pred_train)
    cm.diagonal()
    print(cm)

    pred_test = classifier.predict(teX)
    test.loc[:, 'proba'] = classifier.predict_proba(teX)[:, 1]
    test.to_csv(f'../../var/log_proba_{subset}_{split}.tsv', sep='\t', encoding='utf8')

    test['pred'] = pred_test.tolist()
    print(accuracy_score(tey, pred_test))
    print(f1_score(tey, pred_test))

    cm = confusion_matrix(tey, pred_test)
    cm.diagonal()
    print(cm)

ones = test.loc[test['label'] == 1]
zeros = test.loc[test['label'] == 0]

false_pos = ones[ones['pred'] == 0]
false_neg = zeros[zeros['pred'] == 1]

false_pos.to_csv(f'../../var/false_pos_{subset2}_{split2}.tsv', '\t', encoding='utf8',)
false_neg.to_csv(f'../../var/false_neg_{subset2}_{split2}.tsv', '\t', encoding='utf8',)
#
# print(ones['score'].mean())
# print(zeros['score'].mean())
# print(ones['score'].mean() - zeros['score'].mean())
#
# print()
# print(zeros.loc[zeros['score']>=0.4])
# print(ones.loc[ones['score']<0.11])

#ones = test.loc[dataset_2['label'] == 1]
#zeros = test.loc[dataset_2['label'] == 0]

#zeros.loc[zeros['proba'] >= 0.4].to_csv(f'../../var/test_proba_{Select}_lab1.tsv', sep='\t', encoding='utf8')
#ones.loc[ones['proba'] <= 0.5].to_csv(f'../../var/test_proba_{Select}_lab0.tsv', sep='\t', encoding='utf8')

# train.idx = [i for i in range(len(train))]
fig = px.histogram(test, x='proba', color='label',
                   marginal="box",  # or violin, rug
                   hover_data=test.columns
                   )
fig.show()

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

#Select2 = 'A-By'
#Select = 'Bæ-Forlade'
#Select = 'Forlange-Gå'
Select2 = 'Gård-So'
#Select = 'Sp-Så'
#Select = 'T-Ø'
Select = 'all-4'

dataset = pd.read_csv(f'../../data/feature3/feature_{Select}.txt', '\t', encoding='utf8', header=0)
dataset = dataset.dropna()
dataset = dataset[dataset['ordklasse']=='sb.']

#dataset = pd.read_csv(f'../../data/cosine1/cosine_dataset_def_citat.tsv', '\t', encoding='utf8', header=0)

#dataset = pd.read_csv(f'../../var/_cosine_dataset_all.tsv', '\t', encoding='utf8', header=0)
#dataset = dataset.dropna()


# dataset = dataset[dataset['score']!=0]
dataset_2 = pd.read_csv(f'../../data/feature3/feature_{Select2}.txt', '\t', encoding='utf8', header=0)
dataset_2 = dataset_2.dropna()
dataset_2 = dataset_2[dataset_2['ordklasse']=='sb.']

#group = dataset.groupby('label')
#balanced_data = group.apply(lambda x: x.sample(group.size().min()).reset_index(drop=True)).sort_values(by='lemma').droplevel(0)

#group = balanced_data.groupby('lemma')
#balanced_data = group.apply(lambda x: x.iloc[:5]).droplevel(0)

#train, test = train_test_split(balanced_data, test_size=0.5, random_state=RANDOM_STATE)
#train, test = train_test_split(dataset, test_size=0.5)
# X_train, X_test, y_train, y_test = train_test_split(dataset, test_size=0.2)

train = dataset.copy()
test = dataset_2.copy()

X = train.loc[:, [   'cosine',
                     'onto',
                     'frame',
                     #'score',
                     'main_sense',
                     'figurative'
                 ]
    ]

y = train.loc[:, ['label']]

teX = test.loc[:, [   'cosine',
                      'onto',
                      'frame',
                      #'score',
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
     RandomForestClassifier(max_depth=10),
     #LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
     #KNeighborsClassifier(n_neighbors=5),
     #GaussianNB()
]

for classifier in classifiers:
    classifier.fit(X, y)
    pred_train = classifier.predict(X)
    train.loc[:, 'proba'] = classifier.predict_proba(X)[:, 1]

    train = train.sort_values(by='lemma')
    #train.to_csv('../../var/log_proba_train_semi.tsv', sep='\t', encoding='utf8')
    # print(accuracy_score(y, pred_train))

    pred_test = classifier.predict(teX)
    test.loc[:,'proba'] = classifier.predict_proba(teX)[:,1]
    test.to_csv(f'../../var/log_proba_{Select2}.tsv', sep='\t', encoding='utf8')

    print(accuracy_score(tey, pred_test))
    print(f1_score(tey, pred_test))

    cm = confusion_matrix(tey, pred_test)
    cm.diagonal()
    print(cm)


# ones = dataset.loc[dataset['label']==1]
# zeros = dataset.loc[dataset['label']==0]
#
#
# print(ones['score'].mean())
# print(zeros['score'].mean())
# print(ones['score'].mean() - zeros['score'].mean())
#
# print()
# print(zeros.loc[zeros['score']>=0.4])
# print(ones.loc[ones['score']<0.11])

ones = test.loc[dataset_2['label']==1]
zeros = test.loc[dataset_2['label']==0]

zeros.loc[zeros['proba']>=0.4].to_csv(f'../../var/test_proba_{Select}_lab1.tsv', sep='\t', encoding='utf8')
ones.loc[ones['proba']<=0.5].to_csv(f'../../var/test_proba_{Select}_lab0.tsv', sep='\t', encoding='utf8')

# train.idx = [i for i in range(len(train))]
fig = px.histogram(test, x='proba', color='label',
                   marginal="box",  # or violin, rug
                   hover_data=test.columns
                   )
fig.show()

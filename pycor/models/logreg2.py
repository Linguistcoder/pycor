import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import MinMaxScaler
import plotly.express as px

def create_dataset(list_of_subsets, columns=['0', 'lemma', 'mean', 'std_dv', 'min_mean', 'max_mean', 'type']):
    dataset = pd.DataFrame(columns=columns)
    for subset in list_of_subsets:
        data = pd.read_csv(f'../../var/devel_{subset}.tsv', '\t', encoding='utf8', header=0)
        data = data.dropna()
        dataset = pd.concat([dataset, data], ignore_index=True)
        dataset['type'] = dataset['type'].astype('int64')
    return dataset


RANDOM_STATE = 1
np.random.seed(RANDOM_STATE)

#Select = 'A-B'
#Select = 'C-D-E'
#Select = 'F'
#Select = 'G-H'
#Select = 'I-Sp'
#Select = 'St'
#Select2 = 'T-Ø'
select_tests = ['Gård-So']# ['G-H', 'I-Sp', 'St'] #['A-B', 'F', 'C-D-E', 'T-Ø']
select_trains = ['all-4']#['A-B', 'F', 'C-D-E', 'T-Ø'] #['G-H', 'I-Sp', 'St']

test = create_dataset(select_tests)
train = create_dataset(select_trains)

X = train.loc[:, ['mean',
                  'std_dv',
                  'min_mean',
                  'max_mean'
                 ]
            ]

y = train.loc[:, ['type']]

teX = test.loc[:, ['mean',
                   'std_dv',
                   'min_mean',
                   'max_mean'
                 ]
            ]

tey = test.loc[:, ['type']]


classifiers = [
     #RandomForestClassifier(max_depth=8),
     #LogisticRegression(),
     KNeighborsClassifier(n_neighbors=5),
     #GaussianNB()
]

for classifier in classifiers:
    classifier.fit(X, y)
    pred_train = classifier.predict(X)
    train.loc[:, 'proba'] = classifier.predict_proba(X)[:, 0]

    train = train.sort_values(by='lemma')
    #train.to_csv('../../var/log_proba_train_semi.tsv', sep='\t', encoding='utf8')
    print(accuracy_score(y, pred_train))

    cm = confusion_matrix(y, pred_train)
    cm.diagonal()
    print(cm)
    print(len(train))

    pred_test = classifier.predict(teX)
    test['pred'] = pred_test
    test.loc[:, 'proba'] = classifier.predict_proba(teX)[:, 0]
    test = test.sort_values(by='lemma')
    test.to_csv(f'../../var/log_proba_test_{select_tests}.tsv', sep='\t', encoding='utf8')

    print(test[test['pred']!=test['type']])
    print(accuracy_score(tey, pred_test))
    print(f1_score(tey, pred_test, average='macro'))
    cm = confusion_matrix(tey, pred_test)
    cm.diagonal()
    print(cm)
    print(len(test))



#train.idx = [i for i in range(len(train))]
# fig = px.histogram(train, x='std_dv', color='type',
#                     marginal="box",  # or violin, rug
#                     hover_data=train.columns
#                     )
#fig.show()

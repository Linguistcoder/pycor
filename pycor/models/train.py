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

#from pycor.utils.vectors import vectorize_rows, vectorize_and_cosine

RANDOM_STATE = 1
np.random.seed(RANDOM_STATE)

select = '80_10_2-5'
select_2 = '80_10_2-5'
#train_dataset = pd.read_csv(f'../../data/reduction/reduction_dataset_{select}.tsv', '\t', encoding='utf8')
train_dataset = pd.read_csv(f'../../data/bert/reduction_score_{select_2}.tsv', '\t', encoding='utf8')
train_dataset = train_dataset.dropna()
score = train_dataset.score.values.reshape(-1, 1)
# normalize data
train_dataset.score = MinMaxScaler().fit_transform(score)

#train_dataset = train_dataset[dataset['ordklasse']=='sb.']

#test_dataset = pd.read_csv(f'../../data/reduction/reduction_dataset_{select_2}.tsv', '\t', encoding='utf8')
test_dataset = pd.read_csv(f'../../data/bert/reduction_score_{select_2}.tsv', '\t', encoding='utf8')
test_dataset = test_dataset.dropna()
#test_dataset = test_dataset[dataset['ordklasse']=='sb.']

#train_dataset['score'] = train_dataset.apply(lambda row: vectorize_and_cosine(row), axis=1)
#test_dataset['score'] = test_dataset.apply(lambda row: vectorize_and_cosine(row), axis=1)

#train_dataset['vector'] = train_dataset.apply(lambda row: vectorize_rows(row), axis=1)
#test_dataset['vector'] = test_dataset.apply(lambda row: vectorize_rows(row), axis=1)


#train_dataset.to_csv(f'../../data/word2vec/reduction_score_{select}.tsv', sep='\t')
#test_dataset.to_csv(f'../../data/word2vec/reduction_score_{select_2}.tsv', sep='\t')

X = train_dataset.loc[:, ['score']]
#X = np.stack([x for x in X.reshape(-1)])

y = train_dataset.loc[:, ['label']]

teX = test_dataset.loc[:, ['score']]
#teX = np.stack([x for x in teX.reshape(-1)])

tey = test_dataset.loc[:, ['label']]


classifiers = [
     RandomForestClassifier(max_depth=10),
     LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
     KNeighborsClassifier(n_neighbors=5),
     GaussianNB()
]

for classifier in classifiers:
    classifier.fit(X, y)
    pred_train = classifier.predict(X)
    train_dataset.loc[:, 'proba'] = classifier.predict_proba(X)[:, 1]

    #train = train.sort_values(by='lemma')
    train_dataset.to_csv(f'../../var/log_proba_train_{select}.tsv', sep='\t', encoding='utf8')
    # print(accuracy_score(y, pred_train))

    pred_test = classifier.predict(teX)
    test_dataset.loc[:,'proba'] = classifier.predict_proba(teX)[:,1]
    test_dataset.to_csv(f'../../var/log_proba_{select_2}.tsv', sep='\t', encoding='utf8')

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

#ones = test.loc[dataset_2['label']==1]
#zeros = test.loc[dataset_2['label']==0]

#zeros.loc[zeros['proba']>=0.4].to_csv(f'../../var/test_proba_{Select}_lab1.tsv', sep='\t', encoding='utf8')
#ones.loc[ones['proba']<=0.5].to_csv(f'../../var/test_proba_{Select}_lab0.tsv', sep='\t', encoding='utf8')

# train.idx = [i for i in range(len(train))]
fig = px.histogram(test_dataset, x='proba', color='label',
                   marginal="box",  # or violin, rug
                   hover_data=test_dataset.columns
                   )
fig.show()

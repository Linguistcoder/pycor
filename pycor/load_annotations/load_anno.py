from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
import re
import numpy as np

anno = pd.read_csv('../../data/hum_anno/reduced_rf.txt',
                   sep='\t',
                   encoding='utf-8',
                   na_values='n')


def clean_bet(s):
    if type(s) == float:
        return s
    if s in [str(i) for i in range(20)]:
        return float(s)
    for index, i in enumerate(range(ord('a'), ord('z')+1)):
        try:
            if chr(i) in s:
                s = re.sub(chr(i), str(index+1), s)
                return float(s)
        except:
            print(i)


def clean_class(s):
    if s == 'sb.':
        return 1
    elif s == 'vb.':
        return 2
    elif s == 'adj.':
        return 3
    else:
        return 0

anno['ddo_betyd_nr'] = anno['ddo_betyd_nr'].apply(clean_bet)
anno['ddo_ordklasse'] = anno['ddo_ordklasse'].apply(clean_class)
anno['score'] = pd.to_numeric(anno['score'])

anno = anno.dropna()

labels = anno.loc[:,['cor_bet']]
X = anno.loc[:, ['score', 'ddo_ordklasse', 'ddo_betyd_nr', 'ddo_senselevel', 'ddo_plac', 'ddo_bet']]

X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2)

#clf = RandomForestClassifier(max_depth=3)
#clf = LogisticRegression(random_state=0)
clf = KNeighborsClassifier(n_neighbors=2)
clf.fit(X_train, y_train)
pred_train = clf.predict(X_train)
print(accuracy_score(y_train, pred_train))

pred_test = clf.predict(X_test)
print(accuracy_score(y_test, pred_test))

cm = confusion_matrix(y_test, pred_test)
cm.diagonal()
print(cm)


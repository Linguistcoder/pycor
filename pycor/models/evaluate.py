import pandas as pd
import numpy as np
from sklearn import metrics

model = 'base'
subset = 'keywords_test'
latest_version = '18_08_22'

clusters = pd.read_csv(f'../../var/clusters_{model}_{subset}.tsv', sep='\t')
#clusters = clusters[clusters['wcl'] == 'sb.']


if 'cbc' in subset:
    annotated = pd.read_csv(f'../../data/hum_anno/udtraek_1_{latest_version}.tsv',
                            sep='\t',
                            encoding='utf-8',
                            na_values=['n', ' '],
                            usecols=['ddo_entryid', 'ddo_lemma', 'ddo_homnr',
                                     'ddo_ordklasse', 'ddo_betyd_nr', 'COR_bet_inventar'
                                     ],
                            index_col=False
                            )
    annotated = annotated.dropna(subset=['ddo_lemma', 'COR_bet_inventar'])
    annotated['ddo_homnr'] = annotated['ddo_homnr'].fillna(1)
    annotated.columns = ['entryid', 'lemma', 'homnr', 'wcl', 'DDO', 'COR']
else:
    annotated = pd.read_csv(f'../../data/hum_anno/{subset.split("_")[0]}_{latest_version}.tsv',
                            sep='\t',
                            encoding='utf-8',
                            na_values=['n', ' '],
                            usecols=['ddo_entryid', 'ddo_lemma', 'ddo_homnr',
                                     'ddo_ordklasse', 'ddo_betyd_nr', 'cor_bet_inventar'
                                     ],
                            index_col=False
                            )
    annotated = annotated.dropna(subset=['ddo_lemma', 'cor_bet_inventar'])
    annotated['ddo_homnr'] = annotated['ddo_homnr'].fillna(1)

    annotated.columns = ['entryid', 'lemma', 'homnr', 'wcl', 'DDO', 'COR']


annotated['homnr'] = annotated['homnr'].astype('int64')
clusters['homnr'] = clusters['homnr'].astype('int64')

annotated = annotated.merge(clusters, how='outer', on=['lemma', 'DDO', 'wcl', 'homnr'])
annotated = annotated.dropna(subset=['cor', 'COR'])
annotated = annotated[annotated['COR'] != '???']
annotated = annotated[annotated['COR'] != '??']
annotated = annotated[annotated['COR'] != '3 eller 4?']
annotated['COR'] = annotated['COR'].astype('float')

correct_n = 0
total_n = 0

correct_assign = 0
total_assign = 0
multi_assign = 0
total_multi = 0
any_assign = 0

merge_bias = 0
split_bias = 0
merge_bias_n = 0
split_bias_n = 0

print(f"Number of lemmas: {annotated.groupby(['lemma', 'wcl', 'homnr']).ngroups}")
print(f"Number of lemmas: {clusters.groupby(['lemma', 'wcl', 'homnr']).ngroups}")

def check_sorted(l1, l2):
    not_sorted = False
    for val, lab in zip(l1, l2):
        if not (val[0] in lab or lab[0] in val):
            not_sorted = True
        if val[-1] in lab or lab[-1] in val:
            not_sorted = False
    return not_sorted

def fix_order(l1, l2):
    new_list1 = [[] for _ in l1]
    for value in l1:
        pos = [i for i, v2 in enumerate(l2) for v1 in value if v1 in v2]
        for p in pos:
          new_list1[p] = value
    if [] not in new_list1:
        return new_list1
    else:
        print('ER')

rand_scores = []

for name, group in annotated.groupby(['lemma', 'homnr']):
    rand_scores.append(metrics.rand_score(group['COR'], group['cor']))
    lemma = name[0]
    pred_groups = group.groupby('cor').groups
    lab_groups = group.groupby('COR').groups
    preds = sorted([list(val) for key, val in pred_groups.items() for v in val], key=lambda x: (x[0]))
    labs = sorted([list(val) for key, val in lab_groups.items() for v in val], key=lambda x: (x[0]))

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

    # if len(group) > 2:
    #     print('stop')

    not_sorted = check_sorted(preds, labs)
    if not_sorted:
        preds = fix_order(preds, labs)
        not_sorted = check_sorted(preds, labs)
    if not_sorted:
        print('ERROR')

    for index, val in enumerate(preds):
        multi = len(labs[index]) > 1
        if val[0] == labs[index][0] and val[-1] == labs[index][-1]:
            correct_assign += 1
            if multi:
                multi_assign += 1
        if val[0] == labs[index][0]:
            any_assign += 1
        total_assign += 1
        if multi:
            total_multi += 1


print(f'________________{model.upper()}_{subset.upper()}___________________')
print('rand', np.mean(rand_scores))
print('reduce:', correct_n / total_n)
print('single', correct_assign/total_assign)
print('multi:', multi_assign / total_multi)
print('any', any_assign / total_assign)

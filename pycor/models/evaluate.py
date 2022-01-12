import pandas as pd

Select = 'mellemfrek_3'  # "('80_10_l',) (1)"#'80_10_l' #'80_10_2-3'
# Select = 'C-D-E'
# Select = 'F'
# Select = 'G-H'
# Select = 'I-Sp'
# Select = 'St'
# Select = 'T-Ø'

clusters = pd.read_csv(f'../../var/clusters_{Select}.tsv', sep='\t')

# annotated = pd.read_csv('../../data/hum_anno/15_12_2021.txt',
#                         sep='\t',
#                         encoding='utf-8',
#                         na_values=['n', ' '],
#                         usecols=['ddo_entryid', 'ddo_lemma', 'ddo_homnr',
#                                  'ddo_ordklasse', 'ddo_betyd_nr', 'COR-bet.inventar'
#                                  ]
#                         )
annotated = pd.read_csv('../../data/hum_anno/mellemfrekvente.txt',
                        sep='\t',
                        encoding='utf-8',
                        na_values=['n', ' '],
                        usecols=['ddo_entryid', 'ddo_lemma', 'ddo_homnr',
                                 'ddo_ordklasse', 'ddo_betyd_nr', 'COR-bet.inventar'
                                 ]
                        )
annotated = annotated.dropna(subset=['ddo_lemma', 'COR-bet.inventar'])

#annotated.columns = ['entryid', 'lemma', 'wcl', 'DDO', 'COR', 'homnr']
# uncomment if mellemfrekvent
annotated.columns = ['entryid', 'lemma', 'homnr', 'DDO', 'wcl', 'COR' ]

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
multi = 0
total_multi = 0

print(f"Number of lemmas: {annotated.groupby(['lemma', 'wcl', 'homnr']).ngroups}")
print(f"Number of lemmas: {clusters.groupby(['lemma', 'wcl', 'homnr']).ngroups}")

for name, group in annotated.groupby(['lemma', 'wcl', 'homnr']):

    if group['cor'].max() == group['COR'].max():
        correct_n += 1
    elif group['COR'].max() > len(group):
        if group['cor'].max() == len(group):
            correct_n += 1
    else:
        print(name)
        print('Correct number:', group['COR'].max())  # -group['cor'].max())
        print('Prediction:', group['cor'].max())
        print('senses before:', len(group))

    total_n += 1

    for label, label_group in group.groupby('COR'):
        if len(label_group) > 1:
            if label_group['cor'].nunique() == 1:
                multi += 1
            total_multi += 1
        if label_group['cor'].nunique() == 1:
            correct_assign += 1
        else:
            pass
            # print(label)

        total_assign += 1

print(correct_n / total_n)
print(correct_assign / total_assign)
print(multi / total_multi)

print(correct_n)
print(total_n)
print(correct_assign)
print(total_assign)
print(multi)
print(total_multi)

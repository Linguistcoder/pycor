import pandas as pd

clusters = pd.read_csv('../../var/clusters2.tsv', sep='\t')

annotated = pd.read_csv('../../data/hum_anno/only_sense.txt', sep='\t',
                        na_values=['?','n'])

annotated = annotated.merge(clusters, how='outer', on=['lemma', 'DDO'])
annotated = annotated.dropna(subset=['cor'])
annotated['COR'] = annotated['COR'].astype('float')

correct_n = 0
total_n = 0

correct_assign = 0
total_assign = 0

for name, group in annotated.groupby('lemma'):
    if group['cor'].max() == group['COR'].max():
        correct_n += 1
    total_n += 1

    for label, label_group in group.groupby('COR'):
        if label_group['cor'].nunique() == 1:
            correct_assign += 1
        total_assign += 1

print(correct_n/total_n)
print(correct_assign/total_assign)


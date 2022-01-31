import pandas as pd
import random
from collections import namedtuple

from pycor.utils import preprocess


# anno = pd.read_csv('data/hum_anno/15_12_2021.txt',
#                    sep='\t',
#                    encoding='utf-8',
#                    na_values=['n', ' '])
#
# citater = pd.read_csv('data/citat/citater_ren.tsv',
#                       sep='\t',
#                       encoding='utf-8',
#                       usecols=['ddo_dannetsemid', 'citat'])

anno = pd.read_csv('data/hum_anno/mellemfrekvente.txt',
                   sep='\t',
                   encoding='utf-8',
                   na_values=['n', ' '])

citater = pd.read_csv('H:/CST_COR/data/DDO_citater/citater_mellemfrekvente.tsv',
                      sep='\t',
                      encoding='utf-8')

anno = anno.dropna(subset=['ddo_lemma', 'COR-bet.inventar', 'ddo_dannetsemid'])
anno['ddo_dannetsemid'] = anno['ddo_dannetsemid'].astype('int64')
# anno.merge(citater,
#           how='outer',
#           on='ddo_dannetsemid'
#           ).loc[:, ['ddo_dannetsemid', 'ddo_lemma', 'ddo_ordklasse', 'ddo_homnr', 'citat']].to_csv('var/citater.tsv',
#                                                                                 sep='\t',
#                                                                                 encoding='utf8')

citater = citater.groupby('ddo_dannetsemid').aggregate('||'.join)

# columns = ['score', 'ddo_entryid', 'ddo_lemma', 'ddo_homnr', 'ddo_ordklasse', 'ddo_dannetsemid',
#            'ddo_definition', 'ddo_genprox', 'ddo_bemaerk', 'dn_onto1', 'dn_onto2', 'dn_hyper', 'frame'
#            'ddo_valens', 'ddo_kollokation', 'COR-bet.inventar', 'Hvorfor?', 'Hvem?', 'Ny.ordklasse',
#            'Ny.definition', 'Hvem der?', 'ddo_betyd_nr', 'ddb_nøgleord', 'ddo_bet_tags', 'ddo_senselevel',
#            'ddo_plac', 'ddo_art_tags', 'ddo_bet', 'ddo_mwe_bet', 'Antal citater til bet/ddo_bet_doks',
#            'ddo_art_doks', 'ddo_udvalg', 'dn_id', 'dn_lemma', 'ddo_sublemma', 'ddo_konbet',	'ddo_encykl',
#            'ro_glosse', 'ro_opslag/afvig', 'ddo_morf']

# anno = anno.dropna(subset=['score', 'ddo_lemma', 'COR-bet.inventar'])
anno = anno[anno['COR-bet.inventar'] != '0']

anno = anno.merge(citater, how='outer', on=['ddo_dannetsemid'])
anno = anno.dropna(subset=['ddo_lemma', 'COR-bet.inventar', 'ddo_dannetsemid', 'ddo_betyd_nr'])

anno.ddo_konbet = anno.ddo_konbet.fillna('').astype(str)
anno.ddo_encykl = anno.ddo_encykl.fillna('').astype(str)
anno.ddo_definition = anno.ddo_definition.fillna('').astype(str)

anno.ddo_definition = anno[['ddo_definition', 'ddo_konbet', 'ddo_encykl']].aggregate(' '.join, axis=1)

# anno = anno.dropna(subset=['score', 'ddo_lemma', 'COR-bet.inventar'])

# reduced_anno = anno.loc[:, ['ddo_entryid', 'ddo_dannetsemid', 'ddo_lemma', 'ddo_betyd_nr', 'ddo_definition', 'citat']]
# reduced_anno[reduced_anno.citat.isna()].to_csv(r'C:\Users\nmp828\Documents\pycor\var\mangler_citat.tsv',
#                                                sep='\t',
#                                                encoding='utf8')

# anno = anno.loc[:, ['ddo_lemma', 'ddo_ordklasse', 'ddo_homnr', 'ddo_definition', 'ddo_genprox', 'ddo_kollokation',
#                     'COR-bet.inventar', 'dn_id', 'ddo_betyd_nr', 'citat', 'score', 'ddo_bemaerk',
#                     'dn_onto1', 'dn_onto2', 'dn_hyper', 'frame', 'ddo_konbet', 'ddo_encykl']]
#
# anno.columns = ['lemma', 'ordklasse', 'homnr', 'definition', 'genprox', 'kollokation', 'cor', 'dn_id', 'ddo_nr',
#                 'citat', 't_score', 'bemaerk', 'onto1', 'onto2', 'hyper', 'frame', 'konbet', 'encykl']

anno = anno.loc[:, ['ddo_lemma', 'ddo_ordklasse', 'ddo_homnr', 'ddo_definition', 'ddo_genprox', 'ddo_kollokation',
                    'COR-bet.inventar', 'dn_id', 'ddo_betyd_nr', 'citat', 'ddo_bemaerk',
                    'dn_onto1', 'dn_onto2', 'dn_hyper', 'frame', 'ddo_konbet', 'ddo_encykl']]

anno.columns = ['lemma', 'ordklasse', 'homnr', 'definition', 'genprox', 'kollokation', 'cor', 'dn_id', 'ddo_nr',
                'citat', 'bemaerk', 'onto1', 'onto2', 'hyper', 'frame', 'konbet', 'encykl']


# DanNet = load_obj('DanNet')
# DanNet = expand_synsets(expand_synsets(DanNet), s=False)


def create_dataset(annotations, words='all', sents='all', limit=0):
    # each training instance contains a
    # [0] target,
    # [1] a target sentence,
    # [2] a sense quote for each sense in the inventory (list), [3] the sense name in the inventory (list),
    # [4] and the index of the correct sense in the list of senses
    monosema = 0
    lemma_count = 0
    sen_count1 = 0
    sen_count2 = 0
    noun = 0
    adj = 0
    vb = 0
    lemmas = []
    dataset = []

    instance = namedtuple('instance',
                          ['target', 'target_sentence', 'sense_quotes', 'sense_labels', 'label_index', 'wcl'])
    print(f"Number of lemmas: {annotations.groupby(['lemma', 'ordklasse', 'homnr']).ngroups}")

    for name, group in annotations.groupby(['lemma', 'ordklasse', 'homnr']):
        definitions = {}
        examples = {}
        lemma = name[0].lower()
        wcl = name[1].lower()

        if words != 'all' and words not in wcl:
            continue

        if limit > 0 and len(list(group.ddo_nr.values)) > limit:
            continue

        for cor, sense_group in group.groupby('cor'):
            definitions[f'COR_{cor}'] = []
            examples[f'COR_{cor}'] = []

            for row in sense_group.itertuples():
                definitions[f'COR_{row.cor}'] += [f'[TGT] {lemma} [TGT] '
                                                  + preprocess.remove_special_char(row.definition)]

                if type(row.citat) != float:
                    citats = [row.citat] if '||' not in row.citat else row.citat.split('||')
                    examples[f'COR_{row.cor}'] += citats
                # else:
                # print(f'Missing citat for {lemma}: {row.definition}')

        if sents == 'all' or sents == 'def':
            sen_count1 += len(definitions)
            if not len(definitions) > 1:
                monosema += 1
                continue
            sen_count2 += len(definitions)

            for index, (s, defi) in enumerate(definitions.items()):
                if len(defi) > 1:
                    for extra_def in defi[1:]:
                        dataset.append(instance(target=lemma,
                                                target_sentence=extra_def,
                                                sense_quotes="||".join([de[0] for de in definitions.values()]),
                                                sense_labels="||".join(definitions),
                                                label_index=index,
                                                wcl=wcl
                                                ))

        if sents == 'exam':
            sen_count1 += sum([1 for exam, examlist in examples.items() if len(examlist) > 0])
            if not len(definitions) > 1:
                monosema += 1
                continue

        if sents == 'exam' or sents == 'all':
            for index, (s, example_list) in enumerate(examples.items()):
                if sents == 'exam':
                    sen_count2 += 1 if len(example_list) > 0 else 0
                for example in example_list:
                    dataset.append(instance(target=lemma,
                                            target_sentence=example,
                                            sense_quotes="||".join([de[0] for de in definitions.values()]),
                                            sense_labels="||".join(definitions),
                                            label_index=index,
                                            wcl=wcl))

                    # if len(definitions[s]) > 1 and sents == 'all':
                    #     for extra_def in definitions[s][1:]:
                    #         new_defi_list = [
                    #             de[0] if len(de) == 1 else random.choice(de[1:]) if sens != s else extra_def
                    #             for sens, de in definitions.items()]
                    #
                    #         dataset.append(instance(target=lemma,
                    #                                 target_sentence=example,
                    #                                 sense_quotes="||".join(new_defi_list),
                    #                                 sense_labels="||".join(definitions),
                    #                                 label_index=index
                    #                                 ))

        noun += 1 if 'sb.' in wcl else 0
        adj += 1 if 'adj.' in wcl else 0
        vb += 1 if 'vb.' in wcl else 0
        lemma_count += 1
        lemmas.append((lemma, wcl))

    print(f'Number of monosemic lemmas: {monosema}')
    print(f'Number of lemmas: {lemma_count}')
    print(f'Number of senses: {sen_count1}')
    print(f'Number of poly senses: {sen_count2}')
    print(f'Number of nouns: {noun}')
    print(f'Number of verbs: {vb}')
    print(f'Number of adjectives: {adj}')
    return pd.DataFrame(dataset)


def create_dataset2(annotations, words='all', sents='all', limit=0):
    # each training instance contains a
    # [0] target,
    # [1] a target sentence,
    # [2] a sense quote for each sense in the inventory (list), [3] the sense name in the inventory (list),
    # [4] and the index of the correct sense in the list of senses
    lemma_count = 0
    sen_count1 = 0
    sen_count2 = 0
    noun = 0
    adj = 0
    vb = 0
    dataset = []

    instance = namedtuple('instance', ['lemma', 'ordklasse', 'homnr', 'bet_1', 'bet_2',
                                       'sentence_1', 'sentence_2', 'label'])

    print(f"Number of lemmas: {annotations.groupby(['lemma', 'ordklasse', 'homnr']).ngroups}")

    for name, group in annotations.groupby(['lemma', 'ordklasse', 'homnr']):
        lemma = name[0].lower()
        wcl = name[1].lower()
        if words != 'all' and words not in wcl:
            continue

        hom_nr = name[2]
        senses = list(group.ddo_nr.values)

        ddo_definitions = {row.ddo_nr: [row.definition] for row in group.itertuples()}
        ddo_citat = {row.ddo_nr: row.citat.split('||') for row in group.itertuples() if type(row.citat) == str}
        ddo2cor = {row.ddo_nr: row.cor for row in group.itertuples()}

        if limit > 0 and len(list(group.ddo_nr.values)) > limit:
            continue

        if sents == 'exam':
            sen_count1 += len(ddo_citat)

            if len(ddo_citat) <= 1:
                continue
            sen_count2 += len(ddo_citat)

        else:
            sen_count1 += len(senses)  # len(ddo_citat)

            if len(senses) <= 1:
                continue

            sen_count2 += len(senses)

        noun += 1 if 'sb' in wcl else 0
        adj += 1 if 'adj' in wcl else 0
        vb += 1 if 'vb' in wcl else 0
        lemma_count += 1

        sense_pairs = set([frozenset((sens, sens2))
                           for index, sens in enumerate(senses[:-1]) for sens2 in senses[index + 1:]])
        sense_pairs = [pair for pair in sense_pairs if len(pair)>1]
        try:
            for sense, sense2 in sense_pairs:
                sentence_1 = ddo_definitions[sense] + ddo_citat.get(sense, [])
                sentence_1 = ' '.join(sentence_1)
                sentence_2 = ddo_definitions[sense2] + ddo_citat.get(sense2, [])
                sentence_2 = ' '.join(sentence_2)

                # sentences = []
                # if sents == 'def' or sents == 'all':
                #     sentences += ddo_definitions[sense2]
                # if sents == 'exam' or sents == 'all':
                #    sentences += ddo_citat.get(sense2, [])

                dataset.append(instance(lemma=lemma,
                                        ordklasse=wcl,
                                        homnr=hom_nr,
                                        bet_1=sense2,
                                        bet_2=sense,
                                        sentence_1=sentence_1,
                                        sentence_2=sentence_2,
                                        label=1 if ddo2cor[sense] == ddo2cor[sense2] else 0))
        except:
            print(sense_pairs)
            # for sentence in sentences:
            # dataset.append(instance(lemma=lemma,
            #                         ordklasse=wcl,
            #                         homnr=hom_nr,
            #                         bet_1=sense2,
            #                         bet_2=sense,
            #                         sentence_1=sentence,
            #                         sentence_2=ddo_definitions[sense][0],
            #                         label=1 if ddo2cor[sense] == ddo2cor[sense2] else 0))

    print(f'Number of lemmas: {lemma_count}')
    print(f'Number of senses: {sen_count1}')
    print(f'Number of poly senses: {sen_count2}')
    print(f'Number of nouns: {noun}')
    print(f'Number of verbs: {vb}')
    print(f'Number of adjectives: {adj}')

    return pd.DataFrame(dataset)


#bert_data = create_dataset(anno, words='sb', sents='all', limit=0)
#bert_data.to_csv('var\BERT_dataset_mellemfrekv_sb.tsv', sep='\t', encoding='utf8')

bert_data = create_dataset2(anno, words='all', sents='all', limit=5)
bert_data.to_csv(r'var\reduction_dataset_word2vec_mellem.tsv', sep='\t', encoding='utf8')

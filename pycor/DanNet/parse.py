import pandas as pd

from pycor.DanNet.dan_utils import expand_synsets
from pycor.DanNet.dannet import DanNet, Synset
from pycor.utils.save_load import save_obj


def create_dataframe():
    """Return DanNet as pd.DataFrame. The tables have been outer joined."""
    # Load data from DanNet
    file_words = 'data/DanNet/words.csv'
    file_wordsense = 'data/DanNet/wordsenses.csv'
    file_synsets = 'data/DanNet/synsets.csv'
    file_relations = 'data/DanNet/relations.csv'

    words = pd.read_csv(file_words,
                        sep='@',
                        encoding='utf8',
                        names=['word_id', 'form', 'PoS'],
                        index_col=False,
                        )

    wordsenses = pd.read_csv(file_wordsense,
                             sep='@',
                             encoding='utf8',
                             names=['wordsense_id', 'word_id', 'synset_id', 'register'],
                             index_col=False
                             )

    synsets = pd.read_csv(file_synsets,
                          sep='@',
                          encoding='utf8',
                          names=['synset_id', 'label', 'gloss', 'ont'],
                          index_col=False,
                          skiprows=0
                          )

    relations = pd.read_csv(file_relations,
                            sep='@',
                            encoding='utf8',
                            names=['synset_id', 'name', 'name2', 'value', 'taxonomic', 'in_com'],
                            index_col=False
                            )

    dataset = wordsenses.merge(words, how='outer')
    dataset.to_csv('var/words_words_sense.csv', sep='\t', index=False)
    synsets['synset_id'] = synsets['synset_id'].astype(str)
    dataset = dataset.merge(synsets.loc[:, ['synset_id', 'gloss', 'ont']], how='outer')
    dataset.to_csv('var/all_synsets.csv', sep='\t', index=False)

    return dataset, relations


def generate_relations_dict(relations: pd.DataFrame):
    hyperonyms_dict = dict()
    hyponyms_dict = dict()
    holonyms_dict = dict()
    meronyms_dict = dict()
    synonyms_dict = dict()
    others_dict = dict()

    for row in relations.itertuples():
        synset_id = int(row.synset_id)
        if any(c.isalpha() for c in row.value):
            continue

        relation_id = int(row.value)

        if 'hyponym' in row.name:
            if synset_id in hyperonyms_dict:
                hyperonyms_dict[synset_id].add(relation_id)
            else:
                hyperonyms_dict[synset_id] = {relation_id}

            if relation_id in hyponyms_dict:
                hyponyms_dict[relation_id].add(synset_id)
            else:
                hyponyms_dict[relation_id] = {synset_id}

        elif 'synonym' in row.name:
            if synset_id in synonyms_dict:
                synonyms_dict[synset_id].add(relation_id)
            else:
                synonyms_dict[synset_id] = {relation_id}

        elif 'mero' in row.name.lower():
            if synset_id in meronyms_dict:
                meronyms_dict[synset_id].add(relation_id)
            else:
                meronyms_dict[synset_id] = {relation_id}

            if relation_id in holonyms_dict:
                holonyms_dict[relation_id].add(synset_id)
            else:
                holonyms_dict[relation_id] = {synset_id}
        else:
            if synset_id in others_dict:
                others_dict[synset_id].add((row.name, relation_id))
            else:
                others_dict[synset_id] = {(row.name, relation_id)}

    return hyperonyms_dict, hyponyms_dict, synonyms_dict, holonyms_dict, meronyms_dict, others_dict


def load_dataset_into_synset(dataset: pd.DataFrame, relations):
    synset_dict = DanNet()
    word_dict = dict()

    hyper, hypo, syno, holo, mero, other = generate_relations_dict(relations)

    for row in dataset.itertuples():
        if any(c.isalpha() for c in row.synset_id):
            continue

        synset_id = int(row.synset_id)

        if synset_id in synset_dict:
            synset_dict[synset_id].wordforms.append(row.form)
            synset_dict[synset_id].word_ids.append(row.word_id)
        else:
            synset = Synset(synset_id=synset_id,
                            wordforms=[row.form],
                            word_ids=[row.word_id],
                            gloss=row.gloss,
                            ont=row.ont,
                            synonyms=list(syno.get(synset_id, [])),
                            hypernyms=list(hyper.get(synset_id, [])),
                            hyponyms=list(hypo.get(synset_id, [])),
                            holonyms=list(holo.get(synset_id, [])),
                            meronyms=list(mero.get(synset_id, [])),
                            other_relations=list(other.get(synset_id, [])))

            synset_dict[synset_id] = synset

        # todo: put in a json
        if row.form in word_dict:
            word_dict[row.form].append(synset_id)
        else:
            word_dict[row.form] = [synset_id]

    return synset_dict, word_dict


dataset1, relations1 = create_dataframe()
synsets1, words1 = load_dataset_into_synset(dataset1, relations1)
save_obj(synsets1, 'DanNet')

synsets1 = expand_synsets(expand_synsets(synsets1), s=False)

save_obj(words1, 'word2wordid')
save_obj(words1, 'word2synsetid', save_json=True)

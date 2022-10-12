import sys

from configs.config_clustering import ClusteringConfig
from pycor.load_annotations.datasets import DataSet
from pycor.load_annotations.load_annotations import read_anno
from pycor.models.bert import BertSense
from pycor.models.clustering import ClusterAlgorithm
#from pycor.models.word2vec import word2vec_model
from pycor.utils.save_load import load_obj


def get_score(row):
    """calculate sense density score (developed by Thomas Troelsgård)"""
    return ((row.ddo_bet_doks * 5) + row.ddo_bet_tags) / 10


def delete_senses(row, section):
    """
    indicates which senses in the data that should be deleted based on comments from the dictionary and
    sense density score.

    Each comment type falls within two section: domain and language use.

    In the config under the respective section, we have defined a boundary for each comment type.
    If the a sense has the comment type and the sense density score if below the boundary, then the sense is deleted
    """
    fagspec = section['fagspec']
    sprogbrug = section['sprogbrug']

    delete = 0  # 0 == keep, 1 == remove
    what = ['k', 0]  # k for keep

    if row.ddo_bemaerk:  # ddo_bemaerk is the column with comments
        bemaerk = row.ddo_bemaerk.split('; ') # comment types are separated with ;
        for item in bemaerk:
            if item in fagspec:
                what = fagspec.get(item, ['k', 0])  # get decision (k or r) and boundary. Default is keeping.
            elif item in sprogbrug:
                what = sprogbrug.get(item, ['k', 0])

            if what[0] == 'del':
                delete = 1
            elif row.ddo_betyd_nr != '1':
                if what[0] == 'r':  # r for remove
                    delete = 1
                elif row.score < what[1]:
                    delete = 1

    # always delete with a score under 0.8 unless it is the first sense in the dictionary
    if row.score < 0.8 and row.ddo_betyd_nr != '1':
        delete = 1

    return delete


def load_and_autoannotate_datasets(delete_config, models):
    """
    Loads the file in delete_config['input_data']['file'] and annotates with models

    :param delete_config: config file input file name, comment types, and boundaries
    :param models: the models to use for the annotation
    :return: annotation of input_file defined in config
    """
    if models is None:  # default is always rulebased
        models = {'rulebased': {'onto': 1, 'main': 1, 'fig': 0}}

    dataset_config = delete_config['input_data']
    dataset_name = dataset_config['dataset_name']

    print(f'_______________PROCESSING {dataset_name}____________________')
    print('Loading data...')
    anno = read_anno(anno_file=dataset_config['file'],
                     quote_file=dataset_config['quote'],
                     keyword_file=dataset_config['keyword'],
                     columns=None,
                     annotated=False
                     )
    anno['score'] = anno.apply(lambda row: get_score(row), axis=1)
    anno['delete'] = anno.apply(lambda row: delete_senses(row, config), axis=1)

    # this is used to calculate when all senses of a lemma is deleted (delete2 == 1)
    anno['delete2'] = anno.groupby(['ddo_lemma', 'ddo_ordklasse', 'ddo_homnr'])['delete'].transform('mean')
    anno_deleted = anno[anno['delete'] == 0]
    # slice anno where delete2 is 1, group by lemmas, change delete status for first sense
    anno.loc[anno[anno['delete2'] == 1].groupby(['ddo_entryid']).head(1).index, 'delete'] = 0

    if 'rulebased' in models:
        # each model uses different information. Therefore, we create specific dataset for each model
        rulebased = DataSet(anno_deleted, "rulebased_only").to_dataframe()

        # load clustering configuration for the model
        cluster_config = ClusteringConfig(model_name="base")
        cluster_algo = ClusterAlgorithm(cluster_config)
        rulebased['score'] = rulebased.apply(lambda x: 1 if x.score == 0 else 0, axis=1)

        clusters = cluster_algo.clustering(rulebased).to_dataframe()
        clusters = clusters[['cor', 'ddo_dannetsemid']]
        clusters.columns = ['rulebased', 'ddo_dannetsemid']
        anno = anno.merge(clusters, on='ddo_dannetsemid', how="outer")  # add final clusters to annotation

    if 'word2vec' in models:
        textbased = DataSet(anno_deleted, "textbased_only").to_dataframe()
        # word2vec uses cosine similarity to calculate score
        textbased['score'] = textbased.apply(lambda row: models['word2vec'].vectorize_and_cosine(row.sentence_1,
                                                                                                 row.sentence_2),
                                             axis=1)

        cluster_config = ClusteringConfig(model_name="word2vec")
        cluster_algo = ClusterAlgorithm(cluster_config)

        clusters = cluster_algo.clustering(textbased).to_dataframe()
        clusters = clusters[['cor', 'ddo_dannetsemid']]
        clusters.columns = ['word2vec', 'ddo_dannetsemid']
        anno = anno.merge(clusters, on='ddo_dannetsemid', how="outer")

    if 'bert' in models:
        bertbased = DataSet(anno_deleted, 'bert_reduction').to_dataframe()

        model = models['bert']
        bertbased = model.get_BERT_score(bertbased)
        bertbased['score'] = bertbased.apply(lambda x: 1 - x.score, axis=1)

        cluster_config = ClusteringConfig(model_name="bert")
        cluster_algo = ClusterAlgorithm(cluster_config)

        clusters = cluster_algo.clustering(bertbased).to_dataframe()
        clusters = clusters[['cor', 'ddo_dannetsemid']]
        clusters.columns = ['bert', 'ddo_dannetsemid']
        anno = anno.merge(clusters, on='ddo_dannetsemid', how="outer")

    return anno


if __name__ == "__main__":
    config_name = sys.argv[1]
    config_path = sys.argv[2]

    config = load_obj(config_name, load_json=True, path=config_path)

    # print('Loading word2vec model')
    # model_path = config['model_paths2']['word2vec']
    # word2vec = word2vec_model.load_word2vec_format(model_path,
    #                                                fvocab=model_path + '.vocab',
    #                                                binary=False)
    # print('Loaded word2vec model')
    #
    print('Loading BERT')
    bert_model = 'Maltehb/danish-bert-botxo'
    bert = BertSense.from_pretrained(bert_model)
    bert.load_tokenizer(bert_model)
    print('Loaded bert model')

    models_config = {#'rulebased': None,
                     #'word2vec': word2vec,
                     'bert': bert
                     }

    annotated = load_and_autoannotate_datasets(config, models_config)

    annotated.to_csv('var/auto_annotated_v6.tsv', sep='\t', encoding='utf8')

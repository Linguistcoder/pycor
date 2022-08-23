import sys
import torch

from pycor.load_annotations.datasets import DataSet
from pycor.load_annotations.load_annotations import read_procssed_anno, read_anno, create_or_sample_datasets
from pycor.models.bert import BertSense
from pycor.models.word2vec import word2vec_model
from pycor.utils.save_load import load_obj


def load_and_sample_datasets(config_name, config_path):
    config = load_obj(config_name, load_json=True, path=config_path)

    datasets = create_or_sample_datasets(config)

    for subset in datasets:
        print(f"______________________{subset.upper()}____________________________")
        anno = read_procssed_anno(f"data/{subset}.tsv")

        sense_selection = DataSet(anno, "sense_selection", sentence_type='all')
        sense_selection.to_tsv(f"var/BERT_dataset_{subset}.tsv")
        bert_reduction = DataSet(anno, "bert_reduction", sentence_type='all')
        bert_reduction.to_tsv(f"data/reduction/reduction_{subset}.tsv")

        if 'mellem' not in subset:
            sense_selection2 = DataSet(anno, "sense_selection", sentence_type='all', max_sense=5)
            sense_selection2.to_tsv(f"var/BERT_dataset_{subset}_less.tsv")
            bert_reduction2 = DataSet(anno, "bert_reduction", sentence_type='all', max_sense=5)
            bert_reduction2.to_tsv(f"data/reduction/reduction_{subset}_less.tsv")
            text_based2 = DataSet(anno, "textbased_only", max_sense=5)
            text_based2.to_tsv(f"data/reduction/reduction_word2vec_{subset}_less.tsv")
            rule_based2 = DataSet(anno, "rulebased_only", max_sense=5)
            rule_based2.to_tsv(f"data/base/reduction_score_{subset}_less.tsv")

        text_based = DataSet(anno, "textbased_only")
        text_based.to_tsv(f"data/reduction/reduction_word2vec_{subset}.tsv")
        rule_based = DataSet(anno, "rulebased_only")
        rule_based.to_tsv(f"data/base/reduction_score_{subset}.tsv")

        # feature_based


def load_feature_dataset(config_name, config_path, infotypes, models, save_sample='data/', save_final='var/',
                         sample=True):
    config = load_obj(config_name, load_json=True, path=config_path)

    datasets = create_or_sample_datasets(config, sampled=sample, save_path=save_sample)

    for subset in datasets:
        anno = read_procssed_anno(f"{save_sample}{subset}.tsv")

        feature_dataset = DataSet(anno, "feature", infotypes=infotypes, embedding_type=models)
        feature_dataset.to_tsv(f"{save_final}{subset}_feature_dataset.tsv")


def generate_embeddings(filename, models, save_path):
    print("Loading data...")
    anno = read_anno(anno_file=filename,
                     quote_file='',
                     keyword_file='',
                     annotated=True)

    print('Data loaded.')

    DataSet(anno,
            "generate_embeddings",
            embedding_type=models,
            output_path=save_path)


if __name__ == "__main__":
    config_path = sys.argv[2]

    run_type = sys.argv[1]

    if run_type == 's' or run_type == 'sample':
        config_name = "config_datasets"  # "config_model"
        load_and_sample_datasets(config_name, config_path)

    else:
        config_name = "config_model"
        config = load_obj(config_name, load_json=True, path=config_path)

        print('Loading word2vec model')
        model_path = config['model_paths2']['word2vec']
        word2vec = word2vec_model.load_word2vec_format(model_path,
                                                       fvocab=model_path + '.vocab',
                                                       binary=False)
        print('Loaded word2vec model')

        print('Loading BERT')
        bert_model = 'Maltehb/danish-bert-botxo'
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # config_bert = BertConfig.from_pretrained(bert_model, num_labels=2)
        bert = BertSense.from_pretrained(bert_model)
        bert.load_tokenizer(bert_model)
        bert.load_checkpoint('pycor/models/checkpoints/model_0.pt')
        bert.to(device)

        models = {'bert': bert, 'word2vec': word2vec}

        if run_type == 'c' or run_type == 'create':
            infos = ['cosine', 'bert', 'onto', 'main_sense', 'figurative']
            load_feature_dataset("config_datasets",
                                 config_path,
                                 infos,
                                 models,
                                 save_sample=sys.argv[3],
                                 save_final=sys.argv[4])

        if run_type == 'e' or run_type == 'embed':
            filename = sys.argv[3]
            generate_embeddings(filename, models, save_path=config['model_paths']['save_path'])
            # generate_embeddings(filename, None, word2vec, save_path=config['model_paths2']['save_path'])

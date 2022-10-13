# COR project code

This repository contains the main code base for COR-S: 
the semantic component of [COR project].

We have developed a clustering method to
automatically reduce a fine-grained sense inventory
based on dictionary information.

## The automatic clustering method
The method involves three steps:
1. pair senses within a lemma
2. calculate _sense proximity score_ for each pair
3. cluster based on the score

and includes four different approaches / models to calculate _the sense
proximity score_:
1. **Rule-based** that uses the annotation principles
2. Cosine similarity of **word2vec** centroid representations of the textual data in a sense entry
3. A **BERT model** that has been fine-tuned to compute a score from dictionary definitions and quotes
4. A **feature-based** model that is trained on features from the above model

### Loading the gold standard (annotated data)
We developed the clustering method based on a gold standard of annotated data.
Therefore, we include a pipeline for loading and clustering annotated data.

In order to cluster annotated data, a pair-wise reduction dataset must be made
for each approach / model, which is done by `the generate_datasets.py` script (see section _run_code_).

When the pair-wise reduction datasets have been created, then we can cluster and evaluate the clusters using the script `cluster_and_evaluate.py`.

### New, unannotated data
For new data without an annotation, the entire three-step method is done by the script: `auto_annotate.py`.
This script also automatically deletes unwanted senses based on comments from the dictionary. 

## Run code

### config
The config file is split into two parts:
1. datasets that needs to be processed: 
```
"datasets": [
    {
      "name": "test",  # name of dataset
      "file": "data/hum_anno/test_data.txt",  # input file
      "quote": "data/citat/citater_test_ren.tsv",  # quote file
      "sample_size": 0.5,   # if sampling, the size of test set
      "bias": 0 
    }
  ]
  ```
2. models that are used for calculating scores
```
  "models": {
    "bert_name":  "Maltehb/danish-bert-botxo",  # name of bert model on HuggingFace
    "bert_checkpoint": "pycor/models/checkpoints/model_0.pt",  path to checkpoint
    "save_path": "var/"  where to save processed files
  }
  ```

### generate_datasets.py
The script takes three optional flag arguments:
- `-s (--sample)`: samples and creates pair-wise reduction datasets (not feature). The sampling splits a dataset up into a train, (devel), and test set
- `-e (--embed)`: embeds each row (sense) in a non-paired dataset using word2vec and/or BERT
- `-f (--feature)`: creates pair-wise feature dataset

The script requires the argument `config_path` which should be the path to a config file as described above

The script is run in the command line:
```
pycor/ python generate_datasets.py -s /data -e /data -f /data configs/config_test.json
```

### cluster_and_evaluate.py
```
pycor python cluster_and_evaluate.py base mellem_test -t mellem_train -b 0.5
```

###

[COR project]: <https://cst.ku.dk/english/projects/the-central-word-register-for-danish-cor/>
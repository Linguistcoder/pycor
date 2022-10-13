import pathlib
from dataclasses import dataclass
from typing import Optional, Dict

import dacite

import pycor


@dataclass
class Configuration:
    datasets: pycor.load_annotations.config.Configuration
    models: Optional[pycor.models.config.Configuration]
    fagspec: Optional[Dict]
    sprogbrug: Optional[Dict]


converters = {
    pathlib.Path: pathlib.Path,
    int: int,
    float: float,
    pycor.models.config.ClusteringConfig: pycor.models.config.ClusteringConfig
}


def load_config_from_json(json):
    datasets = json.get('datasets', None)

    if not datasets:
        raise Exception

    datasets = [dacite.from_dict(data_class=pycor.load_annotations.config.DatasetConfig, data=val,
                                 config=dacite.Config(type_hooks=converters))
                for val in datasets]

    json['datasets'] = pycor.load_annotations.config.Configuration(datasets)

    models = json.get('models', None)

    if models:
        json['models'] = dacite.from_dict(data_class=pycor.models.config.Configuration, data=models,
                                          config=dacite.Config(type_hooks=converters))

    config = dacite.from_dict(data_class=Configuration,
                              data=json,
                              config=dacite.Config(type_hooks=converters))

    return config

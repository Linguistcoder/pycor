import pickle
import json
from typing import Optional


def save_obj(obj, name, save_json=False):
    """save obj as ('var/' + name) either as pickle obj (save_json=False) or json obj (save_json=True)"""
    if save_json:
        with open(f'var/{name}.json', 'w') as json_file:
            json.dump(obj, json_file)

    else:
        with open('var/' + name + '.pkl', 'wb') as f:
            pickle.dump(obj, f)


def load_obj(name, load_json: False, path: Optional[str] = None):
    """load obj ('var/' + name) as either pickled obj (load_json=False) or as json obj (load_json=True)"""
    if load_json:
        if path:
            with open(str(path) + str(name) + '.json', 'rb') as f:
                return json.load(f)
        else:
            with open(str(name) + '.json', 'rb') as f:
                return json.load(f)
    else:
        if path:
            with open(str(path) + str(name) + '.pkl', 'rb') as f:
                return pickle.load(f)
        else:
            with open(str(name) + '.pkl', 'rb') as f:
                return pickle.load(f)

import pickle
import json


def save_obj(obj, name, save_json=False):
    if save_json:
        with open(f'var/{name}.json', 'w') as json_file:
            json.dump(obj, json_file)

    else:
        with open('var/' + name + '.pkl', 'wb') as f:
            pickle.dump(obj, f)


def load_obj(name, load_json=False, path=None):
    if load_json:
        if path:
            with open(path + name + '.json', 'rb') as f:
                return json.load(f)
        else:
            with open('var/' + name + '.json', 'rb') as f:
                return json.load(f)
    else:
        if path:
            with open(path + name + '.pkl', 'rb') as f:
                return pickle.load(f)
        else:
            with open('var/' + name + '.pkl', 'rb') as f:
                return pickle.load(f)

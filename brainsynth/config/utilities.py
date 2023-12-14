from pathlib import Path
from types import SimpleNamespace

import yaml

from brainsynth import root_dir
from brainsynth.constants import constants

def load_config(filename=None):
    filename = filename or root_dir / "config" / "synthesizer.yaml"
    with open(filename, "r") as f:
        config = yaml.load(f, Loader=get_loader())
    return recursive_dict_to_namespace(config)

def get_loader():
    loader = yaml.SafeLoader
    loader.add_constructor("!Path", path_constructor)
    loader.add_constructor("!LabelingScheme", labeling_scheme_constructor)
    loader.add_constructor("!include", include_constructor)
    return loader

def path_constructor(loader, node):
    """"""
    return Path(loader.construct_scalar(node))

def labeling_scheme_constructor(loader, node):
    return getattr(constants.labeling_scheme, loader.construct_scalar(node))

def include_constructor(loader, node):
    # os.path.join(os.path.dirname(loader.name), node.value)
    with open(Path(loader.name).parent / loader.construct_scalar(node)) as f:
        return yaml.load(f, Loader=yaml.SafeLoader)

def recursive_dict_to_namespace(d):
    namespace = d
    if isinstance(d, dict):
        namespace = SimpleNamespace(**d)
        for k,v in namespace.__dict__.items():
            setattr(namespace, k, recursive_dict_to_namespace(v))
    return namespace

def recursive_namespace_to_dict(ns):
    d = ns
    if isinstance(d, SimpleNamespace):
        d = vars(d)
        for k,v in d.items():
            d[k] = recursive_namespace_to_dict(v)
    return d

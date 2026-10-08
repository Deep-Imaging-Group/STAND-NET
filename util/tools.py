import os
import threading
import glob
import time

from tqdm import tqdm
import visdom
from loguru import logger
import yaml
from collections import defaultdict

def get_class_or_function_name(obj):
        try:
            name = obj.__name__
        except:
            name = obj.__class__.__name__
        
        return get_initial_chars(name)


def wrap_tqdm_write(msg):
    tqdm.write(msg, end="")

def remove_file(file_name):
    drop_file = glob.glob(file_name)
    if drop_file:
        for i in drop_file:
            os.remove(i)


def get_formated_args(args, except_keys: list=[]):
    infos = [""]
    if isinstance(args, dict):
        for key, val in sorted(args.items()):
            if "func" not in key and key not in except_keys:
                infos.append(("{:>20}: {}".format(key, val)))
    else:
        for key, val in sorted(vars(args).items()):
            if "func" not in key and key not in except_keys:
                infos.append(("{:>20}: {}".format(key, val)))
    
    return "\n".join(infos)

def get_initial_chars(text):
    words = []
    for c in text:
        if c.isupper():
            words.append(c)
    
    if words:
        return "".join(words)
    else:
        return text



def update_args_by_yaml(args):
    yaml_path = args.yaml
    if yaml_path is not None:
        logger.debug(yaml_path)
        if not os.path.exists(yaml_path):
            raise Exception("The path {} does not exists!".format(yaml_path))
        with open(yaml_path, encoding="utf-8") as f:
            opt = yaml.load(f, Loader = yaml.FullLoader)

        args = vars(args)
        args.update(opt)
    return args


def csv_to_json(path):
    res = defaultdict(str)
    if not os.path.exists(path):
        return res
    with open(path) as f:
        lines = f.readlines()

    for line in lines:
        items = line.strip("\n").split(",")
        key = items[0]
        value = ",".join(items[1:])
        res[key] = value
    logger.info(f"Finished to read metric values of all method from {path}")
    return res


def json_to_csv(json_data, out_path):
    with open(out_path, "w") as f:
        for key, value in json_data.items():
            f.write(f"{key},{value}\n")
    logger.info(f"Finished to write metric values of all method to {out_path}")


def get_filename_from_path(path):
    basename = os.path.basename(path)
    file_name = basename.split(".")[0]
    return file_name

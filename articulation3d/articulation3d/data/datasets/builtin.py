# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
This file registers pre-defined datasets at hard-coded paths, and their metadata.
We hard-code metadata for common datasets. This will enable:
1. Consistency check when loading the datasets
2. Use models on these standard datasets directly and run demos,
   without having to download the dataset annotations
We hard-code some paths to the dataset that's assumed to
exist in "./datasets/".
"""
import os
from detectron2.data import DatasetCatalog, MetadataCatalog

from articulation3d.data.datasets import load_scannet_json, load_arti_json


def get_scannet_metadata():
    meta = [
        {"name": "plane", "color": [230, 25, 75], "id": 1},  # noqa
        {"name": "plane2", "color": [230, 25, 75], "id": 2},  # noqa
    ]
    return meta

def get_arti_metadata():
    meta = [
        {"name": "arti_rot", "color": [0, 130, 200], "id": 1},  # noqa
        {"name": "arti_tran", "color": [230, 25, 75], "id": 2},  # noqa
    ]
    return meta


SCANNET_SPLITS = {
    "scannet_surreal_val": ("scannet_surreal", "scannetv2_surreal/cached_set_val.json"),
    "scannet_surreal_train": ("scannet_surreal", "scannetv2_surreal/cached_set_train.json"),
    "scannet_val": ("scannet", "scannetv2/cached_set_val.json"),
    "scannet_train": ("scannet", "scannetv2/cached_set_train.json"),
}

ARTI_SPLITS = {
    "arti_val": ("arti", "articulation/cached_set_val.json"),
    "arti_test": ("arti", "articulation/cached_set_test.json"),
    "arti_train": ("arti", "articulation/cached_set_train.json"),
}


def register_scannet(dataset_name, json_file, image_root, root="datasets"):
    DatasetCatalog.register(
        dataset_name, lambda: load_scannet_json(json_file, dataset_name)
    )
    things_ids = [k["id"] for k in get_scannet_metadata()]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(things_ids)}
    thing_classes = [k["name"] for k in get_scannet_metadata()]
    thing_colors = [k["color"] for k in get_scannet_metadata()]
    json_file = os.path.join(root, json_file)
    image_root = os.path.join(root, image_root)
    metadata = {
        "thing_classes": thing_classes,
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_colors": thing_colors,
    }
    MetadataCatalog.get(dataset_name).set(
        json_file=json_file, image_root=image_root, evaluator_type="mp3d", **metadata
    )

def register_arti(dataset_name, json_file, image_root, root="datasets"):
    DatasetCatalog.register(
        dataset_name, lambda: load_arti_json(json_file, dataset_name)
    )
    things_ids = [k["id"] for k in get_arti_metadata()]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(things_ids)}
    thing_classes = [k["name"] for k in get_arti_metadata()]
    thing_colors = [k["color"] for k in get_arti_metadata()]
    json_file = os.path.join(root, json_file)
    image_root = os.path.join(root, image_root)
    metadata = {
        "thing_classes": thing_classes,
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_colors": thing_colors,
    }
    MetadataCatalog.get(dataset_name).set(
        json_file=json_file, image_root=image_root, evaluator_type="arti", **metadata
    )

for key, (data_root, anno_file) in SCANNET_SPLITS.items():
    register_scannet(key, anno_file, data_root)

for key, (data_root, anno_file) in ARTI_SPLITS.items():
    register_arti(key, anno_file, data_root)

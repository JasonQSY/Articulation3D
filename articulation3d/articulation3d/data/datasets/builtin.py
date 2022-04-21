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
        # {"name": "object", "color": [255, 255, 25], "id": 1},  # noqa
        # {"name": "layout", "color": [230, 25, 75], "id": 2},  # noqa
        {"name": "plane", "color": [230, 25, 75], "id": 1},  # noqa
        {"name": "plane2", "color": [230, 25, 75], "id": 2},  # noqa
        # {"name": "bookcase", "color": [230, 25, 75], "id": 2},  # noqa
        # {"name": "chair", "color": [250, 190, 190], "id": 3},  # noqa
        # {"name": "desk", "color": [60, 180, 75], "id": 4},  # noqa
        # {"name": "misc", "color": [230, 190, 255], "id": 5},  # noqa
        # {"name": "sofa", "color": [0, 130, 200], "id": 6},  # noqa
        # {"name": "table", "color": [245, 130, 48], "id": 7},  # noqa
        # {"name": "tool", "color": [70, 240, 240], "id": 8},  # noqa
        # {"name": "wardrobe", "color": [210, 245, 60], "id": 9},  # noqa
    ]
    return meta

def get_arti_metadata():
    meta = [
        # {"name": "object", "color": [255, 255, 25], "id": 1},  # noqa
        # {"name": "layout", "color": [230, 25, 75], "id": 2},  # noqa
        {"name": "arti_rot", "color": [0, 130, 200], "id": 1},  # noqa
        {"name": "arti_tran", "color": [230, 25, 75], "id": 2},  # noqa
        # {"name": "bookcase", "color": [230, 25, 75], "id": 2},  # noqa
        # {"name": "chair", "color": [250, 190, 190], "id": 3},  # noqa
        # {"name": "desk", "color": [60, 180, 75], "id": 4},  # noqa
        # {"name": "misc", "color": [230, 190, 255], "id": 5},  # noqa
        # {"name": "sofa", "color": [0, 130, 200], "id": 6},  # noqa
        # {"name": "table", "color": [245, 130, 48], "id": 7},  # noqa
        # {"name": "tool", "color": [70, 240, 240], "id": 8},  # noqa
        # {"name": "wardrobe", "color": [210, 245, 60], "id": 9},  # noqa
    ]
    return meta

SCANNET_SPLITS = {
    "scannet_surreal_val": ("scannet_surreal", "scannetv2_surreal/cached_set_val.json"),
    "scannet_surreal_train": ("scannet_surreal", "scannetv2_surreal/cached_set_train.json"),
    "scannet_val": ("scannet", "scannetv2/cached_set_val.json"),
    "scannet_train": ("scannet", "scannetv2/cached_set_train.json"),
    # "scannet_surreal_train": ("scannet_surreal", "scannetv2_surreal/cached_set_train_overfit.json"),
    # "scannet_surreal_val": ("scannet_surreal", "scannetv2_surreal/cached_set_train_overfit.json"),
}

ARTI_SPLITS = {
    "arti_val": ("arti", "v5_arti_RT/cached_set_val.json"),
    #"arti_test": ("arti", "v5_arti_RT/cached_set_test.json"),
    "arti_test": ("arti", "v6_arti_RT_normal/cached_set_test.json"),
    "arti_train": ("arti", "v5_arti_RT/cached_set_train.json"),
}

ARTI_VLFB_SPLITS = {
    "arti_val_vlfb": ("arti", "v6_arti_RT_normal_vlfb/cached_set_val.json"),
    "arti_test_vlfb": ("arti", "v6_arti_RT_normal_vlfb/cached_set_test.json"),
    "arti_train_vlfb": ("arti", "v6_arti_RT_normal_vlfb/cached_set_train.json"),
}

ARTI_SPLITS_OVERFIT = {
    #"arti_val_overfit": ("arti", "v4_arti_overfit/cached_set_val.json"),
    #"arti_test_overfit": ("arti", "v4_arti_overfit/cached_set_test.json"),
    #"arti_train_overfit": ("arti", "v4_arti_overfit/cached_set_train.json"),
    "arti_val_overfit": ("arti", "v5_arti_RT/cached_set_val.json"),
    "arti_test_overfit": ("arti", "v5_arti_RT/cached_set_test.json"),
}

SAPIEN_SPLITS = {
    "sapien_surreal_train": ("arti", "sapien/cached_set_train.json"),
    "sapien_surreal_test": ("arti", "sapien/cached_set_test.json"),
    "sapien_surreal_val": ("arti", "sapien/cached_set_val.json"),
}

CHARADES_SPLITS = {
    "charades_test": ("charades", "v1_charades/cached_set_test.json"),
    "charades_test_hq": ("charades", "v1_charades_hq/cached_set_test.json"),
    "charades_test_480p": ("charades", "v1_charades_480p/cached_set_test.json"),
    'charades_test_hq': ("charades", "v1_charades_hq/cached_set_test.json")
}

CHARADES_SUBSPLITS = {
    "charades_test_6_or_better": ("charades", "v1_charades_6_or_better/cached_set_test.json"),
    "charades_test_3_or_better": ("charades", "v1_charades_3_or_better/cached_set_test.json"),
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

for key, (data_root, anno_file) in ARTI_VLFB_SPLITS.items():
    register_arti(key, anno_file, data_root)

for key, (data_root, anno_file) in ARTI_SPLITS_OVERFIT.items():
    register_arti(key, anno_file, data_root)

for key, (data_root, anno_file) in CHARADES_SPLITS.items():
    register_arti(key, anno_file, data_root)

for key, (data_root, anno_file) in CHARADES_SUBSPLITS.items():
    register_arti(key, anno_file, data_root)

for key, (data_root, anno_file) in SAPIEN_SPLITS.items():
    register_arti(key, anno_file, data_root)

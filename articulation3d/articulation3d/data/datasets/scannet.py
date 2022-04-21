# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import os
import json
from detectron2.data import MetadataCatalog
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import Visualizer

"""
This file contains functions to parse COCO-format annotations into dicts in "Detectron2 format".
"""

logger = logging.getLogger(__name__)

__all__ = ["load_scannet_json"]


def load_scannet_json(json_file, dataset_name=None):
    """
    Load a json file with scannet's instances annotation format.

    Args:
        json_file (str): full path to the json file in COCO instances annotation format.
        image_root (str): the directory where the images in this json file exists.
        dataset_name (str): the name of the dataset (e.g., coco_2017_train).
            If provided, this function will also put "thing_classes" into
            the metadata associated with this dataset.

    Returns:
        list[dict]: a list of dicts in "Detectron2 Dataset" format. (See DATASETS.md)

    Notes:
        1. This function does not read the image files.
           The results do not have the "image" field.
    """
    with open(json_file, 'r') as f:
        summary = json.load(f)
    # meta = MetadataCatalog.get(dataset_name)
    # cats = summary['categories']
    # thing_classes = [c["name"] for c in sorted(cats, key=lambda x: x["id"])]
    # meta.thing_classes = thing_classes
    return summary['data']


if __name__ == "__main__":
    """
    Test the scannet json dataset loader.

    Usage:
        python -m planercnn.data.datasets.scannet \
            datasets/scannetv2_surreal/cached_set_val.json scannet_val

        "dataset_name" can be "coco", "coco_person", or other
        pre-registered ones
    """
    from detectron2.utils.logger import setup_logger
    import detectron2.data.datasets  # noqa # add pre-defined metadata
    import cv2
    import sys

    logger = setup_logger(name=__name__)
    meta = MetadataCatalog.get(sys.argv[2])

    dicts = load_scannet_json(sys.argv[1], sys.argv[2])
    logger.info("Done loading {} samples.".format(len(dicts)))

    dirname = "scannet-data-vis"
    os.makedirs(dirname, exist_ok=True)
    for d in dicts:
        img = cv2.imread(d["file_name"])
        img = cv2.resize(img, (d['width'], d['height']))
        import pdb;pdb.set_trace()
        visualizer = Visualizer(img[:, :, ::-1], metadata=meta, scale=0.5)
        for i in range(len(d['annotations'])):
            d['annotations'][i]['bbox_mode'] = BoxMode(d['annotations'][i]['bbox_mode'])
        vis = visualizer.draw_dataset_dict(d)
        # vis = draw_mp3d_dict(d['0'], meta.thing_classes + ["0"])
        # vis = draw_mp3d_dict(d[1], meta.thing_classes + ["0"])
        fpath = os.path.join(dirname, os.path.basename(d["file_name"]))
        vis.save(fpath)
        exit()

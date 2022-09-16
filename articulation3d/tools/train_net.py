import numpy as np
import os
from collections import OrderedDict
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import inference_on_dataset
from detectron2.utils.logger import setup_logger

# required so that .register() calls are executed in module scope
import articulation3d.modeling  # noqa
from articulation3d.config import get_planercnn_cfg_defaults
from articulation3d.data import PlaneRCNNMapper
from articulation3d.evaluation import ScannetEvaluator, ArtiEvaluator


class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type == "mp3d":
            return ScannetEvaluator(dataset_name, cfg, True, output_dir=cfg.OUTPUT_DIR)
        elif evaluator_type == "arti":
            return ArtiEvaluator(dataset_name, cfg, True, output_dir=cfg.OUTPUT_DIR)
        else:
            raise ValueError("The evaluator type is wrong")

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(
            cfg, dataset_name, mapper=PlaneRCNNMapper(cfg, False, dataset_names=(dataset_name,))
        )

    @classmethod
    def build_train_loader(cls, cfg):
        dataset_names = cfg.DATASETS.TRAIN
        return build_detection_train_loader(
            cfg, mapper=PlaneRCNNMapper(cfg, True, dataset_names=dataset_names)
        )

    @classmethod
    def test(cls, cfg, model):
        """
        Args:
            cfg (CfgNode):
            model (nn.Module):

        Returns:
            dict: a dict of result metrics
        """
        results = OrderedDict()
        for dataset_name in cfg.DATASETS.TEST:
            data_loader = cls.build_test_loader(cfg, dataset_name)
            evaluator = cls.build_evaluator(cfg, dataset_name)
            results_i = inference_on_dataset(model, data_loader, evaluator)
            results[dataset_name] = results_i
            if comm.is_main_process():
                assert isinstance(
                    results_i, dict
                ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                    results_i
                )
        return results


def setup(args):
    cfg = get_cfg()
    get_planercnn_cfg_defaults(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    # Setup logger for "articulation3d" module
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="articulation3d")
    return cfg


def main(args):
    cfg = setup(args)
    print(cfg)
    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        print(res)
        return res

    trainer = Trainer(cfg)

    print("# of layers require gradient:")
    for c in trainer.checkpointer.model.named_children():
        grad = np.array([param.requires_grad for param in 
            getattr(trainer.checkpointer.model, c[0]).parameters()])
        print(c[0], grad.sum())
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )

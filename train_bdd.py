import logging
import os
from collections import OrderedDict
import torch

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import COCOEvaluator
from detectron2.modeling import GeneralizedRCNNWithTTA
from detectron2 import model_zoo
from detectron2.data.datasets import register_coco_instances


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can write your
    own training loop. You can use "tools/plain_train_net.py" as an example.
    """
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()

    # set config file
    cfg.merge_from_file("configs/BDD100K-Detection/faster_rcnn/faster_rcnn_R_101_C4_1x_crop.yaml")
    cfg.DATASETS.TRAIN = ("bdd100k_train",)
    cfg.DATASETS.TEST = ("bdd100k_val",)
    # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")  # Let training initialize from model zoo


    cfg.OUTPUT_DIR = 'bdd100k_val'
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    # new added solver arguments
    cfg.SOLVER.CHECKPOINT_PERIOD = 5000
    cfg.TEST.EVAL_PERIOD = 5000
    # end of new arguments

    # cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)
    # dataset
    register_coco_instances("bdd100k_train", {}, "train_coco.json", "/shared/xudongliu/bdd100k/100k/train")
    register_coco_instances("bdd100k_val", {}, "val_coco.json", "/shared/xudongliu/bdd100k/100k/val")

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    # dataset registration
    # global DatasetCatalog, MetadataCatalog


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
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Modified by Feng Wang
"""
Detection Training Script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in dl_lib.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use dl_lib as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""

import os
import sys
sys.path.insert(0, '.')  # noqa: E402

from colorama import Fore, Style

import dl_lib.utils.comm as comm
from config import config
from dl_lib.checkpoint import DetectionCheckpointer
from dl_lib.data import MetadataCatalog
from dl_lib.engine import (DefaultTrainer, default_argument_parser,
                           default_setup, hooks, launch)
from dl_lib.evaluation import (COCOEvaluator, DatasetEvaluators,
                               PascalVOCDetectionEvaluator, verify_results)
from net import build_model

from dl_lib.data.datasets import register_coco_instances


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains a number pre-defined logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can use the cleaner
    "SimpleTrainer", or write your own training loop.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)


def main(args):
    # register dataset
    register_coco_instances("bdd100k_train", {}, "/shared/xudongliu/bdd100k/labels/bdd100k_labels_images_det_coco_train.json", "/shared/xudongliu/bdd100k/100k/train")
    register_coco_instances("bdd100k_val", {}, "/shared/xudongliu/bdd100k/labels/bdd100k_labels_images_det_coco_val.json", "/shared/xudongliu/bdd100k/100k/val")
    # register dataset end

    config.merge_from_list(args.opts)
    cfg, logger = default_setup(config, args)
    model = build_model(cfg)
    logger.info(f"Model structure: {model}")
    file_sys = os.statvfs(cfg.OUTPUT_DIR)
    free_space_Gb = (file_sys.f_bfree * file_sys.f_frsize) / 2**30
    # We assume that a single dumped model is 700Mb
    eval_space_Gb = (cfg.SOLVER.LR_SCHEDULER.MAX_ITER // cfg.SOLVER.CHECKPOINT_PERIOD) * 700 / 2**10
    if eval_space_Gb > free_space_Gb:
        logger.warning(f"{Fore.RED}Remaining space({free_space_Gb}GB) "
                       f"is less than ({eval_space_Gb}GB){Style.RESET_ALL}")
    if args.eval_only:
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop or subclassing the trainer.
    """
    trainer = Trainer(cfg, model)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )

    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("soft link to {}".format(config.OUTPUT_DIR))
    config.link_log()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )

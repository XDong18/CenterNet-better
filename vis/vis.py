import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import tqdm
import sys
sys.path.insert(0, '.')  # noqa: E402

# from dl_lib.config import get_cfg
from dl_lib.data.detection_utils import read_image
# from dl_lib.utils.logger import setup_logger
from dl_lib.data.datasets import register_coco_instances

from dl_lib.engine import (DefaultTrainer, default_argument_parser,
                           default_setup, hooks, launch)

from predictor import VisualizationDemo

from config import config

# constants
# WINDOW_NAME = "COCO detections"

# def setup_cfg(args):
#     # load config from file and command-line arguments
#     cfg = get_cfg()
#     # To use demo for Panoptic-DeepLab, please uncomment the following two lines.
#     # from detectron2.projects.panoptic_deeplab import add_panoptic_deeplab_config  # noqa
#     # add_panoptic_deeplab_config(cfg)
#     cfg.merge_from_file(args.config_file)
#     cfg.merge_from_list(args.opts)
#     # Set score_threshold for builtin models
#     cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
#     cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
#     cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
#     cfg.freeze()
#     return cfg

def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    # parser.add_argument(
    #     "--config-file",
    #     default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
    #     metavar="FILE",
    #     help="path to config file",
    # )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser

if __name__ == "__main__":
    # dataset
    register_coco_instances("bdd100k_train", {}, "/shared/xudongliu/bdd100k/labels/bdd100k_labels_images_det_coco_train.json", "/shared/xudongliu/bdd100k/100k/train")
    register_coco_instances("bdd100k_val", {}, "/shared/xudongliu/bdd100k/labels/bdd100k_labels_images_det_coco_val.json", "/shared/xudongliu/bdd100k/100k/val")

    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    config.merge_from_list(args.opts)
    cfg, logger = default_setup(config, args)
    # setup_logger(name="fvcore")
    # logger = setup_logger()
    logger.info("Arguments: " + str(args))

    # cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)

    if args.input:
        if len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"
        for path in tqdm.tqdm(args.input, disable=not args.output):
            # use PIL, to be consistent with evaluation
            img = read_image(path, format="BGR")
            start_time = time.time()
            predictions, visualized_output = demo.run_on_image(img)
            logger.info(
                "{}: {} in {:.2f}s".format(
                    path,
                    "detected {} instances".format(len(predictions["instances"]))
                    if "instances" in predictions
                    else "finished",
                    time.time() - start_time,
                )
            )

            if args.output:
                if os.path.isdir(args.output):
                    assert os.path.isdir(args.output), args.output
                    out_filename = os.path.join(args.output, os.path.basename(path))
                else:
                    assert len(args.input) == 1, "Please specify a directory with args.output"
                    out_filename = args.output
                visualized_output.save(out_filename)

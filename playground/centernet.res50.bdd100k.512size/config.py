import os.path as osp

from dl_lib.configs.base_detection_config import BaseDetectionConfig

_config_dict = dict(
    MODEL=dict(
        # WEIGHTS="detectron2://ImageNetPretrained/MSRA/R-18.pth",
        WEIGHTS="detectron2://ImageNetPretrained/MSRA/R-50.pkl",
        # WEIGHTS="",
        MASK_ON=False,
        RESNETS=dict(DEPTH=50),
        # PIXEL_MEAN=[0.485, 0.456, 0.406],
        # PIXEL_STD=[0.229, 0.224, 0.225],
        CENTERNET=dict(
            DECONV_CHANNEL=[2048, 256, 128, 64],
            DECONV_KERNEL=[4, 4, 4],
            NUM_CLASSES=10, # TODO class nunber
            MODULATE_DEFORM=True,
            BIAS_VALUE=-2.19,
            DOWN_SCALE=4,
            MIN_OVERLAP=0.7,
            TENSOR_DIM=128,
        ),
        LOSS=dict(
            CLS_WEIGHT=1,
            WH_WEIGHT=0.1,
            REG_WEIGHT=1,
        ),
    ),
    INPUT=dict(
        AUG=dict(
            TRAIN_PIPELINES=[
                ('CenterAffine', dict(
                    boarder=128,
                    output_size=(512, 512),
                    random_aug=True)),
                ('RandomFlip', dict()),
                ('RandomBrightness', dict(intensity_min=0.6, intensity_max=1.4)),
                ('RandomContrast', dict(intensity_min=0.6, intensity_max=1.4)),
                ('RandomSaturation', dict(intensity_min=0.6, intensity_max=1.4)),
                ('RandomLighting', dict(scale=0.1)),
            ],
            TEST_PIPELINES=[
            ],
        ),
        FORMAT="RGB",
        OUTPUT_SIZE=(128, 128),
        MIN_SIZE_TEST = 720,
        MAX_SIZE_TEST = 1280
    ),
    DATALOADER=dict(
        NUM_WORKERS=8
    ),
    DATASETS=dict(
        TRAIN=("bdd100k_train",),
        TEST=("bdd100k_val",),
    ),
    SOLVER=dict(
        OPTIMIZER=dict(
            NAME="SGD",
            BASE_LR=0.01, # half learning rate because of 4/8 gpus
            WEIGHT_DECAY=1e-4,
        ),
        LR_SCHEDULER=dict(
            GAMMA=0.1,
            STEPS=(81000, 108000),
            MAX_ITER=126000,
            WARMUP_ITERS=1000,
        ),
        IMS_PER_BATCH=64, # half from default because of 4/8 gpus
    ),
    OUTPUT_DIR='./out/centernet_res50_bdd100k_126000i'
    ,
    GLOBAL=dict(DUMP_TEST=False),
)


class CenterNetConfig(BaseDetectionConfig):
    def __init__(self):
        super(CenterNetConfig, self).__init__()
        self._register_configuration(_config_dict)


config = CenterNetConfig()

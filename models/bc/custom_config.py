from yacs.config import CfgNode as CN

_C = CN()
_C.OUTPUT_DIR = "output/resnet18/CASIA-REPLAY/"
_C.NORM_FLAG = True
_C.SEED = 666

_C.DATA = CN()
_C.DATA.DATASET='ZipDataset'

_C.DATA.SUB_DIR = "EXT0.0"
_C.DATA.TRAIN = 'data/data_list/CASIA-ALL.csv'
_C.DATA.VAL = 'data/data_list/REPLAY-ALL.csv'



_C.MODEL = CN()
_C.MODEL.ARCH = 'resnet18'
_C.MODEL.IMAGENET_PRETRAIN = True


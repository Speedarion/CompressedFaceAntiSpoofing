from yacs.config import CfgNode as CN

_C = CN()
_C.OUTPUT_DIR = "output/cmfl_baseline_UNKNOWN_3D"
_C.NORM_FLAG = True
_C.SEED = 3

_C.DATA = CN()
_C.DATA.DATASET='WMCA'

_C.DATA.TRAIN = "/home/hazeeq/FYP-Hazeeq/data/data_list/wmca-protocols-csv/PROTOCOL-UNKNOWN_3D_train.csv"
_C.DATA.VAL = "/home/hazeeq/FYP-Hazeeq/data/data_list/wmca-protocols-csv/PROTOCOL-UNKNOWN_3D_dev.csv"
_C.DATA.TEST = "/home/hazeeq/FYP-Hazeeq/data/data_list/wmca-protocols-csv/PROTOCOL-UNKNOWN_3D_eval.csv"

# ---------------------------------------------------------------------------- #


_C.MODEL = CN()
_C.MODEL.IMAGENET_PRETRAIN = True
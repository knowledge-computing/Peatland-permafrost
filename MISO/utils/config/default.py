from yacs.config import CfgNode as CN

_C = CN()

# Base config files
_C.BASE = ['']

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
# _C.DATA.POINT_DATA_FILE = "/data4/aksdb-covars/aksdb-point-data/mapped-aksdb-point-data-v0.1.json"
_C.DATA.POINT_DATA_FILE = "/home/yaoyi/shared/aksdb-covars/aksdb-point-data/mapped-aksdb-point-data-v0.1.json"

# Satellite image 
_C.DATA.SAT = CN()
_C.DATA.SAT.USE_SAT = True
# _C.DATA.SAT.PATH = "/data4/aksdb-covars/s2_sr_2019_2023_gMedian_v20240713d/s2_seas3midSummer_h5/"
_C.DATA.SAT.PATH = "/home/yaoyi/shared/aksdb-covars/s2_sr_2019_2023_gMedian_v20240713d/s2_seas3midSummer_h5"
_C.DATA.SAT.NUM_BANDS = 9

# Covariates (DEM + Climate)
_C.DATA.COV = CN()
# _C.DATA.COV.PATH = "/data4/aksdb-covars/ifsar-derivatives/merged/"
_C.DATA.COV.PATH = "/home/yaoyi/shared/aksdb-covars/ifsar-derivatives/merged/"
_C.DATA.COV.USE_COV = False
_C.DATA.COV.NUM_BANDS = 10

# Covariates (DEM)
_C.DATA.DEM = CN()
_C.DATA.DEM.USE_DEM = False
_C.DATA.DEM.NUM_BANDS = 7

# Covariates (Climate)
_C.DATA.CLM = CN()
_C.DATA.CLM.USE_CLM = False
_C.DATA.CLM.NUM_BANDS = 3

# Covariates (geo_coord)
_C.DATA.GEOCOORD = CN()
_C.DATA.GEOCOORD.USE_GEOCOORD = False
_C.DATA.GEOCOORD.NUM_BANDS = 2

# Others
_C.DATA.OUT_DIM = 7
_C.DATA.CROP_SIZE = 250
_C.DATA.INPUT_SIZE = 256
_C.DATA.OUTPUT_SIZE = 256
_C.DATA.NUM_LABELS_PER_SAMPLE = 5
_C.DATA.USE_NEIGHBOR_LABELS = True
_C.DATA.USE_OTHER_TILES = False
_C.DATA.USE_OTHER_TILES_RATIO = 0.
_C.DATA.NUM_RANDOM_PSEUDO_LABELS = 10
_C.DATA.ADD_REGULAR_PSEUDO_LABELS = True

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()

# Geo encoder
_C.MODEL.GEO_ENCODER = CN()
_C.MODEL.GEO_ENCODER.TYPE = "geo"
_C.MODEL.GEO_ENCODER.POS_NUM_FREQ = 64
_C.MODEL.GEO_ENCODER.POS_ENCODER = "gridcell"

# # Visual encoder
# _C.MODEL.VISUAL_ENCODER = CN()
# _C.MODEL.VISUAL_ENCODER.TYPE = "swin"
# _C.MODEL.VISUAL_ENCODER.QUERY_METHOD = None
# _C.MODEL.VISUAL_ENCODER.USE_PRETRAIN = "Satlas"
# _C.MODEL.VISUAL_ENCODER.PRETRAIN_WEIGHT = "/home/yijun/work/DeepLATTE/permafrost/spatial_prediction/weights/sentinel2_swinb_si_ms.pth"
# # _C.MODEL.VISUAL_ENCODER.PRETRAIN_WEIGHT = "/home/yaoyi/lin00786/work/DeepLATTE/permafrost/spatial_prediction/weights/sentinel2_swinb_si_ms.pth"

# Satellite visual encoder
_C.MODEL.SAT_ENCODER = CN()
_C.MODEL.SAT_ENCODER.TYPE = "swin"
_C.MODEL.SAT_ENCODER.QUERY_METHOD = None
_C.MODEL.SAT_ENCODER.USE_PRETRAIN = "Satlas"
# _C.MODEL.SAT_ENCODER.PRETRAIN_WEIGHT = "/home/yijun/work/DeepLATTE/permafrost/spatial_prediction/weights/sentinel2_swinb_si_ms.pth"
_C.MODEL.SAT_ENCODER.PRETRAIN_WEIGHT = "/home/yaoyi/lin00786/work/DeepLATTE/permafrost/spatial_prediction/weights/sentinel2_swinb_si_ms.pth"

# DEM visual encoder
_C.MODEL.DEM_ENCODER = CN()
_C.MODEL.DEM_ENCODER.TYPE = "simswin"
_C.MODEL.DEM_ENCODER.QUERY_METHOD = None
_C.MODEL.DEM_ENCODER.USE_PRETRAIN = "SIMMIM"
# _C.MODEL.DEM_ENCODER.PRETRAIN_WEIGHT = "/home/yijun/work/DeepLATTE/permafrost/spatial_prediction/0_pretrain/_runs/simmim_pretrain/simmim_pretrain__swin_base__img192_window6__100ep_1/ckpt_epoch_60.pth"
_C.MODEL.DEM_ENCODER.PRETRAIN_WEIGHT = "/home/yaoyi/lin00786/work/DeepLATTE/permafrost/spatial_prediction/weights/dem__simmim_pretrain__swin_base__img192_window6/ckpt_epoch_60.pth"

# Climate visual encoder
_C.MODEL.CLM_ENCODER = CN()
_C.MODEL.CLM_ENCODER.TYPE = "simswin"
_C.MODEL.CLM_ENCODER.QUERY_METHOD = None
_C.MODEL.CLM_ENCODER.USE_PRETRAIN = None
_C.MODEL.CLM_ENCODER.PRETRAIN_WEIGHT = None

# Covariates (DEM + climate) visual encoder
_C.MODEL.COV_ENCODER = CN()
_C.MODEL.COV_ENCODER.TYPE = "simswin"
_C.MODEL.COV_ENCODER.QUERY_METHOD = None
_C.MODEL.COV_ENCODER.USE_PRETRAIN = None
_C.MODEL.COV_ENCODER.PRETRAIN_WEIGHT = None

# Swin Transformer parameters
_C.MODEL.SWIN = CN()
_C.MODEL.SWIN.PATCH_SIZE = 4
_C.MODEL.SWIN.EMBED_DIM = 128
_C.MODEL.SWIN.DEPTHS = [ 2, 2, 18, 2 ]
_C.MODEL.SWIN.NUM_HEADS = [ 4, 8, 16, 32 ]
_C.MODEL.SWIN.WINDOW_SIZE = 8
_C.MODEL.SWIN.MLP_RATIO = 4.
_C.MODEL.SWIN.QKV_BIAS = True
_C.MODEL.SWIN.QK_SCALE = None
_C.MODEL.SWIN.APE = False
_C.MODEL.SWIN.PATCH_NORM = True
_C.MODEL.SWIN.POOLING = "sum"
_C.MODEL.SWIN.USE_FINAL_LAYER = False

_C.MODEL.VIT = CN()
_C.MODEL.VIT.GLOBAL_POOLING = False

# Decoder
_C.MODEL.DECODER = None
_C.MODEL.PRETRAIN_WEIGHT = ""

# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.BATCH_SIZE = 32
_C.TRAIN.EPOCHS = 40
_C.TRAIN.WARMUP_EPOCHS = 5
_C.TRAIN.RAMPUP_EPOCHS = 15
_C.TRAIN.BASE_LR = 5e-5
_C.TRAIN.WARMUP_LR = 5e-7
_C.TRAIN.MIN_LR = 5e-6
_C.TRAIN.WEIGHT_DECAY = 0.01
_C.TRAIN.BASE_LOSS = "ce"
_C.TRAIN.AUX_LOSSES = []
_C.TRAIN.PATIENCE = 11
_C.TRAIN.EARLY_STOP_CRITERIA = "ce"
_C.TRAIN.NUM_WORKERS = 8
_C.TRAIN.PIN_MEM = True
_C.TRAIN.LAYER_DECAY = 1.0
_C.TRAIN.CLIP_GRAD = 1.0

# Optimizer
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = 'adamw'
_C.TRAIN.OPTIMIZER.EPS = 1e-8
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9

# Scheduler
_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.NAME = 'cosine'
_C.TRAIN.LR_SCHEDULER.DECAY_EPOCHS = 30
_C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1
_C.TRAIN.LR_SCHEDULER.GAMMA = 0.1
_C.TRAIN.LR_SCHEDULER.MULTISTEPS = []
_C.TRAIN.LR_SCHEDULER.PATIENCE = 5

_C.TRAIN.SCHEDULER = CN()
_C.TRAIN.SCHEDULER.PATIENCE = 5
# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------
_C.VAR_NAME = "tax_order"
_C.OUTPUT = "./_runs"
_C.EXP_NAME = "swin__visual_geo_dual__geo_nce__gridcell_f64"
_C.RESUME = False

_C.SPLIT_MODE = "sfold"
# _C.SPLIT_FILE = "/data4/aksdb-covars/aksdb-point-data/experimental_data/sfold5_split_train_test_indices.json"
_C.SPLIT_FILE = "/home/yaoyi/shared/aksdb-covars/aksdb-point-data/experimental_data/sfold5_split_train_test_indices.json"
_C.FOLD_ID = 0 

_C.CHECKPOINT_PATH = None
_C.LOG_PATH = None
_C.SAVE_CONFIG_FILE = None
_C.SAVE_EVERY_EPOCH = 0
_C.EVAL_EVERY_EPOCH = 1

_C.DISTRIBUTED = True
_C.LOCAL_RANK = 0
_C.MASTER_ADDR = "localhost"
_C.MASTER_PORT = 29300

_C.SEED = 0
_C.VERSION = 0

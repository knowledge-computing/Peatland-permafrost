import os
import argparse
import time
import yaml

from utils.config.default import _C

# -----------------------------------------------------------------------------
# Load Configuration
# -----------------------------------------------------------------------------
def _update_config_from_file(config, cfg_file):
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    print('=> merge config from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)


def update_config(config, args):
    _update_config_from_file(config, args.config)

    if args.opts:
        config.merge_from_list(args.opts)

    def _check_args(name):
        if hasattr(args, name) and eval(f'args.{name}'):
            return True
        return False

    # merge from specific arguments
    if _check_args('output'):
        config.OUTPUT = args.output
    if _check_args('master_port'):
        config.MASTER_PORT = args.master_port
    if _check_args('local_rank'):
        config.LOCAL_RANK = args.local_rank
    if _check_args('fold_id'):
        config.FOLD_ID = args.fold_id
    if _check_args('split_mode'):
        config.SPLIT_MODE = args.split_mode
    if _check_args('split_file'):
        config.SPLIT_FILE = args.split_file
    if _check_args('batch_size'):
        config.TRAIN.BATCH_SIZE = args.batch_size
    if _check_args('exp_name'):
        config.EXP_NAME = args.exp_name
    else:
        config.EXP_NAME = config.EXP_NAME + "__" + config.SPLIT_MODE + str(config.FOLD_ID)

    # output folder
    config.OUTPUT = os.path.join(config.OUTPUT, config.VAR_NAME, config.MODEL.DECODER, config.EXP_NAME)
    config.LOG_PATH = os.path.join(config.OUTPUT, 'log')
    config.CHECKPOINT_PATH = os.path.join(config.OUTPUT, 'checkpoint')
    config.SAVE_CONFIG_FILE = os.path.join(config.OUTPUT, 'config.yaml')  
    # config.freeze()

    
def get_config(args):
    config = _C.clone()
    update_config(config, args)
    return config


def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                        help="Path to the YAML configuration file")
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+')
    
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--master_port", type=int, default=12345)
    parser.add_argument("--fold_id", type=int, default=0)
    parser.add_argument("--split_mode", type=str, default="")
    parser.add_argument("--split_file", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=32)

    args = parser.parse_args()
    config = get_config(args)
    return config


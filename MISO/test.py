import os
import argparse
import json
from tqdm import tqdm
import numpy as np
import yaml
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, models

from models.build import build_model
from datasets.dataset import build_train_val_test_datasets
from engine import evaluate
from utils.config.default import _C
from utils.options import _update_config_from_file
from utils.logger import create_logger

# python test.py --exp_name sat_cov__c19__sfold3

def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', default='./_runs/tax_order/visual_geo_dual/sat_cov__c19__sfold3', type=str)
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+')
    args = parser.parse_args() 
    
    config_file = os.path.join(args.model_dir, "config.yaml")
    config = _C.clone()
    _update_config_from_file(config, config_file)
        
    config.DATA.USE_NEIGHBOR_LABELS = False
    return config

def load_test_model(config):
    weight_file = os.path.join(config.OUTPUT, 'best_model.pth')
    model = build_model(config, load_pretrain=False)
    checkpoint = torch.load(weight_file, map_location=torch.device('cpu'))
    msg = model.load_state_dict(checkpoint, strict=False)
    print(f"***** Load test model weights {weight_file}: ", msg)    
    return model

def main():
    config = parse_option()
    logger = create_logger(output_dir=config.LOG_PATH, mode='test')
    
    model = load_test_model(config).cuda()
    train_dataset, _, test_dataset = build_train_val_test_datasets(config)
    train_stats = {'mean': train_dataset.value_mean, 'std': train_dataset.value_std}
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4)

    test_out_stats, output_json = evaluate(model, test_loader, train_stats,
                                           config, logger, header='Test:')
    with open(os.path.join(config.OUTPUT, 'test_pred_prob.json'), 'w') as f:
        json.dump(output_json, f)          
        
        
if __name__ == '__main__':
    main()


    
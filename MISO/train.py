import os
import sys
import json
import numpy as np
import shutil

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils.options import parse_option
from models.build import build_model, load_test_model
from datasets.dataset import build_train_val_test_datasets
from engine import *

from utils import misc
from utils.logger import create_logger

# CUDA_VISIBLE_DEVICES="3" python train.py --config configs/tax_order/swin__visual_geo_dual__c9.yaml --local_rank 0

def main(config):
    #####################
    ##### Load Data #####
    #####################
    train_dataset, val_dataset, test_dataset = build_train_val_test_datasets(config)
    train_stats = {'mean': train_dataset.value_mean, 'std': train_dataset.value_std}
    
    if True:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            train_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True)
        sampler_val = torch.utils.data.SequentialSampler(val_dataset)
        sampler_test = torch.utils.data.SequentialSampler(test_dataset)
    else:
        sampler_train = torch.utils.data.RandomSampler(train_dataset)
        sampler_val = torch.utils.data.SequentialSampler(val_dataset)
        sampler_test = torch.utils.data.SequentialSampler(test_dataset)

    train_loader = DataLoader(
        train_dataset, sampler=sampler_train,
        batch_size=config.TRAIN.BATCH_SIZE,
        num_workers=config.TRAIN.NUM_WORKERS,
        pin_memory=config.TRAIN.PIN_MEM,
        drop_last=True)

    val_loader = DataLoader(
        val_dataset, sampler=sampler_val,
        batch_size=config.TRAIN.BATCH_SIZE,
        num_workers=config.TRAIN.NUM_WORKERS,
        pin_memory=config.TRAIN.PIN_MEM,
        drop_last=False)

    test_loader = DataLoader(
        test_dataset, sampler=sampler_test,
        batch_size=config.TRAIN.BATCH_SIZE,
        num_workers=config.TRAIN.NUM_WORKERS,
        pin_memory=config.TRAIN.PIN_MEM,
        drop_last=False)
    
    ######################
    ##### Load Model #####
    ######################
    model = build_model(config)
    model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, 
                                                      device_ids=[config.LOCAL_RANK], 
                                                      broadcast_buffers=False,
                                                      find_unused_parameters=True)
    model_without_ddp = model.module

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Number of params: {n_parameters}")
    if hasattr(model_without_ddp, 'flops'):
        flops = model_without_ddp.flops()
        logger.info(f"Number of GFLOPs: {flops / 1e9}")
    eff_batch_size = config.TRAIN.BATCH_SIZE * misc.get_world_size()
    logger.info(f"Effective batch size: {eff_batch_size}")
    
    optimizer = optim.AdamW(model.parameters(), 
                            eps=config.TRAIN.OPTIMIZER.EPS, 
                            betas=config.TRAIN.OPTIMIZER.BETAS,
                            lr=config.TRAIN.BASE_LR, 
                            weight_decay=config.TRAIN.WEIGHT_DECAY)

    if config.TRAIN.LR_SCHEDULER.NAME == "validate":
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            patience=config.TRAIN.LR_SCHEDULER.PATIENCE, 
            factor=0.1, 
            threshold=0,
            min_lr=config.TRAIN.MIN_LR, 
            verbose=True)
        
    elif config.TRAIN.LR_SCHEDULER.NAME == "cosine":
        from timm.scheduler.cosine_lr import CosineLRScheduler
        n_iter_per_epoch = len(train_loader)
        num_steps = int(config.TRAIN.EPOCHS * n_iter_per_epoch)
        warmup_steps = int(config.TRAIN.WARMUP_EPOCHS * n_iter_per_epoch)
        lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=num_steps,
            lr_min=config.TRAIN.MIN_LR,
            warmup_lr_init=config.TRAIN.WARMUP_LR,
            warmup_t=warmup_steps,
            cycle_limit=1,
            t_in_epochs=False)

    start_epoch = 0 # misc.resume_checkpoint(model_without_ddp, optimizer, scheduler, config)
    
    #######################
    ##### Train Model #####
    #######################
    logger.info(f"Training Starts: {config.OUTPUT}.")
    best_val_loss = np.inf  
    epochs_no_improve = 0  
    exit_flag = torch.tensor([0]).cuda()
    for epoch in range(start_epoch, config.TRAIN.EPOCHS):
        
        if exit_flag.item() == 1:
            sys.exit(0)

        train_loader.sampler.set_epoch(epoch)
        train_out_stats = train_one_epoch(model, train_loader, optimizer, lr_scheduler, epoch, 
                                          tensorboard_writer, logger, config)      

        if misc.is_main_process() and epoch % config.EVAL_EVERY_EPOCH == 0:
            evaluate(model, test_loader, train_stats, config, logger, header='Test:')

        if misc.is_main_process() and epoch > 10 and config.SAVE_EVERY_EPOCH > 0:
            if epoch % config.SAVE_EVERY_EPOCH == 0 or epoch == config.TRAIN.EPOCHS - 1:
                state_dict = model.module.state_dict()
                torch.save(state_dict, os.path.join(config.OUTPUT, f'model_epoch_{epoch}.pth'))

        for param_group in optimizer.param_groups:
            logger.info(f"Current learning rate: {param_group['lr']}.")
            
        ######################
        ##### Eval Model #####
        ######################
        if misc.is_main_process() and \
            epoch > config.TRAIN.WARMUP_EPOCHS and \
            config.TRAIN.LR_SCHEDULER.NAME == "validate":
            
            val_out_stats, _  = evaluate(model, val_loader, train_stats, config, logger, header='Val:')
            val_loss = val_out_stats[config.TRAIN.EARLY_STOP_CRITERIA]
            if val_loss < best_val_loss:
                logger.info(f"Accuracy improved in val loss: {abs(val_loss-best_val_loss)}.")
                best_val_loss = val_loss
                epochs_no_improve = 0
                state_dict = model.module.state_dict()
                torch.save(state_dict, os.path.join(config.OUTPUT, 'best_model.pth'))
            else:
                epochs_no_improve += 1
                logger.info(f"No improvement in val loss for {epochs_no_improve} epochs.")                
            lr_scheduler.step(val_loss)
            
            ##########################
            ##### Early Stopping #####
            ##########################
            if epochs_no_improve >= config.TRAIN.PATIENCE or \
                epoch == config.TRAIN.EPOCHS - 1:
                
                logger.info(f"No improvement in val loss for {config.TRAIN.PATIENCE} epochs. Early stopping.")
                logger.info(f"Testing - Load best model weight: {config.OUTPUT}")
                test_model = load_test_model(config).cuda()
                test_out_stats, output_json = evaluate(test_model, test_loader, train_stats, 
                                                       config, logger, header='Test:')
                logger.info(f"TEST Stats: {test_out_stats}")
                with open(os.path.join(config.OUTPUT, 'test_pred.json'), 'w') as f:
                    json.dump(output_json, f)
                    
                exit_flag[0] = 1 
                dist.broadcast(exit_flag, src=0)  # Broadcast exit signal to all processes
                print("*** Process 0 set exit_flag to 0.")


if __name__ == '__main__':
    config = parse_option()
    misc.init_distributed_mode(config)
    misc.init_training(config)
        
    logger = create_logger(output_dir=config.LOG_PATH, dist_rank=dist.get_rank())

    if dist.get_rank() == 0:
        tensorboard_writer = SummaryWriter(log_dir=config.LOG_PATH)
        
    logger.info(config.dump())
    main(config)
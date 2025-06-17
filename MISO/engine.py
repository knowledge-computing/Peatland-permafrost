import time
import math
import numpy as np
import datetime
from tqdm import tqdm
from typing import Iterable
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils.misc as misc
import utils.metric as metric
from models.losses import LossCollection


def run_model(batch_data, model, decoder_name):
    x = batch_data[0].cuda()
    x_coords = batch_data[2].cuda()
    geo_coords = batch_data[3].cuda()

    if decoder_name == 'geo_only':
        output = model(geo_coords)
    else:
        output = model(x, x_coords, geo_coords)
    return output
    
    
def train_one_epoch(model, data_loader, optimizer, lr_scheduler, epoch, 
                    tensorboard_writer, logger, config):
    
    torch.autograd.set_detect_anomaly(True)
    model.train()
    
    num_steps = len(data_loader)
    loss_obj = LossCollection(config)
    metric_logger = misc.MetricLogger(
        header=f"EPOCH [{epoch}/{config.TRAIN.EPOCHS}]",
        delimiter="  ", 
        print_freq=50)
    
    start = time.time()
    num_steps = len(data_loader)
    # base_weight = misc.get_warmup_ratio(epoch, config)
    # logger.info(f"Current base_loss_weight: {base_weight}.")
        
    for idx, batch_data in enumerate(metric_logger.log_every(data_loader, logger)):
        optimizer.zero_grad()
        
        #######################
        ##### Model Train #####
        #######################
        all_losses = {}
        output = run_model(batch_data, model, config.MODEL.DECODER)
        y = batch_data[1].reshape(-1, 1).cuda()
        pred = output['pred'].reshape(-1, config.DATA.OUT_DIM)

        loss = loss_obj.compute_base_loss(pred, y, loss_fn=config.TRAIN.BASE_LOSS) # * base_weight
        all_losses['base_loss'] = loss.item() 
        
        aux_loss, aux_loss_dict = loss_obj.compute_aux_losses(output, y)
        loss += aux_loss
        all_losses.update(aux_loss_dict)
        all_losses['total_loss'] = loss.item()
        
        loss.backward()
        
        if config.TRAIN.CLIP_GRAD:    
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
        else:
            grad_norm = misc.get_grad_norm(model.parameters())

        optimizer.step()
        if config.TRAIN.LR_SCHEDULER.NAME == "cosine":
            lr_scheduler.step_update(epoch * num_steps + idx)

        torch.cuda.synchronize()
        
        ###################
        ##### Logging #####
        ###################
        metric_logger.update(grad_norm=grad_norm)
        for k, v in all_losses.items():
            if k == 'total_loss': metric_logger.update(total_loss=v);
            if k == 'base_loss': metric_logger.update(base_loss=v);
            if k == 'sat_loss': metric_logger.update(sat_loss=v);
            if k == 'focal_loss': metric_logger.update(focal_loss=v);
            if k == 'geo_loss': metric_logger.update(geo_loss=v);
            if k == 'visual_loss': metric_logger.update(visual_loss=v);
            if k == 'cos_loss': metric_logger.update(cos_loss=v);
            if k == 'dual_loss': metric_logger.update(dual_loss=v);
            if k == 'visual_geo_nce_loss': metric_logger.update(visual_geo_nce_loss=v);
            if k == 'sat_cov_nce_loss': metric_logger.update(sat_cov_nce_loss=v);
            if k == 'sat_geo_nce_loss': metric_logger.update(sat_geo_nce_loss=v);
            if k == 'cov_geo_nce_loss': metric_logger.update(cov_geo_nce_loss=v);
            if k == 'level_sat_cov_nce_loss': metric_logger.update(level_sat_cov_nce_loss=v);  
            value_reduce = misc.all_reduce_mean(v)
            tensorboard_writer.add_scalar(k, value_reduce, num_steps*epoch+idx)
        
    metric_logger.synchronize_between_processes()
    epoch_time = time.time() - start
    logger.info(f"EPOCH [{epoch}/{config.TRAIN.EPOCHS}] takes {datetime.timedelta(seconds=int(epoch_time))}")
    train_out_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    logger.info(f"EPOCH [{epoch}/{config.TRAIN.EPOCHS}]: {str(train_out_stats)}")
    return train_out_stats


@torch.no_grad()
def evaluate(model,
             data_loader,
             data_stats,
             config,
             logger,
             header):
    
    ######################
    ##### Model Eval #####
    ######################
    model.eval()    
    output_json, gt_list, pred_list = {}, [], []
    for batch_data in tqdm(data_loader, total=len(data_loader)): 
        ground_truth = batch_data[1].cuda()
        pids = batch_data[4]
        output = run_model(batch_data, model, config.MODEL.DECODER)
        
        y = ground_truth.reshape(-1, 1)
        mask = y[:, 0] > -999
        y = y[mask, :]
        pred = output['pred'].reshape(-1, config.DATA.OUT_DIM)
        pred = pred[mask, :]    

        ##########################
        ##### Get Prediction #####
        ##########################
        if config.TRAIN.BASE_LOSS == 'mse':
            pred = pred * data_stats['std'] + data_stats['mean']
        elif config.VAR_NAME == 'aksdb_pf1m_bin':
            pred = torch.sigmoid(pred)
        elif config.VAR_NAME == 'tax_order':
            pred_prob_arr = pred.detach().cpu().numpy().astype(np.float64)
            pred = pred.argmax(dim=-1)
        else:
            raise NotImplementedError
    
        y_arr = y.detach().cpu().numpy().astype(np.float64)
        pred_arr = pred.detach().cpu().numpy().astype(np.float64)
        pred_arr = pred_arr.reshape(y_arr.shape)
        for i in range(y_arr.shape[0]):
            gt_list.append(y_arr[i][0])
            pred_list.append(pred_arr[i][0])
            out = {'gt': y_arr[i][0], 'pred': pred_arr[i][0]}
            if config.VAR_NAME == 'tax_order':    
                out['pred_prob'] = pred_prob_arr[i].tolist()
            output_json[pids[i].item()] =  out

    gt = np.array(gt_list)
    pred = np.array(pred_list)
    
    ###########################
    ##### Compute Metrics #####
    ###########################
    stats = {}
    if config.TRAIN.BASE_LOSS == 'mse':
        stats['mse'] = metric.compute_MSE(gt, pred)
        stats['rmse'] = metric.compute_RMSE(gt, pred)
        stats['rmse_ori'] = metric.compute_RMSE(np.expm1(gt), np.expm1(pred))
        logger.info(f"*** {header} Average MSE = {stats['mse']:.5f}")
        logger.info(f"*** {header} Average RMSE = {stats['rmse']:.5f}")
        logger.info(f"*** {header} Average RMSE (Original Space) = {stats['rmse_ori']:.5f}")
        
    elif config.VAR_NAME == 'aksdb_pf1m_bin':
        stats['bce'] = metric.compute_BCE(gt, pred)
        stats['weighted_bce'] = metric.compute_weighted_BCE(gt, pred)        
        acc_stats = metric.compute_binary_ACC(gt, pred)        
        stats.update(acc_stats)
        logger.info(f"*** {header} Average BCE = {stats['bce']:.5f}")
        logger.info(f"*** {header} Average Weighted BCE = {stats['weighted_bce']:.5f}")
        logger.info(f"*** {header} Precision (label=0) = {stats['precision0']:.5f}")
        logger.info(f"*** {header} Recall (label=0) = {stats['recall0']:.5f}")
        logger.info(f"*** {header} F1 (label=0) = {stats['fscore0']:.5f}")        
        logger.info(f"*** {header} Precision (label=1) = {stats['precision1']:.5f}")
        logger.info(f"*** {header} Recall (label=1) = {stats['recall1']:.5f}")
        logger.info(f"*** {header} F1 (label=1) = {stats['fscore1']:.5f}")
        logger.info(f"*** {header} AUC_ROC = {stats['auc_roc']:.5f}")
        logger.info(f"*** {header} Accuracy = {stats['accuracy']:.5f}")
        
    elif config.VAR_NAME == 'tax_order':
        stats = metric.compute_multi_ACC(gt, pred, config.DATA.OUT_DIM)
        logger.info(f"*** {header} Precision = {stats['precision']:.5f}")
        logger.info(f"*** {header} Recall = {stats['recall']:.5f}")
        logger.info(f"*** {header} F1 = {stats['fscore']:.5f}")
        logger.info(f"*** {header} Precision (weighted) = {stats['precision_weighted']:.5f}")
        logger.info(f"*** {header} Recall (weighted) = {stats['recall_weighted']:.5f}")
        logger.info(f"*** {header} F1 (weighted) = {stats['f1_weighted']:.5f}")
        logger.info(f"*** {header} Accuracy = {stats['accuracy']:.5f}")
        for k, v in stats.items():
            if type(v) == dict:
                logger.info(f"*** {header} {k}: Precision = {v['precision']:.5f}; Recall = {v['recall']:.5f}")
        
    return stats, output_json


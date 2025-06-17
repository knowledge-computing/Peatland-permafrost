import os
import json
import torch
import numpy as np
import pandas as pd
import geopandas as gpd
import glob
import random
from torchvision import transforms
import cv2
import shutil
from PIL import Image
from tqdm import tqdm
import yaml
import h5py
import argparse
import sys
from scipy.ndimage import gaussian_filter

import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
from datasets.data_utils import norm_sat, norm_cov
from models.build import build_model
from utils.config.default import _C
from utils.options import _update_config_from_file

def gaussian_weight_mask(h, w, sigma=0.25):
    center = np.zeros((h, w))
    center[h // 2, w // 2] = 1.0
    return gaussian_filter(center, sigma=min(h, w) * sigma)

##################
##################
##################
parser = argparse.ArgumentParser()
parser.add_argument('--grid_id', type=str)
parser.add_argument('--model_dir', type=str)
parser.add_argument('--output_tif', type=str, default='AK050H50V15_pf1m_interval2.tif')
parser.add_argument('--interval', type=int, default=2)

parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+')
args = parser.parse_args()

# python demo.py --grid_id AK050H48V07 --model_dir ./_runs/tax_order/contrast/swin__visual_geo_dual_sv__geo_nce__gridcell_f64__c19__sfold0__tax_order__v0/

# python demo.py --grid_id AK050H50V15 --model_dir ./_runs/aksdb_pf1m_bin/visual_geo_dual/sat_cov__c19__sfold0/
##################
##################
##################

TILE_SIZE = 5000
GRID_ID = args.grid_id

config_file = os.path.join(args.model_dir, "config.yaml")
config = _C.clone()
_update_config_from_file(config, config_file)

CROP_SIZE = config.DATA.CROP_SIZE
SHIFT_SIZE = CROP_SIZE // args.interval
INPUT_SIZE = config.DATA.INPUT_SIZE
VAR_NAME = config.VAR_NAME
MASK = gaussian_weight_mask(CROP_SIZE, CROP_SIZE, sigma=0.25)

##################
### Load Model ###
##################
model = build_model(config, load_pretrain=False)
weight_file = os.path.join(args.model_dir, 'best_model.pth')
checkpoint = torch.load(weight_file, map_location=torch.device('cpu'))
msg = model.load_state_dict(checkpoint, strict=False)
print(msg)
model = model.cuda()
print('***** Model Loaded: ', weight_file)

#################
### Load DATA ###
#################
all_band_data = []

sat_dir = config.DATA.SAT.PATH
file_path = os.path.join(sat_dir, f"{GRID_ID}.h5")
with h5py.File(file_path, 'r') as h5_file:
    band_data = h5_file["data"][:]
    mask = (band_data == -32767).all(axis=-1) 
    band_data = band_data.astype(np.float32)
    band_data = norm_sat(band_data)
    band_data[mask, :] = 0.
all_band_data.append(band_data)
print('***** Sat Loaded: ', os.path.basename(file_path))

if config.DATA.COV.USE_COV:
    cov_dir = config.DATA.COV.PATH 
    file_path = os.path.join(cov_dir, f"{GRID_ID}.h5")
    with h5py.File(file_path, 'r') as h5_file:
        band_data = h5_file["data"][:]
        mask = (band_data == -3e+38).all(axis=-1) 
        band_data = band_data.astype(np.float32)
        band_data = norm_cov(band_data)
        band_data[mask, :] = 0.
    all_band_data.append(band_data)    
print('***** Cov Loaded: ', os.path.basename(file_path))

band_data = np.concatenate(all_band_data, axis=-1)
band_data = np.transpose(band_data, (2, 0, 1))    
band_data = torch.tensor(band_data, dtype=torch.float32)

#######################
### Load Point Data ###
#######################
point_data_file = config.DATA.POINT_DATA_FILE
with open(point_data_file, 'r') as f:
    point_data = []
    for dat in json.load(f):
        if dat['grid_id'] == GRID_ID and dat.get(VAR_NAME) is not None:
            point_data.append(dat)
print('***** Point Data Loaded: ', len(point_data))
        
##############################
### Construct Pixel Coords ###
##############################
def normalize_coord3338(coord):
    x_coord, y_coord = coord
    # x_coord = round(x_coord, 5)
    # y_coord = round(y_coord, 5)    
    x_norm = (x_coord-(-2199995.)) / (1500005.-(-2199995.)) * 2 - 1
    y_norm = (y_coord-2400005.) / (5.-2400005.) * 2 - 1
    return x_norm, y_norm

def make_grid(v0, v1):
    r = (v1 - v0) / (2 * CROP_SIZE)
    return v0 + r + (2 * r) * torch.arange(CROP_SIZE).float()
    
xs = make_grid(-1, 1)
ys = make_grid(-1, 1)
x, y = torch.meshgrid(xs, ys, indexing='xy')
pixel_coords = torch.stack([x, y], axis=-1)
pixel_coords = pixel_coords.reshape(-1, 2).unsqueeze(0).cuda()
print('***** Pixel Coords Set: ', pixel_coords.shape)

############################
### Construct Geo Coords ###
############################        
shp_data = gpd.read_file("/home/yaoyi/shared/aksdb-covars/tiles/AKVEG_050km_tiles_3338_v20230326.shp")
for i, row in shp_data.iterrows():
    if row['gridID'] == GRID_ID:
        geo_xmin, geo_xmax = row['xmin'], row['xmax']
        geo_ymin, geo_ymax = row['ymin'], row['ymax']
        transform = from_bounds(geo_xmin, geo_ymin, geo_xmax, geo_ymax, 5000, 5000)
        break
        
print('***** Geo Coords: ', geo_xmin, geo_xmax, geo_ymin, geo_ymax)
print('***** Transform: ', transform)
crs = CRS.from_epsg(3338)

########################
### Prediction Start ###
########################        
C = config.DATA.OUT_DIM
output_sum_prediction = np.zeros((5000, 5000, C))
output_count_prediction = np.zeros((5000, 5000, C))
output_sum_geo_prediction = np.zeros((5000, 5000, C))

mask_3d = np.repeat(MASK[:, :, np.newaxis], C, axis=2)

count = 0
with torch.no_grad():
    nx = ny = (TILE_SIZE - CROP_SIZE) // SHIFT_SIZE + 1
    total_iterations = nx * ny
    with tqdm(total=total_iterations, desc="Combined Progress") as pbar:
        for y in range(0, ny * SHIFT_SIZE, SHIFT_SIZE):
            for x in range(0, nx * SHIFT_SIZE, SHIFT_SIZE):
                img_tensor = band_data[:, y: y+CROP_SIZE, x: x+CROP_SIZE].cuda()
                img_tensor = transforms.Resize((INPUT_SIZE, INPUT_SIZE), antialias=None)(img_tensor)
                
                local_geo_min, local_geo_max = geo_xmin + x*10, geo_xmin + (x+CROP_SIZE)*10
                geo_xs = make_grid(local_geo_min, local_geo_max)
        
                local_geo_min, local_geo_max = geo_ymax - y*10, geo_ymax - (y+CROP_SIZE)*10
                geo_ys = make_grid(local_geo_min, local_geo_max)
        
                geo_xs, geo_ys = normalize_coord3338((geo_xs, geo_ys))
                geo_x, geo_y = torch.meshgrid(geo_xs, geo_ys, indexing='xy')
                geo_coords = torch.stack([geo_x, geo_y], axis=-1)
                geo_coords = geo_coords.reshape(-1, 2).unsqueeze(0).cuda()

                if config.MODEL.DECODER in ['visual_only']:
                    output = model(img_tensor.unsqueeze(0), pixel_coords)
                else:
                    output = model(img_tensor.unsqueeze(0), pixel_coords, geo_coords)

                if config.VAR_NAME == 'aksdb_pf1m_bin':
                    y_pred = torch.sigmoid(output['pred'])
                elif config.VAR_NAME == 'tax_order':
                    y_pred = output['pred'].softmax(dim=-1)
                    
                y_pred = y_pred.detach().cpu().numpy().reshape(CROP_SIZE, CROP_SIZE, C)
                output_sum_prediction[y: y+CROP_SIZE, x: x+CROP_SIZE, :] += y_pred * mask_3d
                output_count_prediction[y: y+CROP_SIZE, x: x+CROP_SIZE, :] += mask_3d

                # if config['out_uncertainty']:
                #     log_var_logit = output['pred'][:, :, 1:2]
                #     std_logit = torch.exp(0.5 * log_var_logit)
                #     std_logit = std_logit.detach().cpu().numpy().reshape(CROP_SIZE, CROP_SIZE)
                #     std_logit = np.clip(std_logit, a_min=0., a_max=1.)
                #     image = Image.fromarray(std_logit * 255).convert("RGB")
                #     image.save(os.path.join(model_dir, f'{GRID_ID}_{x}_{y}.png'))
                #     output_uncertainty[y: y+CROP_SIZE, x: x+CROP_SIZE] += std_logit
                
                if output.get('geo_pred') is not None:
                    if config.VAR_NAME == 'aksdb_pf1m_bin':
                        g_pred = torch.sigmoid(output['geo_pred'])
                    elif config.VAR_NAME == 'tax_order':
                        g_pred = output['geo_pred'].softmax(dim=-1)
                        
                    g_pred = g_pred.detach().cpu().numpy().reshape(CROP_SIZE, CROP_SIZE, C)
                    output_sum_geo_prediction[y: y+CROP_SIZE, x: x+CROP_SIZE] += g_pred
                    
                del img_tensor
                torch.cuda.empty_cache()
                pbar.update(1)

########################
### Write Prediction ###
########################
TAX_ORDER_DICT = {
    0: 'Andisols',
    1: 'Entisols',
    2: 'Gelisols',
    3: 'Histosols',
    4: 'Inceptisols',
    5: 'Mollisols',
    6: 'Spodosols',
}
                
output_prediction = output_sum_prediction / output_count_prediction

if  VAR_NAME == 'aksdb_pf1m_bin':
    with rasterio.open(
            os.path.join(args.model_dir, args.output_tif),
            'w',
            driver='GTiff',
            height=5000,
            width=5000,
            count=1,
            dtype=np.float32,
            crs=crs,
            transform=transform,
            nodata=0.
        ) as dst:
            dst.write(output_prediction[..., 0].astype(np.float32), 1)

elif VAR_NAME == 'tax_order':
    with rasterio.open(
        os.path.join(args.model_dir, args.output_tif),
        'w',
        driver='GTiff',
        height=5000,
        width=5000,
        count=C,
        dtype=np.float32,
        crs=crs,
        transform=transform,
        nodata=0.
    ) as dst:
        for i in range(C):
            name = TAX_ORDER_DICT[i]
            dst.write(output_prediction[..., i].astype(np.float32), 1 + i)
            dst.set_band_description(1+i, name)


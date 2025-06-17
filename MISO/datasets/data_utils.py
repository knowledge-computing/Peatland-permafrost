import os
import glob
import random
import numpy as np
from PIL import Image
from collections import defaultdict
from tqdm import tqdm
import h5py
import torch
from torchvision import transforms

from datasets.buildin import COV_STATS, TILE_SIZE

def viz(img, coords, out_file):
    import cv2
    img_arr = np.array(img)
    for coord in coords:
        x, y = int(coord[0]), int(coord[1])
        cv2.circle(img_arr, (x, y), radius=5, color=(255, 0, 0), thickness=-1)
    cv2.imwrite(out_file, img_arr) 

    
def gridID2hv(grid_id):
    return int(grid_id[6:8]), int(grid_id[9:11])

def hv2str(h, v):
    return str(h).zfill(2), str(v).zfill(2)
    
    
def build_neighbor_dict(point_data_list, crop_size):
    neighbors = defaultdict(list)
    print("Building neighborhood dict ...")
    for data_i in tqdm(point_data_list, total=len(point_data_list)):
        for data_j in point_data_list:
            if data_i.id < data_j.id:
                px_i, py_i = data_i.global_x_pixel, data_i.global_y_pixel
                px_j, py_j = data_j.global_x_pixel, data_j.global_y_pixel
                if (px_i - px_j) ** 2 + (py_i - py_j) ** 2 < crop_size ** 2:
                    neighbors[data_i.id].append(data_j.id)
                    neighbors[data_j.id].append(data_i.id)
    return neighbors


# def get_valid_sat_files(sat_dir):
#     sat_files = sorted(glob.glob(os.path.join(sat_dir, '*.h5')))
#     return sat_files
#     # valid_sat_files = []
#     # for sat_file in sat_files:
#     #     flag = True
#     #     grid_id = os.path.basename(sat_file).split('_')[5]
#     #     grid_H, grid_V = gridID2hv(grid_id)
#     #     for i in [-1, 0, 1]:
#     #         for j in [-1, 0, 1]:
#     #             h, v = hv2str(grid_H + i, grid_V + j)
#     #             sat_path = os.path.join(sat_dir, PREFIX.format(
#     #                 grid_id=f"AK050H{h}V{v}") + ".h5")
#     #             if not os.path.exists(sat_path):
#     #                 flag = False
#     #     if flag:
#     #         valid_sat_files.append(sat_file)
#     # return valid_sat_files
    
    
###########################
##### Load Image Data #####
###########################
# def load_image(grid_id, img_dir):
#     img_path = os.path.join(img_dir, PREFIX.format(grid_id=grid_id) + '.png')
#     img = Image.open(img_path).convert("RGB")
#     return img

# def crop_image(x1, y1, x2, y2, img_dir):
#     h1 = int(x1 // TILE_SIZE)
#     h2 = int(x2 // TILE_SIZE)
#     v1 = int(y1 // TILE_SIZE)
#     v2 = int(y2 // TILE_SIZE)
#     h1s, v1s = hv2str(h1, v1)
#     h2s, v2s = hv2str(h2, v2)
    
#     if h1 == h2 and v1 == v2:
#         enlarged_img = load_image(f"AK050H{h1s}V{v1s}", img_dir)

#     elif h1 == h2 and v1 < v2:
#         enlarged_img = Image.new(mode="RGB", size=(TILE_SIZE, TILE_SIZE * 2))
#         img = load_image(f"AK050H{h1s}V{v1s}", img_dir)
#         enlarged_img.paste(img) 
#         img = load_image(f"AK050H{h1s}V{v2s}", img_dir)
#         enlarged_img.paste(img, (0, TILE_SIZE))
        
#     elif h1 < h2 and v1 == v2:
#         enlarged_img = Image.new(mode="RGB", size=(TILE_SIZE * 2, TILE_SIZE))
#         img = load_image(f"AK050H{h1s}V{v1s}", img_dir)
#         enlarged_img.paste(img) 
#         img = load_image(f"AK050H{h2s}V{v1s}", img_dir)
#         enlarged_img.paste(img, (TILE_SIZE, 0))
        
#     elif h1 < h2 and v1 < v2:
#         enlarged_img = Image.new(mode="RGB", size=(TILE_SIZE * 2, TILE_SIZE * 2))
#         img = load_image(f"AK050H{h1s}V{v1s}", img_dir)
#         enlarged_img.paste(img)
#         img = load_image(f"AK050H{h1s}V{v2s}", img_dir)
#         enlarged_img.paste(img, (0, TILE_SIZE))
#         img = load_image(f"AK050H{h2s}V{v1s}", img_dir)
#         enlarged_img.paste(img, (TILE_SIZE, 0))
#         img = load_image(f"AK050H{h2s}V{v2s}", img_dir)
#         enlarged_img.paste(img, (TILE_SIZE, TILE_SIZE))
    
#     crop_img = enlarged_img.crop((x1-h1*TILE_SIZE, y1-v1*TILE_SIZE, 
#                                   x2-h1*TILE_SIZE, y2-v1*TILE_SIZE))
#     return crop_img


########################
##### Load h5 Data #####
########################
def load_h5(grid_id, h5_dir, crop_location, num_bands=9, nodata_value=-32768.):
    r1, r2, c1, c2 = crop_location
    file_path = os.path.join(h5_dir, grid_id+'.h5')
    if not os.path.exists(file_path):
        band_data = np.zeros((TILE_SIZE, TILE_SIZE, num_bands)).astype(np.float32) + nodata_value
        return band_data[r1:r2, c1:c2, :num_bands]
    with h5py.File(file_path, 'r') as h5_file:
        band_data = h5_file["data"]
        r1, r2, c1, c2 = crop_location
        return band_data[r1:r2, c1:c2, :num_bands].astype(np.float32)

def crop_h5(x1, y1, x2, y2, h5_dir, num_bands):
    h1 = int(x1 // TILE_SIZE)
    h2 = int(x2 // TILE_SIZE)
    v1 = int(y1 // TILE_SIZE)
    v2 = int(y2 // TILE_SIZE)
    h1s, v1s = hv2str(h1, v1)
    h2s, v2s = hv2str(h2, v2)
    
    if h1 == h2 and v1 == v2:
        r1, r2 = y1-v1*TILE_SIZE, y2-v1*TILE_SIZE
        c1, c2 = x1-h1*TILE_SIZE, x2-h1*TILE_SIZE
        crop = load_h5(f"AK050H{h1s}V{v1s}", h5_dir, (r1, r2, c1, c2), num_bands)

    elif h1 == h2 and v1 < v2:
        r1, r2 = y1-v1*TILE_SIZE, TILE_SIZE
        c1, c2 = x1-h1*TILE_SIZE, x2-h1*TILE_SIZE        
        img1 = load_h5(f"AK050H{h1s}V{v1s}", h5_dir, (r1, r2, c1, c2), num_bands)
        r1, r2 = 0, y2-v2*TILE_SIZE
        c1, c2 = x1-h1*TILE_SIZE, x2-h1*TILE_SIZE        
        img2 = load_h5(f"AK050H{h1s}V{v2s}", h5_dir, (r1, r2, c1, c2), num_bands)              
        crop = np.concatenate([img1, img2], axis=0)
        
    elif h1 < h2 and v1 == v2:
        r1, r2 = y1-v1*TILE_SIZE, y2-v1*TILE_SIZE
        c1, c2 = x1-h1*TILE_SIZE, TILE_SIZE
        img1 = load_h5(f"AK050H{h1s}V{v1s}", h5_dir, (r1, r2, c1, c2), num_bands)
        r1, r2 = y1-v1*TILE_SIZE, y2-v1*TILE_SIZE
        c1, c2 = 0, x2-h2*TILE_SIZE
        img2 = load_h5(f"AK050H{h2s}V{v1s}", h5_dir, (r1, r2, c1, c2), num_bands)
        crop = np.concatenate([img1, img2], axis=1)
        
    elif h1 < h2 and v1 < v2:
        r1, r2 = y1-v1*TILE_SIZE, TILE_SIZE
        c1, c2 = x1-h1*TILE_SIZE, TILE_SIZE
        img1 = load_h5(f"AK050H{h1s}V{v1s}", h5_dir, (r1, r2, c1, c2), num_bands)
        r1, r2 = y1-v1*TILE_SIZE, TILE_SIZE
        c1, c2 = 0, x2-h2*TILE_SIZE
        img2 = load_h5(f"AK050H{h2s}V{v1s}", h5_dir, (r1, r2, c1, c2), num_bands)
        r1, r2 = 0, y2-v2*TILE_SIZE
        c1, c2 = x1-h1*TILE_SIZE, TILE_SIZE
        img3 = load_h5(f"AK050H{h1s}V{v2s}", h5_dir, (r1, r2, c1, c2), num_bands)
        r1, r2 = 0, y2-v2*TILE_SIZE
        c1, c2 = 0, x2-h2*TILE_SIZE
        img4 = load_h5(f"AK050H{h2s}V{v2s}", h5_dir, (r1, r2, c1, c2), num_bands)
        crop1 = np.concatenate([img1, img2], axis=1)
        crop2 = np.concatenate([img3, img4], axis=1)
        crop = np.concatenate([crop1, crop2], axis=0)        
    return crop


#########################
##### Load SAT Data #####
#########################
def norm_sat(sat):
    # sat[:, :, 0] = np.clip(sat[:, :, 0] / 1624., 0, 1)
    # sat[:, :, 1] = np.clip(sat[:, :, 1] / 1588., 0, 1)
    # sat[:, :, 2] = np.clip(sat[:, :, 2] / 1397., 0, 1)
    ##### mean + std #####
    # sat[:, :, 0] = np.clip(sat[:, :, 0] / 1745., 0, 1) 
    # sat[:, :, 1] = np.clip(sat[:, :, 1] / 1881., 0, 1) 
    # sat[:, :, 2] = np.clip(sat[:, :, 2] / 1764., 0, 1) 
    ##### first clip by 10000 then 2%-98% #####
    sat[:, :, 0] = np.clip(sat[:, :, 0], 0, 10000)
    sat[:, :, 1] = np.clip(sat[:, :, 1], 0, 10000)
    sat[:, :, 2] = np.clip(sat[:, :, 2], 0, 10000)
    sat[:, :, 0] = np.clip((sat[:, :, 0] - 78.) / (3318. - 78.), 0, 1)
    sat[:, :, 1] = np.clip((sat[:, :, 1] - 158.) / (3411. - 158.), 0, 1)
    sat[:, :, 2] = np.clip((sat[:, :, 2] - 167.) / (3335. - 167.), 0, 1)
    sat[:, :, 3:] = np.clip(sat[:, :, 3:] / 8160., 0, 1)
    return sat
    
def crop_sat(x1, y1, x2, y2, sat_dir, num_bands=9):
    crop = crop_h5(x1, y1, x2, y2, sat_dir, num_bands=9)
    mask = (crop == -32767).all(axis=-1)
    crop = norm_sat(crop)
    crop = crop[:, :, :num_bands]
    crop[mask, :] = 0.
    crop = torch.tensor(crop, dtype=torch.float32) # 250x250x9
    mask = torch.tensor(mask, dtype=torch.bool) # 250x250
    return crop, mask


#########################
##### Load COV Data #####
#########################
def standard_norm(value, stat):
    clipped = np.clip(value, stat['clip_min'], stat['clip_max'])
    normed = (clipped - stat['mean']) / stat['std']
    return (normed - stat['norm_min']) / (stat['norm_max'] - stat['norm_min'])

def norm_cov(cov):
    aspct_stat = COV_STATS['aspct']
    cov[:, :, 0] = standard_norm(cov[:, :, 0], aspct_stat)
    elev_stat = COV_STATS['elevation_full_10m_3338']
    cov[:, :, 1] = standard_norm(cov[:, :, 1], elev_stat)
    maxc_4_stat = COV_STATS['maxc_4']
    cov[:, :, 2] = standard_norm(cov[:, :, 2], maxc_4_stat)
    sl_4_stat = COV_STATS['sl_4']
    cov[:, :, 3] = standard_norm(cov[:, :, 3], sl_4_stat)
    spi_stat = COV_STATS['spi']
    cov[:, :, 4] = standard_norm(cov[:, :, 4], spi_stat)
    swi_10_stat = COV_STATS['swi_10']
    cov[:, :, 5] = standard_norm(cov[:, :, 5], swi_10_stat)
    tpi_4_stat = COV_STATS['tpi_4']
    cov[:, :, 6] = standard_norm(cov[:, :, 6], tpi_4_stat)
    ppt_annual_stat = COV_STATS['ppt_annual']
    cov[:, :, 7] = standard_norm(cov[:, :, 7], ppt_annual_stat)
    tmean_swi_stat = COV_STATS['tmean_swi']
    cov[:, :, 8] = standard_norm(cov[:, :, 8], tmean_swi_stat) 
    tmin_january_stat = COV_STATS['tmin_january']
    cov[:, :, 9] = standard_norm(cov[:, :, 9], tmin_january_stat) 
    return cov

def crop_cov(x1, y1, x2, y2, cov_dir):
    crop = crop_h5(x1, y1, x2, y2, cov_dir, num_bands=10)
    mask = (crop == -3e+38).all(axis=-1)    
    crop = norm_cov(crop)
    crop[mask, :] = 0.
    crop = torch.tensor(crop, dtype=torch.float32) # 250x250x10
    mask = torch.tensor(mask, dtype=torch.bool) # 250x250
    return crop, mask

##########################
##### Normalizations #####
##########################
def normalize_image_coords(coord, tile_size):
    x_coord, y_coord = coord
    x_norm = 2. * x_coord / tile_size - 1
    y_norm = 2. * y_coord / tile_size - 1
    return x_norm, y_norm

def denormalize_image_coords(norm_coord, tile_size):
    x_norm, y_norm = norm_coord
    x_coord = (x_norm + 1) * tile_size / 2
    y_coord = (y_norm + 1) * tile_size / 2
    return x_coord, y_coord

def normalize_coord3338(coord):
    x_coord, y_coord = coord
    x_coord = round(x_coord, 5)
    y_coord = round(y_coord, 5)    
    x_norm = (x_coord-(-2199995.)) / (1500005.-(-2199995.)) * 2 - 1
    y_norm = (y_coord-2400005.) / (5.-2400005.) * 2 - 1
    # x_norm = x_coord
    # y_norm = y_coord
    return x_norm, y_norm







    
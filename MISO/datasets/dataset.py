import os
import glob
import json
import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import torch.nn.functional as F

from datasets.data_utils import *
from datasets.buildin import *


############################
##### Point Data Class #####
############################
class PointData:
    def __init__(self, point_data, var_name):
        self.var_name = var_name
        self.id = point_data['id']
        self.dts = point_data['aksdb_dts']
        self.lon = point_data['lon']
        self.lat = point_data['lat']
        self.lon3338 = point_data['x_3338']
        self.lat3338 = point_data['y_3338']        
        self.grid_x_pixel = point_data['x_pixel']
        self.grid_y_pixel = point_data['y_pixel']
        self.grid_id = point_data['grid_id']
        self.grid_H, self.grid_V = gridID2hv(point_data['grid_id'])
        self.global_x_pixel = self.grid_H * TILE_SIZE + self.grid_x_pixel
        self.global_y_pixel = self.grid_V * TILE_SIZE + self.grid_y_pixel
        self.value = point_data.get(var_name, None)
        
    def is_valid(self, sat_dir):
        if self.value is None or type(self.value) == list:
            return False
        
        h, v = hv2str(self.grid_H, self.grid_V)
        img_path = os.path.join(sat_dir, f"AK050H{h}V{v}.h5")
        if not os.path.exists(img_path):
            return False
        
        if self.var_name == 'aksdb_othick_cum_best':
            if 0 <= self.value < 500:
                self.value = np.log1p(self.value)
            else: 
                return False
        
        if self.var_name == 'tax_order':
            if TAX_ORDER_DICT.get(self.value) is not None:
                self.value = TAX_ORDER_DICT[self.value]
            else:
                return False
        return True
    
    
class FakePointData:
    def __init__(self, x, y, crop_size, image_x_pixel=None, image_y_pixel=None):
        if image_x_pixel is None or image_y_pixel is None:
            self.image_x_pixel = torch.rand(1).item() * crop_size
            self.image_y_pixel = torch.rand(1).item() * crop_size                
        else:
            self.image_x_pixel = image_x_pixel
            self.image_y_pixel = image_y_pixel
        self.global_x_pixel = self.image_x_pixel + x
        self.global_y_pixel = self.image_y_pixel + y
        ''' min_x = '01' x 5000 + 1 = 5000
            max_x = '74' x 5000 + 5000 = 375000 '''
        self.lon3338 = (self.global_x_pixel-5000) / (375000-5000) * (1500005.-(-2199995.)) + (-2199995.)
        ''' min_y = '01' x 5000 + 1 = 5000
            max_y = '48' x 5000 + 5000 = 245000 
            noting that lat is decreasing '''
        self.lat3338 = (self.global_y_pixel-5000) / (245000-5000) * (5.-2400005.) + 2400005.    
        self.value = -999.
    

############################
##### Image Data Class #####
############################          
class ImageData:
    def __init__(self, config):
        self.use_sat = config.DATA.SAT.USE_SAT
        self.sat_path = config.DATA.SAT.PATH
        self.sat_dim = config.DATA.SAT.NUM_BANDS   

        self.cov_path = config.DATA.COV.PATH
        self.use_dem = config.DATA.DEM.USE_DEM
        self.dem_dim = config.DATA.DEM.NUM_BANDS
        self.use_clm = config.DATA.CLM.USE_CLM
        self.clm_dim = config.DATA.CLM.NUM_BANDS
        self.use_cov = config.DATA.COV.USE_COV
        self.cov_dim = config.DATA.COV.NUM_BANDS
        self.use_geocoord = config.DATA.GEOCOORD.USE_GEOCOORD
        self.geocoord_dim = config.DATA.GEOCOORD.NUM_BANDS
        
        self.crop_size = config.DATA.CROP_SIZE
        self.input_size = config.DATA.INPUT_SIZE
        self.in_dim = 0
        if self.use_sat: self.in_dim += self.sat_dim;
        if self.use_dem: self.in_dim += self.dem_dim;     
        if self.use_clm: self.in_dim += self.clm_dim;
        if self.use_geocoord: self.in_dim += self.geocoord_dim;
            
    def get_image(self, x1, y1, x2, y2):
        img_list, mask_list = [], []
        mask = torch.zeros((self.crop_size, self.crop_size), dtype=torch.bool)

        if self.use_sat:
            sat, sat_mask = crop_sat(x1, y1, x2, y2, self.sat_path, num_bands=self.sat_dim)
            img_list.append(sat)
            mask = mask | sat_mask
            
        if self.use_dem or self.use_clm or self.use_cov:
            cov, cov_mask = crop_cov(x1, y1, x2, y2, self.cov_path)
            if self.use_dem:                
                img_list.append(cov[:, :, :self.dem_dim])
            if self.use_clm:
                img_list.append(cov[:, :, -self.clm_dim:])
            if self.use_cov:
                img_list.append(cov)
            mask = mask | cov_mask

        if self.use_geocoord:
            lon3338_1 = (x1-5000) / (375000-5000) * (1500005.-(-2199995.)) + (-2199995.)
            lon3338_2 = (x2-5000) / (375000-5000) * (1500005.-(-2199995.)) + (-2199995.)
            lat3338_1 = (y1-5000) / (245000-5000) * (5.-2400005.) + 2400005.    
            lat3338_2 = (y2-5000) / (245000-5000) * (5.-2400005.) + 2400005.    
            r = (lon3338_2 - lon3338_1) / (2 * self.crop_size)
            lon3338_seq = lon3338_1 + r + (2 * r) * torch.arange(self.crop_size).float()
            r = (lat3338_2 - lat3338_1) / (2 * self.crop_size)
            lat3338_seq = lon3338_1 + r + (2 * r) * torch.arange(self.crop_size).float()
            lat_grid, lon_grid = torch.meshgrid(lat3338_seq, lon3338_seq, indexing='ij')
            lon_grid = (lon_grid-(-2199995.)) / (1500005.-(-2199995.)) * 2 - 1
            lat_grid = (lat_grid-2400005.) / (5.-2400005.) * 2 - 1
            img_list.append(lon_grid[:, :, None])
            img_list.append(lat_grid[:, :, None])
            
        img = torch.cat(img_list, dim=-1)
        img = torch.permute(img, (2, 0, 1))
        img = transforms.Resize((self.input_size, self.input_size), antialias=None)(img)
        return img, mask
    
    
####################################
##### Collection of One Sample #####
####################################
class SampleCollection:
    def __init__(self, img, crop_size):
        self.crop_size = crop_size
        self.img = img
        self.labels = []
        self.coords = []
        self.coords3338 = []
        
    def cut(self, target_num):
        self.labels = self.labels[:target_num]
        self.coords = self.coords[:target_num]
        self.coords3338 = self.coords3338[:target_num]
    
    def add_image_coord(self, image_coord):
        x_norm, y_norm = normalize_image_coords(image_coord, self.crop_size)
        self.coords.append([x_norm, y_norm])
    
    def add_coord3338(self, coord):
        x_norm, y_norm = normalize_coord3338(coord)
        self.coords3338.append([x_norm, y_norm])
        
    def add_label(self, label):
        self.labels.append(label)
        
    def __len__(self):
        return len(self.labels)

    def add_random_pt(self, x, y):
        fk = FakePointData(x, y, self.crop_size)
        self.add_image_coord((fk.image_x_pixel, fk.image_y_pixel))
        self.add_coord3338((fk.lon3338, fk.lat3338))
        self.add_label(fk.value)     
    
    def add_regular_pts(self, x, y):
        for ix in range(0, self.crop_size+1, 50):
            for iy in range(0, self.crop_size+1, 50):
                fk = FakePointData(x, y, self.crop_size, image_x_pixel=ix, image_y_pixel=iy)
                self.add_image_coord((fk.image_x_pixel, fk.image_y_pixel))
                self.add_coord3338((fk.lon3338, fk.lat3338))
                self.add_label(fk.value)     
        
    def rasterize(self, output_size, out_dim):
        new_labels = torch.full((output_size, output_size, out_dim), -999.)
        for i in range(len(self.labels)):
            if self.labels[i] > -999:
                coord = (self.coords[i][0], self.coords[i][1])
                x, y = denormalize_image_coords(coord, output_size)
                new_labels[int(x), int(y), 0] = self.labels[i]
        self.labels = new_labels
        self.coords = -1
        self.coords3338 = -1
    
    def vectorize(self):
        self.labels = torch.tensor(self.labels, dtype=torch.float32).reshape(-1, 1) 
        self.coords = torch.tensor(self.coords, dtype=torch.float32).reshape(-1, 2) 
        self.coords3338 = torch.tensor(self.coords3338, dtype=torch.float32).reshape(-1, 2) 
        
    def get_tensors(self, pid=None):
        if pid is None:
            return self.img, self.labels, self.coords, self.coords3338
        else:
            return self.img, self.labels, self.coords, self.coords3338, pid
    
    def get_pretrain_tensors(self):
        self.coords = torch.tensor(self.coords, dtype=torch.float32).reshape(-1, 2) 
        self.coords3338 = torch.tensor(self.coords3338, dtype=torch.float32).reshape(-1, 2) 
        return self.img, self.coords, self.coords3338
        

######################
##### Main Class #####
######################
def sample_crop_location(coord, crop_size, add_random=True):
    x, y = coord
    if add_random:
        cx = x + random.randint(-crop_size // 4, crop_size // 4) + random.random()
        cy = y + random.randint(-crop_size // 4, crop_size // 4) + random.random()
    else:
        cx, cy = x, y
    x1, y1 = int(cx - crop_size // 2), int(cy - crop_size // 2)
    x2, y2 = int(x1 + crop_size), int(y1 + crop_size)
    return (x1, y1, x2, y2)


class AKSDBDataset(Dataset):
    def __init__(self, point_data_list, mode, config):
        self.var_name = config.VAR_NAME
        self.point_data_list = point_data_list 
        self.id2point = {dat.id: dat for dat in self.point_data_list}
        print(f"Number of {mode} point data: {len(self.point_data_list)}")
        
        self.mode = mode
        self.image_data = ImageData(config)
        
        self.in_dim = self.image_data.in_dim
        self.out_dim = config.DATA.OUT_DIM
        self.crop_size = config.DATA.CROP_SIZE
        self.input_size = config.DATA.INPUT_SIZE
        self.output_size = config.DATA.OUTPUT_SIZE
        
        self.out_raster = 'seg' in config.EXP_NAME
        self.use_neighbor_labels = config.DATA.USE_NEIGHBOR_LABELS
        self.use_other_tiles_ratio = config.DATA.USE_OTHER_TILES_RATIO
        self.num_labels_per_sample = config.DATA.NUM_LABELS_PER_SAMPLE
        self.num_random_pseudo_labels = config.DATA.NUM_RANDOM_PSEUDO_LABELS
        self.add_regular_pseudo_labels = config.DATA.ADD_REGULAR_PSEUDO_LABELS
        
        self.value_mean, self.value_std = self.get_stats()
        
        if mode == 'train' and self.use_neighbor_labels:
            self.neighbor_dict = build_neighbor_dict(self.point_data_list, self.crop_size)
        
    def get_stats(self):
        if self.var_name in ['aksdb_pf1m_bin', 'tax_order']:
            return 0., 1.
        else:
            labels = np.array([point_data.value for point_data in self.point_data_list])
            return labels.mean(), labels.std()
        
    def __len__(self):
        return len(self.point_data_list)

    def load_train_sample(self, point_data):
        global_coord = (point_data.global_x_pixel, point_data.global_y_pixel)
        x1, y1, x2, y2 = sample_crop_location(global_coord, self.crop_size) # x,y are global crop locations
        img, mask = self.image_data.get_image(x1, y1, x2, y2)
        
        collection = SampleCollection(img, self.crop_size)
        if self.use_neighbor_labels:
            neighbors = [self.id2point[i] for i in self.neighbor_dict[point_data.id]]
            random.shuffle(neighbors)
        else:
            neighbors = []
            
        for dat in [point_data] + neighbors:
            ix, iy = dat.global_x_pixel - x1, dat.global_y_pixel - y1
            if 0 < ix < self.crop_size and 0 < iy < self.crop_size:
                collection.add_image_coord((ix, iy))
                collection.add_coord3338((dat.lon3338, dat.lat3338))
                label = (dat.value - self.value_mean) / self.value_std
                collection.add_label(label)
                
        collection.cut(self.num_labels_per_sample)
        for _ in range(self.num_labels_per_sample - len(collection)):
            collection.add_random_pt(x1, y1)
        
        for _ in range(self.num_random_pseudo_labels):
            collection.add_random_pt(x1, y1)
             
        if self.add_regular_pseudo_labels:
            collection.add_regular_pts(x1, y1)
        
        if self.out_raster: collection.rasterize(self.output_size, self.out_dim);
        else: collection.vectorize();
        return collection
    
    def load_test_sample(self, point_data):
        global_coord = (point_data.global_x_pixel, point_data.global_y_pixel)
        x1, y1, x2, y2 = sample_crop_location(global_coord, self.crop_size, add_random=False)        
        img, mask = self.image_data.get_image(x1, y1, x2, y2)
        
        collection = SampleCollection(img, self.crop_size)
        ix, iy = point_data.global_x_pixel - x1, point_data.global_y_pixel - y1
        collection.add_image_coord((ix, iy))
        collection.add_coord3338((point_data.lon3338, point_data.lat3338))
        collection.add_label(point_data.value)
        
        if self.out_raster: collection.rasterize(self.output_size, self.out_dim);
        else: collection.vectorize();
        return collection

    def load_train_sample_from_other_tiles(self):
        grid_id = random.choice(GRID_IDS)
        grid_H, grid_V = gridID2hv(grid_id)
        grid_x_pixel = random.random() * TILE_SIZE
        grid_y_pixel = random.random() * TILE_SIZE    
        global_x_pixel = grid_H * TILE_SIZE + grid_x_pixel
        global_y_pixel = grid_V * TILE_SIZE + grid_y_pixel
        global_coord = (global_x_pixel, global_y_pixel)

        x1, y1, x2, y2 = sample_crop_location(global_coord, self.crop_size) # x,y are global crop locations
        img, mask = self.image_data.get_image(x1, y1, x2, y2)
        
        collection = SampleCollection(img, self.crop_size)        
        for _ in range(self.num_labels_per_sample):
            collection.add_random_pt(x1, y1)
        
        for _ in range(self.num_random_pseudo_labels):
            collection.add_random_pt(x1, y1)
             
        if self.add_regular_pseudo_labels:
            collection.add_regular_pts(x1, y1)
        
        if self.out_raster: collection.rasterize(self.output_size, self.out_dim);
        else: collection.vectorize();
        return collection
        
    def __getitem__(self, idx):
        if self.mode == 'train':
            point_data = random.choice(self.point_data_list)
            if random.random() > self.use_other_tiles_ratio:
                sample = self.load_train_sample(point_data)  
            else:
                sample = self.load_train_sample_from_other_tiles()              
            return sample.get_tensors()
        else:
            point_data = self.point_data_list[idx]
            sample = self.load_test_sample(point_data)
            return sample.get_tensors(pid=point_data.id)
        

##########################
##### Build Datasets #####
##########################
from sklearn.cluster import DBSCAN

def spatial_split(all_train_data, thre=1000):
    locations = []
    for data in all_train_data:
        locations.append(np.array([data.lon3338, data.lat3338]))
        
    X = np.array(locations).reshape(-1, 2)
    clustering = DBSCAN(eps=thre, min_samples=2).fit(X)
    labels = clustering.labels_.tolist()
 
    dat_label_dict = defaultdict(list)
    default_label = 3513553
    for dat, label in zip(all_train_data, labels):
        if label == -1:
            dat_label_dict[default_label].append(dat)
            default_label += 1
        else:
            dat_label_dict[label].append(dat)

    unique_labels = sorted(list(set(list(dat_label_dict.keys()))))  
    print("Unique_labels:", len(unique_labels))

    random.shuffle(unique_labels)
    unique_labels_for_train = unique_labels[:int(len(unique_labels) * 0.95)]
    unique_labels_for_val = unique_labels[int(len(unique_labels) * 0.95):]

    train_data, val_data = [], []
    for i in unique_labels_for_train:
        train_data += dat_label_dict[i]
    for i in unique_labels_for_val:
        val_data += dat_label_dict[i]

    return train_data, val_data


def split_train_val_test(point_data_list, config):
    print(f'Load train/val/test from {config.SPLIT_FILE}')
    with open(config.SPLIT_FILE, 'r') as f:
        train_test_split = json.load(f)
        
    if config.SPLIT_MODE in ['kfold', 'sfold', 'gfold', 'tfold']:
        print(f'Train/Test setting: Fold ID = {config.FOLD_ID}')
        train_test_split = train_test_split[f'fold_{config.FOLD_ID}']
    elif config.SPLIT_MODE == 'uniform':
        pass
    else:
        raise NotImplementedError

    point_data_dict = {dat.id: dat for dat in point_data_list}
    all_train_data = [point_data_dict[idx] for idx in train_test_split['train'] 
                      if point_data_dict[idx].is_valid(config.DATA.SAT.PATH)]
    
    test_data = [point_data_dict[idx] for idx in train_test_split['test'] 
                 if point_data_dict[idx].is_valid(config.DATA.SAT.PATH)]
    
    if config.TRAIN.LR_SCHEDULER.NAME == "validate":
        if config.SPLIT_MODE == 'kfold':
            indices = list(range(len(all_train_data)))
            random.seed(1234)
            random.shuffle(indices)
            train_indices = indices[:int(0.9*len(indices))]
            val_indices = indices[int(0.9*len(indices)):]
            train_data = [all_train_data[i] for i in train_indices]
            val_data = [all_train_data[i] for i in val_indices]
        elif config.SPLIT_MODE == 'sfold':
            train_data, val_data = spatial_split(all_train_data, thre=1000)
        elif config.SPLIT_MODE == 'tfold':
            train_data, val_data = spatial_split(all_train_data, thre=10000)
        return train_data, val_data, test_data
        
    else:
        return all_train_data, test_data, test_data

    
def build_train_val_test_datasets(config):
    print(f'Load point data from {config.DATA.POINT_DATA_FILE}')
    with open(config.DATA.POINT_DATA_FILE, 'r') as f:
        point_data = json.load(f)        
        point_data = [PointData(dat, config.VAR_NAME) for dat in point_data]
        
    train_data, val_data, test_data = split_train_val_test(point_data, config)
    
    train_dataset = AKSDBDataset(train_data, mode='train', config=config)
    val_dataset = AKSDBDataset(val_data, mode='val', config=config)
    test_dataset = AKSDBDataset(test_data, mode='test', config=config)

    print(f'Number of train samples: {len(train_dataset)}; ',
          f'Mean={train_dataset.value_mean}; Std={train_dataset.value_std}')
    print(f'Number of val samples: {len(val_dataset)}; ',
          f'Mean={val_dataset.value_mean}; Std={val_dataset.value_std}')
    print(f'Number of test samples: {len(test_dataset)}; ',
          f'Mean={test_dataset.value_mean}; Std={test_dataset.value_std}')
    return train_dataset, val_dataset, test_dataset


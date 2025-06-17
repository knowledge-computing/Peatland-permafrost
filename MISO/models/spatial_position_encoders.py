import math
import numpy as np
import torch
import torch.nn as nn


class PosEncodingNeRF(nn.Module):
    """ 
        Module to add positional encoding as in NeRF [Mildenhall et al. 2020].
        Code Reference: https://github.com/vsitzmann/siren/blob/master/modules.py
    """
    def __init__(
        self, 
        in_features, 
        num_frequencies,
        sidelength=None, 
        fn_samples=None, 
        use_nyquist=True
    ):
        super().__init__()

        self.in_features = in_features
        self.num_frequencies = num_frequencies
        
        # elif self.in_features == 2:
        #     assert sidelength is not None
        #     if isinstance(sidelength, int):
        #         sidelength = (sidelength, sidelength)
        #     self.num_frequencies = 4
        #     if use_nyquist:
        #         self.num_frequencies = self.get_num_frequencies_nyquist(min(sidelength[0], sidelength[1]))
        # elif self.in_features == 1:
        #     assert fn_samples is not None
        #     self.num_frequencies = 4
        #     if use_nyquist:
        #         self.num_frequencies = self.get_num_frequencies_nyquist(fn_samples)

        self.out_dim = in_features + 2 * in_features * self.num_frequencies

    # def get_num_frequencies_nyquist(self, samples):
    #     nyquist_rate = 1 / (2 * (2 * 1 / samples))
    #     return int(math.floor(math.log(nyquist_rate, 2)))

    def forward(self, coords):
        coords = coords.view(coords.shape[0], -1, self.in_features)
        coords_pos_enc = coords
        for i in range(self.num_frequencies):
            for j in range(self.in_features):
                c = coords[..., j]
                sin = torch.unsqueeze(torch.sin((2 ** i) * np.pi * c), -1)
                cos = torch.unsqueeze(torch.cos((2 ** i) * np.pi * c), -1)
                coords_pos_enc = torch.cat((coords_pos_enc, sin, cos), axis=-1)
        return coords_pos_enc.reshape(coords.shape[0], -1, self.out_dim)


    
#############################################################
#############################################################
#############################################################
"""
    The following code is from 
    https://github.com/gengchenmai/space2vec/blob/master/geo_prior/geo_prior/SpatialRelationEncoder.py
"""
def _cal_freq_list(freq_init, freq_num, max_radius, min_radius):
    if freq_init == "random":
        # the frequence we use for each block, alpha in ICLR paper
        # freq_list shape: (freq_num)
        freq_list = np.random.random(size=[freq_num]) * max_radius
    elif freq_init == "geometric":
        max_radius = max_radius * 1.
        min_radius = min_radius * 1.        
        log_timescale_increment = (math.log(max_radius/min_radius) / (freq_num-1))
        timescales = min_radius * np.exp(
            np.arange(freq_num).astype(float) * log_timescale_increment)
        freq_list = 1.0 / timescales
    return torch.tensor(freq_list, dtype=torch.float32) # freq_num


class GridCellSpatialRelationEncoder(nn.Module):
    def __init__(
        self, 
        in_features=2, 
        num_frequencies=16, 
        max_radius=2, 
        min_radius=0.0005,
        freq_init="geometric"
    ):       
        super(GridCellSpatialRelationEncoder, self).__init__()

        self.in_features = in_features 
        self.num_frequencies = num_frequencies
        self.freq_init = freq_init
        self.max_radius = max_radius
        self.min_radius = min_radius

        freq_list = _cal_freq_list(freq_init, num_frequencies, max_radius, min_radius)
        self.freq_mat = freq_list[:, None].repeat(1, 2) # num_frequencies x 2
        self.out_dim = 2 * in_features * self.num_frequencies

    def forward(self, coords):
        coords = coords.view(coords.shape[0], -1, self.in_features)  # B x N x 2
        B, N, _ = coords.shape
        coords_pos_enc = coords[:, :, :, None, None]
        coords_pos_enc = coords_pos_enc.repeat(1, 1, 1, self.num_frequencies, 2) # B x N x 2 x Freq x 2
        coords_pos_enc = coords_pos_enc * self.freq_mat.to(coords.device)
        coords_pos_enc[:, :, :, :, 0::2] = torch.sin(coords_pos_enc[:, :, :, :, 0::2])  # dim 2i
        coords_pos_enc[:, :, :, :, 1::2] = torch.cos(coords_pos_enc[:, :, :, :, 1::2])  # dim 2i+1
        coords_pos_enc = coords_pos_enc.reshape(B, N, -1)
        return coords_pos_enc
    

class TheoryGridCellSpatialRelationEncoder(nn.Module):
    def __init__(
        self, 
        in_features=2, 
        num_frequencies=16, 
        max_radius=2, 
        min_radius=0.0005,
        freq_init="geometric"
    ):       
        super(TheoryGridCellSpatialRelationEncoder, self).__init__()

        self.in_features = in_features 
        self.num_frequencies = num_frequencies
        self.freq_init = freq_init
        self.max_radius = max_radius
        self.min_radius = min_radius

        self.unit_vec1 = torch.tensor([1.0, 0.0], dtype=torch.float32) # 0
        self.unit_vec2 = torch.tensor([-1.0/2.0, math.sqrt(3)/2.0], dtype=torch.float32)      # 120 degree
        self.unit_vec3 = torch.tensor([-1.0/2.0, -math.sqrt(3)/2.0], dtype=torch.float32)     # 240 degree

        freq_list = _cal_freq_list(freq_init, num_frequencies, max_radius, min_radius)
        self.freq_mat = freq_list[:, None].repeat(1, 6) # num_frequencies x 6
        self.out_dim = 6 * self.num_frequencies

    def forward(self, coords):
        coords = coords.view(coords.shape[0], -1, self.in_features)  # B x N x 2
        B, N, _ = coords.shape
        unit_vec1 = self.unit_vec1.to(coords.device)
        unit_vec2 = self.unit_vec2.to(coords.device)
        unit_vec3 = self.unit_vec3.to(coords.device)
        angle_mat1 = torch.matmul(coords, unit_vec1)
        angle_mat2 = torch.matmul(coords, unit_vec2)
        angle_mat3 = torch.matmul(coords, unit_vec3)
            
        angle_mat = torch.stack([angle_mat1, angle_mat1, 
                                 angle_mat2, angle_mat2, 
                                 angle_mat3, angle_mat3], dim=-1)
                
        angle_mat = angle_mat[:, :, None, :].repeat(1, 1, self.num_frequencies, 1)
        coords_pos_enc = angle_mat * self.freq_mat.to(coords.device)
        coords_pos_enc = coords_pos_enc.reshape(B, N, -1)
        coords_pos_enc[:, :, 0::2] = torch.sin(coords_pos_enc[:, :, 0::2])  # dim 2i
        coords_pos_enc[:, :, 1::2] = torch.cos(coords_pos_enc[:, :, 1::2])  # dim 2i+1
        return coords_pos_enc
    
    
    
    
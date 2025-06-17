import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import timm

from models.encoders import VisualQuery, GeoEncoder
from models.model_utils import MLP, FCLayer


class GeoDecoder(nn.Module):
    def __init__(
        self, 
        geo_encoder, 
        decoder_name="geo_only",
        out_dim=1
    ):
        super(GeoDecoder, self).__init__()  
        self.decoder_name = decoder_name
        self.out_dim = out_dim

        self.geo_encoder = geo_encoder
        self.emb_dim = self.geo_encoder.emb_dim
        self.geo_mapper = nn.Sequential(
            MLP(self.emb_dim, self.emb_dim, [self.emb_dim], 'gelu'),
            nn.LayerNorm(self.emb_dim))
            
        self.geo_out = nn.Sequential(
            nn.Linear(self.emb_dim, self.emb_dim),
            nn.GELU(),
            nn.Linear(self.emb_dim, self.emb_dim // 2),
            nn.GELU(),
            nn.Linear(self.emb_dim // 2, self.out_dim))

    def get_geo_embeddings(self, geo_coords):
        geo_emb = self.geo_encoder(geo_coords)
        geo_pred = self.geo_mapper(geo_emb)        
        geo_pred = self.geo_out(geo_pred) 
        return geo_emb, geo_pred
    
    def forward(self, geo_coords):
        out = {}
        geo_emb, geo_pred = self.get_geo_embeddings(geo_coords)
        out['geo_emb'] = geo_emb
        out['pred'] = geo_pred
        return out


class Decoder(nn.Module):
    def __init__(
        self, 
        encoders, 
        geo_encoder=None, 
        decoder_name="visual_only",
        out_dim=1
    ):
        super(Decoder, self).__init__()  
        self.decoder_name = decoder_name
        self.out_dim = out_dim
        self.modalities = list(encoders.keys())
        print(f"***** Modalities: {self.modalities}")
        self.num_modalities = len(self.modalities)

        assert "sat" in self.modalities
        self.emb_dim = encoders['sat'].emb_dim
        
        for key, encoder in encoders.items():
            setattr(self, key + "_encoder", encoder)
            setattr(self, key + "_dim", encoder.in_dim)
            mapper = nn.Sequential(
                MLP(self.emb_dim, self.emb_dim, [self.emb_dim], 'gelu'),
                nn.LayerNorm(self.emb_dim))
            setattr(self, key + "_mapper", mapper)
        
        self.visual_mapper = nn.Sequential(
            MLP(self.emb_dim, self.emb_dim, [self.emb_dim], 'gelu'),
            nn.LayerNorm(self.emb_dim))
            
        self.out = nn.Sequential(
            nn.Linear(self.emb_dim, self.emb_dim),
            nn.GELU(),
            nn.Linear(self.emb_dim, self.emb_dim // 2),
            nn.GELU(),
            nn.Linear(self.emb_dim // 2, self.out_dim))

        if "geo" in self.decoder_name:
            self.geo_encoder = geo_encoder
            self.geo_mapper = nn.Sequential(
                MLP(self.emb_dim, self.emb_dim, [self.emb_dim], 'gelu'),
                nn.LayerNorm(self.emb_dim))
            
            self.geo_out = nn.Sequential(
                nn.Linear(self.emb_dim, self.emb_dim),
                nn.GELU(),
                nn.Linear(self.emb_dim, self.emb_dim // 2),
                nn.GELU(),
                nn.Linear(self.emb_dim // 2, self.out_dim))

        if self.decoder_name == "visual_geo_naive":
            self.fuse_mapper = nn.Sequential(
                MLP(self.emb_dim, self.emb_dim, [self.emb_dim], 'gelu'),
                nn.LayerNorm(self.emb_dim))
            
            self.sat_out = MLP(self.emb_dim, self.out_dim, [self.emb_dim // 2], 'gelu')
            self.cov_out = MLP(self.emb_dim, self.out_dim, [self.emb_dim // 2], 'gelu')

    def retrieve_visual_embs(self, x, x_coords, geo_coords):
        sat = x[:, :self.sat_dim, ...]
        sat_emb = self.sat_encoder(sat, x_coords, geo_coords=geo_coords)
        sat_emb = self.sat_mapper(sat_emb)
        embs = {'sat_emb': sat_emb}
        if "dem" in self.modalities:
            dem = x[:, self.sat_dim: self.sat_dim + self.dem_dim, ...]
            dem_emb = self.dem_encoder(dem, x_coords, geo_coords=geo_coords)
            dem_emb = self.dem_mapper(dem_emb)
            embs['dem_emb'] = dem_emb
        if "clm" in self.modalities:
            clm = x[:, -self.clm_dim:, ...]
            clm_emb = self.clm_encoder(clm, x_coords, geo_coords=geo_coords)
            clm_emb = self.clm_mapper(clm_emb)
            embs['clm_emb'] = clm_emb            
        if "cov" in self.modalities:
            cov = x[:, self.sat_dim: self.sat_dim + self.cov_dim, ...]
            cov_emb = self.cov_encoder(cov, x_coords, geo_coords=geo_coords)
            cov_emb = self.cov_mapper(cov_emb)
            embs['cov_emb'] = cov_emb
        return embs

    def get_geo_embeddings(self, geo_coords):
        geo_emb = self.geo_encoder(geo_coords)
        geo_emb = self.geo_mapper(geo_emb)        
        geo_pred = self.geo_out(geo_emb) 
        return geo_emb, geo_pred
    
    def forward(self, x, x_coords, geo_coords):
        out = {}
        if "geo" in self.decoder_name:
            geo_emb, geo_pred = self.get_geo_embeddings(geo_coords)
            out['geo_emb'] = geo_emb
            out['geo_pred'] = geo_pred

        if self.decoder_name == 'visual_geo_attn':
            embs = self.retrieve_visual_embs(x, x_coords, geo_emb)
        else:
            embs = self.retrieve_visual_embs(x, x_coords, geo_coords)
        out.update(embs)
        
        if self.decoder_name == 'visual_geo_naive':
            visual_emb = sum(embs.values())
            out_emb = self.fuse_mapper(visual_emb + geo_emb * 0.1)
            out['sat_pred'] = self.sat_out(out['sat_emb'])
            out['cov_pred'] = self.cov_out(out['cov_emb'])
        else:
            visual_emb = sum(embs.values())
            out_emb = self.visual_mapper(visual_emb)        
            out['visual_emb'] = out_emb
            
        pred = self.out(out_emb)
        out['pred'] = pred
        out['visual_pred'] = pred
        return out


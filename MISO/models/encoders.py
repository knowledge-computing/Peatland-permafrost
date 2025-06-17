import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import torchvision
import timm 

from models.visual_query import get_query_layers, VisualQuery, SpatialVaryingVisualQuery
from models.swin_transformer import SwinTransformer
from models.position_embed import get_2d_sincos_pos_embed
from models.spatial_position_encoders import *
from models.model_utils import MLP, FCLayer

######################
##### GeoEncoder #####
######################
class GeoEncoder(nn.Module):
    def __init__(
        self, 
        emb_dim=1024,
        num_frequencies=1,
        enc_method='gridcell'
    ):
        super(GeoEncoder, self).__init__()
        self.emb_dim = emb_dim
        if enc_method == 'nerf':
            self.pos_encoder = PosEncodingNeRF(in_features=2, 
                                               num_frequencies=num_frequencies)
        elif enc_method == 'gridcell':
            self.pos_encoder = GridCellSpatialRelationEncoder(in_features=2, 
                                                              num_frequencies=num_frequencies)
        elif enc_method == 'theory':
            self.pos_encoder = TheoryGridCellSpatialRelationEncoder(in_features=2, 
                                                                    num_frequencies=num_frequencies)
            
        layers = []
        layers.append(FCLayer(self.pos_encoder.out_dim, emb_dim, nonlinearity='relu'))
        for i in range(3):
            layers.append(FCLayer(emb_dim, emb_dim, nonlinearity='relu',
                                  dropout_rate=0.1, layer_norm=True, skip=True))
        layers.append(FCLayer(emb_dim, emb_dim, nonlinearity='relu'))
        self.layers = torch.nn.Sequential(*layers)
        
    def forward(self, coords):
        pos_enc = self.pos_encoder(coords)        
        emb = self.layers(pos_enc)
        return emb
                

#######################
##### SwinEncoder #####
#######################    
class SwinEncoder(nn.Module):
    def __init__(self, in_dim, query_method, config):
        super(SwinEncoder, self).__init__()

        self.in_dim = in_dim
        self.query_method = query_method
        self.use_final_layer = config.MODEL.SWIN.USE_FINAL_LAYER
        self.pooling = config.MODEL.SWIN.POOLING
        self.feature_extractor = torchvision.models.swin_v2_b() 
        self.feature_extractor.features[0][0] = nn.Conv2d(self.in_dim, 128, kernel_size=(4, 4), stride=(4, 4))
        emb_dim = self.feature_extractor.head.in_features        
        self.query_layers = get_query_layers(self.query_method, emb_dim, self.pooling)
        self.emb_dim = emb_dim 
        self.norm = nn.LayerNorm(self.emb_dim)
            
    def forward(self, x, coords, geo_coords):
        """
            x: B x C x H x W
            coords: B x N x 2
            geo_coords: B x N x 2 or B x N x C'
            -----
            OUTPUT: B x N x C', default C'=1024
        """
        emb = self.feature_extractor.features[0](x) # emb: B x H' x W' x C, default C=128, H'=H/4
        img_feats, embs = [], []
        for i in range(1, 8):            
            if i % 2 == 0: 
                emb = self.feature_extractor.features[i](emb) # the layer of patch merging
            else:
                emb = self.feature_extractor.features[i](emb) # the layer of attention
                img_feats.append(emb)
                query_emb = self.query_layers[len(embs)](emb, coords, geo_coords)                    
                embs.append(query_emb)
        
        if self.use_final_layer:
            return embs[-1]
        
        if self.pooling == 'sum':            
            embs = torch.stack(embs, dim=1) # B x L x N x C
            embs = torch.sum(embs, dim=1)
            embs = self.norm(embs)
            return embs
        elif self.pooling == 'concat':
            return torch.cat(embs, dim=-1)
        else:
            return torch.stack(embs, dim=1) # B x L x N x C, L=4
        
##########################
##### SimSwinEncoder #####
##########################                
class SimSwinEncoder(nn.Module):
    def __init__(self, in_dim, query_method, config):
        super(SimSwinEncoder, self).__init__()

        self.in_dim = in_dim
        self.query_method = query_method
        self.use_final_layer = config.MODEL.SWIN.USE_FINAL_LAYER
        self.pooling = config.MODEL.SWIN.POOLING
        self.feature_extractor = SwinTransformer(img_size=config.DATA.INPUT_SIZE,
                                                 patch_size=config.MODEL.SWIN.PATCH_SIZE,
                                                 in_chans=in_dim, 
                                                 embed_dim=config.MODEL.SWIN.EMBED_DIM,
                                                 depths=config.MODEL.SWIN.DEPTHS,
                                                 num_heads=config.MODEL.SWIN.NUM_HEADS,
                                                 window_size=config.MODEL.SWIN.WINDOW_SIZE,
                                                 mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
                                                 qkv_bias=config.MODEL.SWIN.QKV_BIAS,
                                                 qk_scale=config.MODEL.SWIN.QK_SCALE,
                                                 ape=config.MODEL.SWIN.APE,
                                                 patch_norm=config.MODEL.SWIN.PATCH_NORM)
        emb_dim = 128 * 8
        self.query_layers = get_query_layers(self.query_method, emb_dim, self.pooling)
        self.emb_dim = emb_dim 
        self.norm = nn.LayerNorm(self.emb_dim)
            
    def forward(self, x, coords, geo_coords):
        """
            x: B x C x H x W
            coords: B x N x 2
            geo_coords: B x N x 2 or B x N x C'
            -----
            OUTPUT: B x N x C', default C'=1024
        """              
        xs = self.feature_extractor(x)
        
        img_feats, out = [], []
        for i, emb in enumerate(xs):
            b, p, c = emb.shape
            emb = emb.reshape(b, int(p**0.5), int(p**0.5), c)
            img_feats.append(emb)
            query_emb = self.query_layers[i](emb, coords, geo_coords)                  
            out.append(query_emb)

        if self.pooling == 'sum':
            out = torch.stack(out, dim=1)
            out = torch.sum(out, dim=1)
            out = self.norm(out)
            return out
        elif self.pooling == 'concat':
            return torch.cat(out, dim=-1)
        else:
            return torch.stack(out, dim=1) # B x L x N x C
        
######################
##### ViTEncoder #####
######################        
class ViTEncoder(nn.Module):
    def __init__(
        self,
        in_dim,
        query_method='visual_query',
        pretrain_model='mae',
        global_pool=False,
    ):
        super(ViTEncoder, self).__init__()

        self.in_dim = in_dim
        self.query_method = query_method
        self.global_pool = global_pool
        self.pretrain_model = pretrain_model
        pretrained = not (pretrain_model is not None)
        self.feature_extractor = timm.create_model('vit_base_patch16_224', pretrained=pretrained)
        emb_dim = self.feature_extractor.head.in_features
        self.emb_dim = emb_dim

        new_patch_embed = nn.Conv2d(in_channels=in_dim, out_channels=emb_dim, kernel_size=16, stride=16)
        self.feature_extractor.patch_embed.proj = new_patch_embed
        
        # Added by Samar, need default pos embedding
        pos_embed = get_2d_sincos_pos_embed(self.feature_extractor.pos_embed.shape[-1], 
                                            int(self.feature_extractor.patch_embed.num_patches ** .5),
                                            cls_token=True)
        self.feature_extractor.pos_embed.data.copy_(torch.from_numpy(pos_embed).unsqueeze(0))
        
        if self.global_pool:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
            embed_dim = 768
            self.fc_norm = norm_layer(embed_dim)
            del self.norm  # remove the original norm

        if self.query_method == 'visual_query':
            self.query_layer = VisualQuery(in_dim=emb_dim + 2, 
                                           out_dim=emb_dim,
                                           hidden_dims=[emb_dim, emb_dim]) 
        elif self.query_method == 'sv_visual_query':
            self.query_layer = SpatialVaryingVisualQuery(in_dim=emb_dim + 2 + 8, 
                                                         out_dim=emb_dim,
                                                         hidden_dims=[emb_dim, emb_dim]) 

            
    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        # stole cls_tokens impl from Phil Wang, thanks
        cls_tokens = self.cls_token.expand(B, -1, -1)  
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.feature_extractor.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]
        return outcome
        
    def forward(self, x, coords, geo_coords):
        if self.pretrain_model == 'mae':
            emb = self.forward_features(x)
        else:
            emb = self.feature_extractor.patch_embed(x)
            for i in range(len(self.feature_extractor.blocks)):
                emb = self.feature_extractor.blocks[i](emb)

        emb = emb.view(x.shape[0], 14, 14, self.emb_dim)
        query_emb = self.query_layer(emb, coords, geo_coords)
        return query_emb


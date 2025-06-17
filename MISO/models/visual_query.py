import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model_utils import MLP, make_coord


#######################
##### VisualQuery #####
#######################
class VisualQuery(nn.Module):
    """
        Code Reference: https://github.com/yinboc/liif/blob/main/models/liif.py
    """
    def __init__(
        self, 
        in_dim, 
        hidden_dims, 
        out_dim
    ):
        super(VisualQuery, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dims = hidden_dims
        self.mlp = MLP(in_dim=in_dim, 
                       out_dim=out_dim,
                       hidden_dims=hidden_dims)
        self.output_layer = nn.Linear(out_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        
    def query(self, feat, coord): 
        """
            feat: B x H x W x C
            coord: B x N x 2
        """
        vx_lst = [-1, 1]
        vy_lst = [-1, 1]
        eps_shift = 1e-6
               
        num_patches = int(feat.shape[-2])
        rx = (1 - (-1)) / num_patches / 2.
        ry = (1 - (-1)) / num_patches / 2.
        
        feat_coord = make_coord([num_patches, num_patches], flatten=False).to(feat.device) \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, num_patches, num_patches)
        
        areas, out_feats = [], []
        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()
                coord_[:, :, 0] += vx * rx + eps_shift # find the nearest grids
                coord_[:, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)

                q_coord = F.grid_sample(
                    feat_coord, 
                    coord_.unsqueeze(1), # note: this is in (x, y)
                    mode='nearest', 
                    align_corners=False)[:, :, 0, :].permute(0, 2, 1)

                q_feat = F.grid_sample(
                    feat.permute(0, 3, 1, 2), # => [B,C,H,W]
                    coord_.unsqueeze(1), 
                    mode='nearest', 
                    align_corners=False)[:, :, 0, :].permute(0, 2, 1)
                
                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= num_patches
                rel_coord[:, :, 1] *= num_patches
                inp = torch.cat([q_feat, rel_coord], dim=-1)
                
                bs, q = q_feat.shape[:2]
                out_feat = self.mlp(inp.view(bs * q, -1)).view(bs, q, -1)
                out_feats.append(out_feat)
                
                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(1 / (area + 1e-9))

        tot_area = torch.stack(areas).sum(dim=0) # = 4.0
        # t = areas[0]; areas[0] = areas[3]; areas[3] = t
        # t = areas[1]; areas[1] = areas[2]; areas[2] = t
        out = 0
        for out_feat, area in zip(out_feats, areas):
            out = out + out_feat * (area / tot_area).unsqueeze(-1)            
        return out
        
    def forward(self, feat, coord, geo_coords=None): 
        out = self.query(feat, coord)
        out = self.output_layer(out)
        out = self.norm(out)
        return out
        
    
#####################################
##### SpatialVaryingVisualQuery #####
#####################################        
class SpatialVaryingVisualQuery(nn.Module):
    def __init__(
        self, 
        in_dim, 
        hidden_dims, 
        out_dim
    ):
        super(SpatialVaryingVisualQuery, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dims = hidden_dims
        self.mlp = MLP(in_dim=in_dim, 
                       out_dim=out_dim,
                       hidden_dims=hidden_dims)
        self.output_layer = nn.Linear(out_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def query(self, feat, coord, geo_coords):  
        """
            feat: B x H x W x C
            coord: B x N x 2
            geo_coords: B x N x 2
        """
        vx_lst = [-1, 1]
        vy_lst = [-1, 1]
        eps_shift = 1e-6
               
        num_patches = int(feat.shape[-2])
        rx = (1 - (-1)) / num_patches / 2.
        ry = (1 - (-1)) / num_patches / 2.
        
        feat_coord = make_coord([num_patches, num_patches], flatten=False).to(feat.device) \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, num_patches, num_patches)
        
        areas, out_feats = [], []
        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()
                coord_[:, :, 0] += vx * rx + eps_shift # find the nearest grids
                coord_[:, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)

                q_coord = F.grid_sample(
                    feat_coord, 
                    coord_.unsqueeze(1), # note: this is in (x, y)
                    mode='nearest', 
                    align_corners=False)[:, :, 0, :].permute(0, 2, 1)

                q_feat = F.grid_sample(
                    feat.permute(0, 3, 1, 2), # => [B,C,H,W]
                    coord_.unsqueeze(1), 
                    mode='nearest', 
                    align_corners=False)[:, :, 0, :].permute(0, 2, 1)
                
                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= num_patches
                rel_coord[:, :, 1] *= num_patches

                ###
                ### add global relative distance for x, y
                geo_coords0 = geo_coords - (-1.)
                geo_coords1 = torch.stack([geo_coords[..., 0] - (-1.), geo_coords[..., 1] - (1.)], dim=-1)
                geo_coords2 = torch.stack([geo_coords[..., 0] - (1.), geo_coords[..., 1] - (-1.)], dim=-1)
                geo_coords3 = geo_coords - (1.)
                geo_rel_coord = torch.concat([geo_coords0, geo_coords1, geo_coords2, geo_coords3], dim=-1) # B x N x 8   
                inp = torch.cat([q_feat, rel_coord, geo_rel_coord], dim=-1) # B x N x 130 (128+2)
                ###
                ###
                
                bs, q = q_feat.shape[:2]
                out_feat = self.mlp(inp.view(bs * q, -1)).view(bs, q, -1) # B x N x 1024 
                out_feats.append(out_feat)                
                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(1 / (area + 1e-9))

        tot_area = torch.stack(areas).sum(dim=0) # = 4.0
        # t = areas[0]; areas[0] = areas[3]; areas[3] = t
        # t = areas[1]; areas[1] = areas[2]; areas[2] = t
        out = 0
        for out_feat, area in zip(out_feats, areas):
            out = out + out_feat * (area / tot_area).unsqueeze(-1)            
        return out
        
    def forward(self, feat, coord, geo_coords): 
        out = self.query(feat, coord, geo_coords)
        out = self.output_layer(out)
        out = self.norm(out)
        return out


##############################
##### GeoAttnVisualQuery #####
##############################   
class GeoAttnVisualQuery(nn.Module):
    def __init__(
        self, 
        in_dim, 
        hidden_dims, 
        out_dim
    ):
        super(GeoAttnVisualQuery, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dims = hidden_dims
        self.mlp = MLP(in_dim=in_dim, 
                       out_dim=out_dim,
                       hidden_dims=hidden_dims)
        self.attn = nn.MultiheadAttention(out_dim, num_heads=16, batch_first=True)
        self.output_layer = nn.Linear(out_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def query(self, feat, coord, geo_query):  
        """
            feat: B x H x W x C
            coord: B x N x 2
            geo_query: B x N x emb_dim
        """
        vx_lst = [-1, 1]
        vy_lst = [-1, 1]
        eps_shift = 1e-6
               
        num_patches = int(feat.shape[-2])
        rx = (1 - (-1)) / num_patches / 2.
        ry = (1 - (-1)) / num_patches / 2.
        
        feat_coord = make_coord([num_patches, num_patches], flatten=False).to(feat.device) \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, num_patches, num_patches)
        
        areas, out_feats = [], []
        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()
                coord_[:, :, 0] += vx * rx + eps_shift # find the nearest grids
                coord_[:, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)

                q_coord = F.grid_sample(
                    feat_coord, 
                    coord_.unsqueeze(1), # note: this is in (x, y)
                    mode='nearest', 
                    align_corners=False)[:, :, 0, :].permute(0, 2, 1)

                q_feat = F.grid_sample(
                    feat.permute(0, 3, 1, 2), # => [B,C,H,W]
                    coord_.unsqueeze(1), 
                    mode='nearest', 
                    align_corners=False)[:, :, 0, :].permute(0, 2, 1)
                
                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= num_patches
                rel_coord[:, :, 1] *= num_patches
                inp = torch.cat([q_feat, rel_coord], dim=-1) # B x N x 130 (128+2)
                
                bs, q = q_feat.shape[:2]
                out_feat = self.mlp(inp.view(bs * q, -1)).view(bs, q, -1) # B x N x 1024 
                out_feats.append(out_feat)
                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(1 / (area + 1e-9))

        out_feats = torch.stack(out_feats, dim=-2) # B x N x 4 x 1024 
        B, N, L, C = out_feats.shape
        out_feats = out_feats.reshape(B * N, L, C)
        geo_query = geo_query.reshape(B * N, 1, C)        
        out, _ = self.attn(geo_query, out_feats, out_feats)
        out += geo_query
        out = out.reshape(B, N, C)
        return out
        
    def forward(self, feat, coord, geo_coords): 
        out = self.query(feat, coord, geo_coords)
        out = self.output_layer(out)
        out = self.norm(out)
        return out
        
        
        
###################
##### Helpers #####
###################
def get_emb_dims(emb_dim, pooling_type):
    # hack here [in, hid, hid, out]
    if pooling_type == 'sum' or pooling_type is None:
        dims = [[emb_dim // 8, emb_dim // 4, emb_dim // 2, emb_dim],
                [emb_dim // 4, emb_dim // 4, emb_dim // 2, emb_dim],
                [emb_dim // 2, emb_dim // 2, emb_dim,      emb_dim],
                [emb_dim,      emb_dim,      emb_dim,      emb_dim]] 
        # dims = [[emb_dim // 8, emb_dim // 8, emb_dim // 4, emb_dim // 4],
        #         [emb_dim // 4, emb_dim // 4, emb_dim // 4, emb_dim // 4],
        #         [emb_dim // 2, emb_dim // 2, emb_dim // 4, emb_dim // 4],
        #         [emb_dim,      emb_dim // 2, emb_dim // 4, emb_dim // 4]] 
    elif pooling_type == 'concat' :
        dims = [[emb_dim // 8, emb_dim // 8, emb_dim // 4, emb_dim // 4],
                [emb_dim // 4, emb_dim // 4, emb_dim // 4, emb_dim // 4],
                [emb_dim // 2, emb_dim // 2, emb_dim // 4, emb_dim // 4],
                [emb_dim,      emb_dim // 2, emb_dim // 4, emb_dim // 4]] 
    else:
        raise NotImplementedError
    return dims
    
    
def get_query_layers(query_method, emb_dim, pooling_type):

    dims = get_emb_dims(emb_dim, pooling_type)

    if query_method is None:
        query_layers = nn.ModuleList([
            VisualQuery(in_dim=dims[i][0] + 2, 
                        out_dim=dims[i][3],
                        hidden_dims=[dims[i][1], dims[i][2]]) 
            for i in range(len(dims))
        ])
    elif query_method == 'sv':
        query_layers = nn.ModuleList([
            SpatialVaryingVisualQuery(in_dim=dims[i][0] + 2 + 8, 
                                      out_dim=dims[i][3],
                                      hidden_dims=[dims[i][1], dims[i][2]]) 
            for i in range(len(dims))
        ])
        
    elif query_method == 'gattn':
        query_layers = nn.ModuleList([
            GeoAttnVisualQuery(in_dim=dims[i][0] + 2, 
                               out_dim=dims[i][3],
                               hidden_dims=[dims[i][1], dims[i][2]]) 
            for i in range(len(dims))
        ])
    return query_layers



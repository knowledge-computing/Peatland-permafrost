import os
import torch
import numpy as np
from collections import OrderedDict
from scipy import interpolate
from scipy.interpolate import RectBivariateSpline

from models.position_embed import interpolate_pos_embed


def build_model(config, load_pretrain=True):
    ###############################
    ##### Load Visual Encoder #####
    ###############################
    visual_encoders = {}
    if config.DATA.SAT.USE_SAT:
        sat_encoder = build_visual_encoder(in_dim=config.DATA.SAT.NUM_BANDS, 
                                           query_method=config.MODEL.SAT_ENCODER.QUERY_METHOD, 
                                           model_type=config.MODEL.SAT_ENCODER.TYPE, 
                                           pretrain_model=config.MODEL.SAT_ENCODER.USE_PRETRAIN, 
                                           pretrain_weight=config.MODEL.SAT_ENCODER.PRETRAIN_WEIGHT, 
                                           load_pretrain=load_pretrain,
                                           config=config)
        visual_encoders['sat'] = sat_encoder

    if config.DATA.DEM.USE_DEM:
        dem_encoder = build_visual_encoder(in_dim=config.DATA.DEM.NUM_BANDS, 
                                           query_method=config.MODEL.DEM_ENCODER.QUERY_METHOD, 
                                           model_type=config.MODEL.DEM_ENCODER.TYPE, 
                                           pretrain_model=config.MODEL.DEM_ENCODER.USE_PRETRAIN, 
                                           pretrain_weight=config.MODEL.DEM_ENCODER.PRETRAIN_WEIGHT, 
                                           load_pretrain=load_pretrain,
                                           config=config)
        visual_encoders['dem'] = dem_encoder

    if config.DATA.CLM.USE_CLM:
        clm_encoder = build_visual_encoder(in_dim=config.DATA.CLM.NUM_BANDS, 
                                           query_method=config.MODEL.CLM_ENCODER.QUERY_METHOD, 
                                           model_type=config.MODEL.CLM_ENCODER.TYPE, 
                                           pretrain_model=config.MODEL.CLM_ENCODER.USE_PRETRAIN, 
                                           pretrain_weight=config.MODEL.CLM_ENCODER.PRETRAIN_WEIGHT, 
                                           load_pretrain=load_pretrain,
                                           config=config)
        visual_encoders['clm'] = clm_encoder

    if config.DATA.COV.USE_COV:
        num_bands = config.DATA.COV.NUM_BANDS
        if config.DATA.GEOCOORD.USE_GEOCOORD:
            num_bands += config.DATA.GEOCOORD.NUM_BANDS
        cov_encoder = build_visual_encoder(in_dim=num_bands, 
                                           query_method=config.MODEL.COV_ENCODER.QUERY_METHOD, 
                                           model_type=config.MODEL.COV_ENCODER.TYPE, 
                                           pretrain_model=config.MODEL.COV_ENCODER.USE_PRETRAIN, 
                                           pretrain_weight=config.MODEL.COV_ENCODER.PRETRAIN_WEIGHT, 
                                           load_pretrain=load_pretrain,
                                           config=config)
        visual_encoders['cov'] = cov_encoder
    
    ############################
    ##### Load Geo Encoder #####
    ############################
    if config.MODEL.GEO_ENCODER.TYPE == 'geo':
        emb_dim = sat_encoder.emb_dim if sat_encoder is not None else 1024
        from models.encoders import GeoEncoder
        geo_encoder = GeoEncoder(emb_dim=emb_dim, 
                                 num_frequencies=config.MODEL.GEO_ENCODER.POS_NUM_FREQ,
                                 enc_method=config.MODEL.GEO_ENCODER.POS_ENCODER)
    else:
        geo_encoder = None
    
    ########################
    ##### Load Decoder #####
    ########################
    if config.MODEL.DECODER.split('_')[0] == 'visual':
        from models.decoders import Decoder
        model = Decoder(visual_encoders, 
                        geo_encoder=geo_encoder, 
                        decoder_name=config.MODEL.DECODER,
                        out_dim=config.DATA.OUT_DIM)
    elif config.MODEL.DECODER == 'geo_only':
        from models.decoders import GeoDecoder
        model = GeoDecoder(geo_encoder=geo_encoder, 
                           decoder_name=config.MODEL.DECODER,
                           out_dim=config.DATA.OUT_DIM)
    
    if os.path.exists(config.MODEL.PRETRAIN_WEIGHT):
        checkpoint = torch.load(config.MODEL.PRETRAIN_WEIGHT, map_location=torch.device('cpu'))
        keys_to_remove = ['out.4.weight', 'out.4.bias', 'geo_out.4.weight', 'geo_out.4.bias']
        for key in keys_to_remove:
            if key in checkpoint:
                print(f"Removing: {key}")
                del checkpoint[key]
            else:
                print(f"Key not found: {key}")
        msg = model.load_state_dict(checkpoint, strict=False)
        print(f"***** Load model weights {config.MODEL.PRETRAIN_WEIGHT}: ", msg)    
        
    return model


def build_visual_encoder(in_dim, 
                         query_method, 
                         model_type, 
                         pretrain_model, 
                         pretrain_weight,
                         load_pretrain,
                         config):
    
    if model_type == 'vit':
        from models.encoders import ViTEncoder
        visual_encoder = ViTEncoder(in_dim=in_dim,
                                    query_method=query_method,
                                    pretrain_model=pretrain_model,
                                    global_pool=config.MODEL.VIT.GLOBAL_POOLING)   
        if pretrain_model == 'IBM' and load_pretrain:
            visual_encoder = load_IBM_weight(visual_encoder, pretrain_weight)
        if pretrain_model == 'SatMAE' and load_pretrain:
            visual_encoder = load_SatMAE_weight(visual_encoder, pretrain_weight)
            
    elif model_type == 'swin':
        from models.encoders import SwinEncoder
        visual_encoder = SwinEncoder(in_dim=in_dim,
                                     query_method=query_method,
                                     config=config)   
        if pretrain_model == 'Satlas' and load_pretrain:
            visual_encoder = load_Satlas_weight(visual_encoder, pretrain_weight)
            
    elif model_type == 'simswin':
        from models.encoders import SimSwinEncoder
        visual_encoder = SimSwinEncoder(in_dim=in_dim,
                                        query_method=query_method,
                                        config=config)   
        if pretrain_model == 'SIMMIM' and load_pretrain:
            visual_encoder = load_SIMMIM_weight(visual_encoder, pretrain_weight)
    else:
        visual_encoder = None
    return visual_encoder


def load_test_model(config):
    weight_file = os.path.join(config.OUTPUT, 'best_model.pth')
    model = build_model(config, load_pretrain=False)
    checkpoint = torch.load(weight_file, map_location=torch.device('cpu'))
    msg = model.load_state_dict(checkpoint, strict=True)
    print(f"***** Load test model weights {weight_file}: ", msg)    
    return model


def load_pretrain_weight(model, weight_file):
    checkpoint = torch.load(weight_file, map_location=torch.device('cpu'))
    msg = model.load_state_dict(checkpoint, strict=False)
    print(msg)
    return model


def load_Satlas_weight(model, weight_file):
    full_state_dict = torch.load(weight_file, map_location=torch.device('cpu'))
    swin_prefix = 'backbone.backbone.'
    swin_state_dict = {k[len(swin_prefix):]: v for k, v in full_state_dict.items() if k.startswith(swin_prefix)}
    swin_state_dict.pop('features.0.0.weight')
    swin_state_dict.pop('features.0.0.bias')    
    msg = model.feature_extractor.load_state_dict(swin_state_dict, strict=False)
    print(f"***** Load pretrained Satlas weights {weight_file}: ", msg)
    # this is previous method !!!!
    # weights_manager = satlaspretrain_models.Weights()
    # MODEL_CHECKPOINT_ID = "Aerial_SwinB_MI"
    # USE_FPN = True
    # base_model = weights_manager.get_pretrained_model(model_identifier=MODEL_CHECKPOINT_ID, fpn=USE_FPN)
    # model = SatLas(base_model, out_features=1, expansion_factor=2)
    return model


def load_IBM_weight(model, weight_file):
    state_dict = torch.load(weight_file, map_location=torch.device('cpu'))
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        valid_key = True
        for remove_key in ['decoder', 'mask_token', 'pos_embed', 'patch_embed']:
            if k.startswith(remove_key): valid_key = False;
        if valid_key: new_state_dict[k] = v;
    msg = model.feature_extractor.load_state_dict(new_state_dict, strict=False)
    print(f"***** Load pretrained IBM weights {weight_file}: ", msg)
    return model


def load_SatMAE_weight(model, weight_file):
    # this code is from SatMAE
    print('... Running SatMAE script to set the model ... ')
    checkpoint = torch.load(weight_file, map_location=torch.device('cpu'))
    checkpoint_model = checkpoint['model']
    state_dict = model.state_dict()
    for k in ['pos_embed', 
              'patch_embed.proj.weight', 'patch_embed.proj.bias', 
              'head.weight', 'head.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]      
    interpolate_pos_embed(model, checkpoint_model)
    msg = model.load_state_dict(checkpoint_model, strict=False)
    print(f"***** Load pretrained SatMAE weights {weight_file}: ", msg)
    return model


def load_SIMMIM_weight(model, weight_file):
    full_state_dict = torch.load(weight_file, map_location=torch.device('cpu'))
    swin_state_dict = full_state_dict['model']
    swin_prefix = 'encoder.'
    swin_state_dict = {k[len(swin_prefix):]: v for k, v in swin_state_dict.items() if k.startswith(swin_prefix)}
    new_state_dict = remap_pretrained_keys_swin(model.feature_extractor, swin_state_dict)
    msg = model.feature_extractor.load_state_dict(new_state_dict, strict=False)
    print(f"***** Load pretrained SIMMIM weights {weight_file}: ", msg)
    return model

def remap_pretrained_keys_swin(model, checkpoint_model):
    state_dict = model.state_dict()
    
    # Geometric interpolation when pre-trained patch size mismatch with fine-tuned patch size
    all_keys = list(checkpoint_model.keys())
    for key in all_keys:
        if "relative_position_bias_table" in key:
            relative_position_bias_table_pretrained = checkpoint_model[key]
            relative_position_bias_table_current = state_dict[key]
            L1, nH1 = relative_position_bias_table_pretrained.size()
            L2, nH2 = relative_position_bias_table_current.size()
            if nH1 != nH2:
                print(f"Error in loading {key}, passing......")
            else:
                if L1 != L2:
                    print(f"{key}: Interpolate relative_position_bias_table using geo.")
                    src_size = int(L1 ** 0.5)
                    dst_size = int(L2 ** 0.5)

                    def geometric_progression(a, r, n):
                        return a * (1.0 - r ** n) / (1.0 - r)

                    left, right = 1.01, 1.5
                    while right - left > 1e-6:
                        q = (left + right) / 2.0
                        gp = geometric_progression(1, q, src_size // 2)
                        if gp > dst_size // 2:
                            right = q
                        else:
                            left = q

                    # if q > 1.090307:
                    #     q = 1.090307

                    dis = []
                    cur = 1
                    for i in range(src_size // 2):
                        dis.append(cur)
                        cur += q ** (i + 1)

                    r_ids = [-_ for _ in reversed(dis)]

                    x = r_ids + [0] + dis
                    y = r_ids + [0] + dis

                    t = dst_size // 2.0
                    dx = np.arange(-t, t + 0.1, 1.0)
                    dy = np.arange(-t, t + 0.1, 1.0)

                    print("Original positions = %s" % str(x))
                    print("Target positions = %s" % str(dx))

                    all_rel_pos_bias = []

                    for i in range(nH1):
                        z = relative_position_bias_table_pretrained[:, i].view(src_size, src_size).float().numpy()
                        # f_cubic = interpolate.interp2d(x, y, z, kind='cubic')
                        f_cubic = RectBivariateSpline(x, y, z, kx=3, ky=3)
                        all_rel_pos_bias.append(torch.Tensor(f_cubic(dx, dy)).contiguous().view(-1, 1).to(
                            relative_position_bias_table_pretrained.device))

                    new_rel_pos_bias = torch.cat(all_rel_pos_bias, dim=-1)
                    checkpoint_model[key] = new_rel_pos_bias

    # delete relative_position_index since we always re-init it
    relative_position_index_keys = [k for k in checkpoint_model.keys() if "relative_position_index" in k]
    for k in relative_position_index_keys:
        del checkpoint_model[k]

    # delete relative_coords_table since we always re-init it
    relative_coords_table_keys = [k for k in checkpoint_model.keys() if "relative_coords_table" in k]
    for k in relative_coords_table_keys:
        del checkpoint_model[k]

    # delete attn_mask since we always re-init it
    attn_mask_keys = [k for k in checkpoint_model.keys() if "attn_mask" in k]
    for k in attn_mask_keys:
        del checkpoint_model[k]

    return checkpoint_model


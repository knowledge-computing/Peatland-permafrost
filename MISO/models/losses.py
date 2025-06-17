import torch
import torch.nn as nn
import torch.nn.functional as F


class LossCollection(object):
    def __init__(self, config):
        
        self.cos_loss_fn = CosineSimilarityLoss()
        self.nce_loss_fn = NCELoss()
        self.info_nce_loss_fn = InfoNCELoss()
        self.bce_loss_fn = nn.BCEWithLogitsLoss()
        self.ce_loss_fn = nn.CrossEntropyLoss()
        self.mse_loss_fn = nn.MSELoss()
        self.focal_loss_fn = BinaryFocalLoss()
        self.base_loss_fn = config.TRAIN.BASE_LOSS
        self.aux_losses = config.TRAIN.AUX_LOSSES
        self.out_dim = config.DATA.OUT_DIM
        self.var_name = config.VAR_NAME
        
    def compute_base_loss(self, pred, y, loss_fn):
        pred = pred.reshape(-1, self.out_dim)        
        mask = y[:, 0] > -999
        if torch.sum(mask) == 0: return torch.tensor(0.0).cuda();
        if loss_fn == 'mse':
            return self.mse_loss_fn(pred[mask, :], y[mask, :])
        elif loss_fn == 'bce':
            return self.bce_loss_fn(pred[mask, :], y[mask, :])
        elif loss_fn == 'ce':
            y = y.reshape(-1).to(torch.long)
            return self.ce_loss_fn(pred[mask, :], y[mask])
        elif loss_fn == 'focal':
            return self.focal_loss_fn(pred[mask, :], y[mask, :]) * 10
        else:
            raise NotImplementedError

    def compute_geo_nce_loss(self, geo_coords, geo_emb, model):
        noise = ((torch.rand_like(geo_coords) * 20) - 10.).cuda()
        disturbed_geo_coords = geo_coords + noise       
        disturbed_geo_emb, _ = model.module.get_geo_embeddings(disturbed_geo_coords)
        loss = self.info_nce_loss_fn(geo_emb, disturbed_geo_emb)
        return loss

    def compute_spatial_similarity_loss(self, geo_coords, geo_emb, model):
        noise = ((torch.rand_like(geo_coords) * 2000.) - 1000.).cuda()
        disturbed_geo_coords = geo_coords + noise
        disturbed_geo_emb, _ = model.module.get_geo_embeddings(disturbed_geo_coords)
        emb = F.normalize(geo_emb, dim=2) 
        disturbed_emb = F.normalize(disturbed_geo_emb, dim=2) 
        emb_sim = torch.sum(emb * disturbed_emb, dim=2)
        geo_dist = torch.norm(geo_coords - disturbed_geo_coords, dim=2)
        geo_sim = torch.exp(-(geo_dist ** 2) / 2 * 500 ** 2)
        max_dist = 500
        mask = (geo_dist < max_dist).float()
        loss = F.mse_loss(emb_sim * mask, geo_sim * mask, reduction='sum') / (mask.sum() + 1e-6)
        
        # loss = F.mse_loss(emb_sim, geo_sim)
        return loss
        
    def compute_aux_loss_base(self, aux_pred, y):
        pred = aux_pred.reshape(-1, self.out_dim)
        if self.var_name == 'aksdb_pf1m_bin': 
            loss = self.compute_base_loss(pred, y, loss_fn='bce')
        elif self.var_name == 'tax_order':
            loss = self.compute_base_loss(pred, y, loss_fn='ce')
        return loss
        
    def compute_aux_losses(self, output, y):
        aux_loss = 0.
        all_losses = {}

        for name in ['geo', 'sat', 'cov', 'visual']: 
            if name in self.aux_losses:
                pred = output[name + '_pred'].reshape(-1, self.out_dim)
                if self.var_name == 'aksdb_pf1m_bin': 
                    loss = self.compute_base_loss(pred, y, loss_fn='bce')
                elif self.var_name == 'tax_order':
                    loss = self.compute_base_loss(pred, y, loss_fn='ce')                
                aux_loss += loss
                all_losses[name + '_loss'] = loss.item()
            
        if 'dual' in self.aux_losses:
            logits1 = output['visual_pred']
            logits2 = output['geo_pred']
            if self.var_name == 'aksdb_pf1m_bin': 
                dual_loss = torch.norm(logits1 - logits2, p=2)
            elif self.var_name == 'tax_order':
                prob1 = F.softmax(logits1, dim=-1)
                prob2 = F.softmax(logits2, dim=-1)
                dual_loss = F.kl_div(prob2.log(), prob1, reduction='batchmean')
                dual_loss += F.kl_div(prob1.log(), prob2, reduction='batchmean')
            aux_loss += dual_loss * 0.1
            all_losses['dual_loss'] = dual_loss.item()
            
        if 'cos' in self.aux_losses:
            visual_emb = output['visual_emb'].flatten(start_dim=0, end_dim=1)
            geo_emb = output['geo_emb'].flatten(start_dim=0, end_dim=1)
            cos_loss = self.cos_loss_fn(visual_emb, geo_emb)
            aux_loss += cos_loss * 0.1
            all_losses['cos_loss'] = cos_loss.item()

        for name in ['sat_cov_nce', 'sat_geo_nce', 'cov_geo_nce', 'visual_geo_nce']: 
            if name in self.aux_losses:   
                a, b, _ = name.split('_')            
                a_emb, b_emb = output[a + '_emb'], output[b + '_emb']
                nce_loss = self.nce_loss_fn(a_emb, b_emb)
                aux_loss += nce_loss * 0.1
                all_losses[name + '_loss'] = nce_loss.item()

        if 'level_sat_cov_nce' in self.aux_losses:
            nce_loss = 0
            for i in range(4):
                a_emb, b_emb = output[f'sat_emb_{i}'], output[f'cov_emb_{i}']
                nce_loss += self.nce_loss_fn(a_emb, b_emb)
            nce_loss /= 4.
            aux_loss += nce_loss * 0.1
            all_losses['level_sat_cov_nce_loss'] = nce_loss.item()

        if 'focal' in self.aux_losses:
            focal_loss = self.compute_base_loss(pred, y, loss_fn='focal')
            aux_loss += focal_loss
            all_losses['focal_loss'] = focal_loss.item()

        return aux_loss, all_losses


class NCELoss(torch.nn.Module):
    def __init__(self):
        super(NCELoss, self).__init__()
        self.ce_loss_fn = nn.CrossEntropyLoss()
        self.temperature = 0.1
    
    def forward(self, A, B):
        """
        Computes the cosine similarity loss.

        Args:
            A (torch.Tensor): First set of embeddings (B, N, C).
            B (torch.Tensor): Second set of embeddings (B, N, C).

        Returns:
            torch.Tensor: Computed loss.
        """
        A_norm = F.normalize(A, p=2, dim=-1)
        B_norm = F.normalize(B, p=2, dim=-1)
        emb1 = A_norm
        emb2 = B_norm   
        B, N, C = A_norm.shape
        labels = torch.arange(B, dtype=torch.long).repeat(N).to(A.device) # Shape: (N x B,)
        emb1_tr = emb1.transpose(0, 1) # Shape: (N, B, C)
        emb2_tr = emb2.transpose(0, 1) # Shape: (N, B, C)
        sim12 = torch.bmm(emb1_tr, emb2_tr.transpose(-1, -2)).reshape(B * N, B) / self.temperature # Shape: (N * B, B)
        sim21 = torch.bmm(emb2_tr, emb1_tr.transpose(-1, -2)).reshape(B * N, B) / self.temperature # Shape: (N * B, B)
        nce_loss = self.ce_loss_fn(sim12, labels)
        nce_loss += self.ce_loss_fn(sim21, labels)
        return nce_loss


class InfoNCELoss(nn.Module):
    def __init__(self):
        super(InfoNCELoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.temperature = 0.1
        
    def forward(self, anchor, positive):
        B, N, C = anchor.shape
        anchor = F.normalize(anchor, dim=-1)
        positive = F.normalize(positive, dim=-1)
        anchor = anchor.permute(1, 0, 2)
        positive = positive.permute(1, 0, 2)
        logits = torch.bmm(anchor, positive.transpose(1, 2)) # (N, B, B)
        labels = torch.arange(B, device=anchor.device).expand(N, -1)
        logits = logits / self.temperature
        loss = self.criterion(logits.reshape(-1, B), labels.reshape(-1))
        return loss


class CosineSimilarityLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cos_loss_fn = nn.CosineEmbeddingLoss()
    
    def forward(self, A, B):
        """
        Computes the cosine similarity loss.

        Args:
            A (torch.Tensor): First set of embeddings (B, C).
            B (torch.Tensor): Second set of embeddings (B, C).

        Returns:
            torch.Tensor: Computed loss.
        """
        # Normalize embeddings
        A_norm = F.normalize(A, p=2, dim=-1)
        B_norm = F.normalize(B, p=2, dim=-1)
        target = torch.ones(A.shape[0]).to(A.device)
        cos_loss = self.cos_loss_fn(A_norm, B_norm, target)
        return cos_loss
            
    

def compute_smoothness_loss(model, img, img_coords, valid_indices=None, epsilon=1e-3):

    # Perturb img_coords along x-axis
    img_coords_x_pos = img_coords.clone()
    img_coords_x_pos[..., 0] += epsilon

    img_coords_x_neg = img_coords.clone()
    img_coords_x_neg[..., 0] -= epsilon

    # Perturb img_coords along y-axis
    img_coords_y_pos = img_coords.clone()
    img_coords_y_pos[..., 1] += epsilon

    img_coords_y_neg = img_coords.clone()
    img_coords_y_neg[..., 1] -= epsilon
    
    stacked_coords = torch.stack([img_coords_x_pos,
                                  img_coords_x_neg,
                                  img_coords_y_pos,
                                  img_coords_y_neg], dim=-2)
    B, N, M, C = stacked_coords.shape  # 16x5x4x2
    stacked_coords_reshaped = stacked_coords.reshape(B, N * M, C)

    # Get the perturbed embeddings
    with torch.no_grad():  # No need to compute gradients during forward passes
        img_emb = model.module.get_img_emb(img, stacked_coords_reshaped)
        img_emb = img_emb.reshape(B, N, M, -1)
        img_emb_x_pos = img_emb[:, :, 0, :]
        img_emb_x_neg = img_emb[:, :, 1, :]
        img_emb_y_pos = img_emb[:, :, 2, :]
        img_emb_y_neg = img_emb[:, :, 3, :]

    # Compute finite differences
    grad_x = (img_emb_x_pos - img_emb_x_neg) / (2 * epsilon)  # Gradient wrt x
    grad_y = (img_emb_y_pos - img_emb_y_neg) / (2 * epsilon)  # Gradient wrt y

    # Compute gradient magnitudes (smoothness loss)
    grad_norm = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-9)  # Add epsilon to avoid sqrt(0)
    smoothness_loss = grad_norm[valid_indices[0], valid_indices[1], :].mean()
    return torch.abs(smoothness_loss - 1.)



def compute_geo_smoothness_loss(model, coords, epsilon=1e-3):
    
    # Perturb coords along x-axis
    coords_x_pos = coords.clone()
    coords_x_pos[..., 0] += epsilon

    coords_x_neg = coords.clone()
    coords_x_neg[..., 0] -= epsilon

    # Perturb coords along y-axis
    coords_y_pos = coords.clone()
    coords_y_pos[..., 1] += epsilon

    coords_y_neg = coords.clone()
    coords_y_neg[..., 1] -= epsilon
    
    stacked_coords = torch.stack([coords_x_pos, coords_x_neg,
                                  coords_y_pos, coords_y_neg], dim=-2)
    
    B, N, M, C = stacked_coords.shape  # 16x5x4x2
    stacked_coords_reshaped = stacked_coords.reshape(B, N * M, C)
    
    # Get the perturbed embeddings
    with torch.no_grad():  # No need to compute gradients during forward passes
        output = model.module(stacked_coords_reshaped)
        out_emb = output['out_emb'].reshape(B, N, M, -1)
        emb_x_pos = out_emb[:, :, 0, :]
        emb_x_neg = out_emb[:, :, 1, :]
        emb_y_pos = out_emb[:, :, 2, :]
        emb_y_neg = out_emb[:, :, 3, :]

    # Compute finite differences
    grad_x = (emb_x_pos - emb_x_neg) / (2 * epsilon)  # Gradient wrt x
    grad_y = (emb_y_pos - emb_y_neg) / (2 * epsilon)  # Gradient wrt y

    # Compute gradient magnitudes (smoothness loss)
    grad_norm = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-9)  # Add epsilon to avoid sqrt(0)
    smoothness_loss = grad_norm.mean()
    return torch.abs(smoothness_loss - 1.)    


def spatial_similarity_loss(geo_coords, embeddings, temp=1.0):
    """
    geo_coords: (N, 2) - coordinates in meters (EPSG:3338)
    embeddings: (N, D) - output embeddings
    """
    # Compute pairwise geo distance (in meters)
    geo_dist = torch.cdist(geo_coords, geo_coords, p=2)  # (N, N)

    # Convert geo distance into similarity (smaller dist = higher sim)
    geo_sim = torch.exp(-geo_dist / temp)  # (N, N)

    # Compute embedding similarity (cosine)
    emb = F.normalize(embeddings, dim=1)
    emb_sim = torch.matmul(emb, emb.T)  # (N, N)

    # MSE between geo-based similarity and embedding similarity
    loss = F.mse_loss(emb_sim, geo_sim)
    return loss


class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        inputs = inputs.reshape(-1) # inputs: (batch_size,) logits
        targets = targets.reshape(-1) # targets: (batch_size,) binary (0 or 1)
        
        probs = torch.sigmoid(inputs)
        probs = probs.clamp(min=1e-6, max=1-1e-6)  # avoid log(0)
        
        pt = probs * targets + (1 - probs) * (1 - targets)  # p_t
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)  # alpha_t

        loss = -alpha_t * (1 - pt) ** self.gamma * torch.log(pt)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


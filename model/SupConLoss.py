import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    def __init__(self, args):
        super(SupConLoss, self).__init__()
        self.args = args
        self.temperature = args.temperature
        
        
    def forward(self, features, labels=None, hasweight=None, weight=None, mask=None, label_rela=None):
        
        features = F.normalize(features, p=2, dim=1)
        
        batch_size = features.shape[0]
        if mask is None:
            # [50, 1]
            labels = labels.contiguous().view(-1, 1)
            # mask [bsz, bsz]的矩阵，其中x_ij = 1 if label_i=label_j, and x_ii = 1  [50,50]
            mask = torch.eq(labels, labels.T).float().to(self.args.device)
        # 给正例负例分配权重
        if hasweight:
            mask = torch.mul(weight.to(self.args.device), mask)
        
        # compute logits, anchor_dot_contrast: (bsz, bsz), x_i_j: (z_i*z_j)/t
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T), self.temperature)
        # for numerical stability, 每一行的数据减去每一行中最大的那个
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(self.args.device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        log_prob = torch.nan_to_num(log_prob, nan=0.0)
        if torch.any(torch.isnan(log_prob)):
            raise ValueError("Log_prob has nan!")
        

        
        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1)+1)
        if label_rela is not None:
            mean_log_prob_pos = mean_log_prob_pos / label_rela
            
        mean_log_prob_pos = torch.nan_to_num(mean_log_prob_pos, nan=0.0)
        if torch.any(torch.isnan(mean_log_prob_pos)):
            raise ValueError("mean_log_prob_pos has nan!")

        # loss
        loss = - mean_log_prob_pos
        # print("SCL Loss:")
        # print(loss.shape)
        if torch.any(torch.isnan(loss)):
                raise ValueError("loss has nan!")
        loss = loss.mean()

        return loss
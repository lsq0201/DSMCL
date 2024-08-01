import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, List, Optional, Sequence, Tuple, Union
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer

from model.attention_for_label_doc import AttentionForLabelDoc
from model.SupConLoss import supconloss


class uniform_loss(nn.Module):
    def __init__(self, t=0.07):
        super(uniform_loss, self).__init__()
        self.t = t

    def forward(self, x):
        return x.matmul(x.T).div(self.t).exp().sum(dim=-1).log().mean()
    

class Model(nn.Module):
    def __init__(self, args, freeze=True):
        super(Model, self).__init__()
        self.args = args
        self.dropout = nn.Dropout(p=args.dropout)
        
        self.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, model_max_length=args.max_length, trust_remote_code=True)
        self.bert = AutoModel.from_pretrained(args.bert_name, trust_remote_code=True)
        if freeze:
            for name, param in self.bert.named_parameters():
                param.requires_grad = False
        else:
            unfreeze = ["pooler", "encoder.layer.11"]
            # unfreeze = ["pooler", "encoder.layer.11", "encoder.layer.10"]
            # unfreeze = ["pooler", "encoder.layer.11", "encoder.layer.10", "encoder.layer.9"]
            # unfreeze = ["pooler", "encoder.layer.11", "encoder.layer.10", "encoder.layer.9", "encoder.layer.8"]
            for name, param in self.bert.named_parameters():
                param.requires_grad = False
                # print(name)
                for unname in unfreeze:
                    if unname in name:
                        param.requires_grad = True
        
        self.criterion = uniform_loss()
        
        self.attention = AttentionForLabelDoc(self.args)

        self.selective = nn.Sequential(
            nn.Linear(768*2, 1024),
            nn.Dropout(self.args.dropout),
            nn.ReLU(),
            nn.Linear(1024, 768),
            nn.ReLU()
        )
        
        self.supconloss = SupConLoss(args)
        
        self.classifier = nn.Linear(768*2, self.args.num_labels)
        
    
    def forward(self, document, labels, flag, true_label, lab, mode, loss_fn):
        
        lab_token = self.tokenizer.batch_encode_plus(labels, padding=True, truncation=True, return_tensors='pt').to(self.args.device)
        label_embed = self.bert(lab_token['input_ids'], lab_token['attention_mask'])['last_hidden_state'][:,0]
        doc_token = self.tokenizer.batch_encode_plus(document, padding=True, truncation=True, return_tensors='pt').to(self.args.device)
        feature = self.bert(doc_token['input_ids'], doc_token['attention_mask'])['last_hidden_state'][:,0]
        

        feature = feature.to(self.args.device)
        label_embed = label_embed.to(self.args.device)
        label = L.to(self.args.device)


        label_loss = self.criterion(F.normalize(label_embed, dim=1))

        label_index = torch.nonzero(label)[:,1].to(self.args.device)
        means_batch = torch.index_select(label_embed, dim=0, index=label_index)

        label = label.float()
        label_num_same = torch.matmul(label, label.T) 
        sum = torch.sum(label, dim=1) 
        label_num_all = torch.zeros(label.shape[0],label.shape[0]).to(self.args.device)
        for i in range(label.shape[0]):
            label_num_all[i] = sum.add(sum[i].item())
        label_num_all = label_num_all - label_num_same 
        
        y_sim = torch.div(label_num_same, label_num_all)    
        sum_x = torch.sum(torch.where(torch.matmul(label, label.T) >= 1, 1, 0), 1)
        label_weight_1 = torch.mul(y_sim, torch.div(len(document), sum_x))
        
        mask1 = torch.where(label_weight_1>0, 1, 0).to(self.args.device)
        
        loss_con_1 = self.supconloss(feature, hasweight=True, weight=label_weight_1, mask=mask1)
        
        
        sum = torch.sum(label, 1).int()
        feature_re = feature.repeat_interleave(sum, 0)
        distance = torch.diag(torch.tensor(cosine_similarity(feature_re.cpu().detach().numpy(), means_batch.cpu().detach().numpy()))).to(self.args.device)
        sim_for_feature = torch.split(distance, tuple(sum))
        normalized_tensor = [F.normalize(i, p=2, dim=0) for i in sim_for_feature]
        means_batch_split = torch.split(means_batch, tuple(sum))


        lab_feature = torch.stack([torch.mean(normalized_tensor[i].unsqueeze(1)*means_batch_split[i], dim=0) for i in range(len(normalized_tensor))])
        
        
        doc_num_same = torch.matmul(label.T, label)     
        sum = torch.sum(label.T, dim=1)  
        doc_num_all = torch.zeros(label.shape[1], label.shape[1]).to(self.args.device)
        for i in range(label.shape[1]):
            doc_num_all[i] = sum.add(sum[i].item())
        doc_num_all = doc_num_all - doc_num_same              
       
        doc_sim = torch.div(doc_num_same, doc_num_all)     
        doc_sim[doc_sim != doc_sim] = 0                  
 
        label_weight_2 = doc_sim.to(self.args.device)
        mask2 = torch.where(label_weight_2>0, 1, 0).to(self.args.device)
        
       
        loss_con_l = self.supconloss(label_embed, hasweight=True, weight=label_weight_2, mask=mask2)
        loss_con_2 = self.supconloss(lab_feature, hasweight=True, weight=label_weight_1, mask=mask1)
       
        loss_distance = torch.mean((lab_feature - feature) ** 2)

        e_l = self.attention(feature, label_embed, label_embed)
        feature = torch.cat((e_l, feature), dim=1)

        output = self.classifier(feature)

        return output, label_loss, loss_con_1, loss_con_2, loss_distance, loss_con_l
        

def regularization_loss(model, factor, p=2):
    reg_loss = torch.tensor(0.,)
    for name, w in model.named_parameters():
        if 'weight' in name:    
            reg_loss = reg_loss + torch.norm(w, p)
    reg_loss = factor * reg_loss
    return reg_loss
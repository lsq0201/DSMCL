import torch
import torch.nn as nn

from transformers import AutoTokenizer, AutoModel
from transformers import AdamW, get_linear_schedule_with_warmup

import os
import numpy as np
from sklearn import metrics
from torchmetrics import Precision
from sklearn.preprocessing import MultiLabelBinarizer
import warnings
warnings.filterwarnings("ignore")

from data_loader.dataloader import MainDataset
# from model.model import Model
# from model.GM_model import Model
from model.model_labelrepre_neg import Model
from main.utils import *
from main.model_validate import validate, modify_pred
from main.evals import compute_metrics


class Trainer(object):
    def __init__(self, args):
        self.args = args
    
        self.dataset = MainDataset(args)
        
        self.train_loader, self.val_loader = self.dataset.train_loader, self.dataset.val_loader
        self.test_loader = self.dataset.test_loader
        self.label = self.dataset.labels
        
        # self.aapd_trainloader, self.aapd_valloader, self.aapd_testloader \
        #     = self.dataset.aapd_train_loader, self.dataset.aapd_val_loader, self.dataset.aapd_test_loader
        # self.emb = self.dataset.emb
        
        if self.args.current == 'train':
            self.model = Model(args, freeze=False)
        elif self.args.current == 'test':
            self.model = torch.load(args.checkpoint)
            
        # device_ids = [0, 1, 2, 3] #必须从零开始(这里0表示第1张卡，1表示第2张卡.)
        self.model.to(args.device)   
        # self.model = nn.DataParallel(self.model)
        # self.model = nn.parallel.DistributedDataParallel(self.model)
        
        self.optimizer, self.scheduler = self._get_optimizer()
        # sigmoid + BCELoss
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.loss = nn.CrossEntropyLoss()
        
        
    def _get_optimizer(self):
        param_optimizer = list(self.model.named_parameters())
        
        # 学习率不衰减的参数（集合）
        no_decay = ['bias', 'layer_norm.bias', 'layer_norm.weight']
        optimizer_grouped_parameters = [
                # 分层权重衰减（以transformers中的BertModel为例）
                {'params': [param for name, param in param_optimizer if not any(nd in name for nd in no_decay)],
                'weight_decay': self.args.weight_decay},
                {'params': [param for name, param in param_optimizer if any(nd in name for nd in no_decay)],
                'weight_decay': 0.0}
        ]
        
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate)
        
        '''学习率预热'''
        # 先从 0 增加到优化器中的初始预设 lr , 再线性降低到 0
        # 使得开始训练的几个epoches或者一些steps内学习率较小,在预热的小学习率下，模型可以慢慢趋于稳定,
        # 等模型相对稳定后再选择预先设置的学习率进行训练,使得模型收敛速度变得更快，模型效果更佳
        '''
            param: 
                optimizer: 优化器
                num_training_steps: 整个训练过程中的总步数
                num_warmup_steps: 初始预热步数
        '''
        scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,
                                                    num_training_steps=self.args.epoch * len(self.train_loader),
                                                    num_warmup_steps=100)
        
        return optimizer, scheduler


    def step(self, batch):
        self.model.train()
        docWithId, labels, flag, y_true = batch['docWithId'], batch['labels'], batch['flag'], batch['y_true'].to(self.args.device)
        
        document = [i[1] for i in docWithId]
        
        # 清除优化器 optimizer 中所有变量 x 的 x.grad(梯度)
        self.optimizer.zero_grad()
        
        # outputs, loss_con_1, loss_con_2, loss_con_3, count_outputs = self.model(document, labels)
        # outputs, loss_con_1 = self.model(document, self.label, y_true)
        outputs, label_loss, loss_con_1, loss_con_2, loss_distance, count_outputs, loss_con_l, logit, logits_neg, loss_weight, reg_loss = self.model(document, self.label, flag, y_true, labels, True)
        
        loss_ce = self.loss_fn(outputs.to(self.args.device), y_true.float())
        # loss_logit = self.loss_fn(logit.to(self.args.device), y_true.float())
        # loss_logit_neg = self.loss_fn(logits_neg.to(self.args.device), y_true.float())

        # 不添加 loss_count
        if self.args.if_loss_count != 'true':
            # loss = loss_ce + 0.05*loss_con_1 + 0.05*loss_con_2 + 0.05*label_loss + 0.05*loss_distance + 0.05*loss_con_l
            loss = loss_ce + 0.1*loss_con_1 + 0.1*loss_con_2 + 0.1*label_loss + 0.1*loss_distance + 0.1*loss_con_l

            # lw
            # loss_list = [loss_ce, label_loss, loss_con_1, loss_con_l, loss_con_2, loss_distance]
            # final_loss = []
            # for i in range(len(loss_list)):
            #     final_loss.append(loss_list[i] / (2 * loss_weight[i].pow(2)) + torch.log(loss_weight[i]))
            # loss = torch.sum(torch.stack(final_loss))

        else:
            # 标签索引从0开始，因此 -1
            labels_count = y_true.sum(dim=1)-1
            count = torch.ones_like(labels_count)*(self.args.count-1)
            # [16], 将数值结果控制在 0~count范围内
            labels_count = torch.where(labels_count > (self.args.count-1), count, labels_count)
            # 标签个数预测损失
            loss_count = self.loss(count_outputs, labels_count)
            loss = loss_ce + 0.01*loss_con_1 + 0.01*loss_con_2 + 0.01*label_loss + 0.01*loss_distance + 0.1*loss_count
            
            outputs = modify_pred(self.args, count_outputs, outputs, labels_count)
        
        loss.backward()
        
        # 对 x 的值进行更新，lr 用于控制步幅  x = x - lr * x.grad
        self.optimizer.step()
        # 更新学习率lr
        self.scheduler.step()
        
        return loss.item(), outputs, y_true


    def train(self):
        print('Training...')
        write_training_info(self.args)
        best_score = float("-inf")
        best_ckpt = ''

        wrong_data = {'document':[], 'labels':[]}
        for epoch in range(self.args.epoch):
            total_loss = 0.0
            predicted_labels, target_labels = list(), list()

            for i, batch in enumerate(self.train_loader):
                loss, y_pred, y_true = self.step(batch)
                
                total_loss += loss
                target_labels.extend(y_true.detach().cpu().numpy())
                predicted_labels.extend(y_pred.detach().cpu().numpy())
                # 每 50 个 batch
                if (i + 1) % 50 == 0 or i == 0 or i == len(train_loader) - 1:
                    print("Epoch: {} - iter: {}/{} - train_loss: {}".format(epoch, i + 1, len(self.train_loader), total_loss/(i + 1)))    
                
                if i == len(self.train_loader) - 1:
                    predicted_labels, target_labels = np.array(predicted_labels), np.array(target_labels)
                    ndcg1 = metrics.ndcg_score(target_labels, predicted_labels, k=1)
                    ndcg3 = metrics.ndcg_score(target_labels, predicted_labels, k=3)
                    ndcg5 = metrics.ndcg_score(target_labels, predicted_labels, k=5)
                    result_train = compute_metrics(predicted_labels,target_labels, 0.5, all_metrics=True)
                    result_train['ndcg1'] = ndcg1
                    result_train['ndcg3'] = ndcg3
                    result_train['ndcg5'] = ndcg5
                    result_train['val_loss'] = total_loss
                    print_eval(result_train)
                    
                    if best_score < result_train['p_at_1']:
                        best_score = result_train['p_at_1']
                        ckpt_path = save_ckpt(self.args, self.model, epoch)
                        best_ckpt = ckpt_path
            
            write_test_sh(self.args, best_ckpt) 

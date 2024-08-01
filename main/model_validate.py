import torch

import numpy as np
from sklearn import metrics

from main.evals import compute_metrics
from main.utils import *


def validate_gm(args, model, data_loader, loss_fn, label):
    model.eval()
    with torch.no_grad():
        total_loss = 0.0
        predicted_labels, target_labels = list(), list()
        for i, batch in enumerate(data_loader):
            docWithId, labels, y_true = batch['docWithId'], batch['labels'], batch['y_true'].to(args.device)
        
            document = [i[1] for i in docWithId]

            # output, loss_con, output_label, target_label, label_loss = model(document, label, y_true)
            # output, label_loss, loss_con, output_gm, loss_re = model(feature, label)
            # output, label_loss, loss_con_1, loss_con_2, loss_distance = model(document, label, y_true)
            output, loss_con_1 = model(document, label, y_true)
            loss_ce = loss_fn(output.to(args.device), y_true.float().to(args.device))
            # loss_ce_label = loss_fn(output_label.to(args.device), target_label.float().to(args.device))
            # loss_ce_gm = loss_fn(output_gm.to(args.device), label.float().to(args.device))

            total_loss = loss_ce + 0.01*loss_con_1
            # total_loss = loss_ce + 0.01*loss_con_1 + 0.01*loss_con_2 + label_loss + loss_distance
            # total_loss = loss_ce + 0.05*label_loss + 0.5*loss_con + 0.05*loss_ce_gm + 0.05*loss_re
            target_labels.extend(y_true.detach().cpu().numpy())
            predicted_labels.extend(output.detach().cpu().numpy())

            val_loss = total_loss/len(data_loader)
    result = cp_loss(predicted_labels, target_labels,val_loss)
    return result


def validate_cls(args, model, data_loader, loss_fn):
    model.eval()
    with torch.no_grad():
        total_loss = 0.0
        predicted_labels, target_labels = list(), list()
        predicted_labels_l, target_labels_l = list(), list()
        for i, batch in enumerate(data_loader):
            docWithId, labels, y_true = batch['docWithId'], batch['labels'], batch['y_true'].to(args.device)
        
            document = [i[1] for i in docWithId]
            # y_true = label.to(args.device)
            
            # output = model(feature)
            # loss = loss_fn(output.to(args.device), y_true.float())
            
            # output, spec_laebl = model(feature, label)
            # y_true = spec_laebl
            # loss, nll_loss, nll_loss_x, c_loss, c_loss_x, kl_loss, cpc_loss, _, pred_x = \
            #             compute_loss(y_true, output, args, loss_fn)
                        
            loss_con, labelemb_output, labelemb_label, feature_putput = model(document, y_true)
            loss_labelemb = loss_fn(labelemb_output, labelemb_label.float().to(args.device))
            loss_feature = loss_fn(feature_putput, y_true.float().to(args.device))
            loss = loss_con*0.05 + loss_labelemb*0.5 + loss_feature
            
            total_loss = total_loss + loss.item()
            target_labels.extend(y_true.detach().cpu().numpy())
            predicted_labels.extend(feature_putput.detach().cpu().numpy())
            # predicted_labels.extend(feature_putput.data.cpu().numpy())
            target_labels_l.extend(labelemb_label.detach().cpu().numpy())
            predicted_labels_l.extend(labelemb_output.detach().cpu().numpy())
        val_loss = total_loss/len(data_loader)
        
    result = cp_loss(predicted_labels, target_labels,val_loss)
    result_l = cp_loss(predicted_labels_l, target_labels_l,val_loss)
    # predicted_labels, target_labels = np.array(predicted_labels), np.array(target_labels)
    # ndcg1 = metrics.ndcg_score(target_labels, predicted_labels, k=1)
    # ndcg3 = metrics.ndcg_score(target_labels, predicted_labels, k=3)
    # ndcg5 = metrics.ndcg_score(target_labels, predicted_labels, k=5)
    # result = compute_metrics(predicted_labels,target_labels, 0.5, all_metrics=True)
    # result['ndcg1'] = ndcg1
    # result['ndcg3'] = ndcg3
    # result['ndcg5'] = ndcg5
    # result['val_loss'] = val_loss

    return result, result_l


def validate(args, model, data_loader, loss_fn, loss, label):
    model.eval()
    with torch.no_grad():
        total_loss = 0.0
        predicted_labels, target_labels = list(), list()
        
        for i, batch in enumerate(data_loader):
            
            docWithId, labels, flag, y_true = batch['docWithId'], batch['labels'], batch['flag'], batch['y_true'].to(args.device)
            
            document = [i[1] for i in docWithId]
            
            # outputs:[bs, args.num_labels]; count_outputs:[bs, args.count]
            # outputs, loss_con_1, loss_con_2, loss_con_3, count_outputs = model(document, labels)
            # outputs, loss_con_1 = model(document, label, y_true)
            outputs, label_loss, loss_con_1, loss_con_2, loss_distance, count_outputs, loss_con_l, logit, logit_neg, _, _, loss_gm, loss_neg = model(document, label, flag, y_true, labels, False, loss_fn)
            loss_ce = loss_fn(outputs.to(args.device), y_true.float())
            # loss_logit = loss_fn(logit.to(args.device), y_true.float())
            # loss_logit_neg = loss_fn(logit_neg.to(args.device), y_true.float())
            
            # 不添加 loss_count
            if args.if_loss_count != 'true':
                # loss_now = loss_ce + 0.01*loss_con_1 + 0.01*loss_con_2 + 0.01*loss_con_3
                # loss_now = loss_ce + 0.01*loss_con_1
                loss_now = loss_ce + 0.01*loss_con_1 + 0.01*loss_con_2 + 0.01*label_loss + 0.01*loss_distance + 0.01*loss_con_l + loss_gm + loss_neg
            else:
                # 样本真实标签 计数 [bs]
                labels_count = y_true.sum(dim=1)-1
                count = torch.ones_like(labels_count)*(args.count-1)
                # [16]
                labels_count = torch.where(labels_count > (args.count-1), count, labels_count)
                # 标签个数预测损失
                loss_count = loss(count_outputs, labels_count)
                loss_now = loss_ce + 0.01*loss_con_1 + 0.01*loss_con_2 + 0.01*label_loss + 0.01*loss_distance - 0.1*loss_count
                
                outputs = modify_pred(args, count_outputs, outputs, labels_count)
            
            total_loss = total_loss + loss_now.item()
            
            target_labels.extend(y_true.detach().cpu().numpy())
            predicted_labels.extend(outputs.detach().cpu().numpy())
            
        val_loss = total_loss/len(data_loader)  
            
    predicted_labels, target_labels = np.array(predicted_labels), np.array(target_labels)
    # accuracy = metrics.accuracy_score(target_labels, predicted_labels.round())
    # micro_f1 = metrics.f1_score(target_labels, predicted_labels.round(), average='micro')
    # macro_f1 = metrics.f1_score(target_labels, predicted_labels.round(), average='macro')
    
    ndcg1 = metrics.ndcg_score(target_labels, predicted_labels, k=1)
    ndcg3 = metrics.ndcg_score(target_labels, predicted_labels, k=3)
    ndcg5 = metrics.ndcg_score(target_labels, predicted_labels, k=5)
    
    # n_classes = self.dataset.label_features.size(0)
    # p1 = Precision('multiclass', num_classes=n_classes, top_k=1)(torch.tensor(predicted_labels), torch.tensor(target_labels))
    # p3 = Precision('multiclass', num_classes=n_classes, top_k=3)(torch.tensor(predicted_labels), torch.tensor(target_labels))
    # p5 = Precision('multiclass', num_classes=n_classes, top_k=5)(torch.tensor(predicted_labels), torch.tensor(target_labels))
    
    result = compute_metrics(predicted_labels,target_labels, 0.5, all_metrics=True)
    result['ndcg1'] = ndcg1
    result['ndcg3'] = ndcg3
    result['ndcg5'] = ndcg5
    result['val_loss'] = val_loss

    return result


def modify_pred(args, count_outputs, outputs, labels_count):
    # print(outputs)
    # 根据模型预测的标签个数，modify y_pred
    # , 每一行最大值的索引
    _, count_pred  = torch.max(count_outputs, 1, keepdim=True)
    # 每个样本的真实标签数量
    labels_count = labels_count.cpu().detach()
    # [bs, 1]
    count_pred = count_pred.cpu().detach()
    # logits排序， 索引排序--标签; 降序; [bs, args.num_labels]
    sorts, indices = torch.sort(outputs, descending=True)  
    # 储存 根据mlp预测的标签数量 的最后一个标签对应的预测值
    x = []
    # i: 行 每个样本
    # count_pred[i][0]：样本i的预测个数
    # indices[i][count_pred[i][0]]：样本i排在第 位的预测标签
    # outputs[i][indices[i][count_pred[i][0]]]：样本i排在第 位标签的预测结果
    for i, t in enumerate(count_pred):
        x.append(outputs[i][indices[i][count_pred[i][0]]])
    x = torch.tensor(x).view(outputs.shape[0], 1).to(args.device)
    one = torch.ones_like(outputs)
    zero = torch.zeros_like(outputs)
    # 预测值大于 的标签位置，置1
    y_pred = torch.where(outputs >= x, one, outputs)
    # 预测值小于 的标签位置，置0
    y_pred = torch.where(y_pred < x, zero, y_pred)
    
    return y_pred
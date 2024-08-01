import os
import math
from tqdm import tqdm
import json
import numpy as np
from sklearn import metrics

import torch

from main.evals import compute_metrics


def build_path(path):
    os.makedirs(path)


'''保存ckpt'''
def save_ckpt(args, model, epoch):
    path = 'ckpt/' +  args.dataset_name
    if not os.path.exists(path):
        os.makedirs(path)
    ckpt_path = path + f'/checkpoint_{epoch}.pt'
    torch.save(model.state_dict(), ckpt_path)
    # torch.save(model, ckpt_path)
    return ckpt_path


def write_training_info(args):
    path = 'ckpt/' +  args.dataset_name
    if not os.path.exists(path):
        os.makedirs(path)
    info = {}
    info['dataset_name'] = args.dataset_name
    info['epoch'] = args.epoch
    info['batch_size'] = args.batch_size
    info['max_length'] = args.max_length
    info['dropout'] = args.dropout
    info['learning_rate'] = args.learning_rate
    info['weight_decay'] = args.weight_decay
    info['temperature'] = args.temperature
    info['if_loss_count'] = args.if_loss_count
    info['emb_size'] = args.emb_size
    with open(path + '/info_{}.json'.format(args.result_path), 'w', encoding='utf8') as f:
        json.dump(info, f, indent=4)
        
    
def save_result(path, result, epoch):
    with open(path, 'r', encoding='utf8') as f:
        re = json.loads(f.read())
    re_new = {}
    for i in result.items():
        re_new[i[0]] = str(i[1])
    re.append(epoch)
    re.append(re_new)
    with open(path, 'w', encoding='utf8') as f:
        json.dump(re, f, indent=4)
        

def print_eval(metrics):
    print('============Performance============\n')
    print("Val_loss: {}\n \
            - HA: {}\n \
            - Accuracy: {}\n \
            - Micro-F1: {}\n \
            - Macro-F1: {}\n \
            - Example-F1: {}\n \
            --------------------------\n \
            - nDCG1: {}\n \
            - nDCG@3: {}\n \
            - nDCG@5: {}\n \
            - P@1: {}\n \
            - P@3: {}\n \
            - P@5: {}".format(metrics['val_loss'], metrics['HA'], metrics['ACC'], \
                metrics['miF1'], metrics['maF1'], metrics['ebF1'], \
                metrics['ndcg1'], metrics['ndcg3'], metrics['ndcg5'], \
                metrics['p_at_1'], metrics['p_at_3'], metrics['p_at_5']))
        
        
def write_test_sh(args, ckpt_path):
    test_sh_path = 'scripts_test/{}'.format(args.dataset_name)
    if os.path.exists(test_sh_path):
        ckptFile = open(test_sh_path + '/test.sh', "r")
        command = []
        for line in ckptFile:
            arg_lst = line.strip().split(' ')
            if '--checkpoint' in arg_lst:
                command.append('--checkpoint')
                # command.append('ckpt/{}/checkpoint_{}.pt'.format(dataset, epoch))
                command.append(ckpt_path)
            else:
                command.extend(arg_lst)
            command.append('\n')
        for i in range(len(command)):
            if 'ckpt/' in command[i]:
                command[i] = "'" + command[i] + "'"
        ckptFile.close()
    else:
        os.makedirs(test_sh_path)
        command = ("python main.py\n \
            --train_data_path %s \n \
            --val_data_path %s \n \
            --test_data_path %s \n \
            --tokenizer_name %s \n \
            --bert_name %s \n \
            --device %s \n \
            --batch_size %d \n \
            --max_length %d \n \
            --current %s \n \
            --count %d \n \
            --num_labels %d \n\
            --if_loss_count %s \n \
            --checkpoint %s" 
            % (args.train_data_path, args.val_data_path, args.test_data_path, args.tokenizer_name, args.bert_name, args.device, 
            args.batch_size, args.max_length, args.current, args.count, args.num_labels, args.if_loss_count, ckpt_path)).strip().split(' ')
        for i in range(len(command)):
            if '--train_data_path' in command[i] or '--val_data_path' in command[i] or '--test_data_path' in command[i] or '--tokenizer_name' in command[i] or '--bert_name' in command[i] or '--device' in command[i] or '--current' in command[i] or '--if_loss_count' in command[i]:
                command[i+1] = "'" + command[i+1] + "'"
            if 'ckpt/' in command[i]:
                command[i] = "'" + command[i] + "'"
    
    ckptFile = open(test_sh_path + '/test.sh', "w")
    ckptFile.write(" ".join(command)+"\n")
    ckptFile.close()
    
    
def cp_loss(predicted_labels, target_labels, total_loss):
    predicted_labels, target_labels = np.array(predicted_labels), np.array(target_labels)
    ndcg1 = metrics.ndcg_score(target_labels, predicted_labels, k=1)
    ndcg3 = metrics.ndcg_score(target_labels, predicted_labels, k=3)
    ndcg5 = metrics.ndcg_score(target_labels, predicted_labels, k=5)
    result = compute_metrics(predicted_labels,target_labels, 0.5, all_metrics=True)
    result['ndcg1'] = ndcg1
    result['ndcg3'] = ndcg3
    result['ndcg5'] = ndcg5
    result['val_loss'] = total_loss
    return result
import json
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np

import torch
import torch.utils.data as data_utils
from torch.utils.data import Dataset, DataLoader

from data_loader.data_utils import *


class MainDataset(object):
    def __init__(self, args):
        self.args = args
        self.batch_size = args.batch_size
        self.max_length = args.max_length
        
        self.train_data_path = args.train_data_path
        self.val_data_path = args.val_data_path
        self.test_data_path = args.test_data_path
        
        self.train_data, self.val_data, self.train_labels, self.val_labels, self.test_data, self.test_labels, self.ori_labels \
            = self.load_train_dataset(self.train_data_path, self.val_data_path, self.test_data_path)
            
        self.train_loader = self.data_loader(self.train_data, self.train_labels, shuffle=True)
        self.val_loader = self.data_loader(self.val_data, self.val_labels, shuffle=False)
        self.test_loader = self.data_loader(self.test_data, self.test_labels, shuffle=False)
        
        self.label_detail = json.load(open(self.args.label_detail))
        # self.labels = list(self.label_detail.values())
        self.labels = list(k + ', ' + v for k, v in self.label_detail.items())

        
    def load_train_dataset(self, train_data_path, val_data_path, test_data_path):
        # 获取数据集
        train = json.load(open(train_data_path))
        val = json.load(open(val_data_path))
        test = json.load(open(test_data_path))
        
        # train_extra = json.load(open('/data/lishuqin/data/processed/AAPD/single.json'))
        # train_extra_sents = [clean_string(text) for text in train_extra['content']]
        
        # doc清理
        train_sents = [clean_string(text) for text in train['content']]
        # train_sents.extend(train_extra_sents)
        val_sents = [clean_string(text) for text in val['content']]
        test_sents = [clean_string(text) for text in test['content']]
        
        # label
        train_labels = train['labels']
        # train_labels.extend(train_extra['labels'])
        val_labels = val['labels']
        test_labels = test['labels']
        
        train_data, val_data, test_data = {}, {}, {}
        train_data['document'] = train_sents
        train_data['labels'] = train_labels
        val_data['document'] = val_sents
        val_data['labels'] = val_labels
        test_data['document'] = test_sents
        test_data['labels'] = test_labels
        
        mlb_label = MultiLabelBinarizer()     # mlb.classes_ 将会由最后一个fit_transform()中的内容确定
        # [batch_size, class] [[0,1,0],[1,0,1],[0,0,1],...]
        train_labels = mlb_label.fit_transform(train_data['labels'])  
        val_labels = mlb_label.transform(val_data['labels'])
        test_labels = mlb_label.transform(test_data['labels'])
        
        labels = list(mlb_label.classes_)
        
        # {'document':['','','',...], 'label':[[],[],[],...]}
        return train_data, val_data, train_labels, val_labels, test_data, test_labels, labels
    
    
    def data_loader(self, train_data, train_labels, shuffle=False):
        
        docWithId = []
        index = 0
        for i in train_data['document']:
            docWithId.append((index, i))
            index += 1
        
        label = train_data['labels']
        # 数据标签
        y_train = torch.tensor(train_labels, dtype=torch.long)
        
        dataset = MyDataset(docWithId, label, y_train)
        # print(dataset)
        train_loader = DataLoader(dataset=dataset, 
                                  batch_size=self.batch_size, 
                                  shuffle=True,
                                  collate_fn=my_collate
        )
        # print(train_loader)
        return train_loader


'''为 data_loader 构建数据集'''
class MyDataset(Dataset):
    def __init__(self, docWithId, label, y_true):
        self.docWithId = docWithId
        self.label = label
        self.y_true = y_true
        
    def __getitem__(self, item):
        return {
            'docWithId' : self.docWithId[item],
            'labels' : self.label[item],
            'y_true': self.y_true[item]
        }
    
    def __len__(self):
        return len(self.docWithId)
    
    
def my_collate(batch):
    docWithId = []
    label = []
    
    for i in range(len(batch)):
        docWithId.append(batch[i]['docWithId'])
        label.append(batch[i]['labels'])
        if i == 0:
            y_true = batch[i]['y_true'].unsqueeze(0)
        else:
            y_true = torch.cat((y_true, batch[i]['y_true'].unsqueeze(0)), dim=0)
        
    data = {
        'docWithId' : docWithId,
        'labels' : label,
        'y_true' : y_true
    }
    
    return  data
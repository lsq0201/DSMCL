import torch


'''将多标签数据展平'''
def flat_data(document, labels):
    # atten_data = []
    label = []
    doc = []
    for i in range(len(labels)):
        # i : index
        for j in labels[i]:
            # j : str
            label.append(j)
            doc.append(document[i])
    # 返回展平后的数据：标签与doc一一对应
    return label, doc


'''label_id
'''
def deal_data(label):
    label.extend(label)
    label_id = []
    for lab in label:
        tmp = []
        for lab2 in label:
            if lab == lab2:
                tmp.append(1)
            else: 
                tmp.append(0)
        label_id.append(tmp)
    return label_id


def deal_data_for_doc(labels):
    label_id = []
    same = []  # 两个doc的相同标签数量
    sum = []  # 两个doc的总标签数量
    num_doc = []  # 与当前样本（list1）有相同标签的样本总数

    for list1 in labels:
        tmp = []
        tmp_same = []
        tmp_sum = []
        tmp_num_doc = 0
        for list2 in labels:
            num_same = len(set(list1) & set(list2))
            num_sum = len(set(list1)|set(list2))
            tmp_sum.append(num_sum)
            if num_same != 0:
                tmp.append(1)
                tmp_same.append(num_same)   # 相同标签个数
                tmp_num_doc += 1
            else:
                tmp.append(0)
                tmp_same.append(0)
        label_id.append(tmp)
        same.append(tmp_same)
        sum.append(tmp_sum)
        num_doc.append(tmp_num_doc)
        
    same = torch.tensor(same)
    sum = torch.tensor(sum)
    num_doc = torch.tensor(num_doc)

    weight = torch.mul(torch.div(same, sum), torch.div(4, num_doc))
    return label_id, weight


def deal_aapd_2(y_true):
    label_id = []
    same = []  # 两个doc的相同标签数量
    sum = []  # 两个doc的总标签数量
    num_doc = []  # 与当前样本（list1）有相同标签的样本总数
    for i in range(y_true.shape[0]):
        tmp = []
        tmp_same = []
        tmp_sum = []
        tmp_num_doc = 0
        for j in range(y_true.shape[0]):
            # num_same = len(set(list1) & set(list2))
            # num_sum = len(set(list1)|set(list2))
            # tmp_sum.append(num_sum)
            
            has_same =  y_true[i].eq(y_true[j])
            count_1 = torch.unique(has_same,return_counts=True)
            num_same = count_1[1][1].item()
            num_sum = torch.where(y_true[i]>y_true[j], y_true[i], y_true[j])
            count_1 = torch.unique(num_sum, return_counts=True)
            num_sum = count_1[1][1].item()
            tmp_sum.append(num_sum)
            
            if not torch.equal(has_same, torch.zeros(has_same.shape)):
                tmp.append(1)
                tmp_same.append(num_same)   # 相同标签个数
                tmp_num_doc += 1
            else:
                tmp.append(0)
                tmp_same.append(0)
        label_id.append(tmp)
        same.append(tmp_same)
        sum.append(tmp_sum)
        num_doc.append(tmp_num_doc)
    
    same = torch.tensor(same)
    sum = torch.tensor(sum)
    num_doc = torch.tensor(num_doc)

    weight = torch.mul(torch.div(same, sum), torch.div(4, num_doc))
    return label_id, weight

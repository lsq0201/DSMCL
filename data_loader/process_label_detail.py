# 对标签信息进行扩充，获取更丰富的标签知识
from lm_api import lmp
import json

def rich_label(labels_path, dataset, label_detail_path):
    # label = '7986.0'
    # prompt = 'The following is a tag from the Eurlex dataset. Please explain the meaning of this tag: {}. Requirements: 1. Only answer the meaning. 2. If the tag is composed of an ID, please indicate its corresponding specific legal topic and category.'.format(label)
    # print(prompt)

    with open(labels_path, 'r') as f:
        labels = json.loads(f.read())

    with open(label_detail_path, 'r') as f:
        con = json.loads(f.read())
    for label in labels:
        prompt = 'The following is a tag from the {} dataset. Please explain the meaning of this tag: {}. Requirements: 1. Only answer the meaning. 2. If the tag is composed of an ID, please indicate its corresponding specific legal topic and category. 3. Less than 30 tokens.'.format(dataset, label)
        label_info = lmp(prompt)
        print(label_info)
        with open(label_detail_path, 'w') as f:
            con[label] = label_info
            json.dump(con, f, indent=2)

dataset = 'Eurlex'
labels_path = '/home/oem/projects/lishuqin/coding/data/Eurlex/Eurlex/labels.json'
label_detail_path = '/home/oem/projects/lishuqin/coding/data/Eurlex/Eurlex/label_detail.json'

# dataset = 'RCV1-V2'
# labels_path = '/home/oem/projects/lishuqin/coding/data/RCV1-V2/RCV1-V2/labels.json'
# label_detail_path = '/home/oem/projects/lishuqin/coding/data/RCV1-V2/RCV1-V2/label_detail.json'

rich_label(labels_path, dataset, label_detail_path)
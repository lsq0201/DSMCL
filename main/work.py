from argparse import ArgumentParser
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import sys
# sys.path.append('/data/lishuqin/project/multi-label/Idea/data/AAPD')
sys.path.append('../..')
sys.path.append('./')
# print(sys.path)

from main.model_train_neg import Trainer
# from main.model_test import Tester

import torch


if __name__ == '__main__':
    parser = ArgumentParser('trainer')
    
    parser.add_argument('--dataset_name', type=str, default='AAPD_l2weight_data')
    parser.add_argument('--train_data_path', type=str, default='/home/oem/projects/lishuqin/coding/data/AAPD/train.json')
    parser.add_argument('--val_data_path', type=str, default='/home/oem/projects/lishuqin/coding/data/AAPD/val.json')
    parser.add_argument('--test_data_path', type=str, default='/home/oem/projects/lishuqin/coding/data/AAPD/test.json')
    
    parser.add_argument('--tokenizer_name', type=str, default='/home/oem/projects/lishuqin/coding/plm/roberta')
    parser.add_argument('--bert_name', type=str, default='/home/oem/projects/lishuqin/coding/plm/roberta')
    
    parser.add_argument('--device', type=str, default='cuda:3')
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_length', type=int, default=384)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--learning_rate', type=float, default=0.00005)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--temperature', type=float, default=0.05)
    parser.add_argument("--current", type = str, default='train')
    parser.add_argument("--checkpoint", type = str, default='/data/lishuqin/project/multi-label/Idea/ckpt/AAPD_bs32_wocount/checkpoint_49.pt')
    
    parser.add_argument('--count', type=int, default=5)
    parser.add_argument('--num_labels', type=int, default=54)
    
    # true则使用 loss_count，false 该参数则不使用
    parser.add_argument('--if_loss_count', type=str, default='false')
    
    parser.add_argument('--label_dim', type=int, default=54)
    parser.add_argument('--emb_size', type=int, default=3072)
    parser.add_argument('--T0', "--T0", default=50, type=int, help='optimizer T0')
    parser.add_argument('--T_mult', "--T_mult", default=2, type=int, help='T_mult')
    parser.add_argument('--eta_min', "--eta_min", default=2e-4, type=float, help='eta min')
    parser.add_argument('--result_path', type=str, default='re.json')

    parser.add_argument('--label_detail', type=str)

    # 是否加入负样本
    parser.add_argument('--negative', action='store_true', default=False)
    parser.add_argument('--positive_prompt', type=str, default='Classified correctly!')
    parser.add_argument('--negative_prompt', type=str, default='Classified incorrectly!')

    args = parser.parse_args()
    
    seed = 1000
    torch.manual_seed(seed)            # 为CPU设置随机种子
    torch.cuda.manual_seed(seed) 
    
    if args.current == "train":
        model_trainer = Trainer(args)
        model_trainer.train()
    elif args.current == "test":
        model_tester = Tester(args)
        model_tester.test()
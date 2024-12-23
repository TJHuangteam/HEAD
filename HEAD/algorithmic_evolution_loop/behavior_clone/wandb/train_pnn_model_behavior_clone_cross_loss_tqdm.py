import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
import logging
import wandb
  # 引入 wandb
from torch.autograd import Variable
from tqdm import tqdm
from ProgressiveNeuralNetworks import PNN
from arg_parser_actions import LengthCheckAction
from evaluation import evaluate_model

import pickle
from torch.utils.data import DataLoader, TensorDataset

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def get_args():
    parser = argparse.ArgumentParser(description='Progressive Neural Networks')
    parser.add_argument('-cuda', default=-1, type=int, help='Cuda device to use (-1 for none)')

    parser.add_argument('--layers', metavar='L', type=int, default=3, help='Number of layers per task')
    parser.add_argument('--sizes', dest='sizes', default=[259, 1024, 512, 2], nargs='+',
                        action=LengthCheckAction)
    parser.add_argument('--epochs', dest='epochs', type=int, default=10)
    parser.add_argument('--bs', dest='batch_size', type=int, default=50)
    parser.add_argument('--lr', dest='lr', type=float, default=1e-3, help='Optimizer learning rate')
    parser.add_argument('--wd', dest='wd', type=float, default=1e-4, help='Optimizer weight decay')
    parser.add_argument('--momentum', dest='momentum', type=float, default=1e-4, help='Optimizer momentum')

    # parser.add_argument('-path', default='/local/veniat/data', type=str, help='path to the data')
    # parser.add_argument('-project_name', default="PNN", type=str, help='WandB project name')  # 添加项目名称
    # parser.add_argument('--n_tasks', dest='n_tasks', type=int, default=1)

    args = parser.parse_known_args()
    return args[0]

#加载多个数据集
def load_multiple_datasets(file_paths):
    tasks_data=[]

    for file_path in file_paths:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            print(f"Data from {file_path} has been loaded.")



            features = torch.Tensor(data['state']).float().unsqueeze(dim=-1)
            labels = torch.Tensor(data['action']).float()

            # 每个文件生成一个 TensorDataset
            dataset = TensorDataset(features, labels)

            # 创建 DataLoader，指定批量大小和是否进行数据随机化
            batch_size = 256
            shuffle = True
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
            tasks_data.append(dataloader)

    return tasks_data



def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args['cuda'])

    # 初始化 WandB
    # wandb.init(project=args['project_name'], config=args)  # 初始化 wandb
    # config = wandb.config  # 将 config 保存在 wandb

    model = PNN(args['layers']) # 3层

    current_path = os.path.abspath(__file__)
    root_path = os.path.dirname(os.path.dirname(current_path))
    data_paths = [
        root_path + "/data_collector/data/processed_data.pkl",
        root_path + "/data_collector/data/processed_data.pkl",
        root_path + "/data_collector/data/processed_data.pkl",
        root_path + "/data_collector/data/processed_data.pkl"

    ]
    tasks_data = load_multiple_datasets(data_paths)
    print("All Tasks' Data has been loaded.")



    x = torch.Tensor()
    y = torch.Tensor()  # LongTensor()



    if args['cuda'] != -1:
        logger.info('Running with cuda (GPU n°{})'.format(args['cuda']))
        model.cuda()
        x = x.cuda()
        y = y.cuda()
    else:
        logger.warning('Running WITHOUT cuda')

    for task_id, data in enumerate(tasks_data):
        model.freeze_columns()  # 冻结模型参数
        model.new_task(args['sizes'])  # 为新任务添加新的列

        #后续可以指定学习率等参数
        optimizer = torch.optim.RMSprop(model.parameters(task_id), lr=args['lr'],
                                        weight_decay=args['wd'], momentum=args['momentum'])

        train_accs = []
        train_losses = []
        for epoch in range(args['epochs']):
            total_samples = 0
            total_loss = 0
            correct_samples = 0
            for inputs, labels in tqdm(data):
                x.resize_(inputs.size()).copy_(inputs)  # x 输入数据
                y.resize_(labels.size()).copy_(labels)  # y 标签

                x = x.view(x.size(0), -1)  # 展平

                #模型更新全过程
                #计算预测值
                predictions = model(Variable(x))

                #计算损失loss
                indiv_loss = F.cross_entropy(predictions,y)
                total_loss += indiv_loss.item()

                #模型更新全过程
                optimizer.zero_grad()  # 反向传播前梯度归零
                indiv_loss.backward()  # 反向传播计算梯度
                optimizer.step()  # 更新模型参数






if __name__ == '__main__':
    main(vars(get_args()))
import argparse
import os
import save_load as sl
import torch
import logging
import wandb
import numpy as np
from torch.autograd import Variable
from tqdm import tqdm
import pickle
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from MoPE import MoPE

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)




base_path = os.getcwd() + '/results'


exp_name = 'PNN_5layers×2_Moe_alldata_new/'
train_method = '2'  # TODO


writer = SummaryWriter(base_path + '/logs/' + exp_name + train_method)
model_path = base_path + '/checkpoint/' + exp_name + train_method
if not os.path.exists(model_path):
    os.makedirs(model_path)


def get_args():
    parser = argparse.ArgumentParser(description='Mixture of Progressive Experts')

    parser.add_argument('-project_name', default="Train_all_XCO_scenario", type=str, help='WandB project name')
    parser.add_argument('-run_name', default="train", type=str, help='WandB run name')

    parser.add_argument('--layers', metavar='L', type=int, default=5, help='Number of layers per task')
    parser.add_argument('--sizes', dest='sizes', default=[259, 1024, 512, 256, 256, 2], nargs='+')

    parser.add_argument('--task_0_epochs', type=int, default=0, help='Number of epochs for task 0 training')
    parser.add_argument('--task_1_epochs', type=int, default=0, help='Number of epochs for task 1 training')
    parser.add_argument('--gating_epochs', type=int, default=1000, help='Number of epochs for gating layer training')


    current_path = os.path.abspath(__file__)
    root_path = os.path.dirname(os.path.dirname(current_path))
    parser.add_argument(
        '--data_paths',
        nargs='+',
        default=[
            os.path.join(root_path, 'data_collector/continual_data/task0_data.pkl'),
            os.path.join(root_path, 'data_collector/continual_data/task1_data.pkl')
        ],
        help='Paths to the training datasets'
    )
    parser.add_argument(
        '--gating_data_path',
        default=[os.path.join(root_path, 'data_collector/continual_data/01_all_data.pkl')],
        help='Path to the gating dataset'
    )


    parser.add_argument('-cuda', default=0, type=int, help='Cuda device to use (-1 for none)')

    args = parser.parse_known_args()
    return args[0]




def load_multiple_datasets(file_paths):
    tasks_data = []
    for file_path in file_paths:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            print(f"Data from {file_path} has been loaded.")
            features = torch.Tensor(data['state']).float().unsqueeze(dim=-1)
            labels = torch.Tensor(data['action']).float()

            # 每个文件生成一个 TensorDataset
            dataset = TensorDataset(features, labels)

            # 创建 DataLoader，指定批量大小和是否进行数据随机化
            batch_size = 512
            shuffle = True
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
            tasks_data.append(dataloader)

    return tasks_data


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args['cuda'])

    wandb.init(project=args['project_name'], name=args['run_name'], config=args)
    print("WandB initialized successfully.")


    model = MoPE(args['layers'])


    current_path = os.path.abspath(__file__)
    root_path = os.path.dirname(os.path.dirname(current_path))


    tasks_data = load_multiple_datasets(args['data_paths'])
    print("All Tasks' Data has been loaded.")

    gating_data = load_multiple_datasets(args['gating_data_path'])
    gating_data_loader = gating_data[0]

    x = torch.Tensor()
    y = torch.Tensor()


    if args['cuda'] != -1:
        logger.info('Running with cuda (GPU n°{})'.format(args['cuda']))
        model.cuda()
        x = x.cuda()
        y = y.cuda()
    else:
        logger.warning('Running WITHOUT cuda')


    task_epochs = [args['task_0_epochs'], args['task_1_epochs']]
    for task_id, data in enumerate(tasks_data):

        model.new_task(args['sizes'])

        optimizer = torch.optim.Adam(model.parameters(task_id))
        for epoch in range(1, task_epochs[task_id] + 1):
            total_samples = 0
            total_loss = 0

            pbar = tqdm(total=len(data), bar_format='{l_bar}{r_bar}', dynamic_ncols=True)
            pbar.set_description(f'Task {task_id} - Epoch %d' % epoch)

            for inputs, labels in data:
                x.resize_(inputs.size()).copy_(inputs)
                y.resize_(labels.size()).copy_(labels)
                x = x.view(x.size(0), -1)

                predictions = model(Variable(x), task_id=task_id)
                loss_fn = torch.nn.MSELoss()
                loss = loss_fn(predictions, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.detach().cpu().item()
                total_samples += 1

                pbar.set_postfix(**{'loss': loss.detach().cpu().item()})
                pbar.update()


            avg_loss = total_loss / total_samples
            wandb.log({f'task_{task_id}_epoch_loss': avg_loss})
            sl.save_checkpoint(model_path, task_id, epoch, model, optimizer)
            pbar.close()




    sample_inputs, _ = next(iter(gating_data_loader))
    sample_inputs = sample_inputs.view(sample_inputs.size(0), -1).to(x.device)
    _ = model(sample_inputs, use_gating=True)




    all_parameters = list(model.parameters())



    optimizer_gating = torch.optim.Adam(all_parameters, lr=0.001)


    gating_weights_history = []

    for epoch in range(1, args['gating_epochs'] + 1):
        total_samples = 0
        total_loss = 0
        all_gating_weights = []

        pbar = tqdm(total=len(gating_data_loader), bar_format='{l_bar}{r_bar}', dynamic_ncols=True)
        pbar.set_description(f'Gating Layer - Epoch {epoch}')

        for inputs, labels in gating_data_loader:
            x.resize_(inputs.size()).copy_(inputs)
            y.resize_(labels.size()).copy_(labels)
            x = x.view(x.size(0), -1)


            combined_output, gating_weights = model(Variable(x), use_gating=True)


            batch_avg_gating_weights = gating_weights.mean(dim=0).detach().cpu().numpy()
            all_gating_weights.append(batch_avg_gating_weights)


            loss_fn = torch.nn.MSELoss()
            loss = loss_fn(combined_output, y)

            optimizer_gating.zero_grad()
            loss.backward()
            optimizer_gating.step()

            total_loss += loss.detach().cpu().item()
            total_samples += 1

            pbar.set_postfix(**{'loss': loss.detach().cpu().item()})
            pbar.update()

        pbar.close()


        avg_gating_weights = np.mean(all_gating_weights, axis=0)
        gating_weights_history.append(avg_gating_weights)


        log_data = {'epoch': epoch, 'gating_epoch_loss': total_loss / total_samples}
        for i, weight in enumerate(avg_gating_weights):
            log_data[f'avg_gating_weight_expert_{i}'] = weight

        wandb.log(log_data)


        sl.save_checkpoint(model_path, 'gating', epoch, model, optimizer_gating)

        pbar.close()


if __name__ == '__main__':
    main(vars(get_args()))

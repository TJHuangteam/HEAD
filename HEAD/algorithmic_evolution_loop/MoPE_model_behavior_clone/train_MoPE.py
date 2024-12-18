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
exp_name = 'Train_XCO_scenario'
train_method = '1'


writer = SummaryWriter(base_path + '/logs/' + exp_name + train_method)
model_path = base_path + '/checkpoint/' + exp_name + train_method
if not os.path.exists(model_path):
    os.makedirs(model_path)


def get_args():
    # whole
    parser = argparse.ArgumentParser(description='Mixture of Progressive Experts')
    parser.add_argument('-project_name', default="Train_XCO_scenario", type=str, help='WandB project name')
    parser.add_argument('-run_name', default="scenario", type=str, help='WandB run name')
    parser.add_argument('--layers', metavar='L', type=int, default=6, help='Number of layers per task')
    parser.add_argument('--sizes', dest='sizes', default=[259,1024,2048,1024,1024,256,2], nargs='+')
    parser.add_argument('--round_1_epochs', type=int, default=0, help='Number of epochs for round 1 training')
    parser.add_argument('--round_2_epochs', type=int, default=0, help='Number of epochs for round 2 training')
    parser.add_argument('--round_3_epochs', type=int, default=900, help='Number of epochs for round 3 training')

    current_path = os.path.abspath(__file__)
    root_path = os.path.dirname(os.path.dirname(current_path))
    parser.add_argument(
        '--data_paths',
        nargs='+',
        default=[
            os.path.join(root_path, 'data_collector/scenario/success_data_X.pkl'),
            os.path.join(root_path, 'data_collector/scenario/success_data_C.pkl'),
            os.path.join(root_path, 'data_collector/scenario/XCO_scenario_all_data.pkl')
        ],
        help='Paths to the training datasets'
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

            dataset = TensorDataset(features, labels)
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

    tasks_data = load_multiple_datasets(args['data_paths'])
    print("All Tasks' Data has been loaded.")

    x = torch.Tensor()
    y = torch.Tensor()

    if args['cuda'] != -1:
        logger.info('Running with cuda (GPU n°{})'.format(args['cuda']))
        model.cuda()
        x = x.cuda()
        y = y.cuda()
    else:
        logger.warning('Running WITHOUT cuda')


    model.new_task(args['sizes'])
    train_single_column_with_gating(model, tasks_data[0], task_id=0, epochs=args['task_0_epochs'], skip_columns=[0])
    model.new_task(args['sizes'])
    train_single_column_with_gating(model, tasks_data[1], task_id=1, epochs=args['task_1_epochs'], skip_columns=[1])
    model.new_task(args['sizes'])
    train_single_column_with_gating(model, tasks_data[2], task_id=2, epochs=args['task_2_epochs'], skip_columns=[2])


def train_single_column_with_gating(model, data_loader, task_id, epochs, skip_columns):
    model.freeze_columns(skip=skip_columns)


    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
    gating_weights_history = []

    for epoch in range(1, epochs + 1):
        total_loss = 0
        total_samples = 0
        all_gating_weights = []
        pbar = tqdm(total=len(data_loader), bar_format='{l_bar}{r_bar}', dynamic_ncols=True)
        pbar.set_description(f'Task {task_id} - Epoch %d' % epoch)

        for inputs, labels in data_loader:
            inputs, labels = inputs.to(next(model.parameters()).device), labels.to(next(model.parameters()).device)
            inputs = inputs.view(inputs.size(0), -1)

            combined_output, gating_weights = model(inputs, task_id=task_id)
            batch_avg_gating_weights = gating_weights.mean(dim=0).detach().cpu().numpy()
            all_gating_weights.append(batch_avg_gating_weights)

            loss_fn = torch.nn.MSELoss()
            loss = loss_fn(combined_output, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.detach().cpu().item()
            total_samples += 1

            pbar.set_postfix(**{'loss': loss.detach().cpu().item()})
            pbar.update()

            avg_gating_weights = np.mean(all_gating_weights, axis=0)
            gating_weights_history.append(avg_gating_weights)

        avg_loss = total_loss / total_samples
        log_data = {'epoch': epoch, 'epoch_loss': avg_loss}
        for i, weight in enumerate(avg_gating_weights):
            log_data[f'avg_gating_weight_expert_{i}'] = weight
        wandb.log(log_data)
        sl.save_checkpoint(model_path, task_id, epoch, model, optimizer)
        pbar.close()




if __name__ == '__main__':
    main(vars(get_args()))

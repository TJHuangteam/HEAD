import os
import argparse
import torch
import wandb
from stable_baselines3.common.monitor import Monitor
from metadrive.envs import MetaDriveEnv
import numpy as np

from ConstantLearning import MoPE


def get_args():

    parser = argparse.ArgumentParser(description='Mixture of Progressive Experts')
    parser.add_argument('-project_name', default="Test_whole_scenario", type=str, help='WandB project name')
    parser.add_argument('-run_name', default="C-6", type=str, help='WandB run name')
    parser.add_argument('--expert_num', dest='expert_num', type=int, default=3, help='Number of experts')
    parser.add_argument('--layers', metavar='L', type=int, default=6, help='Number of layers per task')
    parser.add_argument('--sizes', dest='sizes', default= [259,1024,2048,1024,1024,256,2], nargs='+')
    parser.add_argument('--model_path', default=os.path.join(os.getcwd(), 'results/checkpoint', 'Train_XCO_scenario/ckpt_task_2_epoch_900.pth'),
                        type=str, help='Full path to the model checkpoint')
    parser.add_argument('--eval_eps', type=int, default=1000, help='Number of evaluation episodes')
    parser.add_argument('--render', type=int, choices=[0, 1], default=0, help='Use 1 to enable window visualization, 0 to disable')
    args = parser.parse_known_args()
    return args[0]

def create_env(need_monitor=False):
    env = MetaDriveEnv(dict(map="C",
                            discrete_action=False,
                            horizon=1000,
                            use_render=False,
                            random_traffic=True,
                            random_spawn_lane_index=False,
                            num_scenarios=1,
                            start_seed=5,
                            traffic_density=0.2,
                            accident_prob=0,
                            use_lateral_reward=True,
                            log_level=50))
    if need_monitor:
        env = Monitor(env)
    return env

def eval_in_env(model, eval_env, eval_eps, render):
    print('Start to eval PNN model!')
    rewards = []
    ma_rewards = []
    total_nums = 0
    success_list = []
    reward_all = 0

    for i_ep in range(eval_eps):
        state, _ = eval_env.reset()
        eps_reward = 0.0
        ep_len = 0
        gating_weights_episode = []

        for i_step in range(int(1e6)):
            total_nums += 1
            state = torch.Tensor(state).float().unsqueeze(dim=0).to('cuda')

            action, gating_weights = model(state)
            action = action.squeeze(dim=0).cpu()
            gating_weights_episode.append(gating_weights.cpu().detach().numpy())

            action = action.detach().numpy()
            next_state, reward, done, termin, info = eval_env.step(action)

            if render:
                eval_env.render(mode="topdown", screen_record=True)

            state = next_state
            eps_reward += reward
            ep_len += 1

            if done or termin:
                print("episode_len", ep_len)
                break

        rewards.append(eps_reward)
        print(f"Episode:{i_ep + 1}/{eval_eps}, Reward:{eps_reward:.3f}")
        reward_all += eps_reward
        mean_reward_all = reward_all / total_nums
        mean_reward = eps_reward / ep_len
        print("Overall reward per step:", mean_reward_all)
        print("Current episode reward per step:", mean_reward)

        success_list.append(info['arrive_dest'])
        success_rate = sum(success_list) / len(success_list)
        print(f'Success rate: {success_rate * 100:.2f}%, ({sum(success_list)} / {len(success_list)})')
        print(f'Total steps: {total_nums}')

        if gating_weights_episode:
            gating_weights_avg = np.mean(gating_weights_episode, axis=0)
            gating_weights_avg = gating_weights_avg.flatten()

            log_data = {
                "episode": i_ep + 1,
                "mean_reward": mean_reward,
                "mean_reward_all": mean_reward_all,
                "success_rate": success_rate * 100,
                "episode_length": ep_len,
            }

            for idx, weight in enumerate(gating_weights_avg):
                log_data[f'gating_weight_{idx}'] = float(weight)

            wandb.log(log_data)
        else:
            wandb.log({
                "episode": i_ep + 1,
                "mean_reward": mean_reward,
                "mean_reward_all": mean_reward_all,
                "success_rate": success_rate * 100,
                "episode_length": ep_len
            })

    print('Evaluation completed.')
    wandb.finish()
    return rewards, ma_rewards

def main(args):
    wandb.init(project=args['project_name'], name=args['run_name'], config=args)
    eval_env = create_env()

    model = PNN(args['layers']).to('cuda')

    for i in range(args['expert_num']):
        model.new_task(args['sizes'])

    checkpoint = torch.load(args['model_path'])
    model_state_dict = checkpoint['model']
    model.load_state_dict(model_state_dict)
    print(model)

    eval_in_env(model, eval_env, args['eval_eps'], render=args['render'])

if __name__ == '__main__':
    main(vars(get_args()))


import torch
import tqdm
from torch.utils.tensorboard import SummaryWriter
from algorithm import DDPG
from algorithm.utils import agg_double_list
from simulator import Environment, Vehicle, Park, resetEnvEval, resetEnvVel
from network import ActorNetwork, CriticNetwork
import pickle

MAX_EPISODES = 10000
EPISODES_BEFORE_TRAIN = 256
EVAL_EPISODES = 10
EVAL_INTERVAL = 100

# max steps in each episode, prevent from running too long
MAX_STEPS = 3000  # None

MEMORY_CAPACITY = 1000000
BATCH_SIZE = 128
CRITIC_LOSS = "mse"
MAX_GRAD_NORM = None

TARGET_UPDATE_STEPS = 5
TARGET_TAU = 0.01

REWARD_DISCOUNTED_GAMMA = 0.99

EPSILON_START = 0.99
EPSILON_END = 0.05
EPSILON_DECAY = 500

DONE_PENALTY = None

RANDOM_SEED = 2021

USE_CUDA = False
DEVICE = torch.device("cuda:0") if USE_CUDA else torch.device("cpu")

SAVE_INTERVAL = 1000

LOAD_STATE_DICT = False


def run():
    env = Environment(Vehicle(0, 0, 0, 0, 0), Park(), reset_fn=resetEnvVel, maxSteps=MAX_STEPS, device=DEVICE)
    env.seed(RANDOM_SEED)
    env.reset()

    env_eval = Environment(Vehicle(0, 0, 0, 0, 0), Park(), reset_fn=resetEnvEval, maxSteps=MAX_STEPS, device=DEVICE)
    env_eval.seed(RANDOM_SEED)
    env_eval.reset()

    state_dim = env.observation_space
    action_dim = env.action_space

    actor_network = ActorNetwork(state_dim, action_dim, device=DEVICE).to(DEVICE)
    critic_network = CriticNetwork(state_dim, action_dim, device=DEVICE).to(DEVICE)
    if LOAD_STATE_DICT:
        actor_network.load_state_dict(torch.load('./checkpoint/actor_network.pth'))
        critic_network.load_state_dict(torch.load('./checkpoint/actor_network.pth'))

    writer: SummaryWriter = SummaryWriter(log_dir='./log')

    ddpg = DDPG(env=env, memory_capacity=MEMORY_CAPACITY,
                actor_network=actor_network, critic_network=critic_network,
                state_dim=state_dim, action_dim=action_dim,
                batch_size=BATCH_SIZE, max_steps=MAX_STEPS,
                done_penalty=DONE_PENALTY,
                target_update_steps=TARGET_UPDATE_STEPS, target_tau=TARGET_TAU,
                reward_gamma=REWARD_DISCOUNTED_GAMMA, critic_loss=CRITIC_LOSS,
                epsilon_start=EPSILON_START, epsilon_end=EPSILON_END,
                epsilon_decay=EPSILON_DECAY, max_grad_norm=MAX_GRAD_NORM,
                episodes_before_train=EPISODES_BEFORE_TRAIN, use_cuda=USE_CUDA)

    episodes = []
    eval_rewards = []
    fig, ax = env.InitFrame()
    with tqdm.tqdm(range(MAX_EPISODES)) as pbar:
        while ddpg.n_episodes < MAX_EPISODES:
            ddpg.interact()
            # env.render(fig, ax, mode='train')
            if ddpg.n_episodes >= EPISODES_BEFORE_TRAIN:
                ddpg.train()
            if ddpg.episode_done and ((ddpg.n_episodes + 1) % EVAL_INTERVAL == 0):
                rewards, _ = ddpg.evaluation(env_eval, EVAL_EPISODES)
                rewards_mu, rewards_std = agg_double_list(rewards)
                print("\nEpisode: %d, Average Reward: %.5f" % (ddpg.n_episodes + 1, rewards_mu))
                episodes.append(ddpg.n_episodes + 1)
                eval_rewards.append(rewards_mu)
                writer.add_scalar('rewards_mu_eval', rewards_mu, ddpg.n_episodes + 1)
                writer.add_scalar('rewards_std_eval', rewards_std, ddpg.n_episodes + 1)
            if ddpg.episode_done:
                writer.add_scalar('n_steps', ddpg.n_steps, ddpg.n_episodes + 1)
                writer.add_scalar('reward', ddpg.env.reward, ddpg.n_episodes + 1)
                pbar.update()

            if ((ddpg.n_episodes + 1) % SAVE_INTERVAL == 0):
                torch.save(actor_network.state_dict(), f"./checkpoint/actor_network_{ddpg.n_episodes}.pth")
                torch.save(critic_network.state_dict(), f"./checkpoint/critic_network_{ddpg.n_episodes}.pth")
                with open(f"./checkpoint/ddpg_{ddpg.n_episodes}.pth") as f:
                    pickle.dump(ddpg, f)


if __name__ == "__main__":
    import os

    print("Start training with resetEnvEval")
    os.makedirs('./output') if not os.path.exists('./output') else 0
    os.makedirs('./log') if not os.path.exists('./log') else 0
    os.makedirs('./checkpoint') if not os.path.exists('./checkpoint') else 0
    run()

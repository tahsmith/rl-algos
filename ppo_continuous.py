import gym.spaces
import torch
import numpy as np

device = torch.device('cpu')

env = gym.make('LunarLanderContinuous-v2')

action_size = env.action_space.shape[0]
state_size = env.observation_space.shape[0]
discount = 0.99
eps = 1e-1
learning_rate = 1e-2

network = torch.nn.Sequential(
    torch.nn.Linear(state_size, state_size),
    torch.nn.ELU(),
    torch.nn.Linear(state_size, action_size),
)

opt = torch.optim.Adam(params=network.parameters(), lr=learning_rate)


def pi(state, action):
    network(state)


def log_pi(state):
    return torch.nn.functional.log_softmax(network(state), dim=1)


def policy(state):
    network.eval()
    state = torch.from_numpy(state).to(device).float().unsqueeze(0)
    return torch.gaussian(network(state), 1).cpu().detach().squeeze().numpy()


def learn(episodes):
    network.train()

    all_returns = []
    all_states = []
    all_actions = []
    for episode in episodes:
        states, actions, rewards, next_states, dones = zip(*episode)
        rewards = [reward * (discount ** i)
                   for i, reward
                   in enumerate(rewards)]
        states = torch.tensor(states, device=device, dtype=torch.float)
        rewards = torch.tensor(rewards, device=device, dtype=torch.float)
        actions = torch.tensor(actions, device=device, dtype=torch.long)

        all_returns.append(rewards.flip(0).cumsum(dim=0).flip(0))

        all_states.append(states)
        all_actions.append(actions)

    final_returns = np.array(list(return_[-1] for return_ in all_returns))
    mean = final_returns.mean()
    std = final_returns.std() + 1e-3
    normalised_returns = [(return_ - mean) / std for return_ in all_returns]

    old_pis = [pi(states).gather(1, actions.reshape(-1, 1)).detach()
               for states, actions
               in zip(all_states, all_actions)]
    for i in range(4):
        opt.zero_grad()

        pis = [pi(states).gather(1, actions.reshape(-1, 1))
               for states, actions
               in zip(all_states, all_actions)]

        ratios = [x / y for x, y in zip(pis, old_pis)]
        clipped_ratios = [torch.min(r, torch.clamp(r, 1 - eps, 1 + eps))
                          for r in ratios]
        loss = -sum((x * y).sum()
                    for x, y in zip(clipped_ratios, normalised_returns))
        loss /= len(normalised_returns)
        loss.backward()
        opt.step()

        # old_pis = [prob.detach() for prob in pis]

    return loss


def episode(env: gym.Env):
    state = env.reset()
    done = False
    experiences = []
    reward_total = 0.
    while not done:
        action = policy(state)
        next_state, reward, done, info = env.step(action)
        experiences.append((state, action, reward, next_state, done))
        state = next_state
        reward_total += reward

    return reward_total, experiences


def train():
    running_rewards = [float('-inf') for _ in range(100)]
    episodes = []
    i = 0
    while True:
        reward, experiences = episode(env)
        episodes.append(experiences)
        running_rewards = running_rewards[1:] + [reward]
        print(f'\r{i + 1:06d} {reward:6.1f}', end='')
        if ((i + 1) % 100) == 0:
            print('\r{i:06d} {reward:6.1f}'.format(i=i + 1, reward=sum(
                running_rewards) / 100))

        if ((i + 1) % 4) == 0:
            learn(episodes)
            episodes = []
        i += 1


train()

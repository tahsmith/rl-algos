import gym.spaces
import numpy as np
import torch

device = torch.device("cpu")

env = gym.make("CartPole-v1")

assert isinstance(env.action_space, gym.spaces.Discrete)
action_size = env.action_space.n
state_size = env.observation_space.shape[0]
discount = 0.99

network = torch.nn.Sequential(
    torch.nn.Linear(state_size, state_size),
    torch.nn.ELU(),
    torch.nn.Linear(state_size, action_size),
)

opt = torch.optim.Adam(params=network.parameters(), lr=1e-2)


def pi(state):
    return torch.nn.functional.softmax(network(state), dim=1)


def log_pi(state):
    return torch.nn.functional.log_softmax(network(state), dim=1)


def policy(state):
    network.eval()
    state = torch.from_numpy(state).to(device).float().unsqueeze(0)
    return int(torch.multinomial(pi(state), 1).cpu().detach().squeeze().numpy())


def learn(episodes):
    network.train()

    log_pi_list = []
    returns = []

    opt.zero_grad()

    for episode in episodes:
        states, actions, rewards, next_states, dones = zip(*episode)
        states = torch.tensor(states, device=device, dtype=torch.float)
        rewards = [reward * (discount**i) for i, reward in enumerate(rewards)]
        rewards = torch.tensor(rewards, device=device, dtype=torch.float)
        actions = torch.tensor(actions, device=device, dtype=torch.long)

        returns.append(rewards.flip(0).cumsum(dim=0).flip(0))

        action_log_pi = log_pi(states).gather(1, actions.reshape(-1, 1))
        log_pi_list.append(action_log_pi)

    final_returns = np.array(list(return_[-1] for return_ in returns))
    mean = final_returns.mean()
    std = final_returns.std() + 1e-3
    loss = -sum((x * (y - mean) / std).sum() for x, y in zip(log_pi_list, returns))

    loss.backward()
    opt.step()

    return loss


def episode(env: gym.Env):
    state = env.reset()
    done = False
    experiences = []
    reward_total = 0.0
    while not done:
        action = policy(state)
        next_state, reward, done, info = env.step(action)
        experiences.append((state, action, reward, next_state, done))
        state = next_state
        reward_total += reward

    return reward_total, experiences


def train():
    running_rewards = [float("-inf") for _ in range(100)]
    episodes = []
    i = 0
    while True:
        reward, experiences = episode(env)
        episodes.append(experiences)
        running_rewards = running_rewards[1:] + [reward]
        print(f"\r{i + 1:06d} {reward:6.1f}", end="")
        if ((i + 1) % 100) == 0:
            print(
                "\r{i:06d} {reward:6.1f}".format(
                    i=i + 1, reward=sum(running_rewards) / 100
                )
            )

        if ((i + 1) % 10) == 0:
            learn(episodes)
            episodes = []
        i += 1


train()

import gym

import numpy as np


def main():
    env = gym.make('CliffWalking-v0')

    state_size = env.observation_space.n
    action_size = env.action_space.n

    alpha = 0.2
    alpha_decay = 0.99
    gamma = 0.99

    def policy(state):
        return np.random.randint(0, action_size)

    V = 0.0 * np.ones((state_size,), np.float64)

    def episode(V, alpha):
        state = env.reset()
        done = False
        while not done:
            action = policy(state)
            next_state, reward, done, info = env.step(action)
            td_err = (reward + gamma * V[next_state]) - V[state]
            V[state] += alpha * td_err
            state = next_state

        return V

    for i in range(int(1e5)):
        V_new = episode(V.copy(), alpha)
        alpha = alpha_decay * alpha
        diff = np.abs(V_new - V).mean()
        V = V_new
        if (i + 1) % 100 == 0:
            print(diff)


if __name__ == '__main__':
    main()

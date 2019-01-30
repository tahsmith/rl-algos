import gym
import numpy as np

from math_utils import clipped_exp_decay, constant
from tabular import train


def main():
    env = gym.make('CliffWalking-v0')

    state_size = env.observation_space.n
    action_size = env.action_space.n

    alpha = constant(0.01)

    epsilon_initial = 1.0
    epsilon_min = 1e-5
    epsilon_decay_rate = 0.2
    epsilon = clipped_exp_decay(epsilon_initial, epsilon_min,
                                epsilon_decay_rate)

    gamma = 1.0

    choices = np.arange(action_size)

    def policy_probs(Q, eps, state):
        probs = np.ones(action_size) * eps / action_size
        probs[np.argmax(Q[state])] = 1 - eps + eps / action_size
        return probs

    def policy(Q, eps, state):
        return np.random.choice(choices,
                                p=policy_probs(Q, eps, state))

    def sarsa_episode(i, Q):
        state = env.reset()
        action = policy(Q, epsilon(i), state)
        done = False
        return_ = 0.0
        while not done:
            next_state, reward, done, info = env.step(action)
            next_action = policy(Q, epsilon(i), next_state)
            td_err = (reward + gamma * Q[next_state, next_action]
                      - Q[state, action])
            Q[state, action] += alpha(i) * td_err
            state = next_state
            action = next_action
            return_ += reward

        return return_, Q

    # Optimistic initialisation
    Q_initial = 500 * np.ones((state_size, action_size), np.float64)

    train(sarsa_episode, Q_initial)


if __name__ == '__main__':
    main()

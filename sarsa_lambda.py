import gym

import numpy as np


class Params:
    pass


def epsilon_greedy(Q, epsilon, state):
    greedy = np.random.uniform() > epsilon
    if greedy:
        return np.argmax(Q[state])
    else:
        return np.random.choice(np.arange(len(Q[0])))


def main():
    env = gym.make('CliffWalking-v0')

    state_size = env.observation_space.n
    action_size = env.action_space.n

    alpha_decay = 0.99
    epsilon_decay = 0.99
    gamma = 0.99
    lambda_ = 0.5

    params = Params()
    params.epsilon = 1.0
    params.alpha = 0.2
    # Optimistic initialisation
    params.Q = 1e3 * np.ones((state_size, action_size), np.float64)
    params.e = np.zeros((state_size, action_size), np.float64)

    def policy(state):
        return epsilon_greedy(params.Q, params.epsilon, state)

    def episode(Q):
        state = env.reset()
        action = policy(state)
        done = False
        while not done:
            next_state, reward, done, info = env.step(action)
            next_action = policy(next_state)
            td_err = (reward + gamma * Q[next_state, next_action]
                      - Q[state, action])
            params.e[state, action] += 1
            Q += params.e * params.alpha * td_err
            params.e *= lambda_ * gamma
            state = next_state
            action = next_action

        return Q

    for i in range(int(1e5)):
        Q_new = episode(params.Q.copy())
        params.alpha *= alpha_decay
        params.epsilon *= epsilon_decay
        diff = np.abs(Q_new - params.Q).mean()
        params.Q = Q_new
        if (i + 1) % 100 == 0:
            print(diff)

            if diff < 1e-6:
                break


if __name__ == '__main__':
    main()

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

    alpha_decay = 1.0
    epsilon_decay = 1.0
    gamma = 0.99

    params = Params()
    params.epsilon = 0.1
    params.alpha = 0.2
    # Optimistic initialisation
    params.Q = 1e3 * np.ones((state_size, action_size), np.float64)

    def policy(state):
        return epsilon_greedy(params.Q, params.epsilon, state)

    def episode(Q):
        state = env.reset()
        action = policy(state)
        done = False
        return_ = 0.0
        while not done:
            next_state, reward, done, info = env.step(action)
            next_action = policy(next_state)
            td_err = (reward + gamma * Q[next_state, next_action]
                      - Q[state, action])
            Q[state, action] += params.alpha * td_err
            state = next_state
            action = next_action
            return_ += reward

        return Q, return_

    N = 100
    returns = [float('inf') for _ in range(N)]
    for i in range(int(1e6)):
        Q_new, return_ = episode(params.Q.copy())
        returns = [return_, *returns[0:-1]]
        diff = np.abs(Q_new - params.Q).mean()
        params.Q = Q_new
        params.alpha *= alpha_decay
        params.epsilon *= epsilon_decay
        if (i + 1) % N == 0:
            score = sum(returns) / N
            print(diff, score)

            if 1.0 < score:
                break


if __name__ == '__main__':
    main()
